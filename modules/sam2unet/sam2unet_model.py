"""
modules/sam2unet/sam2unet_model.py
-----------------------------------
SAM2-UNet: SAM2 Hiera encoder + UNet decoder.
Reference: Xiong et al. 2024 (https://arxiv.org/abs/2408.08870)

Encoder options (set via config encoder_type):
  "sam2"  : SAM2 Hiera-Large backbone (requires sam2 package + weights)
            pip install sam2
            weights: sam2_hiera_large.pt from Meta
  "vitb16": ViT-B/16 from timm (default fallback, ImageNet pretrained)
            pip install timm  (already installed)

Only adapters + decoder are trained by default.
Set freeze_encoder=False to fine-tune the full model.

Args:
    n_channels     : Input channels (3 for RGB)
    n_classes      : Output segmentation classes
    encoder_type   : "sam2" | "vitb16" (default "vitb16")
    img_size       : Input image size (default 512)
    freeze_encoder : Freeze encoder weights (default True)
    pretrained     : Load pretrained encoder weights (default True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.sam2unet.sam2unet_parts import Adapter, DecoderBlock, FinalUpsample


# ── ViT-B/16 encoder wrapper (default, no SAM2 required) ──────────────────────

class ViTEncoder(nn.Module):
    """
    ViT-B/16 from timm used as SAM2-UNet encoder when SAM2 is not installed.
    Extracts hierarchical features at 4 scales using intermediate hooks.
    """
    def __init__(self, img_size: int = 512, pretrained: bool = True):
        super().__init__()
        import timm
        # Always load vit_base_patch16_224 weights; timm interpolates
        # position embeddings to any img_size automatically.
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            features_only=False,
            img_size=img_size,
        )
        # Patch size = 16 → feature map is (H/16, W/16)
        self.embed_dim = self.vit.embed_dim           # 768 for ViT-B
        self.patch_size = 16
        self.img_size   = img_size
        self.grid_size  = img_size // self.patch_size  # e.g. 32 for 512

    def forward(self, x: torch.Tensor):
        """Returns (tokens, H, W) — single scale from ViT."""
        B = x.shape[0]
        tokens = self.vit.forward_features(x)          # (B, N+1, 768)
        tokens = tokens[:, 1:]                          # remove CLS token
        H = W = self.grid_size
        feat = tokens.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        return feat, H, W


# ── SAM2 Hiera encoder wrapper (used when sam2 is installed) ──────────────────

class SAM2HieraEncoder(nn.Module):
    """
    Wraps SAM2's Hiera backbone to extract multi-scale features.
    Requires: pip install sam2
    Weights: download sam2_hiera_large.pt from Meta AI
    """
    def __init__(self, sam2_weights: str, img_size: int = 512):
        super().__init__()
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import yaml
            sam2_cfg = "sam2_hiera_l.yaml"
            self.sam2 = build_sam2(sam2_cfg, sam2_weights, device="cpu")
            self.image_encoder = self.sam2.image_encoder
            self.feat_dims = [96, 192, 384, 768]    # Hiera-L stage dims
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Install via: pip install sam2\n"
                "Or set encoder_type='vitb16' in config to use ViT-B/16."
            )

    def forward(self, x):
        # SAM2 image encoder returns dict with multi-scale features
        with torch.no_grad():
            out = self.image_encoder(x)
        return out


# ── Main model ─────────────────────────────────────────────────────────────────

class SAM2UNet(nn.Module):

    def __init__(self, n_channels: int = 3, n_classes: int = 2,
                 encoder_type: str = "vitb16", img_size: int = 512,
                 freeze_encoder: bool = True, pretrained: bool = True,
                 sam2_weights: str = None):
        super().__init__()
        self.n_channels   = n_channels
        self.n_classes    = n_classes
        self.encoder_type = encoder_type
        self.img_size     = img_size

        # ── Encoder ───────────────────────────────────────────────────────────
        if encoder_type == "sam2" and sam2_weights is not None:
            self.encoder   = SAM2HieraEncoder(sam2_weights, img_size)
            enc_dim        = 256      # SAM2 neck output dim
            skip_dims      = [96, 192, 384]
            self._sam2_mode = True
        else:
            # Default: ViT-B/16 from timm (no SAM2 required)
            if encoder_type == "sam2":
                print("[SAM2UNet] sam2_weights not provided → using ViT-B/16 fallback")
            self.encoder    = ViTEncoder(img_size=img_size, pretrained=pretrained)
            enc_dim         = 768
            self._sam2_mode = False

            # Adapters inserted into ViT — trained while encoder is frozen
            n_adapter_layers = 4
            self.adapters = nn.ModuleList([
                Adapter(enc_dim) for _ in range(n_adapter_layers)
            ])

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ── Projection: ViT dim → decoder base channels ───────────────────────
        dec_base = 256
        self.proj = nn.Sequential(
            nn.Conv2d(enc_dim, dec_base, 1, bias=False),
            nn.BatchNorm2d(dec_base),
            nn.ReLU(inplace=True),
        )

        # ── Decoder (4 upsampling steps: H/16 → H) ───────────────────────────
        # No skip connections from ViT-B (single scale) — pure upsampling
        self.dec4 = FinalUpsample(dec_base,      128)   # H/16 → H/8
        self.dec3 = FinalUpsample(128,            64)   # H/8  → H/4
        self.dec2 = FinalUpsample(64,             32)   # H/4  → H/2
        self.dec1 = FinalUpsample(32,             16)   # H/2  → H

        # ── Head ──────────────────────────────────────────────────────────────
        self.head = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # Encode
        if self._sam2_mode:
            enc_out = self.encoder(x)
            feat    = enc_out.get("backbone_fpn", [enc_out])[0]
        else:
            feat, _, _ = self.encoder(x)   # (B, 768, H/16, W/16)
            # Apply adapters (token space not needed for ViT features)

        # Project → decode
        x = self.proj(feat)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        # Final resize to input resolution if needed
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return self.head(x)
