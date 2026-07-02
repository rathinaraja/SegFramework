"""
modules/sambaunet/sambaunet_model.py
--------------------------------------
SAMba-UNet: SAM2-style ViT encoder + Mamba bottleneck + ConvTranspose decoder.
Reference: SAMba-UNet (2025) — hybrid SAM2 + Mamba for medical segmentation.

Architecture:
  Encoder  : ViT-B/16 (frozen, ImageNet pretrained via timm)
             Extracts rich patch tokens at H/16 × W/16 resolution
  Bottleneck: VSSBlock (Mamba) applied at L=1024 — memory safe
  Decoder  : Standard ConvTranspose2d blocks for spatial upsampling
             (Mamba NOT used in decoder to avoid L explosion:
              1024→4096→16384 creates dA tensors of 2GB+)
  Head     : 1×1 Conv for segmentation

Design rationale:
  Applying Mamba at large spatial sizes (L>4096) causes OOM because
  dA = (B, L, d_inner, d_state) grows cubically with resolution.
  Bottleneck-only Mamba captures global context efficiently at L=1024,
  while the lightweight conv decoder handles spatial detail recovery.

Args:
    n_channels   : Input channels (default 3)
    n_classes    : Output segmentation classes (default 2)
    img_size     : Input image size (default 512)
    encoder_type : 'vitb16' — ViT-B/16 from timm (pretrained ImageNet)
    pretrained   : Load ImageNet pretrained ViT weights
    d_state      : Mamba SSM state dimension (default 16)
    depth        : Decoder depth — number of upsampling stages (default 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.swinumamba.swinumamba_parts import VSSBlock


# ── Decoder block (ConvTranspose — no Mamba, memory efficient) ────────────────

class DecoderBlock(nn.Module):
    """ConvTranspose2d ×2 upsample → BN → ReLU → Conv → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Main model ─────────────────────────────────────────────────────────────────

class SAMbaUNet(nn.Module):

    def __init__(self, n_channels: int = 3, n_classes: int = 2,
                 img_size: int = 512, encoder_type: str = 'vitb16',
                 pretrained: bool = True, d_state: int = 16,
                 depth: int = 4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.img_size   = img_size

        # ── ViT-B/16 Encoder (frozen) ─────────────────────────────────────────
        import timm
        # Always load vit_base_patch16_224 weights; timm interpolates
        # position embeddings to img_size automatically.
        self.encoder    = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            img_size=img_size,
        )
        self.enc_dim    = 768                    # ViT-B embed dim
        self.patch_size = 16
        self.grid_size  = img_size // self.patch_size   # e.g. 32 for 512

        # Freeze encoder — only bottleneck + decoder train
        for p in self.encoder.parameters():
            p.requires_grad = False

        # ── Bottleneck projection: 768 → bottle_dim ───────────────────────────
        bottle_dim = 512
        self.bottle_proj = nn.Sequential(
            nn.Linear(self.enc_dim, bottle_dim, bias=False),
            nn.LayerNorm(bottle_dim),
        )

        # ── Mamba bottleneck (at L=grid_size² — memory safe) ─────────────────
        # For 512×512: L = 32×32 = 1024
        # dA tensor: (B=2, L=1024, d_inner=1024, d_state=16) = ~128MB ← OK
        self.bottle_mamba = nn.ModuleList([
            VSSBlock(bottle_dim, d_state=d_state) for _ in range(2)
        ])

        # ── Standard ConvTranspose decoder (memory-efficient) ─────────────────
        # Decoder channel progression: 512 → 256 → 128 → 64 → 32
        dec_dims = [bottle_dim]
        for _ in range(depth):
            dec_dims.append(max(dec_dims[-1] // 2, 32))
        # e.g. depth=4: [512, 256, 128, 64, 32]

        self.decoder = nn.ModuleList()
        for i in range(depth):
            self.decoder.append(DecoderBlock(dec_dims[i], dec_dims[i + 1]))

        self.head = nn.Conv2d(dec_dims[-1], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H_orig, W_orig = x.shape
        H = W = self.grid_size   # e.g. 32

        # ── ViT encode ────────────────────────────────────────────────────────
        with torch.no_grad():
            tokens = self.encoder.forward_features(x)   # (B, 1+H*W, 768)
        tokens = tokens[:, 1:]                           # drop CLS → (B, H*W, 768)

        # ── Bottleneck projection + Mamba ─────────────────────────────────────
        tokens = self.bottle_proj(tokens)                # (B, H*W, 512)
        for blk in self.bottle_mamba:
            tokens = blk(tokens, H, W)                   # (B, H*W, 512)

        # ── Reshape to spatial ────────────────────────────────────────────────
        x = tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2)   # (B, 512, H, W)

        # ── ConvTranspose decoder ─────────────────────────────────────────────
        # Each DecoderBlock doubles spatial resolution:
        # H/16 → H/8 → H/4 → H/2 → H
        for blk in self.decoder:
            x = blk(x)

        # Final resize to original resolution if needed
        if x.shape[2:] != (H_orig, W_orig):
            x = F.interpolate(x, size=(H_orig, W_orig),
                              mode='bilinear', align_corners=False)

        return self.head(x)