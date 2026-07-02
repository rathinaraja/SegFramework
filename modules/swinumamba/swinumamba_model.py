"""
modules/swinumamba/swinumamba_model.py
---------------------------------------
Swin-UMamba: Swin Transformer encoder + Mamba-based UNet decoder.
Reference: Liu et al. 2024 (https://arxiv.org/abs/2402.03302)

Encoder: Swin-T/S/B from timm (ImageNet pretrained available)
Decoder: VSS (Visual State Space / Mamba) blocks with patch expanding

Args:
    n_channels    : Input channels (3 for RGB)
    n_classes     : Output segmentation classes
    img_size      : Input image size (224 recommended for Swin)
    swin_variant  : 'tiny' | 'small' | 'base' (default 'tiny')
    pretrained    : Load ImageNet pretrained Swin weights
    d_state       : Mamba state dimension (default 16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from modules.swinumamba.swinumamba_parts import VSSBlock, PatchExpanding


class SwinUMamba(nn.Module):

    SWIN_CONFIGS = {
        'tiny':  ('swin_tiny_patch4_window7_224',  [96,  192, 384, 768]),
        'small': ('swin_small_patch4_window7_224', [96,  192, 384, 768]),
        'base':  ('swin_base_patch4_window7_224',  [128, 256, 512, 1024]),
    }

    def __init__(self, n_channels: int = 3, n_classes: int = 2,
                 img_size: int = 224, swin_variant: str = 'tiny',
                 pretrained: bool = True, d_state: int = 16):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.img_size   = img_size

        # ── Swin Encoder ──────────────────────────────────────────────────────
        model_name, dims = self.SWIN_CONFIGS[swin_variant]
        # Pass img_size so timm interpolates Swin position embeddings
        # from 224 pretrained weights to any target resolution (e.g. 512).
        self.encoder = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
        )
        # Feature map spatial sizes for img_size=224:
        # stage0: H/4,  dims[0]
        # stage1: H/8,  dims[1]
        # stage2: H/16, dims[2]
        # stage3: H/32, dims[3]
        self.dims = dims

        # ── Mamba Decoder ─────────────────────────────────────────────────────
        # 4 decoder stages with PatchExpanding + VSSBlock
        self.concat_projs = nn.ModuleList()
        self.expand_layers = nn.ModuleList()
        self.mamba_blocks  = nn.ModuleList()

        dec_dims = list(reversed(dims))   # [768, 384, 192, 96]
        for i in range(len(dec_dims) - 1):
            in_dim   = dec_dims[i]
            skip_dim = dec_dims[i + 1]
            out_dim  = dec_dims[i + 1]

            self.expand_layers.append(PatchExpanding(in_dim))
            # After expanding: in_dim//2 channels; after cat skip: in_dim//2+skip_dim
            self.concat_projs.append(
                nn.Linear(in_dim // 2 + skip_dim, out_dim, bias=False)
            )
            self.mamba_blocks.append(VSSBlock(out_dim, d_state=d_state))

        # Final expanding ×4 (H/4 → H)
        self.final_expand = nn.Sequential(
            nn.ConvTranspose2d(dims[0], dims[0] // 2, 2, stride=2),
            nn.BatchNorm2d(dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dims[0] // 2, dims[0] // 4, 2, stride=2),
            nn.BatchNorm2d(dims[0] // 4),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(dims[0] // 4, n_classes, kernel_size=1)

    def to_bchw(self, f: torch.Tensor) -> torch.Tensor:
        """
        Normalise a Swin feature map to BCHW regardless of timm output format.
        timm Swin features_only returns NHWC (B, H, W, C) in newer versions.

        Reliable detection: check if the last dimension matches a known Swin
        channel size from self.dims. This handles cases where spatial dim > channel
        (e.g. feats[0] = (B, 128, 128, 96) where 96 < 128, breaking the old heuristic).
        """
        if f.ndim == 4 and f.shape[-1] in self.dims:
            return f.permute(0, 3, 1, 2).contiguous()   # NHWC → BCHW
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_orig, W_orig = x.shape[2], x.shape[3]

        # Swin encoder — normalise all features to BCHW.
        # Use self.dims (known Swin channel sizes) for reliable NHWC detection.
        feats = [self.to_bchw(f) for f in self.encoder(x)]   # [F0, F1, F2, F3]

        # Convert BCHW feature map to token sequence (B, H*W, C)
        def to_tokens(f):
            B, C, H, W = f.shape
            return f.flatten(2).transpose(1, 2), H, W

        # Start from deepest feature
        x_tok, H, W = to_tokens(feats[3])

        for i, (expand, proj, mamba, skip_feat) in enumerate(
                zip(self.expand_layers, self.concat_projs,
                    self.mamba_blocks, reversed(feats[:3]))):

            x_tok, H, W = expand(x_tok, H, W)
            skip_tok, sH, sW = to_tokens(skip_feat)

            # Align spatial if needed
            if H != sH or W != sW:
                x_tok = x_tok.reshape(x_tok.shape[0], H, W, -1)
                x_tok = F.interpolate(
                    x_tok.permute(0, 3, 1, 2), size=(sH, sW),
                    mode='bilinear', align_corners=False
                ).flatten(2).transpose(1, 2)
                H, W = sH, sW

            x_tok = proj(torch.cat([x_tok, skip_tok], dim=-1))
            x_tok = mamba(x_tok, H, W)

        # Reshape to 2D and final upsample
        B = x_tok.shape[0]
        x = x_tok.reshape(B, H, W, -1).permute(0, 3, 1, 2)   # (B, C, H, W)
        x = self.final_expand(x)

        if x.shape[2:] != (H_orig, W_orig):
            x = F.interpolate(x, size=(H_orig, W_orig),
                              mode='bilinear', align_corners=False)

        return self.head(x)