"""
modules/sam2unet/sam2unet_parts.py
-----------------------------------
SAM2-UNet building blocks.
Reference: Xiong et al. 2024 (https://arxiv.org/abs/2408.08870)
           "SAM2-UNet: Segment Anything 2 Makes Strong Encoder
            for Natural and Medical Image Segmentation"

Architecture:
  Encoder: SAM2 Hiera backbone (or ViT-B from timm as fallback)
           with lightweight adapters for parameter-efficient fine-tuning
  Decoder: Classic U-shaped decoder with skip connections

Adapter:
  Small bottleneck MLP inserted into each encoder block.
  Only adapters + decoder are trained; encoder stays frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Adapter ────────────────────────────────────────────────────────────────────

class Adapter(nn.Module):
    """
    Lightweight bottleneck adapter inserted into frozen encoder layers.
    Input → down-project → GELU → up-project → residual add.
    Adds only ~0.5% of encoder parameters.
    """
    def __init__(self, dim: int, bottleneck: int = None):
        super().__init__()
        bottleneck = bottleneck or max(dim // 8, 16)
        self.down  = nn.Linear(dim, bottleneck)
        self.act   = nn.GELU()
        self.up    = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


# ── Decoder blocks ─────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample × 2 → concat skip → 2× ConvBNReLU."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch // 2 + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FinalUpsample(nn.Module):
    """Upsample without skip — used for the last stage."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBNReLU(out_ch, out_ch)
    def forward(self, x): return self.conv(self.up(x))
