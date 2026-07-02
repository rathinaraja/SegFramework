"""
modules/kongnet/kongnet_parts.py
---------------------------------
KongNet building blocks.
Reference: 2025 MIDOG / MONKEY / PUMA Challenge winner
           (1st place MIDOG 2025, Top-3 PUMA 2025)

Key idea: SCSE attention (Spatial + Channel Squeeze-Excitation)
          on top of ResNet-style encoder-decoder.
          Designed for H&E histopathology — no pretraining needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SCSE Attention ─────────────────────────────────────────────────────────────

class ChannelSE(nn.Module):
    """Channel Squeeze-and-Excitation."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class SpatialSE(nn.Module):
    """Spatial Squeeze-and-Excitation."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))


class SCSE(nn.Module):
    """Concurrent Spatial + Channel SE — output = cSE(x) + sSE(x)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.cse = ChannelSE(channels, reduction)
        self.sse = SpatialSE(channels)
    def forward(self, x):
        return self.cse(x) + self.sse(x)


# ── Residual block ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Conv→BN→ReLU→Conv→BN + SCSE attention + residual."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.scse = SCSE(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x):
        return self.relu(self.scse(self.conv(x)) + self.skip(x))


# ── Encoder / Decoder ──────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """n ResBlocks → returns (downsampled, skip)."""
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 2):
        super().__init__()
        blocks = [ResBlock(in_ch, out_ch)]
        for _ in range(n_blocks - 1):
            blocks.append(ResBlock(out_ch, out_ch))
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """TransposeConv upsample → concat skip → ResBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ResBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))
