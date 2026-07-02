"""
modules/kongnet/kongnet_model.py
---------------------------------
KongNet for binary semantic segmentation.
Adapted from: 2025 MIDOG/MONKEY/PUMA challenge winner.

Architecture:
  ResNet-style encoder with SCSE attention at every block
  U-shaped decoder with skip connections
  Single segmentation head (binary mode)

No pretrained weights required.

Args:
    n_channels   : Input channels (3 for RGB H&E)
    n_classes    : Output classes (2 for binary)
    base_features: Base channel count (default 64)
    depth        : Encoder/decoder depth (default 4)
"""

import torch
import torch.nn as nn
from modules.kongnet.kongnet_parts import EncoderBlock, DecoderBlock, ResBlock


class KongNet(nn.Module):

    def __init__(self, n_channels: int = 3, n_classes: int = 2,
                 base_features: int = 64, depth: int = 4):
        super().__init__()
        self.n_channels    = n_channels
        self.n_classes     = n_classes
        feats = [min(base_features * (2 ** i), 512) for i in range(depth + 1)]

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = ResBlock(n_channels, feats[0])

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        for i in range(depth - 1):
            self.encoders.append(EncoderBlock(feats[i], feats[i + 1]))

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(feats[depth - 1], feats[depth]),
            ResBlock(feats[depth],     feats[depth]),
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch   = feats[i + 1]
            skip_ch = feats[i]
            out_ch  = feats[i]
            self.decoders.append(DecoderBlock(in_ch, skip_ch, out_ch))

        # ── Head ──────────────────────────────────────────────────────────────
        self.head = nn.Conv2d(feats[0], n_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encode ────────────────────────────────────────────────────────────
        x         = self.stem(x)          # (B, 64, H, W)
        stem_skip = x                     # saved for last decoder stage

        skips = []
        for enc in self.encoders:
            x, skip = enc(x)              # enc returns (downsampled, skip)
            skips.append(skip)
        # skips = [enc0_skip, enc1_skip, enc2_skip]
        #       = [(B,128,H,W), (B,256,H/2,W/2), (B,512,H/4,W/4)]

        # ── Bottleneck ────────────────────────────────────────────────────────
        x = self.bottleneck(x)            # (B, 512, H/16, W/16)

        # ── Decode ────────────────────────────────────────────────────────────
        # pair each decoder with its matching skip in order deepest→shallowest
        all_skips = list(reversed(skips)) + [stem_skip]
        # = [enc2_skip, enc1_skip, enc0_skip, stem_skip]
        # = [(B,512,H/4), (B,256,H/2), (B,128,H), (B,64,H)]

        for dec, skip in zip(self.decoders, all_skips):
            x = dec(x, skip)

        return self.head(x)
