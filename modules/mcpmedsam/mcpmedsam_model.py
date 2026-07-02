"""
modules/mcpmedsam/mcpmedsam_model.py
--------------------------------------
MCP-MedSAM: Lightweight Medical SAM — trainable on a single GPU in one day.
Reference: MCP-MedSAM (MELBA 2025)
           "A Powerful Lightweight Medical Segment Anything Model
            Trained with a Single GPU in Just One Day"

Architecture:
  Encoder: EfficientViT-style lightweight ViT (replaces heavy SAM ViT-H)
           Uses depthwise separable convolutions + small attention heads
  Neck:    FPN-style multi-scale feature pyramid
  Decoder: SAM-style mask decoder (lightweight MLP + upsampling)

Design philosophy:
  - Train from scratch on medical images — no SAM weights needed
  - ~10× fewer parameters than MedSAM (ViT-H based)
  - Suitable for single GPU training

Args:
    n_channels  : Input channels (3 for RGB)
    n_classes   : Output segmentation classes
    img_size    : Input image size (default 512)
    embed_dim   : Base embedding dimension (default 192)
    depth       : Transformer depth (default 6)
    num_heads   : Attention heads (default 6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Lightweight attention block ────────────────────────────────────────────────

class EfficientAttention(nn.Module):
    """
    Efficient multi-head attention with depthwise conv for local mixing.
    Linear complexity via key-value reduction.
    """
    def __init__(self, dim: int, num_heads: int = 6,
                 sr_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.sr_ratio  = sr_ratio

        self.q   = nn.Linear(dim, dim)
        self.kv  = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x2d = x.transpose(1, 2).reshape(B, C, H, W)
            x2d = self.sr(x2d).reshape(B, C, -1).transpose(1, 2)
            x2d = self.norm(x2d)
        else:
            x2d = x
        kv = self.kv(x2d).reshape(B, -1, 2, self.num_heads,
                                   self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = self.drop((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)


class MixFFN(nn.Module):
    """Mix-FFN: Linear → DWConv → GELU → Linear."""
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden      = int(dim * mlp_ratio)
        self.fc1    = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act    = nn.GELU()
        self.fc2    = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        B, N, C = x.shape
        x = self.dwconv(x.transpose(1, 2).reshape(B, C, H, W))
        x = self.act(x.flatten(2).transpose(1, 2))
        return self.fc2(x)


class LightTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = EfficientAttention(dim, num_heads, sr_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = MixFFN(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


# ── Lightweight encoder ────────────────────────────────────────────────────────

class LightEncoder(nn.Module):
    """
    4-stage hierarchical encoder (SegFormer / MiT-style but lighter).
    Outputs 4 feature maps at H/4, H/8, H/16, H/32.
    """
    def __init__(self, in_chans: int = 3, embed_dim: int = 192,
                 depth: int = 6, num_heads: int = 6):
        super().__init__()
        # Stage channel sizes
        dims = [embed_dim // 4, embed_dim // 2, embed_dim, embed_dim * 2]
        sr   = [8, 4, 2, 1]

        self.patch_embeds = nn.ModuleList()
        self.layers       = nn.ModuleList()
        self.norms        = nn.ModuleList()

        sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        in_ch = in_chans

        for i in range(4):
            self.patch_embeds.append(nn.Sequential(
                nn.Conv2d(in_ch, dims[i], sizes[i], stride=strides[i],
                          padding=sizes[i] // 2, bias=False),
                nn.BatchNorm2d(dims[i]),
            ))
            n_blk = max(1, depth // 4)
            heads = max(1, min(num_heads, dims[i] // 32))
            self.layers.append(nn.ModuleList([
                LightTransformerBlock(dims[i], heads, sr[i])
                for _ in range(n_blk)
            ]))
            self.norms.append(nn.LayerNorm(dims[i]))
            in_ch = dims[i]

        self.dims = dims

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        outs = []
        for pe, blocks, norm in zip(self.patch_embeds, self.layers, self.norms):
            x = pe(x)
            B_, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x).reshape(B_, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x)
        return outs


# ── All-MLP Decoder (lightweight SAM-style) ───────────────────────────────────

class LightDecoder(nn.Module):
    """Unify multi-scale features → segment."""
    def __init__(self, dims: list, decoder_dim: int, n_classes: int):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Linear(d, decoder_dim) for d in dims
        ])
        self.fuse  = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.head  = nn.Conv2d(decoder_dim, n_classes, 1)

    def forward(self, feats: list, H: int, W: int) -> torch.Tensor:
        projected = []
        for feat, proj in zip(feats, self.projs):
            B, C, fH, fW = feat.shape
            x = proj(feat.flatten(2).transpose(1, 2))
            x = x.transpose(1, 2).reshape(B, -1, fH, fW)
            x = F.interpolate(x, size=(H, W), mode='bilinear',
                              align_corners=False)
            projected.append(x)
        return self.head(self.fuse(torch.cat(projected, dim=1)))


# ── Main model ─────────────────────────────────────────────────────────────────

class MCPMedSAM(nn.Module):

    def __init__(self, n_channels: int = 3, n_classes: int = 2,
                 img_size: int = 512, embed_dim: int = 192,
                 depth: int = 6, num_heads: int = 6, decoder_dim: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.img_size   = img_size

        self.encoder = LightEncoder(n_channels, embed_dim, depth, num_heads)
        target_H     = img_size // 4      # output at H/4

        self.decoder = LightDecoder(
            dims        = self.encoder.dims,
            decoder_dim = decoder_dim,
            n_classes   = n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        feats = self.encoder(x)               # 4 multi-scale feature maps
        out   = self.decoder(feats, H // 4, W // 4)
        return F.interpolate(out, size=(H, W),
                             mode='bilinear', align_corners=False)
