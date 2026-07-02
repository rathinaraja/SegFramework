"""
modules/swinumamba/swinumamba_parts.py
---------------------------------------
Swin-UMamba building blocks.
Reference: Liu et al. 2024 (https://arxiv.org/abs/2402.03302)
           "Swin-UMamba: Mamba-based UNet with ImageNet-based Pretraining"

Key components:
  MambaBlock  : Simplified State Space Model block (pure PyTorch)
                No mamba-ssm package required.
                If mamba-ssm is installed it can be swapped in.
  VSS Block   : Visual State Space block (Mamba in 2D)
  DecoderBlock: Upsampling + skip + VSS

The selective scan is approximated using:
  - Depthwise conv for local mixing
  - Cumulative sum for state propagation
  - Selective gating for content-dependent filtering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Simplified Mamba / SSM block ──────────────────────────────────────────────

class SimplifiedMambaBlock(nn.Module):
    """
    Pure-PyTorch Mamba approximation.
    Captures the key design: selective gating + local conv + state mixing.

    For exact Mamba: pip install mamba-ssm causal-conv1d
    Then swap this block with the official mamba_ssm.Mamba.
    """
    def __init__(self, dim: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim     = dim
        self.d_inner = int(dim * expand)
        self.d_state = d_state

        self.norm    = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        # Local depthwise conv (mimics causal conv1d in Mamba)
        self.dw_conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.act = nn.SiLU()

        # SSM projections
        # x_proj outputs: d_state (for dt) + d_state (for B) + d_state (for C)
        # C must be d_state-dimensional to contract with state h (d_inner, d_state)
        self.x_proj = nn.Linear(self.d_inner, d_state * 3, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)

        # Initialise dt projection like Mamba paper
        dt_init_std = d_state ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # A initialisation (log-uniform)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("A_log", torch.log(A.expand(self.d_inner, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        # Input projection → split into x and z (gate)
        xz     = self.in_proj(x)                                  # (B, L, 2*d_inner)
        x_part = xz[..., :self.d_inner]                           # (B, L, d_inner)
        z      = xz[..., self.d_inner:]                           # (B, L, d_inner)

        # Local depthwise conv
        x_conv = self.dw_conv(
            x_part.transpose(1, 2)                                 # (B, d_inner, L)
        )[..., :L].transpose(1, 2)                                 # (B, L, d_inner)
        x_conv = self.act(x_conv)

        # SSM projections: dt (d_state) | B (d_state) | C (d_state)
        x_dbl = self.x_proj(x_conv)                               # (B, L, d_state*3)
        dt    = x_dbl[..., :self.d_state]                         # (B, L, d_state)
        B_ssm = x_dbl[..., self.d_state:2*self.d_state]          # (B, L, d_state)
        C     = x_dbl[..., 2*self.d_state:3*self.d_state]        # (B, L, d_state) ← d_state not d_inner

        # Selective dt (softplus)
        dt = F.softplus(self.dt_proj(dt))                         # (B, L, d_inner)

        # Simplified selective scan: exponential decay state update
        A = -torch.exp(self.A_log.float())                        # (d_inner, d_state)

        # Discretise A and B
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)                       # (B, L, d_inner, d_state)

        # Selective scan via chunked recurrence (approximate with cumsum for efficiency)
        # y_t = C * sum_{s<=t} dA^{t-s} * dB_s * x_s
        xu = x_conv.unsqueeze(-1) * dB                            # (B, L, d_inner, d_state)
        h  = torch.cumsum(xu * dA.mean(dim=1, keepdim=True), dim=1)
        y  = (h * C.unsqueeze(2)).sum(-1)                         # (B, L, d_inner)

        # Gate with z
        y = y * self.act(z)
        return self.out_proj(y) + residual


class VSSBlock(nn.Module):
    """
    Visual State Space Block: scans image tokens in 4 directions
    (row-left, row-right, col-top, col-bottom) then averages.
    """
    def __init__(self, dim: int, d_state: int = 16, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.ssm    = SimplifiedMambaBlock(dim, d_state=d_state)
        self.norm2  = nn.LayerNorm(dim)
        hidden      = int(dim * mlp_ratio)
        self.ffn    = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        tokens   = x.reshape(B, H, W, C)

        # 4-direction scan
        outs = []
        for seq in [
            tokens.reshape(B, H * W, C),                    # row-wise L→R
            tokens.flip(1).reshape(B, H * W, C),            # row-wise R→L
            tokens.transpose(1, 2).reshape(B, H * W, C),    # col-wise T→B
            tokens.transpose(1, 2).flip(1).reshape(B, H * W, C),  # col-wise B→T
        ]:
            outs.append(self.ssm(seq))

        # Average + reshape
        x = (outs[0] +
             outs[1].reshape(B, H, W, C).flip(1).reshape(B, H * W, C) +
             outs[2].reshape(B, W, H, C).transpose(1, 2).reshape(B, H * W, C) +
             outs[3].reshape(B, W, H, C).flip(1).transpose(1, 2).reshape(B, H * W, C)
             ) / 4.0 + x

        x = x + self.ffn(self.norm2(x))
        return x


# ── Patch merging / expanding ──────────────────────────────────────────────────

class PatchMerging(nn.Module):
    """2× spatial downsampling: merge 2×2 patches → double channels."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        x = x.reshape(B, H, W, C)
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]; x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]; x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1).reshape(B, -1, 4 * C)
        return self.reduction(self.norm(x)), H // 2, W // 2


class PatchExpanding(nn.Module):
    """2× spatial upsampling: expand each token into 2×2."""
    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm   = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, H: int, W: int):
        x = self.expand(x)                                    # (B, H*W, 2C)
        B, L, C = x.shape
        x = x.reshape(B, H, W, C).reshape(B, H, W, 2, 2, C // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, 2 * H, 2 * W, C // 4)
        return self.norm(x.reshape(B, -1, C // 4)), H * 2, W * 2