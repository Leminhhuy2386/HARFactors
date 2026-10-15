"""Vortex Network: a sequence model whose only memory is accumulated winding angle.

Each unit projects its input onto a learned plane, v_t = W x_t + b (b places the unit's center),
and integrates the signed turning of the trajectory around that center:

    dtheta_t = atan2(v_{t-1} x v_t, v_{t-1} . v_t)          (in (-pi, pi])
    W_t = sum_{s<=t} g_s dtheta_s,   A_t = sum_{s<=t} g_s |dtheta_s|

W_t (net winding) and A_t (total turning) depend on where the trajectory goes, not on how fast,
and are unchanged by scaling the trajectory about the center. g_s = rho / (rho + eps) with
rho = |v_{s-1}| |v_s| mutes the ill-defined angle when the path passes through the center.
The state per unit is the previous point and two running sums; training uses cumsum.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def turning(prev: torch.Tensor, cur: torch.Tensor, eps: float) -> torch.Tensor:
    """Gated signed angle from prev to cur around the origin, (..., 2) -> (...)."""
    cross = prev[..., 0] * cur[..., 1] - prev[..., 1] * cur[..., 0]
    dot = (prev * cur).sum(dim=-1)
    rho = torch.sqrt(cross * cross + dot * dot)  # = |prev| |cur|
    safe_dot = torch.where(rho == 0, torch.ones_like(dot), dot)  # atan2 has no gradient at (0, 0)
    return torch.atan2(cross, safe_dot) * (rho / (rho + eps))


class VortexBlock(nn.Module):
    """norm -> U learned planes/centers -> [net winding, total turning, log radius] -> mixing -> GELU -> residual."""

    def __init__(self, d_model: int, n_units: int = 32, eps: float = 1e-2) -> None:
        super().__init__()
        self.n_units = n_units
        self.eps = eps
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 2 * n_units)
        self.mix_out = nn.Linear(3 * n_units, d_model)

    def _features(self, winding, total, v):
        log_r = 0.5 * torch.log(v.pow(2).sum(dim=-1) + self.eps)
        return torch.cat([winding / torch.pi, total / torch.pi, log_r], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        v = self.proj(self.norm(x)).unflatten(-1, (self.n_units, 2))  # (B, L, U, 2)
        d = turning(v[:, :-1], v[:, 1:], self.eps)  # (B, L-1, U)
        d = F.pad(d, (0, 0, 1, 0))  # no turning at t = 0
        feats = self._features(d.cumsum(dim=1), d.abs().cumsum(dim=1), v)
        return x + F.gelu(self.mix_out(feats))

    def step(self, x_t, state):
        """x_t (B, D); state = (v_prev or None, winding, total) -> (output, new state)."""
        v_prev, winding, total = state
        v = self.proj(self.norm(x_t)).unflatten(-1, (self.n_units, 2))
        if v_prev is not None:
            d = turning(v_prev, v, self.eps)
            winding, total = winding + d, total + d.abs()
        return x_t + F.gelu(self.mix_out(self._features(winding, total, v))), (v, winding, total)


class VortexNet(nn.Module):
    def __init__(self, input_channels: int, window_length: int, num_classes: int, *,
                 d_model: int = 32, n_units: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_channels, d_model)
        self.blocks = nn.ModuleList(VortexBlock(d_model, n_units) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = self.encoder(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h.mean(dim=1)))

    @torch.no_grad()
    def forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        zeros = lambda b: torch.zeros(batch, b.n_units, device=x.device, dtype=x.dtype)  # noqa: E731
        states = [(None, zeros(b), zeros(b)) for b in self.blocks]
        pooled = 0.0
        for t in range(length):
            h = self.encoder(x[:, :, t])
            for i, block in enumerate(self.blocks):
                h, states[i] = block.step(h, states[i])
            pooled = pooled + h
        return self.head(self.norm(pooled / length))
