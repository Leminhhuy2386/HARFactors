"""TropicalNet: a sequence model whose only memory is a max-plus (tropical) recurrence.

Each memory unit is a decaying peak-hold of its input v (and a decaying valley-hold twin):

    h_t = max(h_{t-1} - lambda, v_t)          g_t = min(g_{t-1} + lambda, v_t)

i.e. the linear recurrence h_t = a h_{t-1} + v_t of a state-space model with (+, x) replaced by
(max, +). A peak of height P above the current input is remembered for P / lambda steps, so the
memory length depends on amplitude as well as on time; h - g is a running peak-to-peak range.
The memory has no multiplications: one subtraction and one comparison per unit and sample.

Training is parallel: the maps h -> max(h - a, c) compose associatively,

    (a1, c1) then (a2, c2) = (a1 + a2, max(c1 - a2, c2)),

so a Hillis-Steele scan gives every h_t in log2(L) rounds (h_{-1} = -inf, so h_t is the c part).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.hysteresis import inverse_softplus


def maxplus_scan(v: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """h_t = max(h_{t-1} - lam_t, v_t) along the last axis, h_{-1} = -inf.

    v: (..., L); lam: broadcastable to v, >= 0 (constant or per step).
    """
    a = lam.expand_as(v) if lam.shape != v.shape else lam
    c = v
    length = v.shape[-1]
    d = 1
    while d < length:
        a_t = a[..., d:]
        c = torch.cat([c[..., :d], torch.maximum(c[..., :-d] - a_t, c[..., d:])], dim=-1)
        if 2 * d < length:  # the last round does not need the accumulated shifts
            a = torch.cat([a[..., :d], a[..., :-d] + a_t], dim=-1)
        d *= 2
    return c


class TropicalBlock(nn.Module):
    """norm -> mixing -> K peak-holds and K valley-holds per channel -> mixing -> GELU -> residual."""

    def __init__(self, d_model: int, decays: tuple[float, ...] = (0.01, 0.05, 0.25)) -> None:
        super().__init__()
        k = len(decays)
        self.norm = nn.LayerNorm(d_model)
        self.mix_in = nn.Linear(d_model, d_model)
        lam0 = torch.tensor(decays).repeat(d_model, 1)
        lam0 = lam0 * torch.exp(0.3 * torch.randn_like(lam0))
        self.rho = nn.Parameter(inverse_softplus(lam0))  # (D, K)
        self.mix_out = nn.Linear(d_model * (2 * k + 1), d_model)

    def decays(self) -> torch.Tensor:
        return F.softplus(self.rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        v = self.mix_in(self.norm(x)).transpose(1, 2).unsqueeze(2)  # (B, D, 1, L)
        lam = self.decays().unsqueeze(-1)  # (D, K, 1)
        peak = maxplus_scan(v.expand(-1, -1, lam.shape[1], -1), lam)
        valley = -maxplus_scan(-v.expand(-1, -1, lam.shape[1], -1), lam)
        feats = torch.cat([v, peak, valley], dim=2).flatten(1, 2)  # (B, D*(2K+1), L)
        return x + F.gelu(self.mix_out(feats.transpose(1, 2)))

    def step(self, x_t, state):
        """x_t (B, D); state = (peak, valley), each (B, D, K) or None at t = 0."""
        v = self.mix_in(self.norm(x_t)).unsqueeze(-1)  # (B, D, 1)
        lam = self.decays()
        if state is None:
            peak = valley = v.expand(-1, -1, lam.shape[1])
        else:
            peak = torch.maximum(state[0] - lam, v)
            valley = torch.minimum(state[1] + lam, v)
        feats = torch.cat([v, peak, valley], dim=-1).flatten(1)
        return x_t + F.gelu(self.mix_out(feats)), (peak, valley)


class TropicalNet(nn.Module):
    def __init__(self, input_channels: int, window_length: int, num_classes: int, *,
                 d_model: int = 32, num_layers: int = 3,
                 decays: tuple[float, ...] = (0.01, 0.05, 0.25)) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_channels, d_model)
        self.blocks = nn.ModuleList(TropicalBlock(d_model, decays) for _ in range(num_layers))
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
        states = [None] * len(self.blocks)
        total = 0.0
        for t in range(length):
            h = self.encoder(x[:, :, t])
            for i, block in enumerate(self.blocks):
                h, states[i] = block.step(h, states[i])
            total = total + h
        return self.head(self.norm(total / length))
