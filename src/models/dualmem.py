"""DualMemNet: each block carries two new memories side by side.

- quantile trackers (QuantileNet): where the input's values sit, as running quantiles;
- play operators with selective widths (HysteresisNet): the sequence of extrema it went through.

Results on UTD-MHAD motivated the combination: the quantile memory gave the largest gain over the
same network without memory (+6.7), the hysteresis memory the best robustness to execution
speed; the selective widths and the relative-position clock were the HysteresisNet changes chosen
on held-out subjects. With path=True the trackers step by |du| instead of per sample, so the
whole network is rate-independent apart from the mean over time.

Block: norm -> mix -> [u, play(u; r_t) x Kh, quantiles(u) x Kq] -> mix -> GELU -> residual.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.hysteresis import inverse_softplus, play_scan, play_step
from src.models.quantile import quantile_step, quantile_track


class DualMemoryBlock(nn.Module):
    def __init__(self, d_model: int, *, widths=(0.1, 0.4, 1.2),
                 trackers=((0.1, 0.05), (0.5, 0.05), (0.9, 0.05)), path: bool = False,
                 temperature: float = 0.1) -> None:
        super().__init__()
        kh, kq = len(widths), len(trackers)
        self.path = path
        self.temperature = temperature
        self.norm = nn.LayerNorm(d_model)
        self.mix_in = nn.Linear(d_model, d_model)
        self.select = nn.Linear(d_model, d_model)  # selective widths, zero-initialised
        nn.init.zeros_(self.select.weight)
        nn.init.zeros_(self.select.bias)
        r0 = torch.tensor(widths).repeat(d_model, 1)
        self.rho = nn.Parameter(inverse_softplus(r0 * torch.exp(0.3 * torch.randn_like(r0))))  # (D, Kh)
        tau0 = torch.tensor([t for t, _ in trackers]).repeat(d_model, 1)
        eta0 = torch.tensor([e for _, e in trackers]).repeat(d_model, 1)
        self.tau_logit = nn.Parameter(torch.log(tau0) - torch.log1p(-tau0))  # (D, Kq)
        self.eta_raw = nn.Parameter(inverse_softplus(eta0 * torch.exp(0.3 * torch.randn_like(eta0))))
        self.mix_out = nn.Linear(d_model * (1 + kh + kq), d_model)

    def _widths(self, xn):  # (..., D) -> (..., D, Kh)
        return F.softplus(self.rho + self.select(xn).unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        xn = self.norm(x)
        u = self.mix_in(xn)  # (B, L, D)
        y = play_scan(u.transpose(1, 2).unsqueeze(2), self._widths(xn).permute(0, 2, 3, 1))  # (B, D, Kh, L)
        q = quantile_track(u, torch.sigmoid(self.tau_logit), F.softplus(self.eta_raw), self.temperature, self.path)
        feats = torch.cat([u.unsqueeze(-1), y.permute(0, 3, 1, 2), q], dim=-1).flatten(2)  # (B, L, D*(1+Kh+Kq))
        return x + F.gelu(self.mix_out(feats))

    def step(self, x_t, state, hard: bool = False):
        """x_t (B, D); state = (play state (B, D, Kh), q or None, prev)."""
        y, q, prev = state
        xn = self.norm(x_t)
        u = self.mix_in(xn)
        y = play_step(u.unsqueeze(-1), y, self._widths(xn))
        q, prev = quantile_step(u, q, prev, torch.sigmoid(self.tau_logit), F.softplus(self.eta_raw),
                                self.temperature, self.path, hard)
        feats = torch.cat([u.unsqueeze(-1), y, q], dim=-1).flatten(1)
        return x_t + F.gelu(self.mix_out(feats)), (y, q, prev)


class DualMemNet(nn.Module):
    """clock channel (relative position in the window) -> dual-memory blocks -> mean -> head."""

    def __init__(self, input_channels: int, window_length: int, num_classes: int, *,
                 d_model: int = 30, num_layers: int = 3, **block_kwargs) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_channels + 1, d_model)
        self.blocks = nn.ModuleList(DualMemoryBlock(d_model, **block_kwargs) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        ramp = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype).expand(x.shape[0], 1, -1)
        h = self.encoder(torch.cat([x, ramp], dim=1).transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h.mean(dim=1)))

    @torch.no_grad()
    def forward_streaming(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        batch, _, length = x.shape
        ramp = torch.linspace(-1.0, 1.0, length, device=x.device, dtype=x.dtype)
        states = [(torch.zeros(batch, b.rho.shape[0], b.rho.shape[1], device=x.device, dtype=x.dtype), None, None)
                  for b in self.blocks]
        total = 0.0
        for t in range(length):
            h = self.encoder(torch.cat([x[:, :, t], ramp[t].expand(batch, 1)], dim=1))
            for i, block in enumerate(self.blocks):
                h, states[i] = block.step(h, states[i], hard)
            total = total + h
        return self.head(self.norm(total / length))

    def streaming_cost(self) -> dict[str, int]:
        d = self.encoder.out_features
        kh, kq = self.blocks[0].rho.shape[1], self.blocks[0].tau_logit.shape[1]
        n = len(self.blocks)
        return {
            "mac": self.encoder.in_features * d + n * (2 * d * d + d * (1 + kh + kq) * d),
            "play_ops": n * d * kh,
            "tracker_ops": n * d * kq,
            "gelu": n * d,
            "layernorm": n,
            "state_floats": n * d * (kh + kq + 1) + d,
        }
