"""Hysteresis Network: a sequence model whose only memory is a bank of learnable play operators.

A play operator (backlash, the elementary Prandtl-Ishlinskii hysteron) with width r >= 0:

    y_t = clamp(y_{t-1}, u_t - r, u_t + r),   y_{-1} = 0

The output follows the input only once the input has moved more than r away from it, so the
state remembers past extrema rather than past times: the operator is rate-independent (feeding
the same path faster or slower gives the same output path). A bank of widths splits the signal
by amplitude scale, where a state-space model splits it by time scale. Inference needs no
multiplication: two additions and two comparisons per hysteron per sample, one float of state.

Training is parallel over time. Each step is the map f_t(y) = clamp(y, lo_t, hi_t), and clamp
maps are closed under composition:

    clamp(clamp(y, a1, b1), a2, b2) = clamp(y, clamp(a1, a2, b2), clamp(b1, a2, b2))

so the prefix compositions come out of a Hillis-Steele scan in log2(L) rounds.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def clamp_between(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """clamp with tensor bounds (lo <= hi), differentiable in all three arguments."""
    return torch.minimum(torch.maximum(x, lo), hi)


def play_scan(u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Play operator along the last axis of u, zero initial state, all widths in parallel.

    u: (..., L); r: broadcastable to u, >= 0. Returns y with the shape of u + r broadcast.
    """
    lo, hi = u - r, u + r
    length = lo.shape[-1]
    d = 1
    while d < length:
        # element t holds the composition of the last d maps; compose it after element t - d's
        lo_t, hi_t = lo[..., d:], hi[..., d:]
        lo = torch.cat([lo[..., :d], clamp_between(lo[..., :-d], lo_t, hi_t)], dim=-1)
        hi = torch.cat([hi[..., :d], clamp_between(hi[..., :-d], lo_t, hi_t)], dim=-1)
        d *= 2
    return clamp_between(torch.zeros_like(lo), lo, hi)


def _play_forward(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """y_t = clamp(y_{t-1}, lo_t, hi_t) step by step, y_{-1} = 0; lo, hi (L, ...) -> y (L, ...)."""
    y = torch.minimum(torch.maximum(torch.zeros_like(lo[0]), lo[0]), hi[0])
    ys: list[torch.Tensor] = [y]
    for t in range(1, lo.shape[0]):
        y = torch.minimum(torch.maximum(y, lo[t]), hi[t])
        ys.append(y)
    return torch.stack(ys, 0)


def _play_backward(gy: torch.Tensor, passed: torch.Tensor) -> torch.Tensor:
    """Total gradient reaching each y_t: G_t = gy_t + G_{t+1} * passed_{t+1}."""
    acc = gy[-1]
    gs: list[torch.Tensor] = [acc]
    for t in range(gy.shape[0] - 2, -1, -1):
        acc = gy[t] + acc * passed[t + 1]
        gs.append(acc)
    gs.reverse()
    return torch.stack(gs, 0)


try:  # TorchScript fuses each step's elementwise ops
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        _play_forward_jit = torch.jit.script(_play_forward)
        _play_backward_jit = torch.jit.script(_play_backward)
except Exception:  # noqa: BLE001 - fall back to the eager loops
    _play_forward_jit, _play_backward_jit = _play_forward, _play_backward


class _PlaySequentialFn(torch.autograd.Function):
    """play_scan computed step by step with a hand-written backward (EDGE_SSM_FAST=1).

    lo, hi (..., L) -> y (..., L). y_t takes lo_t when y_{t-1} < lo_t, hi_t when y_{t-1} > hi_t, and
    y_{t-1} otherwise, so the gradient of y_t goes to exactly one of them. Same values as play_scan
    (clamps are exact); gradients equal it except at exact ties, where torch splits them.
    """

    @staticmethod
    def forward(ctx, lo, hi):
        lo_t, hi_t = lo.movedim(-1, 0).contiguous(), hi.movedim(-1, 0).contiguous()
        y = _play_forward_jit(lo_t, hi_t)  # (L, ...)
        prev = torch.cat([torch.zeros_like(y[:1]), y[:-1]], 0)
        below, above = prev < lo_t, prev > hi_t
        ctx.save_for_backward(below, above)
        return y.movedim(0, -1)

    @staticmethod
    def backward(ctx, gy):
        below, above = ctx.saved_tensors
        passed = (~(below | above)).to(gy.dtype)
        g = _play_backward_jit(gy.movedim(-1, 0).contiguous(), passed)
        return (g * below).movedim(0, -1), (g * above).movedim(0, -1)


def play_step(u_t: torch.Tensor, y: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """One streaming step of the play operator."""
    return clamp_between(y, u_t - r, u_t + r)


def fast_paths() -> bool:
    """EDGE_SSM_FAST=1 selects the faster training paths (same math, gradients equal up to float
    rounding, so not bitwise): the hand-written backward of QuantileNet's trackers and the skipped
    scan of the memoryless control. Off by default, so stored results reproduce to the last digit;
    run_whar.py records the setting in every result file."""
    import os

    return os.environ.get("EDGE_SSM_FAST") == "1"


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class HysteresisBlock(nn.Module):
    """norm -> channel mixing -> K play operators per channel -> mixing -> GELU -> residual.

    Controls: learn_widths=False keeps the initial widths; memoryless=True sets every width to 0,
    which turns each play operator into the identity (same block, no memory at all).

    selective=True makes the widths depend on the current input, r_t = softplus(rho + s(x_t)) with
    s linear (zero-initialised): the operator decides per sample how sticky its memory is. A
    width that depends on the input value (not on time) keeps the operator rate-independent, and
    the clamp maps still compose, so the scan is unchanged.
    """

    def __init__(self, d_model: int, widths: tuple[float, ...] = (0.05, 0.2, 0.6, 1.5), *,
                 learn_widths: bool = True, memoryless: bool = False, selective: bool = False) -> None:
        super().__init__()
        k = len(widths)
        self.memoryless = memoryless
        self.norm = nn.LayerNorm(d_model)
        self.mix_in = nn.Linear(d_model, d_model)
        self.select = None
        if selective:
            self.select = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.select.weight)
            nn.init.zeros_(self.select.bias)
        r0 = torch.tensor(widths).repeat(d_model, 1)  # (D, K)
        r0 = r0 * torch.exp(0.3 * torch.randn_like(r0))  # jitter so channels differ
        if learn_widths and not memoryless:
            self.rho = nn.Parameter(inverse_softplus(r0))
        else:
            self.register_buffer("rho", inverse_softplus(r0))
        self.mix_out = nn.Linear(d_model * (k + 1), d_model)

    def widths(self) -> torch.Tensor:
        r = F.softplus(self.rho)
        return torch.zeros_like(r) if self.memoryless else r

    def _widths_at(self, xn: torch.Tensor) -> torch.Tensor:
        """Widths for normalized input xn (..., D) -> (..., D, K); static unless selective."""
        if self.select is None:
            return self.widths()
        return F.softplus(self.rho + self.select(xn).unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        xn = self.norm(x)
        u = self.mix_in(xn).transpose(1, 2)  # (B, D, L)
        if self.memoryless and self.select is None and fast_paths():
            # width 0: clamp(y, u, u) = u, so every play operator is the identity; skip the scan
            y = u.unsqueeze(2).expand(-1, -1, self.rho.shape[1], -1)  # (B, D, K, L)
        else:
            r = self._widths_at(xn)  # (D, K) or (B, L, D, K)
            r = r.unsqueeze(-1) if r.dim() == 2 else r.permute(0, 2, 3, 1)  # (D, K, 1) or (B, D, K, L)
            if fast_paths():
                y = _PlaySequentialFn.apply(u.unsqueeze(2) - r, u.unsqueeze(2) + r)  # (B, D, K, L)
            else:
                y = play_scan(u.unsqueeze(2), r)  # (B, D, K, L)
        feats = torch.cat([u.unsqueeze(2), y], dim=2).flatten(1, 2)  # (B, D*(K+1), L)
        return x + F.gelu(self.mix_out(feats.transpose(1, 2)))

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_t (B, D), state (B, D, K) -> (output (B, D), new state)."""
        xn = self.norm(x_t)
        u = self.mix_in(xn)
        state = play_step(u.unsqueeze(-1), state, self._widths_at(xn))
        feats = torch.cat([u.unsqueeze(-1), state], dim=-1).flatten(1)
        return x_t + F.gelu(self.mix_out(feats)), state


class HysteresisNet(nn.Module):
    """encoder -> hysteresis blocks -> pooling over time -> linear head.

    clock=True appends the relative position in the window, 2 t / (L - 1) - 1, as an input
    channel, so play operators can also remember *when* within the window something happened.
    It spans [-1, 1] whatever the window length, so a uniform speed change leaves it unchanged.
    pooling="meanmax" concatenates the mean and the max over time.
    """

    def __init__(self, input_channels: int, window_length: int, num_classes: int, *,
                 d_model: int = 32, num_layers: int = 3,
                 widths: tuple[float, ...] = (0.05, 0.2, 0.6, 1.5), clock: bool = False,
                 pooling: str = "mean", **block_kwargs) -> None:
        super().__init__()
        if pooling not in ("mean", "meanmax"):
            raise ValueError(f"unknown pooling: {pooling}")
        self.clock = clock
        self.pooling = pooling
        self.encoder = nn.Linear(input_channels + int(clock), d_model)
        self.blocks = nn.ModuleList(HysteresisBlock(d_model, widths, **block_kwargs)
                                    for _ in range(num_layers))
        pooled = d_model * (2 if pooling == "meanmax" else 1)
        self.norm = nn.LayerNorm(pooled)
        self.head = nn.Linear(pooled, num_classes)

    @staticmethod
    def _ramp(length: int, like: torch.Tensor) -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, length, device=like.device, dtype=like.dtype)

    def forward(self, x: torch.Tensor, clock: torch.Tensor | None = None,
                weights: torch.Tensor | None = None) -> torch.Tensor:  # (B, C, L)
        """clock (B, L): the clock channel's values at the actual sample times (default: the uniform ramp);
        weights (B, L): pooling weights summing to 1 per window, i.e. each sample's share of the window's
        time (default: the plain mean). Both are for irregular or event-driven sampling (schedule_test.py);
        with the defaults the model is unchanged."""
        if self.clock:
            ramp = self._ramp(x.shape[-1], x).expand(x.shape[0], -1) if clock is None else clock
            x = torch.cat([x, ramp.unsqueeze(1).to(x.dtype)], dim=1)
        h = self.encoder(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        pooled = h.mean(dim=1) if weights is None else (h * weights.unsqueeze(-1).to(h.dtype)).sum(dim=1)
        if self.pooling == "meanmax":
            pooled = torch.cat([pooled, h.amax(dim=1)], dim=-1)
        return self.head(self.norm(pooled))

    @torch.no_grad()
    def forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        """Same as forward, one sample at a time (reference for deployment; the clock needs the
        window length in advance)."""
        batch, _, length = x.shape
        states = [torch.zeros(batch, b.rho.shape[0], b.rho.shape[1], device=x.device, dtype=x.dtype)
                  for b in self.blocks]
        ramp = self._ramp(length, x)
        total, peak = 0.0, None
        for t in range(length):
            x_t = x[:, :, t]
            if self.clock:
                x_t = torch.cat([x_t, ramp[t].expand(batch, 1)], dim=1)
            h = self.encoder(x_t)
            for i, block in enumerate(self.blocks):
                h, states[i] = block.step(h, states[i])
            total = total + h
            peak = h if peak is None else torch.maximum(peak, h)
        pooled = total / length
        if self.pooling == "meanmax":
            pooled = torch.cat([pooled, peak], dim=-1)
        return self.head(self.norm(pooled))

    def streaming_cost(self) -> dict[str, int]:
        """Operations per input sample (head amortized per window, as for EdgeSSMClassifier)."""
        d = self.encoder.out_features
        k = self.blocks[0].rho.shape[1]
        n = len(self.blocks)
        select = d * d if self.blocks[0].select is not None else 0
        mac = self.encoder.in_features * d + n * (d * d + select + d * (k + 1) * d)
        return {
            "mac": mac,
            "play_ops": n * d * k,  # each: 2 add + 2 compare, no multiply
            "gelu": n * d,
            "layernorm": n,  # per-sample normalizations of a D-vector (+1 on the pooled vector per window)
            "state_floats": n * d * k + d,  # hysterons + running sum for the mean
        }

    def width_summary(self) -> list[list[float]]:
        """Learned widths per block: 10th, 50th, 90th percentile (input is z-scored)."""
        q = torch.tensor([0.1, 0.5, 0.9])
        return [torch.quantile(b.widths().detach().cpu().flatten(), q).tolist() for b in self.blocks]
