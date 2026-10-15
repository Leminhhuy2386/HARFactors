"""QuantileNet: a sequence model whose only memory is online quantile tracking.

Each memory unit follows its input with fixed-size steps and settles on a running quantile:

    q_t = q_{t-1} + eta * (tau - 1[u_t < q_{t-1}])

with learnable quantile level tau in (0, 1) and step eta > 0 (stochastic-approximation quantile
estimation; at tau = 1/2 it is the slew-rate-limited follower of a delta modulator). The state
stores an order statistic of the recent input instead of a decayed sum: an outlier moves it by
one step however large it is. Inference needs one comparison and one addition per unit.

path=True (upgrade, "PathQuantileNet") scales each step by how far the input moved,

    q_t = q_{t-1} + eta * |u_t - u_{t-1}| * (tau - 1[u_t < q_{t-1}]),   q_0 = u_0,

so the tracker estimates the tau-quantile of the input *along its path* (weighted by arc
length) instead of over time: playing the same path faster or slower gives the same states.

For backpropagation the indicator is replaced by sigmoid((q - u) / temperature). The update is
not associative, so training runs the recurrence step by step (cheap: the state is (B, D, K)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.hysteresis import fast_paths, inverse_softplus


def quantile_track(u: torch.Tensor, tau: torch.Tensor, eta: torch.Tensor, temperature: float,
                   path: bool = False, hard: bool = False) -> torch.Tensor:
    """Run K trackers per channel over time. u (B, L, D); tau, eta (D, K) -> q (B, L, D, K)."""
    u = u.unsqueeze(-1)  # (B, L, D, 1)
    if path:
        q = u[:, 0].expand(-1, -1, tau.shape[1])
        prev = u[:, 0]
    else:
        q = torch.zeros(u.shape[0], u.shape[2], tau.shape[1], device=u.device, dtype=u.dtype)
    qs = []
    for t in range(u.shape[1]):
        u_t = u[:, t]
        below = (u_t < q).to(u.dtype) if hard else torch.sigmoid((q - u_t) / temperature)
        step = eta * (tau - below)
        if path:
            step = step * (u_t - prev).abs()
            prev = u_t
        q = q + step
        qs.append(q)
    return torch.stack(qs, dim=1)


def _track_forward(u4: torch.Tensor, tau: torch.Tensor, eta: torch.Tensor,
                   temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Tracker states and comparator outputs, u4 (B, L, D, 1) -> q, s (B, L, D, K)."""
    q = torch.zeros(u4.shape[0], u4.shape[2], tau.shape[1], device=u4.device, dtype=u4.dtype)
    qs: list[torch.Tensor] = []
    ss: list[torch.Tensor] = []
    for t in range(u4.shape[1]):
        below = torch.sigmoid((q - u4[:, t]) / temperature)
        q = q + eta * (tau - below)
        qs.append(q)
        ss.append(below)
    return torch.stack(qs, 1), torch.stack(ss, 1)


def _track_backward(grad_q: torch.Tensor, carry: torch.Tensor) -> torch.Tensor:
    """Total gradient reaching each q_t: g_t = grad_q_t + g_{t+1} * carry_{t+1}."""
    acc = grad_q[:, -1]
    gs: list[torch.Tensor] = [acc]
    for t in range(grad_q.shape[1] - 2, -1, -1):
        acc = grad_q[:, t] + acc * carry[:, t + 1]
        gs.append(acc)
    gs.reverse()
    return torch.stack(gs, 1)


try:  # TorchScript fuses each step's elementwise ops: ~16x faster loops, same math
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        _track_forward_jit = torch.jit.script(_track_forward)
        _track_backward_jit = torch.jit.script(_track_backward)
except Exception:  # noqa: BLE001 - fall back to the eager loops
    _track_forward_jit, _track_backward_jit = _track_forward, _track_backward


class _QuantileTrackFn(torch.autograd.Function):
    """quantile_track (soft comparator, time-based steps) with a hand-written backward.

    Both loops are TorchScript-compiled and build no autograd graph. The backward walks the
    recurrence once in reverse: with s_t = sigmoid((q_{t-1} - u_t) / T) and s'_t = s_t (1 - s_t) / T,
        dq_t/dq_{t-1} = 1 - eta s'_t,  dq_t/du_t = eta s'_t,  dq_t/dtau = eta,  dq_t/deta = tau - s_t.
    Same math as quantile_track; values and gradients equal it up to float rounding (fused
    kernels round differently), so it is used only when EDGE_SSM_FAST=1 (hysteresis.fast_paths).
    """

    @staticmethod
    def forward(ctx, u, tau, eta, temperature):
        q, s = _track_forward_jit(u.unsqueeze(-1), tau, eta, float(temperature))
        ctx.save_for_backward(s, tau, eta)
        ctx.temperature = temperature
        return q

    @staticmethod
    def backward(ctx, grad_q):
        s, tau, eta = ctx.saved_tensors
        ds = s * (1 - s) / ctx.temperature  # s'_t
        g = _track_backward_jit(grad_q.contiguous(), 1 - eta * ds)  # carry = dq_t / dq_{t-1}
        grad_u = (g * eta * ds).sum(-1)
        grad_tau = (g * eta).sum((0, 1))
        grad_eta = (g * (tau - s)).sum((0, 1))
        return grad_u, grad_tau, grad_eta, None


def quantile_step(u_t, q, prev, tau, eta, temperature: float, path: bool, hard: bool):
    """One streaming update. u_t (B, D); q (B, D, K) or None at t = 0; prev (B, D) for path."""
    u_t = u_t.unsqueeze(-1)
    if q is None:
        if path:
            q, prev = u_t.expand(-1, -1, tau.shape[1]), u_t
        else:
            q = torch.zeros(u_t.shape[0], u_t.shape[1], tau.shape[1], device=u_t.device, dtype=u_t.dtype)
    below = (u_t < q).to(u_t.dtype) if hard else torch.sigmoid((q - u_t) / temperature)
    step = eta * (tau - below)
    if path:
        step = step * (u_t - prev).abs()
        prev = u_t
    return q + step, prev


class QuantileBlock(nn.Module):
    """norm -> mixing -> K quantile trackers per channel -> mixing -> GELU -> residual."""

    def __init__(self, d_model: int,
                 trackers: tuple[tuple[float, float], ...] = ((0.1, 0.05), (0.5, 0.02), (0.5, 0.1), (0.9, 0.05)),
                 temperature: float = 0.1, path: bool = False) -> None:
        super().__init__()
        k = len(trackers)
        self.temperature = temperature
        self.path = path
        self.norm = nn.LayerNorm(d_model)
        self.mix_in = nn.Linear(d_model, d_model)
        tau0 = torch.tensor([t for t, _ in trackers]).repeat(d_model, 1)
        eta0 = torch.tensor([e for _, e in trackers]).repeat(d_model, 1)
        eta0 = eta0 * torch.exp(0.3 * torch.randn_like(eta0))
        self.tau_logit = nn.Parameter(torch.log(tau0) - torch.log1p(-tau0))  # (D, K)
        self.eta_raw = nn.Parameter(inverse_softplus(eta0))  # (D, K)
        self.mix_out = nn.Linear(d_model * (k + 1), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        u = self.mix_in(self.norm(x))  # (B, L, D)
        tau, eta = torch.sigmoid(self.tau_logit), F.softplus(self.eta_raw)
        if not self.path and fast_paths():
            q_seq = _QuantileTrackFn.apply(u, tau, eta, self.temperature)
        else:
            q_seq = quantile_track(u, tau, eta, self.temperature, self.path)  # (B, L, D, K)
        feats = torch.cat([u.unsqueeze(-1), q_seq], dim=-1).flatten(2)  # (B, L, D*(K+1))
        return x + F.gelu(self.mix_out(feats))

    def step(self, x_t, state, hard: bool = False):
        """x_t (B, D); state = (q, prev) with q None at t = 0. hard=True is the deployed comparator."""
        u = self.mix_in(self.norm(x_t))
        q, prev = quantile_step(u, state[0], state[1], torch.sigmoid(self.tau_logit), F.softplus(self.eta_raw),
                                self.temperature, self.path, hard)
        feats = torch.cat([u.unsqueeze(-1), q], dim=-1).flatten(1)
        return x_t + F.gelu(self.mix_out(feats)), (q, prev)


class QuantileNet(nn.Module):
    def __init__(self, input_channels: int, window_length: int, num_classes: int, *,
                 d_model: int = 32, num_layers: int = 3, **block_kwargs) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_channels, d_model)
        self.blocks = nn.ModuleList(QuantileBlock(d_model, **block_kwargs) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = self.encoder(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h.mean(dim=1)))

    def streaming_cost(self) -> dict[str, int]:
        """Operations per input sample (head amortized per window), deployed with the hard comparator."""
        d = self.encoder.out_features
        k = self.blocks[0].tau_logit.shape[1]
        n = len(self.blocks)
        return {
            "mac": self.encoder.in_features * d + n * (d * d + d * (k + 1) * d),
            "tracker_ops": n * d * k,  # each: 1 compare + 1 add of a precomputed step (eta*tau or eta*(tau-1))
            "gelu": n * d,
            "layernorm": n,
            "state_floats": n * d * k + d,
        }

    @torch.no_grad()
    def forward_streaming(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        batch, _, length = x.shape
        states = [(None, None)] * len(self.blocks)
        total = 0.0
        for t in range(length):
            h = self.encoder(x[:, :, t])
            for i, block in enumerate(self.blocks):
                h, states[i] = block.step(h, states[i], hard)
            total = total + h
        return self.head(self.norm(total / length))
