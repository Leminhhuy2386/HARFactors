"""Round-7 candidate: ReturnMapNet, a memory made of learned Poincaré sections (RESEARCH_PLAN.md, round 7).

(Not "PoincaréNet": that name is taken by hyperbolic-geometry networks.)

A repeated movement (walking, waving, a drawn shape) is a closed loop in the signal's phase space.
The classic way to read such a loop without Fourier analysis is a Poincaré section: a surface the
trajectory pierces once per cycle. Each block learns S sections of the normalized input x_t:

    s_k(t) = a_k . x_t + b_k                  (signed distance to section k)
    relay_k(t): switches up when s_k > +h_k and down when s_k < -h_k (Schmitt trigger, learned band)

and, at every upward crossing (an event), records
    - the event itself (crossing rate = cycle frequency),
    - where the trajectory pierced the section, through a learned 2-D projection P_k x_t at that
      sample (the return-map coordinate: the shape of the loop at a fixed phase, whatever the speed),
    - the time since the last crossing and the last return time (period), relative to the window.
These are held until the next event and passed, with the instantaneous signal u, to linear mixing,
GELU and a residual, as in the earlier families.

The relay is computed in parallel: its state is "up" when the last sample above +h is more recent
than the last sample below -h (cumulative maxima of indices), so no sequential loop is needed. The
event times are discrete; gradients reach the projections through the gathered values and reach
the sections through a soft event weight sigmoid((s - h) / T) at the event samples. Deployed: per
sample and section one dot product, two comparisons, a flag and a few registers; no multiplications
beyond the dot products the other families also have.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.hysteresis import inverse_softplus
from src.models.round6 import Round6Net, delayed


def relay_events(s: torch.Tensor, band: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Schmitt-trigger relay on s (B, L, S) with band (S,), in parallel along time.

    Returns state (bool, up), upward-crossing events (bool) and, for every t, the index of the last
    event at or before t and of the event before that (-1 if none)."""
    length = s.shape[1]
    t = torch.arange(length, device=s.device).view(1, -1, 1).expand_as(s)
    up_at = torch.cummax(torch.where(s > band, t, torch.full_like(t, -1)), dim=1).values
    down_at = torch.cummax(torch.where(s < -band, t, torch.full_like(t, -1)), dim=1).values
    state = up_at > down_at
    event = state & ~delayed(state, 1)
    event[:, 0] = state[:, 0]  # an initial "up" counts as a crossing from the default "down"
    last = torch.cummax(torch.where(event, t, torch.full_like(t, -1)), dim=1).values
    before = torch.cat([torch.full_like(last[:, :1], -1), last[:, :-1]], dim=1)
    prev = torch.cummax(torch.where(event, before, torch.full_like(t, -1)), dim=1).values
    return state, event, last, prev


class ReturnMapBlock(nn.Module):
    def __init__(self, d_model: int, sections: int = 16, band: float = 0.3, temperature: float = 0.1) -> None:
        super().__init__()
        self.sections, self.temperature = sections, temperature
        self.norm = nn.LayerNorm(d_model)
        self.mix_in = nn.Linear(d_model, d_model)
        self.section = nn.Linear(d_model, sections)
        self.band_raw = nn.Parameter(inverse_softplus(torch.full((sections,), band)))
        self.position = nn.Linear(d_model, 2 * sections)
        self.mix_out = nn.Linear(d_model + 6 * sections, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        xn = self.norm(x)
        u = self.mix_in(xn)
        s = self.section(xn)  # (B, L, S)
        band = F.softplus(self.band_raw)
        with torch.no_grad():
            state, event, last, prev = relay_events(s.detach(), band.detach())
        length = s.shape[1]
        ev = event.to(s.dtype) * torch.sigmoid((s - band) / self.temperature)  # soft weight at events
        pos = self.position(xn).unflatten(-1, (self.sections, 2))  # (B, L, S, 2)
        idx = last.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, 2)
        seen = (last >= 0).to(s.dtype)
        pos_at = torch.gather(pos, 1, idx) * seen.unsqueeze(-1)  # pierce point of the last crossing
        t = torch.arange(length, device=s.device, dtype=s.dtype).view(1, -1, 1)
        since = torch.where(last >= 0, t - last.to(s.dtype), t + 1) / length
        period = torch.where(prev >= 0, (last - prev).to(s.dtype), torch.zeros_like(s)) / length
        feats = torch.cat([ev.unsqueeze(-1), pos_at, since.unsqueeze(-1), period.unsqueeze(-1),
                           state.to(s.dtype).unsqueeze(-1)], dim=-1).flatten(2)  # (B, L, 6 S)
        return x + F.gelu(self.mix_out(torch.cat([u, feats], dim=-1)))


def _lock_forward(v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, omega0: torch.Tensor, k_phi: torch.Tensor,
                  k_omega: torch.Tensor, k_amp: torch.Tensor, k_mu: torch.Tensor) -> list[torch.Tensor]:
    """Phase-locked template oscillators, step by step. v (L, N, K) input per unit; a, b (K, H) template
    harmonics; omega0 and the gains (K,). Returns post-step states phi, omega, amp, mu, the error e and
    the quantities the backward needs: pre-step states, tanh of the phase drive, w, w', w''."""
    length, n, k = v.shape
    h = torch.arange(1, a.shape[1] + 1, device=v.device, dtype=v.dtype)  # (H,)
    phi = torch.zeros(n, k, device=v.device, dtype=v.dtype)
    omega = omega0.expand(n, k).clone()
    amp = torch.ones(n, k, device=v.device, dtype=v.dtype)
    mu = torch.zeros(n, k, device=v.device, dtype=v.dtype)
    l_phi: list[torch.Tensor] = []
    l_omega: list[torch.Tensor] = []
    l_amp: list[torch.Tensor] = []
    l_mu: list[torch.Tensor] = []
    l_e: list[torch.Tensor] = []
    l_th: list[torch.Tensor] = []
    l_w: list[torch.Tensor] = []
    l_w1: list[torch.Tensor] = []
    l_w2: list[torch.Tensor] = []
    n_phi: list[torch.Tensor] = []
    n_omega: list[torch.Tensor] = []
    n_amp: list[torch.Tensor] = []
    for t in range(length):
        hp = phi.unsqueeze(-1) * h  # (N, K, H)
        c, s = torch.cos(hp), torch.sin(hp)
        w = (a * c + b * s).sum(-1)
        w1 = (h * (b * c - a * s)).sum(-1)
        w2 = (h * h * (-a * c - b * s)).sum(-1)
        e = v[t] - amp * w - mu
        th = torch.tanh(e * amp * w1)
        l_phi.append(phi)
        l_omega.append(omega)
        l_amp.append(amp)
        l_mu.append(mu)
        l_e.append(e)
        l_th.append(th)
        l_w.append(w)
        l_w1.append(w1)
        l_w2.append(w2)
        new_phi = phi + omega + k_phi * th
        new_omega = omega + k_omega * th
        new_amp = amp + k_amp * e * w
        new_mu = mu + k_mu * e
        phi, omega, amp, mu = new_phi, new_omega, new_amp, new_mu
        n_phi.append(phi)
        n_omega.append(omega)
        n_amp.append(amp)
    return [torch.stack(l_phi, 0), torch.stack(l_omega, 0), torch.stack(l_amp, 0), torch.stack(l_mu, 0),
            torch.stack(l_e, 0), torch.stack(l_th, 0), torch.stack(l_w, 0), torch.stack(l_w1, 0),
            torch.stack(l_w2, 0), torch.stack(n_phi, 0), torch.stack(n_omega, 0), torch.stack(n_amp, 0)]


def _lock_backward(g_cos: torch.Tensor, g_sin: torch.Tensor, g_omega: torch.Tensor, g_amp: torch.Tensor,
                   g_e: torch.Tensor, st: list[torch.Tensor], a: torch.Tensor, b: torch.Tensor,
                   k_phi: torch.Tensor, k_omega: torch.Tensor, k_amp: torch.Tensor, k_mu: torch.Tensor) -> list[torch.Tensor]:
    """Reverse pass of _lock_forward for the features (cos phi', sin phi', omega', amp', e)."""
    phi, omega, amp, mu, e, th, w, w1, w2, phi_n, omega_n, amp_n = st
    length, n, k = e.shape
    h = torch.arange(1, a.shape[1] + 1, device=e.device, dtype=e.dtype)
    l_phi = torch.zeros(n, k, device=e.device, dtype=e.dtype)
    l_omega, l_amp, l_mu = torch.zeros_like(l_phi), torch.zeros_like(l_phi), torch.zeros_like(l_phi)
    dv = torch.empty_like(e)
    da, db = torch.zeros_like(a), torch.zeros_like(b)
    dk = [torch.zeros(k, device=e.device, dtype=e.dtype) for _ in range(4)]
    for t in range(length - 1, -1, -1):
        # adjoints of the post-step state, including this step's feature gradients
        lp = l_phi - torch.sin(phi_n[t]) * g_cos[t] + torch.cos(phi_n[t]) * g_sin[t]
        lo = l_omega + g_omega[t]
        la = l_amp + g_amp[t]
        lm = l_mu
        le = g_e[t]
        A, E, W, W1, W2, T = amp[t], e[t], w[t], w1[t], w2[t], th[t]
        dT = 1 - T * T
        # partial derivatives of the drive g = e A w' and of e
        p_phi = -A * A * W1 * W1 + E * A * W2
        p_amp = W1 * (E - A * W)
        p_mu = -A * W1
        p_v = A * W1
        drive = lp * k_phi * dT + lo * k_omega * dT  # adjoint reaching g
        l_phi = lp + drive * p_phi + la * k_amp * (E * W1 - A * W * W1) - lm * k_mu * A * W1 - le * A * W1
        l_omega = lp + lo
        l_amp = drive * p_amp + la * (1 - k_amp * W * W) - lm * k_mu * W - le * W
        l_mu = drive * p_mu - la * k_amp * W + lm * (1 - k_mu) - le
        dv[t] = drive * p_v + la * k_amp * W + lm * k_mu + le
        hp = phi[t].unsqueeze(-1) * h
        c, s = torch.cos(hp), torch.sin(hp)
        # adjoint reaching e (from A', mu', the e feature and the drive's e factor), then e's template terms
        l_e = la * k_amp * W + lm * k_mu + le + drive * A * W1
        da = da + (-(l_e * A).unsqueeze(-1) * c + (la * k_amp * E).unsqueeze(-1) * c
                   - (drive * E * A).unsqueeze(-1) * h * s).sum(0)
        db = db + (-(l_e * A).unsqueeze(-1) * s + (la * k_amp * E).unsqueeze(-1) * s
                   + (drive * E * A).unsqueeze(-1) * h * c).sum(0)
        dk[0] = dk[0] + (lp * T).sum(0)
        dk[1] = dk[1] + (lo * T).sum(0)
        dk[2] = dk[2] + (la * E * W).sum(0)
        dk[3] = dk[3] + (lm * E).sum(0)
    return [dv, da, db, l_omega.sum(0), dk[0], dk[1], dk[2], dk[3]]


try:  # TorchScript fuses each step's elementwise ops
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        _lock_forward_jit = torch.jit.script(_lock_forward)
        _lock_backward_jit = torch.jit.script(_lock_backward)
except Exception:  # noqa: BLE001
    _lock_forward_jit, _lock_backward_jit = _lock_forward, _lock_backward


class _PhaseLockFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, a, b, omega0, k_phi, k_omega, k_amp, k_mu):
        st = _lock_forward_jit(v, a, b, omega0, k_phi, k_omega, k_amp, k_mu)
        ctx.save_for_backward(*st, a, b, k_phi, k_omega, k_amp, k_mu)
        phi_n, omega_n, amp_n, e = st[9], st[10], st[11], st[4]
        return torch.cos(phi_n), torch.sin(phi_n), omega_n, amp_n, e

    @staticmethod
    def backward(ctx, g_cos, g_sin, g_omega, g_amp, g_e):
        saved = ctx.saved_tensors
        st, (a, b, k_phi, k_omega, k_amp, k_mu) = list(saved[:12]), saved[12:]
        zero = lambda g, like: torch.zeros_like(like) if g is None else g.contiguous()  # noqa: E731
        g = [zero(g_cos, st[9]), zero(g_sin, st[9]), zero(g_omega, st[9]), zero(g_amp, st[9]), zero(g_e, st[9])]
        dv, da, db, dw0, dkp, dko, dka, dkm = _lock_backward_jit(*g, st, a, b, k_phi, k_omega, k_amp, k_mu)
        return dv, da, db, dw0, dkp, dko, dka, dkm


def phase_lock_reference(v, a, b, omega0, k_phi, k_omega, k_amp, k_mu):
    """Same recurrence with autograd (for checking _PhaseLockFn)."""
    st = _lock_forward(v, a, b, omega0, k_phi, k_omega, k_amp, k_mu)
    return torch.cos(st[9]), torch.sin(st[9]), st[10], st[11], st[4]


class PhaseLockBlock(nn.Module):
    """Phase-locked template oscillators (analysis by synthesis).

    Each of `units` oscillators has a learned periodic template w_k(phi) (a few harmonics) and runs
    alongside a learned projection v_t of the input, locking onto it like an adaptive-frequency
    oscillator / phase-locked loop:
        e = v - (A w(phi) + mu)
        phi += omega + k_phi tanh(e A w'(phi)),   omega += k_omega tanh(e A w'(phi))
        A += k_A e w(phi),   mu += k_mu e
    The updates are online gradient steps on e^2/2 (tanh-limited for the phase and frequency), so the
    frequency adapts within the sequence. Templates, initial frequencies and gains are learned by
    backpropagation through time (hand-written reverse pass). Features: cos/sin of the phase, the
    locked frequency, the amplitude and the error (how well the template explains the input).
    """

    def __init__(self, d_model: int, units: int = 16, harmonics: int = 3) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mix_in = nn.Linear(d_model, d_model)
        self.drive = nn.Linear(d_model, units)
        a = torch.zeros(units, harmonics)
        a[:, 0] = 1.0
        self.a = nn.Parameter(a + 0.1 * torch.randn(units, harmonics))
        self.b = nn.Parameter(0.1 * torch.randn(units, harmonics))
        # initial frequencies log-uniform between 0.5 and 5 Hz at 50 Hz sampling (radians per sample)
        lo, hi = 2 * torch.pi * 0.5 / 50, 2 * torch.pi * 5.0 / 50
        self.omega0 = nn.Parameter(torch.exp(torch.linspace(float(torch.log(torch.tensor(lo))),
                                                            float(torch.log(torch.tensor(hi))), units)))
        self.gain_raw = nn.Parameter(inverse_softplus(torch.tensor([0.1, 0.01, 0.05, 0.05]).repeat(units, 1)))
        self.mix_out = nn.Linear(d_model + 5 * units, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        xn = self.norm(x)
        u = self.mix_in(xn)
        v = self.drive(xn).transpose(0, 1).contiguous()  # (L, B, K)
        gains = F.softplus(self.gain_raw)  # (K, 4)
        feats = _PhaseLockFn.apply(v, self.a, self.b, self.omega0, *gains.unbind(-1))
        feats = torch.stack(feats, dim=-1).transpose(0, 1).flatten(2)  # (B, L, 5 K)
        return x + F.gelu(self.mix_out(torch.cat([u, feats], dim=-1)))


class Round7Net(Round6Net):
    BLOCKS = {"returnmap": ReturnMapBlock, "phaselock": PhaseLockBlock}
