"""Round 12: families defined by network structure (known layers allowed, RESEARCH_PLAN.md §1, 2026-10-11).

IsochronNet (an isochron is the set of states with equal phase): the network computes in the movement's intrinsic time instead of sample time.
- Oscillators entrained by the movement: K learned channel projections, each smoothed by a learned FIR filter; the
  analytic signal (Hilbert transform) gives a unit phasor u_k(t) and an amplitude A_k(t) per sample.
- Phase-indexed slot memory ("ring"): per-sample features are written into S slots addressed by the phase (soft von
  Mises binning computed from the phasor, weighted by relative amplitude and by the phase step, so the slots
  integrate over phase, not over samples) and averaged over cycles, i.e. a differentiable epoch folding. Slot
  occupancy (time spent per phase, in sample time) is kept as an extra channel. The ring is exactly unchanged by a
  change of tempo (cycle rate); speed changes inside a cycle cannot be told apart from a different waveform on one
  channel, so the Hilbert phase does not undo them.
- Progress line: an arc-length clock p(t) (cumulative norm of the change of a learned embedding, normalized to
  [0, 1]) addresses a line of slots, each sample weighted by its arc-length step. For a monotone time warp of a
  continuous signal p(t) is unchanged and the line is an integral over p, so it is a warp-free resampling.
- Readouts: circular convolution over the ring slots (mean and std over slots), a convolution over the line,
  relative phases of oscillator pairs at 1:1 and 1:2 locking, a symmetry spectrum (similarity of a ring with a
  learned channel-mirror of itself shifted by a quarter and half cycle), and clock statistics (phase velocity,
  its regularity, amplitude).
- A tonic path in sample time runs alongside for posture and static activities; both kinds add the per-channel
  window statistics of round 8. The ring is filled from the tonic path's last hidden states.
  tonic="encoder" = per-sample encoder only, so all temporal structure comes from intrinsic time;
  tonic="recurrence" = RecurrenceNet's blocks (the trunk of recstatnet, the strongest model so far).

clock="uniform" is the control: same parameters, phases replaced by sample time (fixed 1-4 cycles per window,
unit amplitude) and the progress clock by t / L, so the ring and line become plain time bins.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.round6 import RecurrenceBlock
from src.models.round8 import ChannelStats


def analytic_signal(y: torch.Tensor, pad: int) -> torch.Tensor:
    """y (..., L) real -> complex analytic signal (..., L); FFT Hilbert transform after reflect padding."""
    length = y.shape[-1]
    if pad > 0:
        y = F.pad(y, (pad, pad), mode="reflect")  # y must be (N, C, L) here
    n = y.shape[-1]
    h = torch.zeros(n, device=y.device, dtype=y.dtype)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    a = torch.fft.ifft(torch.fft.fft(y, dim=-1) * h, dim=-1)
    return a[..., pad:pad + length]


def ring_weights(cos: torch.Tensor, sin: torch.Tensor, slots: int, kappa: float) -> torch.Tensor:
    """Soft von Mises assignment of a phasor (cos, sin) (..., L) to `slots` equally spaced phases -> (..., L, S)."""
    angles = torch.arange(slots, device=cos.device, dtype=cos.dtype) * (2 * math.pi / slots)
    logits = kappa * (cos.unsqueeze(-1) * torch.cos(angles) + sin.unsqueeze(-1) * torch.sin(angles))
    return torch.softmax(logits, dim=-1)


def line_weights(p: torch.Tensor, slots: int) -> torch.Tensor:
    """Soft Gaussian assignment of a progress value p in [0, 1] (..., L) to `slots` centres -> (..., L, S)."""
    centres = (torch.arange(slots, device=p.device, dtype=p.dtype) + 0.5) / slots
    return torch.softmax(-0.5 * ((p.unsqueeze(-1) - centres) * slots).square(), dim=-1)


def fold(weights: torch.Tensor, values: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """weights (B, K, L, S), values (B, L, D) -> weighted slot means (B, K, S, D).

    To integrate over intrinsic time rather than sample time, the weights must already carry the measure of each
    sample (its phase step or arc-length step); then the slot means do not depend on how fast the movement went.
    eps is relative to the mean slot mass, so an empty slot reads as 0."""
    mass = weights.sum(dim=2)  # (B, K, S)
    floor = eps * mass.mean(dim=-1, keepdim=True) + 1e-8
    return torch.einsum("bkls,bld->bksd", weights, values) / (mass + floor).unsqueeze(-1)


def occupancy(weights: torch.Tensor) -> torch.Tensor:
    """weights (B, K, L, S) in sample time -> share of the window spent in each slot (B, K, S)."""
    mass = weights.sum(dim=2)
    return mass / (mass.sum(dim=-1, keepdim=True) + 1e-8)


class PhaseClock(nn.Module):
    """K oscillators entrained by the input: projection -> learned FIR -> analytic signal -> phasor and amplitude."""

    def __init__(self, n_in: int, oscillators: int, fir: int = 15) -> None:
        super().__init__()
        self.proj = nn.Linear(n_in, oscillators)
        self.fir = nn.Conv1d(oscillators, oscillators, fir, padding=fir // 2, groups=oscillators,
                             bias=False, padding_mode="reflect")
        with torch.no_grad():  # start as a Hann low-pass, so the phase follows the slow movement
            hann = torch.hann_window(fir + 2, periodic=False)[1:-1]
            self.fir.weight.copy_((hann / hann.sum()).expand(oscillators, 1, fir))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x (B, L, C) -> cos, sin (B, K, L) of the phase and relative amplitude w (B, K, L), mean 1 per window."""
        y = self.fir(self.proj(x).transpose(1, 2))
        y = y - y.mean(dim=-1, keepdim=True)
        a = analytic_signal(y, pad=min(y.shape[-1] // 4, y.shape[-1] - 1))
        amp = (a.real.square() + a.imag.square() + 1e-8).sqrt()
        cos, sin = a.real / (amp + 1e-4), a.imag / (amp + 1e-4)
        w = amp / (amp.mean(dim=-1, keepdim=True) + 1e-4)
        return cos, sin, w


class IsochronNet(nn.Module):
    """Tonic path in sample time + phasic path in intrinsic time (rings, progress line, relative phases)."""

    def __init__(self, input_channels: int, window_length: int, num_classes: int, *, tonic: str = "encoder",
                 d_model: int = 32, num_layers: int = 3, oscillators: int = 6, slots: int = 16,
                 line_slots: int = 16, values: int = 16, ring_dim: int = 8, line_dim: int = 4,
                 mirrors: int = 2, embed: int = 8, kappa: float = 4.0, clock: str = "phase") -> None:
        super().__init__()
        self.clock_kind, self.slots, self.line_slots, self.kappa = clock, slots, line_slots, kappa
        self.encoder = nn.Linear(input_channels, d_model)
        if tonic not in ("encoder", "recurrence"):
            raise ValueError(tonic)
        n_blocks = num_layers if tonic == "recurrence" else 0
        self.blocks = nn.ModuleList(RecurrenceBlock(d_model) for _ in range(n_blocks))
        self.stats = ChannelStats(input_channels)
        # phasic path
        self.clock = PhaseClock(input_channels, oscillators)
        self.embed = nn.Linear(input_channels, embed)  # arc-length clock
        self.values = nn.Linear(d_model, values)
        self.ring_conv = nn.Sequential(
            nn.Conv1d(values + 1, ring_dim, 5, padding=2, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(ring_dim, ring_dim, 3, padding=1, padding_mode="circular"))
        self.line_conv = nn.Conv1d(values, line_dim, 3, padding=1)
        self.mirror = nn.Parameter(torch.zeros(mirrors, values))  # sign-flip-like channel mirrors, tanh-bounded
        with torch.no_grad():
            g = torch.Generator().manual_seed(12)
            self.mirror.copy_(2.0 * torch.randint(0, 2, (mirrors, values), generator=g).to(self.mirror.dtype) - 1.0)
        pairs = oscillators * (oscillators - 1) // 2
        width = (d_model + self.stats.out + oscillators * 2 * ring_dim + line_slots * line_dim
                 + pairs * 4 + oscillators * mirrors * 2 + oscillators * 4)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)
        self.register_buffer("pair_idx", torch.triu_indices(oscillators, oscillators, 1), persistent=False)

    def phases(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """x (B, L, C) -> cos, sin, amplitude weights (B, K, L); progress p in [0, 1] and its step dp (B, L)."""
        b, length, _ = x.shape
        if self.clock_kind == "uniform":
            k = self.clock.proj.out_features
            t = torch.arange(length, device=x.device, dtype=x.dtype) / length
            cycles = torch.arange(1, k + 1, device=x.device, dtype=x.dtype) % 4 + 1  # 2, 3, 4, 1, 2, 3 ...
            angle = 2 * math.pi * cycles[:, None] * t[None, :]
            cos, sin = torch.cos(angle).expand(b, k, length), torch.sin(angle).expand(b, k, length)
            w = torch.ones_like(cos)
            p = ((torch.arange(length, device=x.device, dtype=x.dtype) + 0.5) / length).expand(b, length)
            return cos, sin, w, p, torch.full_like(p, 1.0 / length)
        cos, sin, w = self.clock(x)
        z = self.embed(x)
        step = ((z[:, 1:] - z[:, :-1]).square().sum(-1) + 1e-8).sqrt()  # (B, L-1); d step / d z stays in [-1, 1]
        arc = torch.cat([torch.zeros_like(step[:, :1]), step.cumsum(dim=1)], dim=1)
        total = arc[:, -1:] + 1e-6
        # each sample's arc-length measure: half of the steps on either side (trapezoid rule)
        dp = 0.5 * (torch.cat([step[:, :1], step], dim=1) + torch.cat([step, step[:, -1:]], dim=1)) / total
        return cos, sin, w, arc / total, dp

    def phasic(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """x (B, L, C) raw input, v (B, L, D_v) values to fold -> phasic features (B, F)."""
        cos, sin, w, p, dp = self.phases(x)
        bsz, k, _ = cos.shape
        # per-sample phase step |sin(dtheta)|, the measure of intrinsic time on the ring
        dsin = sin[..., 1:] * cos[..., :-1] - cos[..., 1:] * sin[..., :-1]
        dcos = cos[..., 1:] * cos[..., :-1] + sin[..., 1:] * sin[..., :-1]
        speed = (dsin.square() + 1e-8).sqrt()
        speed = 0.5 * (torch.cat([speed[..., :1], speed], dim=-1) + torch.cat([speed, speed[..., -1:]], dim=-1))
        rw = ring_weights(cos, sin, self.slots, self.kappa) * w.unsqueeze(-1)  # (B, K, L, S), sample time
        ring = fold(rw * speed.unsqueeze(-1), v)  # (B, K, S, D), integrated over phase
        occ = occupancy(rw)  # time spent per phase: carries the timing that the ring leaves out
        r = torch.cat([ring, occ.unsqueeze(-1) * self.slots], dim=-1)  # occupancy 1 = uniform phase speed
        r = self.ring_conv(r.flatten(0, 1).transpose(1, 2))  # (B*K, R, S)
        ring_feats = torch.cat([r.mean(-1), r.std(-1)], dim=-1).view(bsz, -1)
        lw = (line_weights(p, self.line_slots) * dp.unsqueeze(-1)).unsqueeze(1)  # (B, 1, L, S), arc-length measure
        line = fold(lw, v)
        line_feats = self.line_conv(line[:, 0].transpose(1, 2)).flatten(1)
        # relative phases at 1:1 and 1:2 locking, amplitude-weighted (phase-locking values)
        i, j = self.pair_idx
        wij = w[:, i] * w[:, j]
        norm = wij.sum(-1) + 1e-4
        c2, s2 = cos * cos - sin * sin, 2 * cos * sin  # double-angle phasor
        rel = []
        for (ca, sa), (cb, sb) in (((cos[:, i], sin[:, i]), (cos[:, j], sin[:, j])),
                                   ((cos[:, i], sin[:, i]), (c2[:, j], s2[:, j]))):
            rel.append(((ca * cb + sa * sb) * wij).sum(-1) / norm)  # cos of the phase difference
            rel.append(((sa * cb - ca * sb) * wij).sum(-1) / norm)  # sin of the phase difference
        rel_feats = torch.cat(rel, dim=-1)
        # symmetry spectrum: ring vs mirrored ring shifted by a quarter and half cycle
        flat = ring - ring.mean(dim=2, keepdim=True)
        sym = []
        for shift in (self.slots // 4, self.slots // 2):
            rolled = torch.roll(flat, shifts=shift, dims=2)
            for m in torch.tanh(self.mirror):
                sym.append(F.cosine_similarity(flat.flatten(2), (rolled * m).flatten(2), dim=-1, eps=1e-4))
        sym_feats = torch.stack(sym, dim=-1).flatten(1)
        # clock statistics: phase velocity (sin of the per-sample phase step), its spread, amplitude spread
        ww = w[..., 1:] * w[..., :-1]
        wn = ww.sum(-1) + 1e-4
        vel = (dsin * ww).sum(-1) / wn
        vel_sd = (((dsin - vel.unsqueeze(-1)).square() * ww).sum(-1) / wn + 1e-6).sqrt()
        clock_feats = torch.cat([vel, (dcos * ww).sum(-1) / wn, vel_sd, w.std(-1)], dim=-1)
        return torch.cat([ring_feats, line_feats, rel_feats, sym_feats, clock_feats], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        xt = x.transpose(1, 2)
        h = self.encoder(xt)
        for block in self.blocks:
            h = block(h)
        feats = torch.cat([h.mean(dim=1), self.stats(x), self.phasic(xt, self.values(h))], dim=-1)
        return self.head(self.norm(feats))


def leaky_trace(s: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """m_t = lam * m_{t-1} + (1 - lam) * s_t with m_0 = (1 - lam) s_0, as a causal FFT convolution.

    s (B, L, N), lam (N,) in (0, 1) -> (B, L, N)."""
    length = s.shape[1]
    k = torch.arange(length, device=s.device, dtype=s.dtype)
    kernel = (1 - lam) * lam.unsqueeze(0) ** k.unsqueeze(1)  # (L, N)
    n = 2 * length
    out = torch.fft.irfft(torch.fft.rfft(s, n=n, dim=1) * torch.fft.rfft(kernel, n=n, dim=0), n=n, dim=1)
    return out[:, :length]


class DendriticLayer(nn.Module):
    """A layer of dendritic plateau neurons (proposal A of the Nature-architecture atlas).

    Basal branches see the data: each branch filters its input with its own leaky trace, then applies a
    rising-then-falling ("bump") nonlinearity g(m) = relu(m) exp(-alpha relu(m)), like the graded dendritic calcium
    spikes of human L2/3 neurons that let one neuron compute XOR. Apical input is the window context (per-channel
    window statistics and the mean of the layer input over the window). The soma bursts when both agree:
    y = basal * (1 + beta * sigmoid(apical))."""

    def __init__(self, d_model: int, context: int, units: int = 32, branches: int = 4, bump: bool = True,
                 gate: bool = True) -> None:
        super().__init__()
        self.units, self.branches, self.bump, self.gate = units, branches, bump, gate
        self.norm = nn.LayerNorm(d_model)
        self.basal = nn.Linear(d_model, units * branches)
        taus = torch.logspace(0, math.log10(32.0), branches).repeat(units)  # time constants 1-32 samples per unit
        self.lam_raw = nn.Parameter(torch.logit(torch.exp(-1.0 / taus)))
        self.alpha_raw = nn.Parameter(torch.full((units * branches,), math.log(math.expm1(0.5))))
        self.branch_w = nn.Parameter(torch.full((units, branches), 1.0 / branches))
        self.soma = nn.Linear(d_model, units)  # direct somatic input
        if gate:
            self.apical = nn.Linear(context + d_model, units)
            self.beta_raw = nn.Parameter(torch.full((units,), math.log(math.expm1(1.0))))
        self.out = nn.Linear(units, d_model)

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:  # h (B, L, D), ctx (B, Dc)
        hn = self.norm(h)
        m = leaky_trace(self.basal(hn), torch.sigmoid(self.lam_raw))  # (B, L, U*Br)
        if self.bump:
            r = F.relu(m)
            g = r * torch.exp(-F.softplus(self.alpha_raw) * r)
        else:
            g = F.relu(m)
        basal = (g.unflatten(-1, (self.units, self.branches)) * self.branch_w).sum(-1) + self.soma(hn)
        if self.gate:
            a = self.apical(torch.cat([ctx, hn.mean(dim=1)], dim=-1))  # (B, U)
            basal = basal * (1 + F.softplus(self.beta_raw) * torch.sigmoid(a)).unsqueeze(1)
        return h + self.out(basal)


class DendriticNet(nn.Module):
    """encoder -> dendritic layers (context = window statistics) -> mean over time -> head.

    gate=False is the control without apical gating; its window statistics join at the head instead (as in
    recstatnet). bump=False replaces the bump by a ReLU."""

    def __init__(self, input_channels: int, window_length: int, num_classes: int, *, d_model: int = 32,
                 num_layers: int = 3, units: int = 32, branches: int = 4, bump: bool = True, gate: bool = True) -> None:
        super().__init__()
        self.gate = gate
        self.encoder = nn.Linear(input_channels, d_model)
        self.stats = ChannelStats(input_channels)
        self.layers = nn.ModuleList(DendriticLayer(d_model, self.stats.out, units, branches, bump, gate)
                                    for _ in range(num_layers))
        width = d_model + (0 if gate else self.stats.out)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        ctx = self.stats(x)
        h = self.encoder(x.transpose(1, 2))
        for layer in self.layers:
            h = layer(h, ctx)
        pooled = h.mean(dim=1) if self.gate else torch.cat([h.mean(dim=1), ctx], dim=-1)
        return self.head(self.norm(pooled))
