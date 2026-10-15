"""Sequence-model baselines missing from WHAR Arena: GRU, S4D and Mamba.

All take (B, C, L) like the WHAR models and pool over time like EdgeSSMClassifier, so the
only difference between them is the sequence layer. Mamba is a plain-PyTorch reference
implementation (sequential selective scan): the official CUDA kernels need Linux and a newer
GPU than this machine's Pascal card.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.hysteresis import fast_paths
from src.models.layers import causal_fft_conv


class GRUClassifier(nn.Module):
    def __init__(self, input_channels, window_length, num_classes, *, hidden=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_channels, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.gru(x.transpose(1, 2))
        return self.head(out.mean(dim=1))


class S4DKernel(nn.Module):
    """Diagonal state-space kernel, S4D-Lin initialization (Gu et al., 2022)."""

    def __init__(self, d_model, d_state=32, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        half = d_state // 2
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.c = nn.Parameter(torch.view_as_real(torch.randn(d_model, half, dtype=torch.cfloat)))
        self.log_a_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, half)))
        self.a_imag = nn.Parameter(math.pi * torch.arange(half).float().expand(d_model, half).clone())

    dt_scale = 1.0  # plain attribute (not in the state dict): Δ multiplier for a new sampling rate (schedule_test.py)

    def forward(self, length):
        dt = torch.exp(self.log_dt) * self.dt_scale
        c = torch.view_as_complex(self.c)
        a = -torch.exp(self.log_a_real) + 1j * self.a_imag
        dt_a = a * dt.unsqueeze(-1)
        steps = torch.arange(length, device=a.device)
        c = c * (torch.exp(dt_a) - 1.0) / a
        return 2 * torch.einsum("hn,hnl->hl", c, torch.exp(dt_a.unsqueeze(-1) * steps)).real


class S4DBlock(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.kernel = S4DKernel(d_model, d_state)
        self.skip = nn.Parameter(torch.randn(d_model))
        self.out = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)

    def forward(self, x):  # (B, H, L)
        u = self.norm(x.transpose(1, 2)).transpose(1, 2)
        k = self.kernel(u.shape[-1])  # (H, L)
        y = causal_fft_conv(u, k.unsqueeze(1)).squeeze(2) + u * self.skip.unsqueeze(-1)
        y = F.glu(self.out(F.gelu(y)), dim=1)
        return x + y


class S4DClassifier(nn.Module):
    def __init__(self, input_channels, window_length, num_classes, *, d_model=64, d_state=32,
                 num_layers=2):
        super().__init__()
        self.encoder = nn.Conv1d(input_channels, d_model, kernel_size=1)
        self.blocks = nn.ModuleList(S4DBlock(d_model, d_state) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h.mean(dim=-1)))


class MambaBlock(nn.Module):
    """Mamba (S6) mixer, Gu & Dao (2023), with the reference initialization."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.a_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).expand(d_inner, d_state).clone())
        self.d = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    dt_scale = 1.0  # plain attribute: Δ multiplier for a new sampling rate (the 4-tap conv stays in samples)

    def forward(self, x):  # (B, L, D)
        length = x.shape[1]
        xz = self.in_proj(self.norm(x))
        u, z = xz.chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :length].transpose(1, 2))  # (B, L, E)
        dt, b, c = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)) * self.dt_scale  # (B, L, E)
        a = -torch.exp(self.a_log)  # (E, N)
        log_decay = dt.unsqueeze(-1) * a  # (B, L, E, N), <= 0
        drive = (dt * u).unsqueeze(-1) * b.unsqueeze(2)  # (B, L, E, N)
        scan = _SequentialScanFn.apply if fast_paths() else selective_scan
        y = scan(log_decay, drive, c) + u * self.d
        return x + self.out_proj(y * F.silu(z))


def selective_scan(log_decay, drive, c):
    """y_t = <h_t, c_t> with h_t = exp(log_decay_t) * h_{t-1} + drive_t, h_{-1} = 0.

    Exact Hillis-Steele scan: log2(L) rounds of elementwise ops instead of L sequential steps.
    Each round composes (A_t, h_t) with (A_{t-d}, h_{t-d}); A only multiplies decays in (0, 1],
    so nothing can overflow. log_decay, drive: (B, L, E, N); c: (B, L, N).
    """
    length = drive.shape[1]
    a, h = torch.exp(log_decay), drive
    d = 1
    while d < length:
        h = torch.cat([h[:, :d], a[:, d:] * h[:, :-d] + h[:, d:]], dim=1)
        a = torch.cat([a[:, :d], a[:, d:] * a[:, :-d]], dim=1)
        d *= 2
    return (h * c.unsqueeze(2)).sum(dim=-1)


def _scan_forward(a: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    """h_t = a_t h_{t-1} + drive_t step by step; a, drive (B, L, E, N) -> h (B, L, E, N)."""
    h = drive[:, 0]
    hs: list[torch.Tensor] = [h]
    for t in range(1, drive.shape[1]):
        h = a[:, t] * h + drive[:, t]
        hs.append(h)
    return torch.stack(hs, 1)


def _scan_backward(gh: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Total gradient reaching each h_t: G_t = gh_t + a_{t+1} G_{t+1}."""
    acc = gh[:, -1]
    gs: list[torch.Tensor] = [acc]
    for t in range(gh.shape[1] - 2, -1, -1):
        acc = gh[:, t] + a[:, t + 1] * acc
        gs.append(acc)
    gs.reverse()
    return torch.stack(gs, 1)


try:  # TorchScript fuses each step's elementwise ops
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        _scan_forward_jit = torch.jit.script(_scan_forward)
        _scan_backward_jit = torch.jit.script(_scan_backward)
except Exception:  # noqa: BLE001 - fall back to the eager loops
    _scan_forward_jit, _scan_backward_jit = _scan_forward, _scan_backward


class _SequentialScanFn(torch.autograd.Function):
    """selective_scan computed step by step, with a hand-written backward (EDGE_SSM_FAST=1).

    Same recurrence as the Hillis-Steele scan; values and gradients equal it up to float rounding.
    With a_t = exp(log_decay_t) and G_t the total gradient reaching h_t:
        d drive_t = G_t,  d log_decay_t = G_t h_{t-1} a_t,  d c_t = sum_E dy_t h_t.
    """

    @staticmethod
    def forward(ctx, log_decay, drive, c):
        a = torch.exp(log_decay)
        h = _scan_forward_jit(a, drive)
        ctx.save_for_backward(a, h, c)
        return (h * c.unsqueeze(2)).sum(dim=-1)

    @staticmethod
    def backward(ctx, gy):
        a, h, c = ctx.saved_tensors
        g = _scan_backward_jit((gy.unsqueeze(-1) * c.unsqueeze(2)).contiguous(), a)
        h_prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        return g * h_prev * a, g, (gy.unsqueeze(-1) * h).sum(dim=2)


class MambaClassifier(nn.Module):
    def __init__(self, input_channels, window_length, num_classes, *, d_model=32, d_state=16,
                 num_layers=2):
        super().__init__()
        self.encoder = nn.Linear(input_channels, d_model)
        self.blocks = nn.ModuleList(MambaBlock(d_model, d_state) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.encoder(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h.mean(dim=1)))
