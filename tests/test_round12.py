"""Round 12: phase clock, tempo-free ring, warp-free progress line, uniform control, dendritic trace; train step.

Run with: uv run python tests/test_round12.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import build  # noqa: E402
from src.models.round12 import (PhaseClock, IsochronNet, analytic_signal, fold, leaky_trace,  # noqa: E402
                              line_weights, occupancy, ring_weights)

torch.set_default_dtype(torch.float64)


def identity_clock() -> PhaseClock:
    """One oscillator reading channel 0 with a pass-through FIR."""
    clock = PhaseClock(1, 1)
    with torch.no_grad():
        clock.proj.weight.fill_(1.0)
        clock.proj.bias.zero_()
        clock.fir.weight.zero_()
        clock.fir.weight[0, 0, clock.fir.kernel_size[0] // 2] = 1.0
    return clock


def test_analytic_phase_of_a_sinusoid():
    length = 300
    t = torch.arange(length) / length
    for cycles in (3.0, 4.7, 9.2):
        theta = 2 * math.pi * cycles * t
        a = analytic_signal(torch.cos(theta)[None, None], pad=length // 4)[0, 0]
        err = torch.angle(a * torch.exp(-1j * theta))[30:-30].abs().max()
        assert err < 0.15, (cycles, float(err))  # FFT Hilbert: edge leakage for non-integer cycles


def test_ring_is_tempo_free():
    """The same cycle shape played at two tempi fills the same ring (epoch folding in phase)."""
    clock = identity_clock()
    length = 600
    t = torch.arange(length) / length
    rings = []
    for cycles in (4.0, 7.0):
        theta = 2 * math.pi * cycles * t
        x = torch.cos(theta)[None, :, None]  # (B, L, C)
        values = torch.stack([torch.cos(theta), torch.sin(2 * theta), torch.cos(3 * theta)], dim=-1)[None]
        cos, sin, w = clock(x)
        rw = ring_weights(cos, sin, 16, 4.0) * w.unsqueeze(-1)
        rings.append((fold(rw, values), occupancy(rw)))
    assert (rings[0][0] - rings[1][0]).abs().max() < 0.05, (rings[0][0] - rings[1][0]).abs().max()
    assert (rings[0][1] - rings[1][1]).abs().max() < 0.01


def test_line_is_warp_free():
    """A monotone time warp of the same path leaves the arc-length progress line unchanged (model's own clock)."""
    model = IsochronNet(2, 2000, 4)
    length = 2000
    t = torch.arange(length) / (length - 1)
    lines = []
    for warp in (t, t ** 1.7, 0.5 * (1 - torch.cos(math.pi * t))):
        s = 2 * math.pi * warp
        x = torch.stack([torch.cos(s) + 0.3 * s, torch.sin(2 * s)], dim=-1)[None]  # a non-periodic 2-D path
        _, _, _, p, dp = model.phases(x)
        lines.append(fold((line_weights(p, 16) * dp.unsqueeze(-1)).unsqueeze(1), x))
    for other in lines[1:]:
        assert (lines[0] - other).abs().max() < 0.02, (lines[0] - other).abs().max()


def test_model_progress_clock_is_warp_free():
    model = IsochronNet(2, 2000, 4)
    length = 2000
    t = torch.arange(length) / (length - 1)
    ps = []
    for warp in (t, t ** 1.5):
        s = 2 * math.pi * warp
        x = torch.stack([torch.cos(s) + 0.3 * s, torch.sin(2 * s)], dim=-1)[None]
        ps.append(model.phases(x)[3])
    # same progress at the same point of the path: compare p as a function of the warped time
    idx = torch.searchsorted(t ** 1.5, t[100:-100]).clamp(max=length - 1)
    assert (ps[0][0, 100:-100] - ps[1][0, idx]).abs().max() < 0.01


def test_uniform_control_has_same_parameters_and_ignores_input_timing():
    a = build("isochron", input_channels=6, window_length=150, num_classes=27)
    b = build("isochron_uniform", input_channels=6, window_length=150, num_classes=27)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    x1, x2 = torch.randn(2, 150, 6), torch.randn(2, 150, 6)
    c1, c2 = b.phases(x1), b.phases(x2)
    assert all(torch.equal(u, v) for u, v in zip(c1, c2))


def test_leaky_trace_matches_recursion():
    torch.manual_seed(0)
    s = torch.randn(2, 40, 5)
    lam = torch.tensor([0.0, 0.3, 0.7, 0.9, 0.99])
    got = leaky_trace(s, lam)
    m = torch.zeros(2, 5)
    ref = []
    for i in range(40):
        m = lam * m + (1 - lam) * s[:, i]
        ref.append(m)
    assert torch.allclose(got, torch.stack(ref, dim=1), atol=1e-10)


def test_train_step_all_models():
    torch.set_default_dtype(torch.float32)
    try:
        for model_id in ("isochron", "isochron_uniform", "isochron_rec", "isochron_rec_uniform",
                         "dendriticnet", "dendritic_relu", "dendritic_nogate"):
            for c, length, k in ((6, 150, 27), (133, 90, 5), (3, 60, 6)):
                torch.manual_seed(0)
                model = build(model_id, input_channels=c, window_length=length, num_classes=k)
                x, y = torch.randn(8, c, length), torch.randint(0, k, (8,))
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                assert torch.isfinite(loss)
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), model_id
    finally:
        torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
