"""schedule_test.py: the zero-order-hold form of HysteresisNet's native pipeline equals event-driven processing,
the defaults leave the model unchanged, and Δ scaling resamples S4D consistently.

Run with: uv run python tests/test_schedule.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import build  # noqa: E402
from src.evaluation.schedule_test import reconstruct, send_on_delta, set_dt_scale  # noqa: E402

torch.set_default_dtype(torch.float64)


def test_hold_equals_event_driven():
    """Held inputs on the grid give exactly the time-weighted output of processing only the events."""
    torch.manual_seed(0)
    model = build("hystnet_sel_clock", input_channels=6, window_length=150, num_classes=27).double().eval()
    x = torch.cumsum(0.3 * torch.randn(3, 6, 150), dim=-1).numpy()
    for b in range(3):
        idx = send_on_delta(x[b], 0.5)
        assert 1 < len(idx) < 150
        _, hold, clk = reconstruct(x[b][:, idx], idx.astype(np.float64), 150)
        a = model(torch.tensor(hold)[None], clock=torch.tensor(clk)[None])
        t = idx.astype(np.float64)
        dur = np.diff(np.concatenate([t, [150.0]]))
        dur[0] += t[0]  # samples before the first event are held from it
        ev = torch.tensor(x[b][:, idx])[None]
        e = model(ev, clock=torch.tensor(2 * t / 149 - 1)[None], weights=torch.tensor(dur / 150)[None])
        assert torch.allclose(a, e, atol=1e-10), (a - e).abs().max()


def test_defaults_unchanged():
    torch.manual_seed(1)
    model = build("hystnet_sel_clock", input_channels=6, window_length=150, num_classes=27).double().eval()
    x = torch.randn(2, 6, 150)
    ref = model(x)
    clock = torch.linspace(-1, 1, 150).expand(2, -1)
    w = torch.full((2, 150), 1 / 150)
    assert torch.allclose(model(x, clock=clock, weights=w), ref, atol=1e-12)


def test_dt_scale_matches_resampled_s4d():
    """A smooth signal at twice the rate with Δ halved gives nearly the same S4D output as at the native rate."""
    torch.manual_seed(2)
    model = build("s4d", input_channels=3, window_length=200, num_classes=4).double().eval()
    t = torch.arange(400) / 2.0
    x2 = torch.stack([torch.sin(0.05 * t), torch.cos(0.03 * t), torch.sin(0.07 * t + 1)])[None]
    x1 = x2[..., ::2]
    base = model(x1)
    set_dt_scale(model, 0.5)
    scaled = model(x2)
    set_dt_scale(model, 1.0)
    naive = model(x2)
    assert (scaled - base).abs().max() < 0.3 * (naive - base).abs().max(), ((scaled - base).abs().max(), (naive - base).abs().max())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
