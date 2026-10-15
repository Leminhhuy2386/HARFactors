"""Run with: uv run python -m pytest tests -q   (or: uv run python tests/test_edge_ssm.py)"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.layers import EdgeSSMLayer, KANLayer  # noqa: E402
from src.models.model import EdgeSSMClassifier  # noqa: E402
from src.models.registry import CONFIGS, build  # noqa: E402

torch.set_default_dtype(torch.float64)


def naive_edge_layer(layer: EdgeSSMLayer, x: torch.Tensor) -> torch.Tensor:
    """The equations of the summary, written out loop by loop."""
    _, a, b = layer.decay()
    a = a.expand(layer.in_features, layer.out_features)
    b = b.expand(layer.in_features, layer.out_features)
    batch, n_in, length = x.shape
    out = torch.zeros(batch, layer.out_features, length)
    # oscillatory: rotate the state (re, im) by omega and read the real part
    cos = torch.cos(layer.omega) if layer.oscillatory else torch.ones_like(a)
    sin = torch.sin(layer.omega) if layer.oscillatory else torch.zeros_like(a)
    s = torch.zeros(batch, n_in, layer.out_features)
    s_im = torch.zeros_like(s)
    for t in range(length):
        s, s_im = a * (cos * s - sin * s_im) + b * x[:, :, t, None], a * (sin * s + cos * s_im)
        phi = layer.weight * s
        if layer.use_phi:
            for r, g in enumerate(layer.grid):
                phi = phi + layer.coef[..., r] * torch.tanh((s - g) / layer.basis_width)
        out[:, :, t] = phi.sum(dim=1) + layer.bias
    return out


def test_fft_matches_recurrence():
    torch.manual_seed(0)
    for kwargs in ({}, {"use_phi": False}, {"tau_sharing": "input"}, {"input_gain": "ema"},
                   {"oscillatory": True}):
        layer = EdgeSSMLayer(5, 7, **kwargs)
        x = torch.randn(3, 5, 40)
        assert torch.allclose(layer(x), naive_edge_layer(layer, x), atol=1e-9), kwargs


def test_streaming_matches_parallel():
    for oscillatory in (False, True):
        torch.manual_seed(0)
        model = EdgeSSMClassifier(6, 50, 4, hidden=8, kan_hidden=5, oscillatory=oscillatory).eval()
        x = torch.randn(2, 6, 50)
        assert torch.allclose(model(x), model.forward_streaming(x), atol=1e-9), oscillatory


def test_kan_layer_matches_definition():
    torch.manual_seed(0)
    kan = KANLayer(4, 3)
    x = torch.randn(2, 4, 9)
    w, c = kan.weight[:, 0, :], kan.weight[:, 1:, :]  # (in, out), (in, R, out)
    expect = torch.einsum("bil,io->bol", x, w) + kan.bias[:, None]
    for r, g in enumerate(kan.grid):
        expect = expect + torch.einsum("bil,io->bol", torch.tanh((x - g) / kan.basis_width), c[:, r])
    assert torch.allclose(kan(x), expect, atol=1e-12)
    assert torch.allclose(kan(x[:, :, 3]), expect[:, :, 3], atol=1e-12)


def test_tau_init_range_and_gain():
    torch.manual_seed(0)
    layer = EdgeSSMLayer(16, 32)
    tau = layer.tau().detach()
    assert tau.min() >= 2.0 - 1e-6 and tau.max() <= 100.0 + 1e-6
    # sqrt(1 - a^2) keeps the stationary variance of white noise at 1
    _, a, b = layer.decay()
    assert torch.allclose(b**2 / (1 - a**2), torch.ones_like(a))
    # after tau steps an impulse keeps exp(-1) of its weight
    assert math.isclose(float(torch.exp(-1 / tau[0, 0]) ** tau[0, 0]), math.exp(-1), rel_tol=1e-9)


def test_parallel_selective_scan_matches_recurrence():
    from src.models.baselines import selective_scan

    torch.manual_seed(0)
    log_decay = -torch.rand(2, 37, 3, 4) * 2.0
    drive, c = torch.randn(2, 37, 3, 4), torch.randn(2, 37, 4)
    state, ys = torch.zeros(2, 3, 4), []
    for t in range(37):
        state = torch.exp(log_decay[:, t]) * state + drive[:, t]
        ys.append(torch.einsum("ben,bn->be", state, c[:, t]))
    assert torch.allclose(selective_scan(log_decay, drive, c), torch.stack(ys, 1), atol=1e-10)


def test_play_scan_matches_recurrence_and_is_rate_independent():
    from src.models.hysteresis import play_scan

    torch.manual_seed(0)
    u = torch.cumsum(torch.randn(3, 4, 53), dim=-1)  # random walks: plenty of reversals
    r = torch.rand(4, 1) * 2
    y, ys = torch.zeros(3, 4), []
    for t in range(53):
        y = torch.minimum(torch.maximum(y, u[..., t] - r[:, 0]), u[..., t] + r[:, 0])
        ys.append(y)
    assert torch.allclose(play_scan(u, r), torch.stack(ys, -1), atol=1e-12)
    # play the same path twice as slowly (every sample repeated): same output path
    slow = u.repeat_interleave(2, dim=-1)
    assert torch.allclose(play_scan(slow, r)[..., 1::2], play_scan(u, r), atol=1e-12)


def test_hysteresis_net_streaming_matches_parallel():
    from src.models.hysteresis import HysteresisNet

    x = torch.randn(2, 6, 40)
    for kwargs in ({}, {"selective": True}, {"clock": True}, {"pooling": "meanmax"}):
        torch.manual_seed(0)
        model = HysteresisNet(6, 40, 5, d_model=8, num_layers=2, **kwargs).eval()
        if kwargs.get("selective"):  # move off the zero init so the selection is exercised
            for b in model.blocks:
                torch.nn.init.normal_(b.select.weight)
        assert torch.allclose(model(x), model.forward_streaming(x), atol=1e-9), kwargs


def test_vortex_net_streaming_matches_parallel_and_winding_counts_turns():
    from src.models.vortex import VortexNet, turning

    torch.manual_seed(0)
    model = VortexNet(6, 40, 5, d_model=8, n_units=4, num_layers=2).eval()
    x = torch.randn(2, 6, 40)
    assert torch.allclose(model(x), model.forward_streaming(x), atol=1e-9)
    # two full turns around the origin at radius 10 (gate ~ 1): winding = 4 pi, at any speed
    for n in (50, 200):
        a = torch.linspace(0, 4 * math.pi, n)
        path = 10 * torch.stack([torch.cos(a), torch.sin(a)], dim=-1)
        total = turning(path[:-1], path[1:], eps=1e-2).sum()
        assert abs(float(total) / (4 * math.pi) - 1) < 1e-3, (n, float(total))  # gate = 100/100.01


def test_maxplus_scan_and_tropical_streaming():
    from src.models.tropical import TropicalNet, maxplus_scan

    torch.manual_seed(0)
    v, lam = torch.randn(3, 4, 45), torch.rand(4, 1)
    h, hs = torch.full((3, 4), -torch.inf), []
    for t in range(45):
        h = torch.maximum(h - lam[:, 0], v[..., t])
        hs.append(h)
    assert torch.allclose(maxplus_scan(v, lam), torch.stack(hs, -1), atol=1e-12)
    lam_t = torch.rand(3, 4, 45)  # per-step decays compose too
    h, hs = torch.full((3, 4), -torch.inf), []
    for t in range(45):
        h = torch.maximum(h - lam_t[..., t], v[..., t])
        hs.append(h)
    assert torch.allclose(maxplus_scan(v, lam_t), torch.stack(hs, -1), atol=1e-12)
    model = TropicalNet(6, 40, 5, d_model=8, num_layers=2).eval()
    x = torch.randn(2, 6, 40)
    assert torch.allclose(model(x), model.forward_streaming(x), atol=1e-9)


def test_quantile_tracker_converges_and_streams():
    from src.models.quantile import QuantileNet

    torch.manual_seed(0)
    model = QuantileNet(6, 40, 5, d_model=8, num_layers=2).eval()
    x = torch.randn(2, 6, 40)
    assert torch.allclose(model(x), model.forward_streaming(x), atol=1e-9)
    from src.models.quantile import quantile_step, quantile_track

    block = model.blocks[0]  # hard tracker on i.i.d. normal input settles near the tau-quantile
    q = torch.zeros(1, 8, 4)
    with torch.no_grad():
        tau, eta = torch.sigmoid(block.tau_logit), torch.full((8, 4), 0.01)
        for u in torch.randn(20000, 1, 8):
            q, _ = quantile_step(u, q, None, tau, eta, 0.1, path=False, hard=True)
    expect = torch.distributions.Normal(0.0, 1.0).icdf(tau)
    assert (q[0] - expect).abs().max() < 0.15, (q[0], expect)
    # path trackers: the same path played twice as slowly gives the same states
    u = torch.cumsum(torch.randn(2, 60, 8), dim=1)
    tau, eta = torch.rand(8, 3), torch.rand(8, 3)
    fast = quantile_track(u, tau, eta, 0.1, path=True)
    slow = quantile_track(u.repeat_interleave(2, dim=1), tau, eta, 0.1, path=True)
    assert torch.allclose(slow[:, 1::2], fast, atol=1e-12)
    # and streaming matches the parallel form for the path variant
    torch.manual_seed(0)
    pmodel = QuantileNet(6, 40, 5, d_model=8, num_layers=2, path=True,
                         trackers=((0.1, 0.25), (0.5, 0.5))).eval()
    assert torch.allclose(pmodel(x), pmodel.forward_streaming(x), atol=1e-9)


def test_dual_memory_net():
    from src.models.dualmem import DualMemoryBlock, DualMemNet

    x = torch.randn(2, 6, 40)
    for path in (False, True):
        torch.manual_seed(0)
        model = DualMemNet(6, 40, 5, d_model=8, num_layers=2, path=path).eval()
        for b in model.blocks:  # exercise the selective widths
            torch.nn.init.normal_(b.select.weight)
        assert torch.allclose(model(x), model.forward_streaming(x), atol=1e-9), path
    # with path trackers a block is rate-independent: duplicated samples change nothing
    torch.manual_seed(0)
    block = DualMemoryBlock(8, path=True).eval()
    torch.nn.init.normal_(block.select.weight)
    h = torch.randn(2, 30, 8).cumsum(dim=1)
    with torch.no_grad():
        assert torch.allclose(block(h.repeat_interleave(2, dim=1))[:, 1::2], block(h), atol=1e-10)


def test_time_warps():
    from src.evaluation.warp_test import random_positions, resample, uniform_positions

    torch.manual_seed(0)
    x = torch.randn(4, 3, 150)
    assert torch.allclose(resample(x, uniform_positions(150, 1.0)), x, atol=1e-12)
    assert resample(x, uniform_positions(150, 1.5)).shape[-1] == 100
    pos = random_positions(4, 150, 0.5, torch.Generator().manual_seed(1))
    assert (pos.diff(dim=1) > 0).all() and torch.allclose(pos[:, 0], torch.zeros(4), atol=1e-9)
    assert torch.allclose(pos[:, -1], torch.full((4,), 149.0), atol=1e-6)
    ramp = torch.arange(150.0).expand(4, 1, 150)
    assert torch.allclose(resample(ramp, pos)[:, 0], pos, atol=1e-9)  # linear signal maps to positions


def test_every_registered_model_trains_one_step():
    torch.set_default_dtype(torch.float32)
    try:
        for model_id in CONFIGS:
            torch.manual_seed(0)
            model = build(model_id, input_channels=9, window_length=30, num_classes=6)
            x = torch.randn(4, 9, 30)
            logits = model(x)
            assert logits.shape == (4, 6), model_id
            logits.logsumexp(dim=1).mean().backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in grads), model_id
    finally:
        torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
