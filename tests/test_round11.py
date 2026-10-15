"""Round 11: banded soft-DTW against the plain recursion and hard DTW; style invariance; train step.

Run with: uv run python tests/test_round11.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import build  # noqa: E402
from src.models.round11 import soft_dtw_band  # noqa: E402

torch.set_default_dtype(torch.float64)


def naive_soft_dtw(x: torch.Tensor, y: torch.Tensor, w: int, gamma: float) -> float:
    t = x.shape[0]
    inf = float("inf")
    d = [[inf] * (t + 1) for _ in range(t + 1)]
    d[0][0] = 0.0
    def smin(*v: float) -> float:
        vals = [-a / gamma for a in v if a < inf]
        if not vals:
            return inf
        top = max(vals)
        return -gamma * (top + math.log(sum(math.exp(u - top) for u in vals)))
    for i in range(1, t + 1):
        for j in range(1, t + 1):
            if abs(i - j) > w:
                continue
            c = float(((x[i - 1] - y[j - 1]) ** 2).sum())
            d[i][j] = c + smin(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[t][t]


def test_soft_dtw_matches_recursion():
    torch.manual_seed(0)
    for t, e, w, gamma in ((12, 2, 3, 0.5), (20, 3, 6, 1.0), (9, 4, 2, 0.1)):
        x, p = torch.randn(3, t, e), torch.randn(4, t, e)
        got = soft_dtw_band(x, p, w, gamma)
        ref = torch.tensor([[naive_soft_dtw(x[a], p[b], w, gamma) for b in range(4)] for a in range(3)])
        assert torch.allclose(got, ref, atol=1e-8), (got - ref).abs().max()


def test_small_gamma_approaches_hard_dtw():
    from src.models.classical.elastic_baseline import dtw_band
    torch.manual_seed(1)
    x, p = torch.randn(3, 15, 2), torch.randn(5, 15, 2)
    hard = dtw_band(x, p, 4)
    soft = soft_dtw_band(x, p, 4, 1e-3)
    assert torch.allclose(soft, hard, atol=0.05)


def test_elastic_evidence_ignores_gain_and_offset():
    torch.manual_seed(2)
    model = build("stylealignnet", input_channels=6, window_length=150, num_classes=5)
    x = torch.randn(2, 6, 150).cumsum(-1) * 0.1
    ev, _, _ = model.elastic(x)
    ev2, _, _ = model.elastic(2.5 * x + 0.7)  # one gain for all channels, any constant offset
    assert torch.allclose(ev, ev2, rtol=5e-3, atol=1e-4)  # exact up to the 1e-3 floor on the fitted gain
    ev3, _, _ = model.elastic(x.flip(-1))  # but not blind: a time-reversed window aligns differently
    assert not torch.allclose(ev, ev3, rtol=5e-3, atol=1e-4)


def test_models_train_step():
    torch.set_default_dtype(torch.float32)
    for model_id in ("stylealignnet", "stylealign_elastic"):
        torch.manual_seed(0)
        model = build(model_id, input_channels=6, window_length=150, num_classes=27)
        for batch in (8, 1):
            loss = torch.nn.functional.cross_entropy(model(torch.randn(batch, 6, 150)), torch.randint(0, 27, (batch,)))
            loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print(model_id, sum(p.numel() for p in model.parameters()), "params")
    torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
