"""Round 10: EcoNet's reverse pass and multistability; GrammarNet's inside algorithm against brute force.

Run with: uv run python tests/test_round10.py
"""

import math
import sys
from functools import lru_cache
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import build  # noqa: E402
from src.models.round10 import (eco_scan, eco_scan_fast, inside, kz_holonomy, prefix_product,  # noqa: E402
                              transpositions)

torch.set_default_dtype(torch.float64)


def test_eco_reverse_pass_matches_autograd():
    torch.manual_seed(0)
    b, length, h, s = 3, 40, 2, 4
    r = (1.5 * torch.randn(b, length, h, s)).requires_grad_()
    a = (0.3 * torch.randn(h, s, s) + torch.eye(s)).requires_grad_()
    l0 = (-0.5 + 0.1 * torch.randn(h, s)).requires_grad_()
    w = torch.randn(b, length, h, s)
    (eco_scan(r, a, 0.2, l0, -3.0, 1.5) * w).sum().backward()
    ref = [x.grad.clone() for x in (r, a, l0)]
    for x in (r, a, l0):
        x.grad = None
    out = eco_scan_fast(r, a, l0, 0.2, -3.0, 1.5)
    assert torch.allclose(out, eco_scan(r, a, 0.2, l0, -3.0, 1.5))
    assert ((out <= -3.0) | (out >= 1.5)).any()  # the clamp is exercised
    (out * w).sum().backward()
    for x, g in zip((r, a, l0), ref):
        assert torch.allclose(x.grad, g, atol=1e-12)


def test_eco_memory_depends_on_order():
    """Two competing species, strong mutual competition: whoever was favoured first keeps winning."""
    a = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])
    l0 = torch.full((1, 2), -2.0)
    first, second = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    neutral = torch.tensor([0.8, 0.8])
    seq_a = torch.stack([first] * 30 + [second] * 3 + [neutral] * 60).view(1, -1, 1, 2)
    seq_b = torch.stack([second] * 30 + [first] * 3 + [neutral] * 60).view(1, -1, 1, 2)
    end_a = eco_scan(seq_a, a, 0.2, l0, -8.0, 4.0)[0, -1, 0]
    end_b = eco_scan(seq_b, a, 0.2, l0, -8.0, 4.0)[0, -1, 0]
    assert end_a[0] > end_a[1] and end_b[1] > end_b[0]  # same final input, different winner


def brute_inside(emit: torch.Tensor, log_rules: torch.Tensor, i: int, j: int, a: int) -> float:
    """Sum over all binary parse trees of tokens i..j-1 rooted at a (exp space)."""

    @lru_cache(maxsize=None)
    def go(i: int, j: int, a: int) -> float:
        if j - i == 1:
            return math.exp(float(emit[i, a]))
        total = 0.0
        for k in range(i + 1, j):
            for bb in range(emit.shape[1]):
                for cc in range(emit.shape[1]):
                    total += math.exp(float(log_rules[a, bb, cc])) * go(i, k, bb) * go(k, j, cc)
        return total

    return go(i, j, a)


def test_inside_matches_enumeration():
    torch.manual_seed(1)
    t, k = 5, 3
    emit = torch.randn(1, t, k)
    log_rules = torch.log_softmax(torch.randn(k, k * k), -1).view(k, k, k)
    beta = inside(emit, log_rules)
    for w in (2, 3, 5):
        for i in range(t - w + 1):
            for a in range(k):
                assert math.isclose(math.exp(float(beta[w][0, i, a])), brute_inside(emit[0], log_rules, i, i + w, a),
                                    rel_tol=1e-9)


def test_prefix_product():
    torch.manual_seed(2)
    m = torch.randn(2, 13, 3, 3)
    out = prefix_product(m)
    acc = torch.eye(3).expand(2, 3, 3)
    for t in range(13):
        acc = m[:, t] @ acc
        assert torch.allclose(out[:, t], acc, atol=1e-10)


def strand_paths(arc_height: float, steps: int = 400) -> torch.Tensor:
    """3 strands; strand 0 moves from -1.5 to +1.5 around strand 1 (at 0) along an arc of the given
    height (positive: above, negative: below); strand 2 stays at 3i. Returns z (1, steps, 1, 3)."""
    s = torch.linspace(0, 1, steps)
    z0 = torch.complex(-1.5 + 3 * s, arc_height * torch.sin(torch.pi * s))
    z = torch.stack([z0, torch.zeros_like(z0), torch.full_like(z0, 3j)], dim=-1)
    return z.view(1, steps, 1, 3)


def test_braid_holonomy_is_topological():
    tau = transpositions(3).to(torch.complex128)
    iu = torch.triu_indices(3, 3, 1)
    kappa = torch.tensor([0.4 + 0.3j], dtype=torch.complex128)
    hol = lambda h: kz_holonomy(strand_paths(h).to(torch.complex128), kappa, tau, iu[0], iu[1], 0.0)[0][0, 0, -1]  # noqa: E731
    over_a, over_b, under = hol(1.0), hol(0.6), hol(-1.0)
    assert torch.allclose(over_a, over_b, atol=2e-3)  # homotopic paths (same side): same holonomy
    assert (over_a - under).abs().max() > 0.05  # passing on the other side of strand 1: another braid
    assert not torch.allclose(over_a, torch.eye(3, dtype=torch.complex128) / 3 ** 0.5, atol=0.05)


def test_models_train_step():
    torch.set_default_dtype(torch.float32)
    for model_id in ("econet", "grammarnet", "bhnet"):
        torch.manual_seed(0)
        model = build(model_id, input_channels=6, window_length=150, num_classes=27)
        for batch in (8, 1):
            loss = model(torch.randn(batch, 6, 150)).logsumexp(-1).mean()
            loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print(model_id, sum(p.numel() for p in model.parameters()), "params")
    torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
