"""Round 14: LevelNet's level maps and phase plane on known signals; train step for every id.

Run with: uv run python tests/test_round14.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import build  # noqa: E402
from src.models.round14 import level_maps, plane_density  # noqa: E402

torch.set_default_dtype(torch.float64)


def test_level_maps_of_a_sinusoid():
    """Sharp steps: occupation above 0 is one half, and 4 cycles cross level 0 eight times."""
    length = 400
    t = torch.arange(length) / length
    s = torch.sin(2 * math.pi * 4 * t + 0.3)[None, None]
    levels = torch.tensor([-2.0, 0.0, 0.5, 2.0])
    m = level_maps(s, levels, 1e-4, segments=3, crossings=True)[0, 0]  # (1 + 3 + 1, M)
    assert abs(m[0, 1] - 0.5) < 0.01
    assert abs(m[0, 0] - 1.0) < 1e-6 and m[0, 3] < 1e-6  # always above -2, never above 2
    assert abs(m[4, 1] * (length - 1) - 8) < 1e-6, m[4, 1] * (length - 1)
    assert m[4, 0] < 1e-6 and m[4, 3] < 1e-6


def test_plane_density_is_a_distribution():
    torch.manual_seed(0)
    s, ds = torch.randn(2, 3, 50), torch.randn(2, 3, 50)
    d = plane_density(s, ds, torch.linspace(-2.5, 2.5, 16), 1 / 3)
    assert d.shape == (2, 3, 16, 16)
    assert torch.allclose(d.sum(dim=(-1, -2)), torch.ones(2, 3))


def test_early_mixing_control_tokenizes_mixed_channels():
    model = build("levelnet_early", input_channels=6, window_length=150, num_classes=27)
    assert model.channels(torch.randn(4, 6, 150)).shape == (4, 32, 150)
    late = build("levelnet", input_channels=6, window_length=150, num_classes=27)
    assert late.channels(torch.randn(4, 6, 150)).shape == (4, 14, 150)


def test_train_step_all_models():
    torch.set_default_dtype(torch.float32)
    try:
        for model_id in ("levelnet", "levelnet_plane", "levelnet_early", "levelnet_nocross", "levelnet_mlp"):
            for c, length, k in ((6, 150, 27), (133, 90, 5), (3, 60, 6)):
                torch.manual_seed(0)
                model = build(model_id, input_channels=c, window_length=length, num_classes=k)
                x, y = torch.randn(8, c, length), torch.randint(0, k, (8,))
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                assert torch.isfinite(loss)
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), model_id
                model.eval()
                assert model(x).shape == (8, k)
    finally:
        torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
