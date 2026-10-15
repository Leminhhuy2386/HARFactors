"""Round 8 (LawNet): sensor finding, closed-form algebra, invariance, streaming sums, forward/backward.

Run with: uv run python tests/test_round8.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import CONFIGS, build  # noqa: E402
from src.models.round8 import LawFeatures, charpoly3, find_triples, inv3, random_triples  # noqa: E402

torch.set_default_dtype(torch.float64)

ACTRECTUT = ["acc_1_x", "acc_1_y", "acc_1_z", "gyr_1_x", "gyr_1_y", "acc_2_x", "acc_2_y", "acc_2_z",
             "gyr_2_x", "gyr_2_y", "acc_3_x", "acc_3_y", "acc_3_z", "gyr_3_x", "gyr_3_y"]
OPP_PART = ["IMU_BACK_mag_x", "IMU_BACK_mag_y", "IMU_BACK_mag_z", "IMU_BACK_quat_1", "IMU_BACK_quat_2",
            "IMU_BACK_quat_3", "IMU_BACK_quat_4", "IMU_RUA_acc_x", "IMU_RUA_acc_y", "IMU_RUA_acc_z",
            "IMU_LSHOE_angvel_nav_z", "IMU_LSHOE_compass", "T_xacc", "T_yacc", "T_zacc", "GyroX", "GyroY", "GyroZ"]


def rotation(gen: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=gen))
    return q * torch.sign(torch.diagonal(r))


def test_find_triples():
    assert find_triples(["Ax", "Ay", "Az", "GyroX", "GyroY", "GyroZ"], 6) == [(0, 1, 2), (3, 4, 5)]
    assert find_triples(ACTRECTUT, 15) == [(0, 1, 2), (5, 6, 7), (10, 11, 12)]
    assert find_triples(OPP_PART, len(OPP_PART)) == [(0, 1, 2), (7, 8, 9), (12, 13, 14), (15, 16, 17)]
    assert find_triples(None, 7) == [(0, 1, 2), (3, 4, 5)]
    assert sorted(i for t in random_triples(9, 3) for i in t) == list(range(9))


def test_closed_forms():
    gen = torch.Generator().manual_seed(0)
    m = torch.randn(50, 3, 3, generator=gen) + 3 * torch.eye(3)
    assert torch.allclose(inv3(m), torch.linalg.inv(m), atol=1e-10)
    ev = torch.linalg.eigvals(m)
    want = torch.stack([ev.sum(-1), ev[:, 0] * ev[:, 1] + ev[:, 0] * ev[:, 2] + ev[:, 1] * ev[:, 2], ev.prod(-1)], -1)
    assert torch.allclose(charpoly3(m), want.real, atol=1e-9)


def test_invariance_to_sensor_rotation_and_scale():
    """Rotating each sensor (and re-scaling it) leaves the operator and coupling invariants unchanged.
    Exact for scale only without the small absolute ridge (abs_ridge, a floor for still sensors)."""
    gen = torch.Generator().manual_seed(1)
    triples = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    feats = LawFeatures(triples, window_length=120, abs_ridge=0.0)
    x = torch.cumsum(torch.randn(4, 9, 120, generator=gen), dim=-1) * 0.1 + torch.randn(4, 9, 1, generator=gen)
    own, pair = feats.invariants(x)
    xr = x.clone()
    for g, (a, b, c) in enumerate(triples):
        scale = 0.5 + g
        xr[:, [a, b, c]] = scale * torch.einsum("ij,bjt->bit", rotation(gen), x[:, [a, b, c]])
    own_r, pair_r = feats.invariants(xr)
    n_op = 3 * len(feats.lags)
    assert torch.allclose(own[..., :n_op], own_r[..., :n_op], atol=1e-8)  # temporal operators
    assert torch.allclose(own[..., n_op + 1:n_op + 3], own_r[..., n_op + 1:n_op + 3], atol=1e-5)  # shape (1e-6 floor)
    assert torch.allclose(pair, pair_r, atol=1e-8)  # canonical correlations
    # rotation alone also keeps the energy and the mean length
    xr = x.clone()
    for a, b, c in triples:
        xr[:, [a, b, c]] = torch.einsum("ij,bjt->bit", rotation(gen), x[:, [a, b, c]])
    assert torch.allclose(own, feats.invariants(xr)[0], atol=1e-8)


def test_streaming_sums_match():
    """The operator from per-sample running sums (as on an MCU) equals the batch computation."""
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(1, 3, 60, generator=gen).cumsum(-1)
    lag, n = 4, 60 - 4
    s_cur, s_past = torch.zeros(3), torch.zeros(3)
    s_cp, s_pp = torch.zeros(3, 3), torch.zeros(3, 3)
    for t in range(lag, 60):  # one multiply-add per accumulator per sample, ring buffer of `lag` samples
        cur, past = x[0, :, t], x[0, :, t - lag]
        s_cur, s_past = s_cur + cur, s_past + past
        s_cp, s_pp = s_cp + torch.outer(cur, past), s_pp + torch.outer(past, past)
    mean = x[0].mean(-1)  # window mean, also a running sum
    cp = (s_cp - torch.outer(s_cur, mean) - torch.outer(mean, s_past) + n * torch.outer(mean, mean)) / n
    pp = (s_pp - torch.outer(s_past, mean) - torch.outer(mean, s_past) + n * torch.outer(mean, mean)) / n
    y = x[0] - mean[:, None]
    assert torch.allclose(cp, y[:, lag:] @ y[:, :-lag].T / n, atol=1e-10)
    assert torch.allclose(pp, y[:, :-lag] @ y[:, :-lag].T / n, atol=1e-10)


def test_models_train_step():
    names = ["Ax", "Ay", "Az", "GyroX", "GyroY", "GyroZ", "Mx", "My", "Mz", "extra"]
    for model_id in ("lawnet", "lawnet_rand", "statnet", "lawnet_rec", "recstatnet"):
        torch.manual_seed(0)
        model = build(model_id, input_channels=10, window_length=96, num_classes=5, channel_names=names)
        assert len(model.triples) == 3
        for batch in (8, 1):  # a last training batch of one window must not break BatchNorm
            x = torch.randn(batch, 10, 96)
            loss = model(x).logsumexp(-1).mean()
            loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        model.eval()
        assert model(torch.randn(3, 10, 96)).shape == (3, 5)
        print(model_id, sum(p.numel() for p in model.parameters()), "params")
    assert CONFIGS["lawnet"][1]["side"] == "law"


def test_random_triples_control_is_not_physical():
    names = [f"S{s}_{a}" for s in range(5) for a in "xyz"]
    model = build("lawnet_rand", input_channels=15, window_length=96, num_classes=3, channel_names=names)
    assert len(model.side.index) == 5 and model.side.index.tolist() != find_triples(names, 15)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
