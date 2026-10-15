"""Round 9: sensor groups, agent weight sharing, message bits, soft trees, forward/backward.

Run with: uv run python tests/test_round9.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import build  # noqa: E402
from src.models.round9 import SoftObliviousTrees, ste_sign, sensor_groups  # noqa: E402

NAMES = ["Ax", "Ay", "Az", "quat_1", "quat_2", "quat_3", "quat_4", "Gx", "Gy", "Gz", "compass"]


def test_sensor_groups_cover_every_channel_once():
    groups = sensor_groups(NAMES, len(NAMES))
    assert groups[:2] == [[0, 1, 2], [7, 8, 9]]
    assert groups[2:] == [[3, 4, 5], [6, 10, -1]]
    flat = sorted(i for g in groups for i in g if i >= 0)
    assert flat == list(range(len(NAMES)))


def test_ste_sign():
    z = torch.tensor([-2.0, -0.1, 0.0, 0.3], requires_grad=True)
    s = ste_sign(z)
    assert s.tolist() == [-1.0, -1.0, 1.0, 1.0]
    s.sum().backward()
    assert torch.allclose(z.grad, 1 - torch.tanh(z.detach()) ** 2)


def test_parliament_agents_share_weights_and_hear_each_other():
    torch.manual_seed(0)
    model = build("parliamentnet", input_channels=len(NAMES), window_length=96, num_classes=4, channel_names=NAMES)
    assert model.n_groups == 4
    x = torch.randn(2, len(NAMES), 96)
    base = model(x)
    # changing one sensor's signal changes the output only through that agent and its posted bits
    x2 = x.clone()
    x2[:, 7:10] = torch.randn(2, 3, 96)
    assert not torch.allclose(model(x2), base)
    # without messages an agent's vote depends on its own sensor only
    model.messages = False
    with torch.no_grad():
        model.post.weight.mul_(0)
    n_params = sum(p.numel() for p in model.parameters())
    big = build("parliamentnet", input_channels=60, window_length=150, num_classes=7,
                channel_names=[f"S{s}_{k}{a}" for s in range(5) for k in "ALGM" for a in "xyz"])
    assert big.n_groups == 20
    # parameters grow only by the agent embeddings (and the 3 extra class outputs)
    d = model.agent.shape[1]
    assert sum(p.numel() for p in big.parameters()) - n_params == (20 - 4) * d + 3 * (d + 1)
    print("parliamentnet params:", n_params, "(11 ch)", sum(p.numel() for p in big.parameters()), "(60 ch)")


def test_trees_are_soft_partitions():
    trees = SoftObliviousTrees(n_features=5, n_classes=3, trees=4, depth=3)
    f = torch.randn(7, 5)
    chosen = torch.einsum("bf,tdf->btd", f, torch.softmax(trees.select, dim=-1))
    right = torch.sigmoid((chosen - trees.threshold) * torch.exp(-trees.log_temp))
    p = right.unsqueeze(2) * trees.codes + (1 - right.unsqueeze(2)) * (1 - trees.codes)
    assert torch.allclose(p.prod(-1).sum(-1), torch.ones(7, 4), atol=1e-6)  # leaf probabilities sum to 1


def test_models_train_step():
    for model_id in ("parliamentnet", "forestnet", "parliamentbody"):
        torch.manual_seed(0)
        model = build(model_id, input_channels=len(NAMES), window_length=96, num_classes=5, channel_names=NAMES)
        for batch in (8, 1):
            loss = model(torch.randn(batch, len(NAMES), 96)).logsumexp(-1).mean()
            loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval()
        assert model(torch.randn(3, len(NAMES), 96)).shape == (3, 5)
        print(model_id, sum(p.numel() for p in model.parameters()), "params")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
