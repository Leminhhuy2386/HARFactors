"""Round-9 candidates (RESEARCH_PLAN.md §5f): deep models that scale with the number of sensors.

Round 8 confirmed the gap the WHAR Arena table shows: our skeleton squeezes every channel through one
linear layer, and the Random Forest wins as sensors are added. Two answers, screened fast:

ParliamentNet. Every physical 3-axis sensor is an agent (sensors found by round 8's name rule; the
remaining channels are packed three by three, zero-padded). All agents share one network: a patch
embedding of their own 3 channels (5 samples per token), a learned agent embedding, and RecurrenceNet
blocks as their memory. After the window each agent posts a few bits (sign of a learned projection,
straight-through gradient) to a shared board and reads the mean of the others' bits; it then casts
class scores and a confidence, and the network's output is the confidence-weighted vote. Weight
sharing gives every sensor the full model without the parameter count growing with the sensor
count; the bit budget is what a body-area network with one MCU per sensor would pay by radio.

ForestNet. RecurrenceNet's trunk and the per-channel window statistics (mean, std, min, max) feed
soft oblivious decision trees: at each depth a tree compares one softly selected feature with a
threshold (sigmoid split), and leaves hold class scores. Trees split on one feature at a time, the
way the Random Forest that wins on many-channel datasets does. Control: round 8's `recstatnet`
(same features, linear head).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.round6 import RecurrenceBlock
from src.models.round8 import SafeBatchNorm1d, find_triples


def sensor_groups(names: list[str] | None, n_channels: int) -> list[list[int]]:
    """Physical triples first, then the remaining channels in order, three per group; -1 pads."""
    triples = [list(t) for t in find_triples(names, n_channels)]
    used = {i for t in triples for i in t}
    rest = [i for i in range(n_channels) if i not in used]
    for k in range(0, len(rest), 3):
        g = rest[k:k + 3]
        triples.append(g + [-1] * (3 - len(g)))
    return triples


def ste_sign(z: torch.Tensor) -> torch.Tensor:
    """sign(z) forward, d tanh(z) backward (straight-through)."""
    t = torch.tanh(z)
    return t + (torch.where(z >= 0, 1.0, -1.0).to(z) - t).detach()


class ParliamentNet(nn.Module):
    NEEDS_CHANNEL_NAMES = True

    def __init__(self, input_channels: int, window_length: int, num_classes: int, *, d_model: int = 32,
                 num_layers: int = 2, patch: int = 5, bits: int = 4, messages: bool = True, n_embed: int = 4,
                 lags: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16), body_dim: int = 0,
                 channel_names: list[str] | None = None) -> None:
        """body_dim > 0 adds a whole-body agent: a RecurrenceNet over all channels at full rate
        (3 blocks of width body_dim) that posts, listens and votes like the sensor agents."""
        super().__init__()
        groups = sensor_groups(channel_names, input_channels)
        self.groups = groups
        index = torch.tensor(groups, dtype=torch.long)
        self.register_buffer("index", torch.where(index < 0, input_channels, index), persistent=False)
        self.patch, self.messages = patch, messages
        self.n_groups = len(groups)
        self.embed = nn.Linear(3 * patch, d_model)
        self.agent = nn.Parameter(0.02 * torch.randn(self.n_groups, d_model))
        self.blocks = nn.ModuleList(RecurrenceBlock(d_model, n_embed=n_embed, lags=lags) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.post = nn.Linear(d_model, bits)
        self.vote = nn.Sequential(nn.Linear(d_model + bits, d_model), nn.GELU(), nn.Linear(d_model, num_classes + 1))
        self.body = None
        if body_dim:
            self.body = nn.ModuleDict({
                "encoder": nn.Linear(input_channels, body_dim),
                "blocks": nn.ModuleList(RecurrenceBlock(body_dim) for _ in range(3)),
                "norm": nn.LayerNorm(body_dim),
                "post": nn.Linear(body_dim, bits),
                "vote": nn.Sequential(nn.Linear(body_dim + bits, body_dim), nn.GELU(),
                                      nn.Linear(body_dim, num_classes + 1)),
            })

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        batch, _, length = x.shape
        body_h = None
        if self.body is not None:
            hb = self.body["encoder"](x.transpose(1, 2))
            for block in self.body["blocks"]:
                hb = block(hb)
            body_h = self.body["norm"](hb.mean(dim=1))  # (B, body_dim)
        pad = (-length) % self.patch
        x = F.pad(torch.cat([x, x.new_zeros(batch, 1, length)], dim=1), (0, pad))  # zero channel for padding
        xs = x[:, self.index]  # (B, G, 3, L')
        tokens = xs.unflatten(-1, (-1, self.patch)).permute(0, 1, 3, 2, 4).flatten(3)  # (B, G, T, 3 * patch)
        h = self.embed(tokens) + self.agent[None, :, None, :]
        h = h.flatten(0, 1)  # every agent of every window is one sequence through the shared blocks
        for block in self.blocks:
            h = block(h)
        h = self.norm(h.mean(dim=1)).view(batch, self.n_groups, -1)
        bits = ste_sign(self.post(h))  # (B, G, bits): what each agent posts
        if body_h is not None:
            bits = torch.cat([bits, ste_sign(self.body["post"](body_h)).unsqueeze(1)], dim=1)
        n_posts = bits.shape[1]
        if self.messages and n_posts > 1:
            heard = (bits.sum(dim=1, keepdim=True) - bits) / (n_posts - 1)  # mean of the others' posts
        else:
            heard = torch.zeros_like(bits)
        out = self.vote(torch.cat([h, heard[:, :self.n_groups]], dim=-1))  # (B, G, K + 1)
        if body_h is not None:
            out = torch.cat([out, self.body["vote"](torch.cat([body_h, heard[:, -1]], dim=-1)).unsqueeze(1)], dim=1)
        weight = torch.softmax(out[..., -1], dim=1)
        return (weight.unsqueeze(-1) * out[..., :-1]).sum(dim=1)


class SoftObliviousTrees(nn.Module):
    """T trees of depth D; each depth of a tree softly selects one feature and compares it with a
    threshold. Leaf probabilities are products of the D sigmoid decisions; output = mean leaf scores."""

    def __init__(self, n_features: int, n_classes: int, trees: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        self.select = nn.Parameter(0.1 * torch.randn(trees, depth, n_features))
        self.threshold = nn.Parameter(torch.zeros(trees, depth))
        self.log_temp = nn.Parameter(torch.zeros(trees, depth))
        self.leaves = nn.Parameter(0.1 * torch.randn(trees, 2 ** depth, n_classes))
        codes = torch.tensor([[(leaf >> d) & 1 for d in range(depth)] for leaf in range(2 ** depth)], dtype=torch.float32)
        self.register_buffer("codes", codes, persistent=False)  # (2^D, D): branch taken at each depth

    def forward(self, f: torch.Tensor) -> torch.Tensor:  # (B, F)
        chosen = torch.einsum("bf,tdf->btd", f, torch.softmax(self.select, dim=-1))
        right = torch.sigmoid((chosen - self.threshold) * torch.exp(-self.log_temp))  # (B, T, D)
        p = right.unsqueeze(2) * self.codes + (1 - right.unsqueeze(2)) * (1 - self.codes)  # (B, T, 2^D, D)
        leaf = p.prod(dim=-1)  # (B, T, 2^D)
        return torch.einsum("btl,tlk->bk", leaf, self.leaves) / self.select.shape[0]


class ForestNet(nn.Module):
    """RecurrenceNet trunk + per-channel window statistics -> soft oblivious trees.

    The trees see the raw (batch-normalized) statistics of every channel, not a projection, so a
    split can test one channel's mean or range as a Random Forest split does."""

    def __init__(self, input_channels: int, window_length: int, num_classes: int, *, d_model: int = 32,
                 num_layers: int = 3, trees: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_channels, d_model)
        self.blocks = nn.ModuleList(RecurrenceBlock(d_model) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.stat_norm = SafeBatchNorm1d(4 * input_channels)
        self.forest = SoftObliviousTrees(d_model + 4 * input_channels, num_classes, trees, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = self.encoder(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        stats = torch.stack([x.mean(-1), x.std(-1), x.amin(-1), x.amax(-1)], dim=-1).flatten(1)
        return self.forest(torch.cat([self.norm(h.mean(dim=1)), self.stat_norm(stats)], dim=-1))
