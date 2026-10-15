"""Round-8 candidates: classify by the laws that generate the motion (RESEARCH_PLAN.md §5e).

Rounds 1-7 looked for a new temporal memory. The comparison with the WHAR Arena table
(results/metrics/arena/compare.md §5-6) pointed elsewhere: models differ on unseen people, and deep models
fall behind the Random Forest as the number of sensors grows. Round 8 therefore reads quantities
that should not change from one wearer to the next.

LawNet. For every 3-axis sensor the window's second moments are accumulated (streaming sums), and
two kinds of laws are solved in closed form at the end of the window:
  - a temporal operator per sensor and lag tau, x_t ~ A_tau x_{t-tau} (ridge least squares);
  - the coupling between every pair of sensors, through their canonical correlations at a few lags.
Only invariants are read out:
  - the characteristic polynomial of A_tau (trace, second elementary symmetric function,
    determinant). These are unchanged by any rotation of the sensor (A -> R A R^T), and by any
    re-scaling of the whole sensor, since the ridge is relative;
  - the elementary symmetric functions of the squared canonical correlations of a pair, unchanged
    by an independent rotation of each sensor;
  - the shape of the sensor's covariance (log energy, tr(C^2)/tr(C)^2, det(C)/tr(C)^3) and the
    length of its mean vector, unchanged by rotation.
A sensor's invariants and the mean of its pair embeddings go through one MLP shared by all sensors;
the per-sensor embeddings are concatenated and projected, and join the pooled output of a
memoryless (or recurrence) skeleton at the head. On a microcontroller: per sensor 3 + 6 + 9 per
lag accumulators and a ring buffer of the largest lag; 3x3 solves once per window.

Sensors are found from the channel names by one rule for every dataset (no per-dataset map):
three consecutive channels whose names differ in exactly one character, reading x, y, z (any
case). Channels outside such triples (quaternions, compass, 2-axis gyros) stay in the main path only.

Controls (same code):
  - lawnet_rand: the same number of triples, drawn from a fixed random permutation of the
    channels, so the "sensors" are not physical;
  - statnet / recstatnet: per-channel mean, std, min, max (the statistics behind the Random
    Forest's tsfresh features) in place of the laws.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F
from torch import nn

from src.models.round6 import RecurrenceBlock

AXES = ("xyz", "XYZ")


def find_triples(names: list[str] | None, n_channels: int) -> list[tuple[int, int, int]]:
    """3-axis sensors: consecutive channels whose names differ only in one position reading x, y, z.

    Without names (models built outside run_whar.py) every consecutive triple is taken."""
    if names is None or len(names) != n_channels:
        return [(i, i + 1, i + 2) for i in range(0, n_channels - 2, 3)]
    triples, i = [], 0
    while i + 2 < n_channels:
        a, b, c = names[i:i + 3]
        diff = [k for k in range(len(a)) if len(a) == len(b) == len(c) and not (a[k] == b[k] == c[k])]
        if len(diff) == 1 and a[diff[0]] + b[diff[0]] + c[diff[0]] in AXES:
            triples.append((i, i + 1, i + 2))
            i += 3
        else:
            i += 1
    return triples


def random_triples(n_channels: int, n_groups: int, seed: int = 8) -> list[tuple[int, int, int]]:
    """n_groups triples from a fixed random permutation (independent of the training seed)."""
    perm = torch.randperm(n_channels, generator=torch.Generator().manual_seed(seed)).tolist()
    return [tuple(perm[3 * g:3 * g + 3]) for g in range(min(n_groups, n_channels // 3))]


def inv3(m: torch.Tensor) -> torch.Tensor:
    """Inverse of (..., 3, 3) matrices by the adjugate (closed form, as on a microcontroller)."""
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    co = torch.stack([e * i - f * h, -(d * i - f * g), d * h - e * g,
                      -(b * i - c * h), a * i - c * g, -(a * h - b * g),
                      b * f - c * e, -(a * f - c * d), a * e - b * d], dim=-1).unflatten(-1, (3, 3))
    det = a * co[..., 0, 0] + b * co[..., 0, 1] + c * co[..., 0, 2]
    return co.transpose(-1, -2) / det[..., None, None]


def charpoly3(a: torch.Tensor) -> torch.Tensor:
    """Coefficients of the characteristic polynomial of (..., 3, 3): trace, e2, det -> (..., 3)."""
    tr = a.diagonal(dim1=-2, dim2=-1).sum(-1)
    tr2 = (a * a.transpose(-1, -2)).sum((-1, -2))  # tr(A^2)
    return torch.stack([tr, 0.5 * (tr * tr - tr2), _det3(a)], dim=-1)


def _det3(m: torch.Tensor) -> torch.Tensor:
    return (m[..., 0, 0] * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
            - m[..., 0, 1] * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
            + m[..., 0, 2] * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0]))


def ridge(c: torch.Tensor, rel: float, absolute: float) -> torch.Tensor:
    """c + (rel * tr(c) / 3 + absolute) I: relative, so a re-scaled sensor gives the same operator."""
    lam = rel * c.diagonal(dim1=-2, dim2=-1).sum(-1) / 3 + absolute
    return c + lam[..., None, None] * torch.eye(3, dtype=c.dtype, device=c.device)


def lagged_cov(a: torch.Tensor, b: torch.Tensor, lag: int) -> torch.Tensor:
    """E[a_t b_{t-lag}^T] over the window: a, b (B, G, 3, L) centred -> (B, G, 3, 3); lag may be < 0."""
    length = a.shape[-1]
    if lag > 0:
        a, b = a[..., lag:], b[..., :length - lag]
    elif lag < 0:
        a, b = a[..., :length + lag], b[..., -lag:]
    return a @ b.transpose(-1, -2) / a.shape[-1]


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that falls back to its running statistics for a training batch of one window
    (the harness's DataLoader keeps a last batch of any size)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.shape[0] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


class LawFeatures(nn.Module):
    """Invariants of the per-sensor temporal operators and of the sensor couplings -> (B, out)."""

    N_OWN = 4  # covariance shape (3) + mean length (1)

    def __init__(self, triples: list[tuple[int, int, int]], window_length: int,
                 lags: tuple[int, ...] = (1, 2, 4, 8, 16), cross_lags: tuple[int, ...] = (-4, 0, 4),
                 rel_ridge: float = 1e-3, abs_ridge: float = 1e-4, hidden: int = 16, out: int = 32) -> None:
        super().__init__()
        self.lags = tuple(lag for lag in lags if lag < window_length - 2)
        self.cross_lags = tuple(lag for lag in cross_lags if abs(lag) < window_length - 2)
        self.rel_ridge, self.abs_ridge = rel_ridge, abs_ridge
        self.register_buffer("index", torch.tensor(triples, dtype=torch.long).reshape(-1, 3), persistent=False)
        n_groups = len(triples)
        pairs = list(itertools.combinations(range(n_groups), 2))
        self.register_buffer("pair_g", torch.tensor([p[0] for p in pairs], dtype=torch.long), persistent=False)
        self.register_buffer("pair_h", torch.tensor([p[1] for p in pairs], dtype=torch.long), persistent=False)
        # pair -> group incidence, averaged (each pair informs both of its sensors)
        incidence = torch.zeros(n_groups, len(pairs))
        for k, (g, h) in enumerate(pairs):
            incidence[g, k] = incidence[h, k] = 1.0
        self.register_buffer("incidence", incidence / incidence.sum(1, keepdim=True).clamp_min(1.0),
                             persistent=False)
        self.n_groups, self.n_pairs = n_groups, len(pairs)
        n_own, n_pair_feat, n_pair_emb = 3 * len(self.lags) + self.N_OWN, 3 * len(self.cross_lags), 8 if pairs else 0
        self.norm = SafeBatchNorm1d(n_groups * n_own) if n_groups else None  # per sensor and invariant
        self.pair_norm = SafeBatchNorm1d(n_pair_feat) if pairs else None  # shared by all pairs
        self.pair_mlp = nn.Sequential(nn.Linear(n_pair_feat, n_pair_emb), nn.GELU()) if pairs else None
        self.group_mlp = nn.Sequential(nn.Linear(n_own + n_pair_emb, hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden), nn.GELU())
        self.proj = nn.Linear(max(n_groups, 1) * hidden, out)
        self.out = out

    def invariants(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x (B, C, L) -> own (B, G, 3 * lags + 4) and pair (B, P, 3 * cross lags) invariants."""
        xs = x[:, self.index]  # (B, G, 3, L)
        mean = xs.mean(-1, keepdim=True)
        y = xs - mean
        length = y.shape[-1]
        feats = []
        for lag in self.lags:
            past = y[..., :length - lag]
            cur_past = y[..., lag:] @ past.transpose(-1, -2) / past.shape[-1]
            past_past = past @ past.transpose(-1, -2) / past.shape[-1]
            a = cur_past @ inv3(ridge(past_past, self.rel_ridge, self.abs_ridge))
            feats.append(charpoly3(a))
        c0 = y @ y.transpose(-1, -2) / length
        s = c0.diagonal(dim1=-2, dim2=-1).sum(-1) + 1e-6
        shape = torch.stack([torch.log(s), (c0 * c0).sum((-1, -2)) / s ** 2, _det3(c0) / s ** 3], dim=-1)
        m2 = torch.log(mean.squeeze(-1).square().sum(-1, keepdim=True) + 1e-4)
        own = torch.cat(feats + [shape, m2], dim=-1)
        if not self.n_pairs:
            return own, None
        cg = ridge(c0, self.rel_ridge, self.abs_ridge)
        inv = inv3(cg)  # (B, G, 3, 3)
        yg, yh = y[:, self.pair_g], y[:, self.pair_h]  # (B, P, 3, L)
        inv_g, inv_h = inv[:, self.pair_g], inv[:, self.pair_h]
        pair = []
        for lag in self.cross_lags:
            cgh = lagged_cov(yg, yh, lag)
            k = inv_g @ cgh @ inv_h @ cgh.transpose(-1, -2)  # squared canonical correlations as eigenvalues
            pair.append(charpoly3(k))
        return own, torch.cat(pair, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        if not self.n_groups:
            return x.new_zeros(x.shape[0], self.out)
        own, pair = self.invariants(x)
        batch = own.shape[0]
        own = self.norm(torch.asinh(own).flatten(1)).view(batch, self.n_groups, -1)
        if pair is not None:
            pe = self.pair_mlp(self.pair_norm(torch.asinh(pair).flatten(0, 1)).view(batch, self.n_pairs, -1))
            own = torch.cat([own, torch.einsum("gp,bpf->bgf", self.incidence, pe)], dim=-1)
        return self.proj(self.group_mlp(own).flatten(1))


class ChannelStats(nn.Module):
    """Per-channel window statistics (the Random Forest's kind of features) -> (B, out).

    stats: any of "mean" (offset, i.e. posture), "std" (intensity), "min", "max" (range); round 11's
    ablation uses subsets."""

    FUNCS = {"mean": lambda x: x.mean(-1), "std": lambda x: x.std(-1),
             "min": lambda x: x.amin(-1), "max": lambda x: x.amax(-1)}

    def __init__(self, n_channels: int, out: int = 32, stats: tuple[str, ...] = ("mean", "std", "min", "max")) -> None:
        super().__init__()
        self.stats = tuple(stats)
        self.norm = SafeBatchNorm1d(len(self.stats) * n_channels)
        self.proj = nn.Linear(len(self.stats) * n_channels, out)
        self.out = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        feats = torch.stack([self.FUNCS[s](x) for s in self.stats], dim=-1)
        return self.proj(self.norm(feats.flatten(1)))


class PlainBlock(nn.Module):
    """The skeleton's block without memory: LayerNorm -> mixing -> mixing -> GELU -> residual."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mix_in = nn.Linear(d_model, d_model)
        self.mix_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.mix_out(self.mix_in(self.norm(x))))


class Round8Net(nn.Module):
    """encoder -> blocks -> mean over time, joined at the head by a window-level side path.

    side = "law" (LawNet), "law_rand" (random triples), "stats" (per-channel statistics).
    blocks = "plain" (memoryless) or "recurrence" (RecurrenceNet's blocks)."""

    BLOCKS = {"plain": PlainBlock, "recurrence": RecurrenceBlock}
    NEEDS_CHANNEL_NAMES = True

    def __init__(self, input_channels: int, window_length: int, num_classes: int, *, side: str,
                 blocks: str = "plain", d_model: int = 32, num_layers: int = 3,
                 channel_names: list[str] | None = None, **side_kwargs) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_channels, d_model)
        self.blocks = nn.ModuleList(self.BLOCKS[blocks](d_model) for _ in range(num_layers))
        triples = find_triples(channel_names, input_channels)
        if side == "law":
            self.side = LawFeatures(triples, window_length, **side_kwargs)
        elif side == "law_rand":
            self.side = LawFeatures(random_triples(input_channels, len(triples)), window_length, **side_kwargs)
        elif side == "stats":
            self.side = ChannelStats(input_channels, **side_kwargs)
        else:
            raise ValueError(side)
        self.triples = triples
        self.norm = nn.LayerNorm(d_model + self.side.out)
        self.head = nn.Linear(d_model + self.side.out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = self.encoder(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(torch.cat([h.mean(dim=1), self.side(x)], dim=-1)))
