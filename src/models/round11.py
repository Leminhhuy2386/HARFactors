"""Round 11 (RESEARCH_PLAN.md §5h): StyleAlignNet, a deep path plus an elastic style-fitting path.

Why (round-11 evaluation, NEW_ARCHITECTURES.md): on gesture datasets with few training people, plain 1-NN
matching after fitting each window's tempo (DTW) and gain/offset (per-window standardization) beats every
deep model (ActRecTut 69.0 vs RecurrenceNet 40.4), but the same matching loses on locomotion and erases
posture. So the network keeps RecurrenceNet's non-invariant path and adds an elastic one:

  - a linear embedding of the signal (E dims), averaged over patches of P samples (tokens);
  - the window's style is fitted in closed form, per embedding dimension: offset (mean) and gain (std);
  - the standardized tokens are aligned with M learned programs per class (learned token sequences) by a
    banded soft-DTW; a class's evidence is the soft-min of its programs' alignment costs;
  - the logits are the deep head's output plus the elastic evidence (a direct vote); the head also sees the
    fitted style (offset, log gain) and the smallest alignment cost over all programs (evidence that the
    window matches no class well, for null classes).

Soft-DTW is computed row by row inside the band: with e_j = c_j + softmin(D[i-1, j], D[i-1, j-1]), the row
recurrence D[i,j] = softmin(e_j, c_j + D[i,j-1]) is linear in the exp domain, so D = S + softcummin(e - S)
with S = cumsum(c) and softcummin(v) = -gamma * logcumsumexp(-v / gamma).

Literature check (2026-10-11): partly new. Learned class prototypes with (soft) DTW exist (DP-DTW, CVPR 2021;
teNN, 2024; DTWNet, NeurIPS 2019; soft-DTW barycenters, 2017). Not found: the closed-form gain/offset fit
inside the layer with the fitted style as features, the null-class evidence from the smallest cost, the fusion
with a deep path, or learned soft-DTW templates for IMU HAR at all.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.round6 import RecurrenceBlock

BIG = 1e5  # stands for +inf (finite, so softmin gradients stay defined)


def softmin2(a: torch.Tensor, b: torch.Tensor, gamma: float) -> torch.Tensor:
    return -gamma * torch.logaddexp(-a / gamma, -b / gamma)


def soft_dtw_band(x: torch.Tensor, prog: torch.Tensor, w: int, gamma: float) -> torch.Tensor:
    """x (B, T, E) windows, prog (P, T, E) programs -> (B, P) banded soft-DTW cost (squared Euclidean)."""
    length = x.shape[1]
    offs = torch.arange(-w, w + 1, device=x.device)
    jj = torch.arange(length, device=x.device)[:, None] + offs  # (T, K): column of every band cell
    valid_all = (jj >= 0) & (jj < length)
    jc_all = jj.clamp(0, length - 1)
    # every band cell's local cost at once: (B, P, T, K)
    cost_all = ((x * x).sum(-1)[:, None, :, None] + (prog * prog).sum(-1)[:, jc_all][None]
                - 2 * torch.einsum("bte,ptke->bptk", x, prog[:, jc_all])).clamp_min(0)
    big = torch.full((), BIG, device=x.device, dtype=x.dtype)
    prev = None
    for i in range(length):
        j, valid, cost = jj[i], valid_all[i], cost_all[:, :, i]
        if prev is None:
            e = torch.where(j == 0, cost, big)
        else:
            up = torch.cat([prev[..., 1:], torch.full_like(prev[..., :1], BIG)], dim=-1)  # D[i-1, j]
            e = cost + softmin2(up, prev, gamma)  # prev[..., k] is D[i-1, j-1]
        c0 = torch.where(valid, cost, torch.zeros_like(cost))
        e = torch.where(valid, e, big)
        s = torch.cumsum(c0, dim=-1)
        row = s - gamma * torch.logcumsumexp(-(e - s) / gamma, dim=-1)
        prev = torch.where(valid, row.clamp_max(BIG), big)
    return prev[..., w]


class StyleAlignNet(nn.Module):
    def __init__(self, input_channels: int, window_length: int, num_classes: int, *, d_model: int = 32,
                 num_layers: int = 3, embed: int = 8, patch: int = 5, programs: int = 2, band: float = 0.3,
                 gamma: float = 0.5, deep: bool = True) -> None:
        """deep=False drops the RecurrenceNet path (control: the elastic path alone)."""
        super().__init__()
        self.patch, self.gamma, self.programs, self.num_classes = patch, gamma, programs, num_classes
        self.tokens = -(-window_length // patch)
        self.band = max(1, math.ceil(band * self.tokens))
        self.embed = nn.Linear(input_channels, embed)
        self.prog = nn.Parameter(torch.randn(num_classes * programs, self.tokens, embed))
        self.log_temp = nn.Parameter(torch.zeros(()))  # elastic vote = -cost / temperature
        self.deep = deep
        if deep:
            self.encoder = nn.Linear(input_channels, d_model)
            self.blocks = nn.ModuleList(RecurrenceBlock(d_model) for _ in range(num_layers))
            self.norm = nn.LayerNorm(d_model)
        n_feat = (d_model if deep else 0) + 2 * embed + 1
        self.head_norm = nn.LayerNorm(n_feat)
        self.head = nn.Linear(n_feat, num_classes)

    def elastic(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x (B, C, L) -> class evidence (B, K), style (B, 2E), smallest normalized cost (B, 1)."""
        z = self.embed(x.transpose(1, 2))  # (B, L, E)
        length = z.shape[1]
        tok = F.pad(z, (0, 0, 0, (-length) % self.patch)).unflatten(1, (-1, self.patch)).mean(2)  # (B, T, E)
        mu, sd = tok.mean(1, keepdim=True), tok.std(1, keepdim=True) + 1e-3
        cost = soft_dtw_band((tok - mu) / sd, self.prog, self.band, self.gamma) / self.tokens  # (B, K*M)
        per_class = cost.view(-1, self.num_classes, self.programs)
        class_cost = -self.gamma * torch.logsumexp(-per_class / self.gamma, dim=-1)  # soft-min over programs
        evidence = -class_cost / torch.exp(self.log_temp)
        style = torch.cat([mu.squeeze(1), torch.log(sd.squeeze(1))], dim=-1)
        return evidence, style, cost.amin(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        evidence, style, best = self.elastic(x)
        feats = [style, best]
        if self.deep:
            h = self.encoder(x.transpose(1, 2))
            for block in self.blocks:
                h = block(h)
            feats.insert(0, self.norm(h.mean(dim=1)))
        return self.head(self.head_norm(torch.cat(feats, dim=-1))) + evidence
