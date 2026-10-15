"""Round 14 (RESEARCH_PLAN.md §5m): LevelNet, a network that works along each channel's value axis.

Sequence networks usually process the time axis and mix all channels in the first layer. LevelNet keeps channels
apart and describes each one by how its values are distributed over a fixed grid of levels:
- channels: the input channels plus a few learned projections of them ("virtual channels");
- per channel, for the signal and for its first difference: the share of the window above each level (a soft
  empirical CDF), the same share per third of the window (coarse order), and the soft level-crossing rate;
- optional (plane=True): the soft 2-D histogram of (signal, difference), i.e. the channel's phase-plane density;
- a small convolution along the level axis, with weights shared by all channels, turns each channel's map into one
  token; channels meet only afterwards, in one transformer layer over the channel tokens.

Controls: mixing="early" mixes all channels into d_model channels before the level maps (as every earlier family
mixed them first); crossings=False drops the crossing rates; tokenizer="mlp" uses an MLP instead of the level-axis
convolution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def level_maps(s: torch.Tensor, levels: torch.Tensor, temp: float, segments: int, crossings: bool) -> torch.Tensor:
    """s (B, C, L) -> (B, C, F, M) with F = 1 + segments + crossings (segments > 1 adds one map per segment)."""
    step = torch.sigmoid((s.unsqueeze(-1) - levels) / temp)  # (B, C, L, M)
    maps = [step.mean(dim=2)]
    if segments > 1:
        for idx in torch.tensor_split(torch.arange(s.shape[-1], device=s.device), segments):
            maps.append(step[:, :, idx].mean(dim=2))
    if crossings:
        maps.append((step[:, :, 1:] - step[:, :, :-1]).abs().mean(dim=2))
    return torch.stack(maps, dim=2)


def plane_density(s: torch.Tensor, ds: torch.Tensor, centres: torch.Tensor, width: float) -> torch.Tensor:
    """Soft 2-D histogram of (s, ds), both (B, C, L), over centres x centres -> (B, C, M, M), summing to 1."""
    ws = torch.softmax(-0.5 * ((s.unsqueeze(-1) - centres) / width).square(), dim=-1)
    wd = torch.softmax(-0.5 * ((ds.unsqueeze(-1) - centres) / width).square(), dim=-1)
    return torch.einsum("bclm,bcln->bcmn", ws, wd) / s.shape[-1]


class ChannelTokenizer(nn.Module):
    """(B*C, F, M) level maps (and optionally (B*C, M, M) phase planes) -> (B*C, d_model) channel tokens."""

    def __init__(self, n_maps: int, levels: int, d_model: int, conv: int, plane: bool, kind: str) -> None:
        super().__init__()
        self.kind, self.plane = kind, plane
        if kind == "conv":
            self.line = nn.Sequential(nn.Conv1d(n_maps, conv, 3, padding=1), nn.GELU(),
                                      nn.Conv1d(conv, conv, 3, padding=1), nn.GELU(), nn.Flatten())
            width = conv * levels
        elif kind == "mlp":
            self.line = nn.Sequential(nn.Flatten(), nn.Linear(n_maps * levels, 2 * d_model), nn.GELU())
            width = 2 * d_model
        else:
            raise ValueError(kind)
        if plane:
            self.grid = nn.Sequential(nn.Conv2d(1, conv // 2, 3, padding=1), nn.GELU(), nn.AvgPool2d(2),
                                      nn.Conv2d(conv // 2, conv, 3, padding=1), nn.GELU(), nn.AvgPool2d(2),
                                      nn.Flatten())
            width += conv * (levels // 4) ** 2
        self.out = nn.Linear(width, d_model)

    def forward(self, maps: torch.Tensor, planes: torch.Tensor | None = None) -> torch.Tensor:
        h = self.line(maps)
        if self.plane:
            h = torch.cat([h, self.grid(planes.unsqueeze(1))], dim=-1)
        return self.out(h)


class LevelNet(nn.Module):
    def __init__(self, input_channels: int, window_length: int, num_classes: int, *, levels: int = 16,
                 span: float = 2.5, virtual: int = 8, d_model: int = 32, conv: int = 16, heads: int = 2,
                 segments: int = 3, crossings: bool = True, plane: bool = False, mixing: str = "late",
                 tokenizer: str = "conv") -> None:
        super().__init__()
        if mixing not in ("late", "early"):
            raise ValueError(mixing)
        self.mixing, self.segments, self.crossings, self.plane = mixing, segments, crossings, plane
        spacing = 2 * span / (levels - 1)
        self.temp, self.width = 0.5 * spacing, spacing
        self.register_buffer("levels", torch.linspace(-span, span, levels), persistent=False)
        if mixing == "early":
            self.mix = nn.Linear(input_channels, d_model, bias=False)
            n_channels = d_model
        else:
            self.mix = nn.Linear(input_channels, virtual, bias=False) if virtual else None
            n_channels = input_channels + virtual
        self.mix_norm = nn.BatchNorm1d(n_channels if mixing == "early" else max(virtual, 1), affine=False)
        self.diff_norm = nn.BatchNorm1d(n_channels, affine=False)
        n_maps = 2 * (1 + (segments if segments > 1 else 0) + int(crossings))
        self.tokenizer = ChannelTokenizer(n_maps, levels, d_model, conv, plane, tokenizer)
        self.channel_embed = nn.Parameter(0.02 * torch.randn(n_channels, d_model))
        self.mixer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout=0.0, activation="gelu",
                                                batch_first=True, norm_first=True)
        self.query = nn.Parameter(0.02 * torch.randn(d_model))
        self.norm = nn.LayerNorm(2 * d_model)
        self.head = nn.Linear(2 * d_model, num_classes)

    def channels(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, C, L) -> the channel set the level maps describe (B, C', L)."""
        if self.mixing == "early":
            return self.mix_norm(self.mix(x.transpose(1, 2)).transpose(1, 2))
        if self.mix is None:
            return x
        v = self.mix_norm(self.mix(x.transpose(1, 2)).transpose(1, 2))
        return torch.cat([x, v], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        s = self.channels(x)
        ds = self.diff_norm(s[..., 1:] - s[..., :-1])
        maps = torch.cat([level_maps(s, self.levels, self.temp, self.segments, self.crossings),
                          level_maps(ds, self.levels, self.temp, self.segments, self.crossings)], dim=2)
        b, c = maps.shape[:2]
        planes = plane_density(s[..., 1:], ds, self.levels, self.width).flatten(0, 1) if self.plane else None
        tokens = self.tokenizer(maps.flatten(0, 1), planes).view(b, c, -1) + self.channel_embed
        tokens = self.mixer(tokens)
        attn = torch.softmax(tokens @ self.query, dim=1).unsqueeze(-1)
        pooled = torch.cat([tokens.mean(dim=1), (attn * tokens).sum(dim=1)], dim=-1)
        return self.head(self.norm(pooled))
