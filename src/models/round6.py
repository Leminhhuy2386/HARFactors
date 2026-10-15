"""Round-6 candidate families: memories built from comparisons and counting instead of a
register updated by arithmetic (RESEARCH_PLAN.md, round 6).

All share the skeleton of the earlier families (encoder -> blocks -> mean over time -> LayerNorm
-> linear head; each block: LayerNorm -> linear mixing -> memory -> [u, memory features] -> linear
mixing -> GELU -> residual), so only the memory differs.

OrdinalNet. For each channel u and lag tau the block reads the order pattern of
(u_{t-2tau}, u_{t-tau}, u_t): one of 6 permutations (Bandt-Pompe symbol of order 3). Soft pairwise
comparisons sigmoid((a - b) / T) give a probability over the 6 patterns; a learned score per
(channel, lag, pattern) turns it into a feature. The features depend only on the order of the
samples, so they are invariant to any increasing transform of each channel (gain, offset,
saturation). Deployed: 3 comparisons, a 6-entry table lookup per (channel, lag).

RecurrenceNet. The block embeds the signal into a few learned 2-D spaces and asks, for a bank of
lags tau, whether the trajectory is back where it was tau samples ago:
r_t(tau) = sigmoid((eps - |z_t - z_{t-tau}|) / T), the current row of a recurrence plot, with a
learned radius eps per embedding. It also keeps r_t(tau) * r_{t-1}(tau), a recurrence that has
lasted two steps (the shortest diagonal line of recurrence quantification analysis). Averaged
over the window these give a periodicity profile over lags. Differences make it invariant to a
constant offset of the embedded signal. Deployed: distances to a ring buffer and comparisons.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.hysteresis import inverse_softplus

# the 6 valid sign patterns (x1 > x0, x2 > x1, x2 > x0) of an order-3 window (x0, x1, x2) = (u_{t-2tau}, u_{t-tau}, u_t)
_PATTERNS = torch.tensor([[1, 1, 1], [0, 0, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=torch.float32)


def delayed(u: torch.Tensor, lag: int) -> torch.Tensor:
    """u (B, L, ...) delayed by lag samples along dim 1; the first value is held for t < lag."""
    if lag == 0:
        return u
    return torch.cat([u[:, :1].expand(-1, min(lag, u.shape[1]), *u.shape[2:]), u[:, :-lag]], dim=1)[:, :u.shape[1]]


def ordinal_probs(x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Soft order pattern of (x0, x1, x2): (...) -> (..., 6), rows sum to 1."""
    p = torch.stack([torch.sigmoid((x1 - x0) / temperature), torch.sigmoid((x2 - x1) / temperature),
                     torch.sigmoid((x2 - x0) / temperature)], dim=-1)  # (..., 3)
    pat = _PATTERNS.to(p)
    # probability of each sign pattern = product over the 3 comparisons; renormalised over the 6 consistent ones
    f = p.unsqueeze(-2) * pat + (1 - p.unsqueeze(-2)) * (1 - pat)  # (..., 6, 3)
    w = f[..., 0] * f[..., 1] * f[..., 2]  # (..., 6); explicit product: prod's backward is slow on GPU
    return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class OrdinalBlock(nn.Module):
    def __init__(self, d_model: int, lags: tuple[int, ...] = (1, 4, 16), temperature: float = 0.1) -> None:
        super().__init__()
        self.lags, self.temperature = lags, temperature
        self.norm = nn.LayerNorm(d_model)
        self.mix_in = nn.Linear(d_model, d_model)
        self.score = nn.Parameter(0.5 * torch.randn(d_model, len(lags), 6))
        self.mix_out = nn.Linear(d_model * (1 + len(lags)), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        u = self.mix_in(self.norm(x))
        feats = [u]
        for i, lag in enumerate(self.lags):
            p = ordinal_probs(delayed(u, 2 * lag), delayed(u, lag), u, self.temperature)  # (B, L, D, 6)
            feats.append((p * self.score[:, i]).sum(dim=-1))
        return x + F.gelu(self.mix_out(torch.cat(feats, dim=-1)))


class RecurrenceBlock(nn.Module):
    def __init__(self, d_model: int, n_embed: int = 8, dim: int = 2,
                 lags: tuple[int, ...] = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48), temperature: float = 0.1,
                 radius: float = 0.5, layer_norm: bool = True) -> None:
        """layer_norm=False drops the per-time-step LayerNorm (round 11b mechanism test (b))."""
        super().__init__()
        self.lags, self.n_embed, self.dim, self.temperature = lags, n_embed, dim, temperature
        self.norm = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.mix_in = nn.Linear(d_model, d_model)
        self.embed = nn.Linear(d_model, n_embed * dim)
        self.eps_raw = nn.Parameter(inverse_softplus(torch.full((n_embed, 1), radius)))
        self.mix_out = nn.Linear(d_model + 2 * n_embed * len(lags), d_model)

    def recurrence(self, z: torch.Tensor) -> torch.Tensor:
        """z (B, L, E, m) -> soft recurrence r (B, L, E, n_lags); no recurrence before the lag exists."""
        length = z.shape[1]
        t = torch.arange(length, device=z.device)
        rows = []
        for lag in self.lags:
            dist = ((z - delayed(z, lag)).square().sum(-1) + 1e-6).sqrt()  # (B, L, E)
            r = torch.sigmoid((F.softplus(self.eps_raw).squeeze(-1) - dist) / self.temperature)
            rows.append(r * (t >= lag).to(z.dtype)[None, :, None])
        return torch.stack(rows, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        xn = self.norm(x)
        u = self.mix_in(xn)
        z = self.embed(xn).unflatten(-1, (self.n_embed, self.dim))
        r = self.recurrence(z)  # (B, L, E, n_lags)
        line = r * delayed(r, 1)  # recurrence held over two consecutive steps at the same lag
        return x + F.gelu(self.mix_out(torch.cat([u, r.flatten(2), line.flatten(2)], dim=-1)))


def _rainflow_pairs(u: torch.Tensor, capacity: int, closures: int,
                    keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Streaming rainflow counting (ASTM E1049 four-point rule) on N sequences at once.

    u (N, L), no gradient. Returns
      cyc (N, L, closures, 2): time indices of the two turning points of each cycle closed at t (-1: none);
      stk (N, L, keep): time indices of the top `keep` stack entries after step t, newest first (-1: none).
    The stack holds the unresolved turning points; the newest entry is the current (still moving) point.
    """
    n_seq, length = u.shape
    ar = torch.arange(n_seq, device=u.device)
    stack = torch.zeros(n_seq, capacity, dtype=torch.long, device=u.device)
    n = torch.zeros(n_seq, dtype=torch.long, device=u.device)
    cyc = torch.full((n_seq, length, closures, 2), -1, dtype=torch.long, device=u.device)
    stk = torch.full((n_seq, length, keep), -1, dtype=torch.long, device=u.device)
    for t in range(length):
        v = u[:, t]
        top = stack[ar, (n - 1).clamp(min=0)]
        sec = stack[ar, (n - 2).clamp(min=0)]
        top_v, sec_v = u[ar, top], u[ar, sec]
        d_prev, d_new = torch.sign(top_v - sec_v), torch.sign(v - top_v)
        # same direction as the last segment (or no move): the current point moves on; otherwise a new turning point
        extend = ((n >= 2) & ((d_new == d_prev) | (d_new == 0))) | ((n == 1) & (d_new == 0))
        full = (~extend) & (n == capacity)
        if bool(full.any()):  # drop the oldest entry (rare: the stack only keeps unresolved extrema)
            stack = torch.where(full.unsqueeze(1), torch.roll(stack, -1, dims=1), stack)
            n = torch.where(full, n - 1, n)
        pos = torch.where(extend, n - 1, n)
        stack[ar, pos] = t
        n = torch.where(extend, n, n + 1)
        for j in range(closures):  # four-point rule on the newest four entries
            i1, i2 = stack[ar, (n - 1).clamp(min=0)], stack[ar, (n - 2).clamp(min=0)]
            i3, i4 = stack[ar, (n - 3).clamp(min=0)], stack[ar, (n - 4).clamp(min=0)]
            inner = (u[ar, i2] - u[ar, i3]).abs()
            close = (n >= 4) & (inner <= (u[ar, i3] - u[ar, i4]).abs()) & (inner <= (u[ar, i1] - u[ar, i2]).abs())
            cyc[:, t, j, 0] = torch.where(close, i3, -1)
            cyc[:, t, j, 1] = torch.where(close, i2, -1)
            at = (n - 3).clamp(min=0)
            stack[ar, at] = torch.where(close, i1, stack[ar, at])
            n = torch.where(close, n - 2, n)
        for k in range(keep):
            stk[:, t, k] = torch.where(n - 1 - k >= 0, stack[ar, (n - 1 - k).clamp(min=0)], -1)
    return cyc, stk


try:  # TorchScript removes the Python overhead of the loop
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        _rainflow_pairs_jit = torch.jit.script(_rainflow_pairs)
except Exception:  # noqa: BLE001
    _rainflow_pairs_jit = _rainflow_pairs


def _sparse_tables(u: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Range max / min tables with arg indices: level k covers [i, i + 2^k) (clipped at the end)."""
    n_seq, length = u.shape
    idx = torch.arange(length, device=u.device).expand(n_seq, -1)
    mx, mn, amx, amn = [u], [u], [idx], [idx]
    h = 1
    while 2 * h <= length:
        def shifted(t, fill):
            return torch.cat([t[:, h:], torch.full_like(t[:, :h], fill)], dim=1)
        nxt_mx, nxt_amx = shifted(mx[-1], float("-inf")), shifted(amx[-1], 0)
        take = nxt_mx > mx[-1]
        mx.append(torch.where(take, nxt_mx, mx[-1]))
        amx.append(torch.where(take, nxt_amx, amx[-1]))
        nxt_mn, nxt_amn = shifted(mn[-1], float("inf")), shifted(amn[-1], 0)
        take = nxt_mn < mn[-1]
        mn.append(torch.where(take, nxt_mn, mn[-1]))
        amn.append(torch.where(take, nxt_amn, amn[-1]))
        h *= 2
    return mx, mn, amx, amn


def _range_query(table: list[torch.Tensor], arg: list[torch.Tensor], lo: torch.Tensor, hi: torch.Tensor,
                 is_max: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Max (or min) of u over [lo, hi] inclusive and its index; empty ranges give -inf/+inf and index 0."""
    length = table[0].shape[1]
    empty = hi < lo
    span = (hi - lo + 1).clamp(min=1)
    level = torch.floor(torch.log2(span.to(torch.float32))).to(torch.long).clamp(max=len(table) - 1)
    tv, ta = torch.stack(table), torch.stack(arg)  # (K, N, L)
    rows = torch.arange(lo.shape[0], device=lo.device).unsqueeze(1).expand_as(lo)
    a = lo.clamp(0, length - 1)
    b = (hi - (1 << level) + 1).clamp(0, length - 1)
    va, vb = tv[level, rows, a], tv[level, rows, b]
    ia, ib = ta[level, rows, a], ta[level, rows, b]
    pick_b = vb > va if is_max else vb < va
    val, at = torch.where(pick_b, vb, va), torch.where(pick_b, ib, ia)
    fill = float("-inf") if is_max else float("inf")
    return torch.where(empty, torch.full_like(val, fill), val), torch.where(empty, torch.zeros_like(at), at)


def _nearest(u: torch.Tensor, table: list[torch.Tensor], left: bool, is_max: bool, strict: bool) -> torch.Tensor:
    """For every i, the nearest index on one side whose value beats u_i, by binary lifting on the tables.

    is_max: beats = greater (strict: >, else >=); else smaller (strict: <, else <=). Missing: -1 / L."""
    n_seq, length = u.shape
    pos = torch.arange(length, device=u.device).expand(n_seq, -1).clone()
    for level in reversed(range(len(table))):
        step = 1 << level
        cand = pos - step if left else pos + step
        inside = (cand >= 0) if left else (cand <= length - 1)
        start = cand.clamp(0, length - 1) if left else (pos + 1).clamp(0, length - 1)
        window = torch.gather(table[level], 1, start)  # extreme over the `step` samples being skipped
        if is_max:
            skip = (window <= u) if strict else (window < u)
        else:
            skip = (window >= u) if strict else (window > u)
        pos = torch.where(inside & skip, cand, pos)
    return pos - 1 if left else pos + 1


def break_ties(u: torch.Tensor) -> torch.Tensor:
    """float64 copy with a tiny increasing ramp (1e-9 per sample): equal values (plateaus, quantized
    sensors) get a strict order, so the parallel pairing and the streaming four-point rule agree."""
    return u.double() + 1e-9 * torch.arange(u.shape[-1], device=u.device, dtype=torch.float64)


def _rainflow_parallel(u: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rainflow cycles of N sequences in O(log L) parallel steps (no gradient; indices only).

    A peak i closes a full cycle when the signal first returns to >= u_i (time j = NGE(i)), with the
    lowest point of (i, j) as its valley, provided that valley is not deeper than the lowest point
    between the previous higher sample and i (otherwise i belongs to a larger cycle). Valleys are the
    mirror image. This is the closing rule of streaming four-point rainflow counting, written with
    nearest-greater elements and range minima (a causal form of the elder rule); each cycle is reported
    at the time the streaming algorithm closes it (checked equal on 512 sequences). It is not the full
    0-dim persistence of the window: an open signal can have persistence pairs that rainflow never
    closes (e.g. 0, 5, 1, 4, 2, 3).
    Also returns, for every t, the index of the lowest point since the last higher sample (`up_from`)
    and of the highest point since the last lower sample (`down_from`): the open excursions at t.
    """
    n_seq, length = u.shape
    mx, mn, amx, amn = _sparse_tables(u)
    i = torch.arange(length, device=u.device).expand(n_seq, -1)
    pge, nge = _nearest(u, mx, True, True, True), _nearest(u, mx, False, True, False)
    ple, nle = _nearest(u, mn, True, False, True), _nearest(u, mn, False, False, False)
    lmin, _ = _range_query(mn, amn, pge + 1, i - 1, False)
    rmin, rmin_at = _range_query(mn, amn, i + 1, nge - 1, False)
    lmax, _ = _range_query(mx, amx, ple + 1, i - 1, True)
    rmax, rmax_at = _range_query(mx, amx, i + 1, nle - 1, True)
    peak = (nge <= length - 1) & (i - 1 >= pge + 1) & (nge - 1 >= i + 1) & (rmin >= lmin)
    valley = (nle <= length - 1) & (i - 1 >= ple + 1) & (nle - 1 >= i + 1) & (rmax <= lmax)
    _, up_from = _range_query(mn, amn, pge + 1, i, False)
    _, down_from = _range_query(mx, amx, ple + 1, i, True)
    return {"peak": peak, "peak_close": nge, "peak_other": rmin_at, "valley": valley, "valley_close": nle,
            "valley_other": rmax_at, "up_from": up_from, "down_from": down_from}


class RainflowBlock(nn.Module):
    """Cycle memory: rainflow counting on each mixed channel.

    The discrete part (which samples pair into cycles and when each cycle closes) runs without
    gradient, in O(log L) parallel steps (_rainflow_parallel); amplitudes are then read from u at
    those indices, so gradients reach u through the extremum values, as through max pooling.
    Per time step and channel the features are
      - the cycles that close at t, soft-binned by log-amplitude into `bins` learned bins (counts),
      - the open excursions at t: the rise since the lowest point after the last higher sample and
        the fall since the highest point after the last lower sample.
    Averaged over the window, the closed-cycle counts form an amplitude-resolved cycle spectrum:
    how many oscillations of each size the window held, whatever their shape and timing. Deployed,
    the same quantities come from the streaming four-point algorithm on a small stack of turning
    points: comparisons and pushes/pops, no multiplications (`_rainflow_pairs` is that reference).
    """

    def __init__(self, d_model: int, bins: int = 3, log_lo: float = -2.5, log_hi: float = 1.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mix_in = nn.Linear(d_model, d_model)
        self.centers = nn.Parameter(torch.linspace(log_lo, log_hi, bins).repeat(d_model, 1))  # (D, bins), log amplitude
        self.log_width = nn.Parameter(torch.full((d_model, 1), float(torch.log(torch.tensor((log_hi - log_lo) / bins)))))
        self.mix_out = nn.Linear(d_model * (3 + bins), d_model)

    def cycle_features(self, flat: torch.Tensor, batch: int) -> torch.Tensor:
        """flat (N = batch * D, L) -> (N, L, bins + 2)."""
        with torch.no_grad():
            rf = _rainflow_parallel(break_ties(flat.detach()))
        take = lambda idx: torch.gather(flat, 1, idx)  # noqa: E731
        bins = self.centers.shape[1]
        centers = self.centers.repeat(batch, 1).unsqueeze(1)  # (N, 1, bins)
        width = self.log_width.exp().repeat(batch, 1).unsqueeze(1)
        counts = torch.zeros(*flat.shape, bins, device=flat.device, dtype=flat.dtype)
        for kind, sign in (("peak", 1.0), ("valley", -1.0)):
            amp = sign * (flat - take(rf[f"{kind}_other"]))  # >= 0 where a cycle closes
            valid = rf[kind].to(flat.dtype)
            soft = torch.exp(-0.5 * ((torch.log(amp.clamp(min=0) + 1e-3).unsqueeze(-1) - centers) / width) ** 2)
            close = rf[f"{kind}_close"].clamp(max=flat.shape[1] - 1)
            counts = counts.scatter_add(1, close.unsqueeze(-1).expand(-1, -1, bins), soft * valid.unsqueeze(-1))
        up = flat - take(rf["up_from"])
        down = take(rf["down_from"]) - flat
        return torch.cat([counts, up.unsqueeze(-1), down.unsqueeze(-1)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        u = self.mix_in(self.norm(x))
        batch, length, d = u.shape
        mem = self.cycle_features(u.transpose(1, 2).reshape(batch * d, length), batch)  # (N, L, F)
        mem = mem.view(batch, d, length, -1).permute(0, 2, 1, 3).flatten(2)  # (B, L, D * F)
        return x + F.gelu(self.mix_out(torch.cat([u, mem], dim=-1)))


class Round6Net(nn.Module):
    """encoder -> blocks -> mean over time -> LayerNorm -> linear head (the shared skeleton)."""

    BLOCKS = {"ordinal": OrdinalBlock, "recurrence": RecurrenceBlock, "rainflow": RainflowBlock}

    def __init__(self, input_channels: int, window_length: int, num_classes: int, *, memory: str,
                 d_model: int = 32, num_layers: int = 3, pooling: str = "mean", **block_kwargs) -> None:
        """pooling="meanstd" reads the mean and the standard deviation over time of the last hidden
        states (round 11b mechanism test (a)); the default "mean" is the original skeleton."""
        super().__init__()
        self.pooling = pooling
        self.encoder = nn.Linear(input_channels, d_model)
        self.blocks = nn.ModuleList(self.BLOCKS[memory](d_model, **block_kwargs) for _ in range(num_layers))
        width = 2 * d_model if pooling == "meanstd" else d_model
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = self.encoder(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        pooled = torch.cat([h.mean(dim=1), h.std(dim=1)], dim=-1) if self.pooling == "meanstd" else h.mean(dim=1)
        return self.head(self.norm(pooled))
