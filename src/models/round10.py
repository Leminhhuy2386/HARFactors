"""Round-10 candidates (RESEARCH_PLAN.md §5g): bold memories, checked against the literature first.

EcoNet. The memory of a block is an ecological community. Each head holds S species with
positive abundances n. At every step the input sets each species' growth rate, and the species
interact through a learned community matrix A (self-limitation on the diagonal, competition or
facilitation off it), as in a discrete-time Lotka-Volterra (Ricker) model:

    log n_{t+1} = log n_t + dt * (r(u_t) - A n_t),      r(u_t) = r_max * tanh(W u_t + b).

What the community remembers is its composition: competition makes it multistable (which species
won depends on the order of past conditions), extinct species stay rare until conditions favour
them again (hysteresis), and abundances stay positive and bounded by self-limitation. The block
reads log-abundances, relative abundances within each community and each community's Shannon
diversity. Deployed: per step and species one exp (a table on an MCU), S multiply-adds and a few
additions. Literature check (2026-10-10): partly new; Lotka-Volterra neural dynamics (Fukai & Tanaka
1997, winnerless competition, Rabinovich 2001) and untrained ecological reservoirs (Ushio et al.
2023) exist, but not a trained input-driven Lotka-Volterra cell for sequence classification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def eco_scan(r: torch.Tensor, a: torch.Tensor, dt: float, l0: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """r (B, L, H, S) growth rates, a (H, S, S) community matrix, l0 (H, S) initial log-abundance.
    Returns the log-abundances after each step (B, L, H, S)."""
    batch, length, heads, species = r.shape
    ell = l0.unsqueeze(0).expand(batch, heads, species)
    out = []
    for t in range(length):
        crowding = torch.matmul(a, torch.exp(ell).unsqueeze(-1)).squeeze(-1)  # (B, H, S): A n per community
        ell = torch.clamp(ell + dt * (r[:, t] - crowding), lo, hi)
        out.append(ell)
    return torch.stack(out, dim=1)


def eco_backward(grad: torch.Tensor, ell: torch.Tensor, r: torch.Tensor, a: torch.Tensor, dt: float,
                 l0: torch.Tensor, lo: float, hi: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reverse pass of eco_scan: grad (B, L, H, S) w.r.t. the outputs -> grads of r, a, l0.

    Step t: z = l_{t-1} + dt (r_t - A n_{t-1}), l_t = clamp(z); with delta = dLoss/dz (zero where clamped):
    dr_t = dt delta, dl_{t-1} += delta - dt (A^T delta) * n_{t-1}, dA -= dt sum_b delta n_{t-1}^T."""
    batch, length, heads, species = r.shape
    lam = torch.zeros_like(grad[:, 0])
    dr = torch.empty_like(r)
    da = torch.zeros_like(a)
    l_start = l0.unsqueeze(0).expand(batch, heads, species)
    for t in range(length - 1, -1, -1):
        lam = lam + grad[:, t]
        prev = ell[:, t - 1] if t > 0 else l_start
        n = torch.exp(prev).unsqueeze(-1)  # (B, H, S, 1)
        z = prev + dt * (r[:, t] - torch.matmul(a, n).squeeze(-1))
        delta = lam * ((z > lo) & (z < hi)).to(lam.dtype)
        dr[:, t] = dt * delta
        da = da - dt * (delta.unsqueeze(-1) * n.transpose(-1, -2)).sum(0)
        lam = delta - dt * torch.matmul(a.transpose(-1, -2), delta.unsqueeze(-1)).squeeze(-1) * n.squeeze(-1)
    return dr, da, lam.sum(0)


try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        _eco_scan_jit = torch.jit.script(eco_scan)
        _eco_backward_jit = torch.jit.script(eco_backward)
except Exception:  # noqa: BLE001
    _eco_scan_jit, _eco_backward_jit = eco_scan, eco_backward


class _EcoScan(torch.autograd.Function):
    """eco_scan with the hand-written reverse pass (no autograd graph over the time loop)."""

    @staticmethod
    def forward(ctx, r, a, l0, dt: float, lo: float, hi: float):
        with torch.no_grad():
            ell = _eco_scan_jit(r, a, dt, l0, lo, hi)
        ctx.save_for_backward(ell, r, a, l0)
        ctx.consts = (dt, lo, hi)
        return ell

    @staticmethod
    def backward(ctx, grad):
        ell, r, a, l0 = ctx.saved_tensors
        dt, lo, hi = ctx.consts
        dr, da, dl0 = _eco_backward_jit(grad.contiguous(), ell, r, a, dt, l0, lo, hi)
        return dr, da, dl0, None, None, None


def eco_scan_fast(r, a, l0, dt: float, lo: float, hi: float) -> torch.Tensor:
    return _EcoScan.apply(r, a, l0, dt, lo, hi)


class EcoBlock(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, species: int = 8, dt: float = 0.2, r_max: float = 2.0,
                 lo: float = -8.0, hi: float = 4.0, stride: int = 3) -> None:
        """stride: the community advances once per `stride` samples, with the mean growth rate of
        those samples; every sample reads the community as it was after the previous stride (causal)."""
        super().__init__()
        self.heads, self.species, self.dt, self.r_max, self.lo, self.hi = heads, species, dt, r_max, lo, hi
        self.stride = stride
        self.norm = nn.LayerNorm(d_model)
        self.mix_in = nn.Linear(d_model, d_model)
        self.growth = nn.Linear(d_model, heads * species)
        self.self_limit = nn.Parameter(torch.zeros(heads, species))  # softplus(0) + 0.3 ~ 1: carrying capacity ~ r
        self.interact = nn.Parameter(0.1 * torch.randn(heads, species, species))
        self.l0 = nn.Parameter(torch.full((heads, species), -0.7))
        n_sp = heads * species
        self.mix_out = nn.Linear(d_model + 2 * n_sp + heads, d_model)

    def community(self) -> torch.Tensor:
        off = self.interact * (1 - torch.eye(self.species, device=self.interact.device))
        return off + torch.diag_embed(F.softplus(self.self_limit) + 0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        u = self.mix_in(self.norm(x))
        batch, length, _ = u.shape
        r = self.r_max * torch.tanh(self.growth(u)).view(batch, length, self.heads, self.species)
        s = self.stride
        if s > 1:  # mean growth per stride; the stride's samples read the community before it
            r = F.pad(r, (0, 0, 0, 0, 0, (-length) % s)).unflatten(1, (-1, s)).mean(2)
        ell = eco_scan_fast(r, self.community(), self.l0, self.dt, self.lo, self.hi)  # (B, L / s, H, S)
        if s > 1:
            start = self.l0.expand(batch, 1, self.heads, self.species)
            ell = torch.cat([start, ell[:, :-1]], dim=1).repeat_interleave(s, dim=1)[:, :length]
        p = torch.softmax(ell, dim=-1)  # relative abundances within each community
        diversity = -(p * torch.log(p + 1e-8)).sum(-1)  # (B, L, H)
        return x + F.gelu(self.mix_out(torch.cat([u, ell.flatten(2), p.flatten(2), diversity], dim=-1)))


def inside(emit: torch.Tensor, log_rules: torch.Tensor) -> list[torch.Tensor]:
    """Inside algorithm (CKY in log space) of a binary grammar over T tokens.

    emit (B, T, K): log-potential of each nonterminal spanning one token; log_rules (K, K, K):
    log P(A -> B C). Returns beta[w] (B, T - w + 1, K) for w = 1..T (beta[0] is unused):
    beta[w][:, i, A] = log of the summed weight of all parses of tokens i..i+w-1 rooted at A."""
    batch, length, k = emit.shape
    rules = torch.exp(log_rules)  # (A, B, C)
    beta = [emit, emit]
    for w in range(2, length + 1):
        n = length - w + 1
        left = torch.stack([beta[s][:, :n] for s in range(1, w)], dim=2)  # (B, n, w-1, K): span i..i+s-1
        right = torch.stack([beta[w - s][:, s:s + n] for s in range(1, w)], dim=2)  # span i+s..i+w-1
        lm, rm = left.amax(-1, keepdim=True), right.amax(-1, keepdim=True)
        inner = torch.einsum("bnsj,bnsk,ajk->bnsa", torch.exp(left - lm), torch.exp(right - rm), rules)
        beta.append(torch.logsumexp(torch.log(inner + 1e-30) + lm + rm, dim=2))
    return beta


class GrammarNet(nn.Module):
    """GrammarNet: the window is parsed by a learned stochastic context-free grammar.

    A per-sample network embeds the signal; patches of `patch` samples become tokens; each of K
    nonterminals scores every token; a shared binary grammar (learned rule probabilities A -> B C)
    parses the token sequence with the inside algorithm. The parse is the memory: it can relate a
    sub-stroke at the start of the window to one at the end through nested constituents, at every
    scale at once. Features for the head: the posterior over the root nonterminal of the whole window,
    the average constituent posterior at widths 2, 4 and 8 tokens, and log Z per token, joined with
    the pooled per-sample embedding. Literature check (2026-10-10): partly new; grammars for activities
    (Ivanov & Bobick 2000), differentiable CKY (Maillard et al. 2017) and differentiable regular grammars
    for video (Piergiovanni et al. 2020) exist, not a context-free grammar trained end to end on raw IMU.
    """

    def __init__(self, input_channels: int, window_length: int, num_classes: int, *, d_model: int = 48,
                 nonterminals: int = 12, patch: int = 5, widths: tuple[int, ...] = (2, 4, 8)) -> None:
        super().__init__()
        self.patch = patch
        self.tokens = -(-window_length // patch)
        self.widths = tuple(w for w in widths if w <= self.tokens)
        self.encoder = nn.Linear(input_channels, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.emit = nn.Linear(d_model, nonterminals)
        self.rules = nn.Parameter(0.1 * torch.randn(nonterminals, nonterminals, nonterminals))
        n_feat = d_model + nonterminals * (1 + len(self.widths)) + 1
        self.norm = nn.LayerNorm(n_feat)
        self.head = nn.Linear(n_feat, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = self.encoder(x.transpose(1, 2))
        h = h + self.mlp(self.norm_in(h))
        batch, length, d = h.shape
        tok = F.pad(h, (0, 0, 0, (-length) % self.patch)).unflatten(1, (-1, self.patch)).mean(2)  # (B, T, D)
        log_rules = torch.log_softmax(self.rules.flatten(1), dim=-1).view_as(self.rules)  # P(B C | A)
        beta = inside(self.emit(tok), log_rules)
        root = beta[-1][:, 0]  # (B, K)
        feats = [h.mean(dim=1), torch.softmax(root, dim=-1)]
        feats += [torch.softmax(beta[w], dim=-1).mean(dim=1) for w in self.widths]
        feats.append(torch.logsumexp(root, dim=-1, keepdim=True) / tok.shape[1])
        return self.head(self.norm(torch.cat(feats, dim=-1)))


def prefix_product(m: torch.Tensor) -> torch.Tensor:
    """Cumulative matrix products along dim -3 in O(log L) steps: out[t] = m[t] @ m[t-1] @ ... @ m[0]."""
    length, off = m.shape[-3], 1
    while off < length:
        m = torch.cat([m[..., :off, :, :], m[..., off:, :, :] @ m[..., :-off, :, :]], dim=-3)
        off *= 2
    return m


def transpositions(strands: int) -> torch.Tensor:
    """(P, S, S) permutation matrices of the transpositions (i j), i < j."""
    mats = []
    for i in range(strands):
        for j in range(i + 1, strands):
            p = torch.eye(strands)
            p[[i, j]] = p[[j, i]]
            mats.append(p)
    return torch.stack(mats)


def normalized_prefix_product(m: torch.Tensor) -> torch.Tensor:
    """prefix_product with every partial product scaled to unit Frobenius norm (projective holonomy:
    a scalar factor commutes with everything, so only the direction is kept; no overflow)."""
    unit = lambda a: a / torch.linalg.matrix_norm(a)[..., None, None].clamp_min(1e-30)  # noqa: E731
    m = unit(m)
    length, off = m.shape[-3], 1
    while off < length:
        m = torch.cat([m[..., :off, :, :], unit(m[..., off:, :, :] @ m[..., :-off, :, :])], dim=-3)
        off *= 2
    return m


def kz_holonomy(z: torch.Tensor, kappa: torch.Tensor, tau: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor,
                eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """z (B, L, H, S) complex strand positions, kappa (H,) complex, tau (P, S, S) transpositions.

    Returns the projective holonomy after each step (B, H, L, S, S; unit Frobenius norm) and the
    abelian part (B, L, H) = kappa * cumulative sum of the pairwise dlog increments. Each step uses
    the trace-free part of the generator (unit determinant); the scalar part is the abelian one.
    Angle increments are damped when two strands nearly collide (|d_t d_{t-1}| << eps^2), where the
    winding is undefined."""
    d = z[..., pair_i] - z[..., pair_j]  # (B, L, H, P)
    mod2 = d.real ** 2 + d.imag ** 2
    d_log_r = 0.5 * (torch.log(mod2[:, 1:] + eps ** 2) - torch.log(mod2[:, :-1] + eps ** 2))
    w = d[:, 1:] * torch.conj(d[:, :-1])
    w2 = w.real ** 2 + w.imag ** 2
    d_arg = torch.atan2(w.imag, w.real) * (w2 / (w2 + eps ** 4)) if eps > 0 else torch.atan2(w.imag, w.real)
    step = torch.complex(d_log_r, d_arg)
    step = torch.cat([torch.zeros_like(step[:, :1]), step], dim=1)  # (B, L, H, P) dlog increments
    strands = tau.shape[-1]
    tau0 = tau - (tau.diagonal(dim1=-2, dim2=-1).sum(-1) / strands)[:, None, None] * torch.eye(strands, device=tau.device)
    gen = torch.einsum("blhp,pij->blhij", step * kappa[:, None], tau0.to(step.dtype))
    psi = normalized_prefix_product(torch.linalg.matrix_exp(gen).transpose(1, 2))  # (B, H, L, S, S)
    return psi, torch.cumsum(step.sum(-1), dim=1) * kappa


class BraidBlock(nn.Module):
    """Braid memory: holonomy of a flat Knizhnik-Zamolodchikov-type connection.

    Each head turns the signal into S moving points z_1..z_S in the plane ("strands") and carries
    the matrix Psi_t = exp(kappa * sum_{i<j} dlog(z_i - z_j) tau_ij) Psi_{t-1}, with tau_ij the
    transposition matrices of the S strands. Transpositions satisfy the infinitesimal pure braid
    relations, so the connection is flat: Psi depends (up to the discretization) only on how the
    strands wound around each other, not on their exact paths or speeds. The abelian part (the
    determinant) is the sum of pairwise log-ratios, i.e. the windings VortexNet counted; the rest is
    non-commutative and records the order in which strands circled each other. The block reads the
    determinant-normalized traces of Psi, Psi^2, Psi^3 (conjugation invariants) and the abelian part.
    Computed with a parallel prefix product. Near-collisions are softened by eps.
    """

    def __init__(self, d_model: int, heads: int = 8, strands: int = 4, eps: float = 0.1, kappa: float = 0.3) -> None:
        super().__init__()
        self.heads, self.strands, self.eps = heads, strands, eps
        self.norm = nn.LayerNorm(d_model)
        self.mix_in = nn.Linear(d_model, d_model)
        self.points = nn.Linear(d_model, heads * strands * 2)
        self.kappa_re = nn.Parameter(torch.full((heads,), 0.0))
        self.kappa_im = nn.Parameter(torch.full((heads,), kappa))
        self.register_buffer("tau", transpositions(strands), persistent=False)
        iu = torch.triu_indices(strands, strands, 1)
        self.register_buffer("pair_i", iu[0], persistent=False)
        self.register_buffer("pair_j", iu[1], persistent=False)
        self.mix_out = nn.Linear(d_model + 8 * heads, d_model)

    def holonomy(self, xn: torch.Tensor) -> torch.Tensor:
        batch, length, _ = xn.shape
        p = self.points(xn).view(batch, length, self.heads, self.strands, 2)
        z = torch.complex(p[..., 0], p[..., 1])  # (B, L, H, S)
        kappa = torch.complex(self.kappa_re, self.kappa_im)  # (H,)
        psi, abelian = kz_holonomy(z, kappa, self.tau, self.pair_i, self.pair_j, self.eps)
        psi = psi * self.strands ** 0.5  # unit Frobenius norm -> the identity has norm sqrt(S), traces ~ S
        p2 = psi @ psi
        traces = [psi.diagonal(dim1=-2, dim2=-1).sum(-1), p2.diagonal(dim1=-2, dim2=-1).sum(-1),
                  (p2 @ psi).diagonal(dim1=-2, dim2=-1).sum(-1)]  # each (B, H, L)
        feats = [t.transpose(1, 2) / self.strands for t in traces] + [abelian]
        out = torch.cat([torch.stack([f.real, f.imag], dim=-1) for f in feats], dim=-1)  # (B, L, H, 8)
        return torch.asinh(out).flatten(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        xn = self.norm(x)
        u = self.mix_in(xn)
        return x + F.gelu(self.mix_out(torch.cat([u, self.holonomy(xn)], dim=-1)))


class Round10Net(nn.Module):
    """encoder -> blocks -> mean over time -> LayerNorm -> linear head (the shared skeleton)."""

    BLOCKS = {"eco": EcoBlock, "braid": BraidBlock}

    def __init__(self, input_channels: int, window_length: int, num_classes: int, *, memory: str,
                 d_model: int = 32, num_layers: int = 3, **block_kwargs) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_channels, d_model)
        self.blocks = nn.ModuleList(self.BLOCKS[memory](d_model, **block_kwargs) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = self.encoder(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h.mean(dim=1)))
