"""Edge-SSM building blocks.

Every edge i -> j owns a first-order memory with its own learnable time constant, followed by
its own learnable nonlinear function; a neuron sums its incoming edges
(remember -> transform -> sum):

    a_ij = sigmoid(alpha_ij),  b_ij = sqrt(1 - a_ij^2)
    s_ij[t] = a_ij * s_ij[t-1] + b_ij * x_i[t]
    phi_ij(s) = w_ij * s + sum_r c_ijr * tanh((s - g_r) / H)
    h_j[t] = beta_j + sum_i phi_ij(s_ij[t])

Training runs the memory as a causal FFT convolution with kernel b * a^k (all edges in
parallel); inference can run one sample at a time with `step`, whose state is one float per
edge. Tensors are laid out (batch, channels, time), as the WHAR harness feeds them.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def tanh_basis(x: torch.Tensor, grid: torch.Tensor, width: float) -> torch.Tensor:
    """[x, tanh((x - g_1)/H), ..., tanh((x - g_R)/H)] stacked on a new last dimension."""
    basis = torch.tanh((x.unsqueeze(-1) - grid) / width)
    return torch.cat([x.unsqueeze(-1), basis], dim=-1)


def causal_fft_conv(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """y[..., t] = sum_{k<=t} kernel[..., k] * x[..., t-k], zero initial state.

    x: (B, C, L); kernel: (C, H, L) -> (B, C, H, L).
    """
    length = x.shape[-1]
    n = 2 * length
    x_f = torch.fft.rfft(x, n=n)
    k_f = torch.fft.rfft(kernel, n=n)
    return torch.fft.irfft(x_f.unsqueeze(2) * k_f.unsqueeze(0), n=n)[..., :length]


class EdgeSSMLayer(nn.Module):
    """One learnable time constant and one learnable function per edge.

    tau_sharing: "edge" (one tau per edge, the model), "input" (one tau per input channel,
    shared by its outgoing edges) or "fixed" (initial taus, not trained) - the last two are
    ablations. use_phi=False drops the tanh terms, so phi is linear and the layer collapses
    to a channel-wise linear SSM. input_gain: "sqrt" (b = sqrt(1 - a^2), white-noise variance
    preserved) or "ema" (b = 1 - a, unit DC gain).

    oscillatory=True (exploratory variant "osc") gives every edge a complex pole
    lambda = a * exp(i * omega) with its own learnable omega; phi reads Re(s), so the memory
    kernel is b * a^k * cos(omega * k), a band-pass that is the plain low-pass at omega = 0.
    The streaming state is then two floats per edge.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_basis: int = 6,
        grid_range: float = 2.5,
        basis_width: float = 0.8,
        tau_min: float = 2.0,
        tau_max: float = 100.0,
        tau_sharing: str = "edge",
        use_phi: bool = True,
        input_gain: str = "sqrt",
        oscillatory: bool = False,
    ) -> None:
        super().__init__()
        if tau_sharing not in ("edge", "input", "fixed"):
            raise ValueError(f"unknown tau_sharing: {tau_sharing}")
        if input_gain not in ("sqrt", "ema"):
            raise ValueError(f"unknown input_gain: {input_gain}")
        self.in_features = in_features
        self.out_features = out_features
        self.tau_sharing = tau_sharing
        self.use_phi = use_phi
        self.input_gain = input_gain
        self.oscillatory = oscillatory
        self.basis_width = basis_width
        if oscillatory:  # angular frequency per edge, radians per step, init uniform in [0, pi)
            self.omega = nn.Parameter(math.pi * torch.rand(in_features, out_features))

        tau_shape = (in_features, 1) if tau_sharing == "input" else (in_features, out_features)
        log_tau = torch.empty(tau_shape).uniform_(math.log(tau_min), math.log(tau_max))
        a = torch.exp(-1.0 / torch.exp(log_tau))
        alpha = torch.log(a) - torch.log1p(-a)  # logit(a)
        if tau_sharing == "fixed":
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = nn.Parameter(alpha)

        self.weight = nn.Parameter(torch.randn(in_features, out_features) / math.sqrt(in_features))
        if use_phi:
            self.coef = nn.Parameter(
                0.3 * torch.randn(in_features, out_features, num_basis) / math.sqrt(in_features)
            )
            self.register_buffer("grid", torch.linspace(-grid_range, grid_range, num_basis))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def decay(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(log a, a, b), each broadcastable to (in_features, out_features)."""
        log_a = F.logsigmoid(self.alpha)
        a = torch.exp(log_a)
        one_minus_a = torch.sigmoid(-self.alpha)  # exact even when a is close to 1
        b = one_minus_a if self.input_gain == "ema" else torch.sqrt(one_minus_a * (1.0 + a))
        return log_a, a, b

    def tau(self) -> torch.Tensor:
        """Time constants in steps, shape (in_features, out_features)."""
        log_a, _, _ = self.decay()
        return (-1.0 / log_a).expand(self.in_features, self.out_features)

    def _edge_weights(self) -> torch.Tensor:
        """(in, out, 1 + R): linear weight followed by the tanh coefficients."""
        if not self.use_phi:
            return self.weight.unsqueeze(-1)
        return torch.cat([self.weight.unsqueeze(-1), self.coef], dim=-1)

    def _phi_sum(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, in, out, ...) edge states -> (B, out, ...) neuron outputs."""
        if self.use_phi:
            feats = tanh_basis(s, self.grid, self.basis_width)  # (B, in, out, ..., 1+R)
        else:
            feats = s.unsqueeze(-1)
        weights = self._edge_weights()
        extra = feats.dim() - 4  # 1 when a time axis is present
        w = weights.view(*weights.shape[:2], *([1] * extra), weights.shape[-1])
        out = (feats * w).sum(dim=(1, -1))
        return out + self.bias.view(-1, *([1] * extra))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in, L) -> (B, out, L)."""
        length = x.shape[-1]
        log_a, _, b = self.decay()
        steps = torch.arange(length, device=x.device, dtype=x.dtype)
        kernel = b.unsqueeze(-1) * torch.exp(log_a.unsqueeze(-1) * steps)
        if self.oscillatory:
            kernel = kernel * torch.cos(self.omega.unsqueeze(-1) * steps)
        kernel = kernel.expand(self.in_features, self.out_features, length)
        s = causal_fft_conv(x, kernel)  # (B, in, out, L)
        return self._phi_sum(s)

    def init_state(self, batch: int, device=None) -> torch.Tensor:
        dtype = torch.get_default_dtype()
        if self.oscillatory:
            dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        return torch.zeros(batch, self.in_features, self.out_features, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One sample: x_t (B, in), state (B, in, out) -> (h_t (B, out), new state)."""
        _, a, b = self.decay()
        pole = a * torch.exp(1j * self.omega) if self.oscillatory else a
        state = pole * state + b * x_t.unsqueeze(-1)
        return self._phi_sum(state.real if self.oscillatory else state), state

    def macs_per_step(self) -> dict[str, int]:
        """Streaming cost of one time step, counted per edge as on a microcontroller."""
        edges = self.in_features * self.out_features
        r = self.coef.shape[-1] if self.use_phi else 0
        memory = 5 if self.oscillatory else 2  # complex: 4 MAC rotation + b*x; real: a*s + b*x
        return {
            "mac": edges * (memory + 1 + r),  # memory, w*s, c_r*tanh
            "tanh": edges * r,
            "add": edges * r,  # s - g_r (the 1/H scale folds into the grid)
            "state_floats": edges * (2 if self.oscillatory else 1),
        }


class KANLayer(nn.Module):
    """Static KAN layer: the same per-edge phi as EdgeSSMLayer, without memory.

    The tanh basis depends only on the input value, so it is computed once per input and
    the per-edge coefficients reduce to a single matrix product.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_basis: int = 6,
        grid_range: float = 2.5,
        basis_width: float = 0.8,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis_width = basis_width
        self.register_buffer("grid", torch.linspace(-grid_range, grid_range, num_basis))
        weight = torch.randn(in_features, 1, out_features) / math.sqrt(in_features)
        coef = 0.3 * torch.randn(in_features, num_basis, out_features) / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.cat([weight, coef], dim=1))  # (in, 1+R, out)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in, L) -> (B, out, L); also accepts (B, in) for streaming."""
        feats = tanh_basis(x, self.grid, self.basis_width)  # (B, in, [L,] 1+R)
        if x.dim() == 2:
            return torch.einsum("bir,iro->bo", feats, self.weight) + self.bias
        return torch.einsum("bilr,iro->bol", feats, self.weight) + self.bias.unsqueeze(-1)

    def macs_per_step(self) -> dict[str, int]:
        r = self.grid.numel()
        return {
            "mac": self.in_features * (1 + r) * self.out_features,
            "tanh": self.in_features * r,
            "add": self.in_features * r,
            "state_floats": 0,
        }
