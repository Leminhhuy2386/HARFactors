"""Edge-SSM classifier (architecture v2): Edge-SSM layer (memory) -> static KAN layer ->
mean over time -> linear head.

The mean over the window is a running sum at inference, so the whole network streams:
per sample it updates one float per edge of the first layer and adds to the running sum.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.layers import EdgeSSMLayer, KANLayer


class EdgeSSMClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int,
        window_length: int,
        num_classes: int,
        *,
        hidden: int = 64,
        kan_hidden: int = 64,
        **edge_kwargs,
    ) -> None:
        super().__init__()
        self.window_length = window_length
        self.edge = EdgeSSMLayer(input_channels, hidden, **edge_kwargs)
        self.kan = KANLayer(hidden, kan_hidden)
        self.head = nn.Linear(kan_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> logits (B, num_classes)."""
        h = self.kan(self.edge(x))
        return self.head(h.mean(dim=-1))

    @torch.no_grad()
    def forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        """Same result as forward, computed one sample at a time (reference for deployment)."""
        batch, _, length = x.shape
        state = self.edge.init_state(batch, device=x.device)
        pooled = torch.zeros(batch, self.kan.out_features, device=x.device)
        for t in range(length):
            h, state = self.edge.step(x[:, :, t], state)
            pooled += self.kan(h)
        return self.head(pooled / length)

    def streaming_cost(self) -> dict[str, int]:
        """Operations per input sample, plus the head once per window (amortized)."""
        edge, kan = self.edge.macs_per_step(), self.kan.macs_per_step()
        per_step = {k: edge[k] + kan[k] for k in edge}
        per_step["add"] += self.kan.out_features  # running sum for the mean
        per_step["state_floats"] += self.kan.out_features
        head = self.head.in_features * self.head.out_features
        per_step["mac_per_window"] = per_step["mac"] * self.window_length + head
        return per_step

    def tau_seconds(self, sampling_freq: float) -> torch.Tensor:
        """Learned time constants of the memory layer, (input_channels, hidden), in seconds."""
        return self.edge.tau().detach().cpu() / float(sampling_freq)
