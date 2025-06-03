import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange


class ResLinear(nn.Module):
    """Residual MLP with SiLU activation."""

    def __init__(self, layer_size: int, num_layers: int):
        super(ResLinear, self).__init__()
        dims = np.tile([layer_size], num_layers)
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(d, d)) for d in dims])
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x: torch.Tensor):
        for idx, w in enumerate(self.weights):
            first_layer = idx == 0
            if not first_layer:
                x = F.silu(x)
            residual = x
            x = x @ w + residual

        return x


class LinearProjection(nn.Module):
    """Linear layer with no bias."""

    def __init__(self, in_dim: int, out_dim: int, n_chunks: int) -> None:
        super(LinearProjection, self).__init__()
        self.reshape = Rearrange("b (n c) h -> b n c h", c=n_chunks)
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor):
        return nn.Sequential(
            self.reshape,
            self.linear,
            nn.SiLU(),
        )(x)


class AdaptiveWeight(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_chunks: int,
        max_weight: float,
    ):
        super(AdaptiveWeight, self).__init__()
        self.reshape = Rearrange("b (n c) h -> b n c h", c=n_chunks)
        self.linear = nn.Linear(in_dim, out_dim)
        self.max_weight = max_weight

    def forward(self, x: torch.Tensor):
        lr = nn.Sequential(
            self.reshape,
            self.linear,
            nn.Sigmoid(),
        )(x)
        return lr * self.max_weight  # rescale lr


class SlidingWindowAttention(nn.Module):
    """Self attention block with sliding window."""

    def __init__(self, input_dim: int, num_heads: int, window_size: int):
        super(SlidingWindowAttention, self).__init__()
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        attn_mask = torch.zeros(seq_len, seq_len).bool()
        attn_mask = attn_mask.to(x.device)
        indices = torch.arange(seq_len, device=x.device)
        # sliding attention mask
        attn_mask = (indices[:, None] - indices[None, :]).abs() <= (
            self.window_size // 2
        )

        output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        return output
