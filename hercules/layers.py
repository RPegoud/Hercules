import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return nn.Sequential(
            self.depthwise,
            self.pointwise,
        )(x)


class ResLinear(nn.Module):
    """Residual MLP with SiLU activation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        out_size: int,
        n_hidden_layers: int,
    ):
        super(ResLinear, self).__init__()
        dims = np.array([input_size, *np.tile(hidden_size, n_hidden_layers), out_size])
        layer_sizes = list(zip(dims[:-1], dims[1:]))
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(*x)) for x in layer_sizes]
        )
        self.projections = nn.ParameterList(
            [
                (
                    nn.Parameter(torch.eye(in_dim, out_dim))
                    if in_dim == out_dim
                    else nn.Parameter(torch.randn(in_dim, out_dim))
                )
                for in_dim, out_dim in layer_sizes
            ]
        )

        for w in self.weights:
            nn.init.xavier_uniform_(w)

        for idx, (in_dim, out_dim) in enumerate(layer_sizes):
            if in_dim != out_dim:
                nn.init.xavier_uniform_(self.projections[idx])

    def forward(self, x: torch.Tensor):
        for idx, (w, p) in enumerate(zip(self.weights, self.projections)):
            first_layer = idx == 0
            if not first_layer:
                x = F.silu(x)
            residual = x
            x = x @ w + residual @ p

        return x


class LinearProjection(nn.Module):
    """Linear layer with no bias."""

    def __init__(
        self, in_dim: int, out_dim: int, n_chunks: int, kernel_size: int = None
    ) -> None:
        super(LinearProjection, self).__init__()
        self.reshape = Rearrange("b (n c) h -> b c n h", c=n_chunks)
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        # self.depthwise_separable_conv = DepthwiseSeparableConv1d(
        #     in_dim, out_dim, kernel_size
        # )

    def forward(self, x: torch.Tensor):
        return nn.Sequential(
            self.reshape,
            self.linear,
            nn.SiLU(),
            # self.depthwise_separable_conv, # TODO: needs fixing
        )(x)


class AdaptiveWeight(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_chunks: int, max_weight: float):
        super(AdaptiveWeight, self).__init__()
        self.reshape = Rearrange("b (n c) h -> b c n h", c=n_chunks)
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
        # TODO: add MLP after attention?

        output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        return output
