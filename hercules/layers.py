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
    def __init__(self, hidden_size: int, depth: int, expansion_factor: int):
        super().__init__()
        dims = [
            hidden_size,
            *((hidden_size * expansion_factor,) * (depth - 1)),
            hidden_size,
        ]
        layer_sizes = zip(dims[:1], dims[1:])
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(dim_in, dim_out))
                for dim_in, dim_out in layer_sizes
            ]
        )
        nn.init.zeros_(self.weights)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, (w, p) in enumerate(zip(self.weights, self.projections)):
            if idx != 0:
                x = F.silu(x)
            residual = x
            x = x @ w + residual @ p
        return x


class LinearProjection(nn.Module):
    """Linear layer with no bias."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_chunks: int,
    ) -> None:
        super(LinearProjection, self).__init__()
        self.n_chunks = n_chunks
        self.reshape = Rearrange("b (n c) h -> b c n h", c=n_chunks)
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor):
        # chunk the input only if `n_chunks` is greater than one and
        # the LLM is not generating
        # is_generating = x.shape[1] == 1
        # if not is_generating and self.n_chunks > 1:
        #     x = self.reshape(x)
        x = self.linear(x)
        x = F.silu(x)

        return x


class AdaptiveWeight(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_chunks: int,
        max_weight: float = 1.0,
    ):
        super(AdaptiveWeight, self).__init__()
        self.reshape = Rearrange("b (n c) h -> b c n h", c=n_chunks)
        self.linear = nn.Linear(in_dim, out_dim)
        self.max_weight = max_weight

    def forward(self, x: torch.Tensor):
        # x = self.reshape(x)
        x = self.linear(x)
        lr = F.sigmoid(x)
        return lr * self.max_weight  # rescale lr
