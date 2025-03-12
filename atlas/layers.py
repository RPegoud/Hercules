import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    """Linear Layer with no bias."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(LinearProjection, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor):
        return F.silu(self.linear(x))


class AdaptiveLR(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, max_lr: float):
        super(AdaptiveLR, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.max_lr = max_lr

    def forward(self, x: torch.Tensor):
        return F.sigmoid(self.linear(x)) * self.max_lr

class SlidingWindowAttention(nn.Module):
    """Self attention block with sliding window."""

    def __init__(self, input_dim: int, num_heads: int, window_size: int, device: str):
        super(SlidingWindowAttention, self).__init__()
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, # device=device # TODO: reactivate for testing
        )

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