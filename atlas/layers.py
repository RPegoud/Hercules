import torch
import torch.nn as nn


class ResLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        super(ResLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor):
        return F.silu(x + self.linear(x))
