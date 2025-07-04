import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LinearProjection, ResLinear, SlidingWindowAttention


def flatten_and_expand(x: torch.Tensor, n: int):
    """Flattens a tensor and adds `n` trailing dimensions."""
    return x.view(-1, *([1] * n))


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


class NeuralMemory(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        conv_kernel_size: int,
        n_hidden_layers: int,
        meta_memory_dim: int,
        num_attention_heads: int,
        attention_window_size: int,
    ) -> None:
        super(NeuralMemory, self).__init__()
        self.meta_memory_dim = meta_memory_dim

        self.memory_module = ResLinear(
            input_dim, hidden_dim, output_dim, n_hidden_layers
        )
        self.key_projection = LinearProjection(input_dim, output_dim, conv_kernel_size)
        self.query_projection = LinearProjection(
            input_dim, output_dim, conv_kernel_size
        )
        self.value_projection = LinearProjection(
            input_dim, output_dim, conv_kernel_size
        )

        self.meta_memory = nn.Parameter(torch.randn(meta_memory_dim, input_dim))
        nn.init.xavier_uniform_(
            self.meta_memory
        )  # TODO: is it necessary to initialise here?

        self.swa = SlidingWindowAttention(
            input_dim, num_attention_heads, attention_window_size
        )

    def _inject_meta_memory(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        meta_memory = self.meta_memory.expand(batch_size, -1, -1)
        meta_x = torch.concat([meta_memory, x], dim=1)
        return meta_x

    def get_associative_memory_loss(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        k = self.key_projection(x)
        v = self.value_projection(x)
        k, v = l2_norm(k), l2_norm(v)

        preds = self.memory_module(k)
        loss = F.mse_loss(preds, v)
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._inject_meta_memory(x)

        q = self.query_projection(x)
        q = l2_norm(q)

        retrieved = self.memory_module(q)
        retrieved = F.silu(retrieved)
        retrieved = self.swa(retrieved)

        # discard meta-memory
        output = retrieved[:, self.meta_memory_dim :, :]

        return output
