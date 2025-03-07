import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from atlas.layers import ResLinear


class MemoryAsLayer(nn.Module):
    def __init__(
        self,
        layer_sizes: int,
        persistent_memory_dim: int,
    ) -> None:
        super(MemoryAsLayer, self).__init__()
        self.persistent_memory_dim = persistent_memory_dim
        self.layer_sizes = torch.Tensor(layer_sizes)
        self.layer_sizes += persistent_memory_dim

        self.lmm = nn.Sequential(
            *[
                ResLinear(in_dim, out_dim)
                for in_dim, out_dim in list(
                    zip(self.layer_sizes[:-1], self.layer_sizes[1:])
                )
            ]
        )

    def forward(self, x: torch.Tensor, persistent_memory: torch.Tensor) -> torch.Tensor:
        x = x.view(-1)
        persistent_memory = persistent_memory.view(-1)
        x_with_memory = torch.concat((persistent_memory, x))
        return self.lmm(x_with_memory)


class LlamaWithMAL(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        layer_sizes: list[int],
        freeze_llama: bool = True,
    ):
        self.model = model
        self.freeze_llama = freeze_llama
        self.memory_module = MemoryAsLayer(layer_sizes)

        if freeze_llama:
            self.model.requires_grad_(False)
