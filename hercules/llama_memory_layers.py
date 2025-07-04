from typing import Union

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

from hercules import NeuralMemory


class LlamaMemoryAsLayer(nn.Module):
    def __init__(
        self,
        original_layer: nn.Module,
        lmm: NeuralMemory,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.lmm = lmm

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        with torch.set_grad_enabled(self.training):
            llama_output = self.original_layer(
                hidden_states, attention_mask=attention_mask, **kwargs
            )
        attn_output = llama_output[0]
        mal_output = self.lmm(attn_output)

        assert (
            attn_output.shape == mal_output.shape
        ), f"Memory module output shape: {mal_output.shape}, expected {attn_output.shape}"

        # replace the attention output with the memory augmented attention
        return (mal_output,) + llama_output[1:]


def inject_memory_module(
    llama: LlamaForCausalLM,
    memory_module: NeuralMemory,
    layer_id: Union[int, list, tuple],
) -> LlamaForCausalLM:
    original_layer = llama.model.layers[layer_id]
    llama.model.layers[layer_id] = LlamaMemoryAsLayer(original_layer, memory_module)

    return llama
