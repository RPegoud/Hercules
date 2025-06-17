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
        layer_size: int,
        output_size: int,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.lmm = lmm
        self.lm_head = nn.Linear(layer_size, output_size)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        with torch.set_grad_enabled(self.training):
            output = self.original_layer(
                hidden_states, attention_mask=attention_mask, **kwargs
            )
        attn_output = output[0]
        mal_output = self.lmm(attn_output)
        combined_output = (mal_output,) + output[1:]
        return self.lm_head(combined_output)


def inject_memory_module(
    llama: LlamaForCausalLM,
    memory_module: NeuralMemory,
    layer_size: int,
    output_size: int,
    layer_id: Union[int, list, tuple] = -2,
) -> LlamaForCausalLM:
    if isinstance(layer_id, int):
        original_layer = llama.model.layers[layer_id]
        llama.model.layers[layer_id] = LlamaMemoryAsLayer(
            original_layer, memory_module, layer_size, output_size
        )
    elif isinstance(layer_id, (list, tuple)):
        for id in layer_id:
            original_layer = llama.model.layers[layer_id]
            llama.model.layers[id] = (
                LlamaMemoryAsLayer(original_layer, memory_module),
                layer_size,
                output_size,
            )

    return llama
