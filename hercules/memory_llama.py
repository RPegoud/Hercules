import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from typing import Union

from hercules import NeuralMemory


class MemoryLlama(nn.Module):
    def __init__(
        self,
        llama_hf_path: str,
        freeze_llama_layers: bool,
        neural_memory_config: dict,
        memory_layer_id: int,
        hf_token: str,
    ):
        super(MemoryLlama, self).__init__()

        llama = LlamaForCausalLM.from_pretrained(
            llama_hf_path,
            token=hf_token,
        )
        self.config = llama.config

        if freeze_llama_layers:
            llama = llama.eval()
            for param in llama.parameters():
                param.requires_grad = False
            assert sum(p.numel() for p in llama.parameters() if p.requires_grad) == 0

        self.llama = llama  # pre-trained llama

        neural_memory_config["hidden_size"] = llama.config.hidden_size
        self.neural_memory_config = neural_memory_config

        self.neural_memory = NeuralMemory(**neural_memory_config)
        self.memory_layer_id = memory_layer_id

        self.model = inject_memory_module(  # memory-augmented llama
            self.llama,
            self.neural_memory,
            layer_id=memory_layer_id,
        )

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, input_ids, attention_mask, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )


class LlamaMemoryAsLayer(nn.Module):
    def __init__(
        self,
        original_layer: nn.Module,
        neural_memory: NeuralMemory,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.neural_memory = neural_memory

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        with torch.set_grad_enabled(self.training):
            llama_output = self.original_layer(
                hidden_states, attention_mask=attention_mask, **kwargs
            )
        attn_output = llama_output[0]
        mal_output = self.neural_memory(attn_output)

        assert (
            attn_output.shape == mal_output.shape
        ), f"Memory module output shape: {mal_output.shape}, expected {attn_output.shape}"

        # replace the attention output with the memory augmented attention
        return (mal_output,) + llama_output[1:]


def inject_memory_module(
    llama: LlamaForCausalLM,
    memory_module: NeuralMemory,
    layer_id: Union[int, list],
) -> LlamaForCausalLM:
    if isinstance(layer_id, list):
        for id in layer_id:
            original_layer = llama.model.layers[id]
            llama.model.layers[id] = LlamaMemoryAsLayer(original_layer, memory_module)
    else:
        original_layer = llama.model.layers[layer_id]
        llama.model.layers[layer_id] = LlamaMemoryAsLayer(original_layer, memory_module)

    return llama
