from collections import Counter
from typing import Union

import torch
import torch.nn as nn
from colorama import Fore, Style
from dotenv import dotenv_values
from transformers import LlamaForCausalLM  # , QuantoConfig

from hercules import NeuralMemory, inject_memory_module, log_memory_model


class MemoryLlama(nn.Module):
    def __init__(
        self,
        llama_hf_path: str,
        freeze_llama_layers: bool,
        neural_memory_config: dict,
        memory_layer_ids: Union[int, list, tuple],
        token: str,
    ):
        super(MemoryLlama, self).__init__()

        llama = LlamaForCausalLM.from_pretrained(
            llama_hf_path,
            token=token,
            torch_dtype=torch.float16,
        )
        self.config = llama.config

        if freeze_llama_layers:
            for param in llama.parameters():
                param.requires_grad = False
            assert sum(p.numel() for p in llama.parameters() if p.requires_grad) == 0

        self.llama = llama

        self.neural_memory_config = neural_memory_config
        self.neural_memory_config["input_dim"] = self.config.hidden_size
        self.neural_memory = NeuralMemory(**neural_memory_config)

        self.model = inject_memory_module(
            self.llama,
            self.neural_memory,
            layer_id=memory_layer_ids,
        )

        log_memory_model(self.model)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
