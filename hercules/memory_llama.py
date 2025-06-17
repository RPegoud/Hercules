import torch.nn as nn
from transformers import AutoModelForCausalLM, QuantoConfig

from hercules import LlamaMemoryAsLayer, NeuralMemory


class MemoryLlama(nn.Module):
    def __init__(
        self,
        memory_architecture: str,
        llama_hf_path: str,
        freeze_llama_layers: bool,
        neural_memory_config: dict,
        quantize: bool,
    ):
        super(MemoryLlama, self).__init__()
        self.MEMORY_ARCHITECTURES = ["layer", "gate", "context"]

        assert (
            memory_architecture in self.MEMORY_ARCHITECTURES
        ), f"Memory architecture must be one of {self.MEMORY_ARCHITECTURES}, got {memory_architecture}"

        self.memory_architecture = memory_architecture

        if quantize:
            quantization_config = QuantoConfig(weights="int8")
        else:
            quantization_config = None

        self.quantization_config = quantization_config
        self.llama = AutoModelForCausalLM.from_pretrained(
            llama_hf_path, quantization_config=quantization_config, device_map="auto"
        )
        self.tokenizer = AutoModelForCausalLM.from_pretrained(llama_hf_path)
        self.config = self.llama.config

        if freeze_llama_layers:
            for param in self.llama.parameters():
                param.requires_grad = False

        self.neural_memory_config = neural_memory_config
        self.neural_memory_config["input_dim"] = self.config.hidden_size
        self.neural_memory = NeuralMemory(**neural_memory_config)

        if memory_architecture == "layer":
            original_layer = self.llama.model.layers[-2]
            self.llama.model.layers[-2] = LlamaMemoryAsLayer(
                original_layer, self.neural_memory
            )

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.llama(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )
