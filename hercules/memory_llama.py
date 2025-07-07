import torch.nn as nn
from transformers import LlamaForCausalLM, QuantoConfig

from hercules import NeuralMemory, inject_memory_module


class MemoryLlama(nn.Module):
    def __init__(
        self,
        llama_hf_path: str,
        freeze_llama_layers: bool,
        neural_memory_config: dict,
        memory_layer_id: int,
        quantize: bool,
        hf_token: str,
    ):
        super(MemoryLlama, self).__init__()

        if quantize:
            quantization_config = QuantoConfig(weights="int8")
        else:
            quantization_config = None

        llama = LlamaForCausalLM.from_pretrained(
            llama_hf_path,
            token=hf_token,
            # torch_dtype=torch.float16,
            # quantization_config=quantization_config,
        )
        self.quantization_config = quantization_config
        self.config = llama.config

        if freeze_llama_layers:
            llama = llama.eval()
            for param in llama.parameters():
                param.requires_grad = False
            assert sum(p.numel() for p in llama.parameters() if p.requires_grad) == 0

        self.llama = llama

        self.neural_memory_config = neural_memory_config
        self.neural_memory_config["input_dim"] = self.config.hidden_size
        self.neural_memory = NeuralMemory(**neural_memory_config)
        self.memory_layer_id = memory_layer_id
        # ensures the memory module uses the same precision as the quantized llama
        # self.neural_memory.to(torch.float16)

        self.model = inject_memory_module(
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
