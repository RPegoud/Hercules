import torch.nn as nn
import torch
from transformers import LlamaForCausalLM
from typing import List, Union
from colorama import Style, Fore
from peft import LoraConfig, get_peft_model, TaskType
from hercules import NeuralMemory
from omegaconf.listconfig import ListConfig
from peft import PeftModel


class MemoryLlama(nn.Module):
    def __init__(
        self,
        llama_hf_path: str,
        neural_memory_config: dict,
        memory_layer_id: int,
        hf_token: str,
        use_lora: bool,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        lora_target_modules: list[str] | None = None,
        lora_dropout: float | None = None,
    ):
        super(MemoryLlama, self).__init__()

        llama = LlamaForCausalLM.from_pretrained(
            llama_hf_path,
            token=hf_token,
        )

        self.config = llama.config

        neural_memory_config["hidden_size"] = llama.config.hidden_size
        self.neural_memory_config = neural_memory_config

        if not isinstance(memory_layer_id, ListConfig):
            memory_layer_id = [memory_layer_id]
        self.memory_layer_ids = memory_layer_id

        self.neural_memory = nn.ModuleList(
            [NeuralMemory(**neural_memory_config) for _ in self.memory_layer_ids]
        )

        n_memory_params = 0
        for memory_module in self.neural_memory:
            n_memory_params += sum(p.numel() for p in memory_module.parameters())
        self.n_memory_params = n_memory_params

        memory_llama = inject_memory_modules(
            llama, self.neural_memory, self.memory_layer_ids
        )

        if use_lora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=list(lora_target_modules),
                modules_to_save=["neural_memory"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            print(f"{Fore.BLUE}Applying LoRA config ...")
            memory_llama = get_peft_model(memory_llama, lora_config)

            # enable gradients for gate parameters
            for n, p in memory_llama.named_parameters():
                if "neural_memory" in n:
                    p.requires_grad = True
                    if "memory_module" in n:
                        p.requires_grad = False

        self.model = memory_llama

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            model = super().__getattribute__("model")
            return getattr(model, attr)

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

    def save(self, path: str) -> None:
        self.model.save_pretrained(path, safe_serialization=True)

    @classmethod
    def load(
        cls,
        adapter_path: str,
        llama_hf_path: str,
        neural_memory_config: dict,
        memory_layer_id: int,
        hf_token: str,
        **kwargs,
    ):
        base_memory_llama = cls(
            use_lora=False,
            llama_hf_path=llama_hf_path,
            neural_memory_config=neural_memory_config,
            memory_layer_id=memory_layer_id,
            hf_token=hf_token,
            **kwargs,
        )

        initial_params = base_memory_llama.model.parameters()

        base_memory_llama.model = PeftModel.from_pretrained(
            base_memory_llama.model, adapter_path
        )

        assert (
            initial_params != base_memory_llama.model.parameters()
        ), "Loaded parameters are the same as random initialisation, something went wrong."


class LlamaMemoryAsLayer(nn.Module):
    def __init__(
        self,
        original_layer: nn.Module,
        neural_memory: NeuralMemory,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.neural_memory = neural_memory

    @staticmethod
    def _assert_equal_dim(x: torch.Tensor, y: torch.Tensor) -> None:
        assert (
            x.shape == y.shape
        ), f"Memory module output shape: {x.shape}, expected {y.shape}"

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        mal_output = self.neural_memory(hidden_states)
        with torch.set_grad_enabled(self.training):
            llama_output = self.original_layer(
                mal_output, attention_mask=attention_mask, **kwargs
            )
            attn_output = llama_output[0]
        self._assert_equal_dim(mal_output, attn_output)
        return llama_output


def inject_memory_modules(
    llama: LlamaForCausalLM,
    memory_modules: nn.ModuleList,
    layer_ids: List[int],
) -> LlamaForCausalLM:

    for layer_id, memory_module in zip(layer_ids, memory_modules):
        original_layer = llama.model.layers[layer_id]
        llama.model.layers[layer_id] = LlamaMemoryAsLayer(original_layer, memory_module)

    return llama
