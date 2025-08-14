import torch.nn as nn
import torch
from transformers import LlamaForCausalLM
from typing import List, Union
from colorama import Style, Fore
from peft import LoraConfig, get_peft_model, TaskType
from hercules import NeuralMemory


class MemoryLlama(nn.Module):
    def __init__(
        self,
        llama_hf_path: str,
        neural_memory_config: dict,
        memory_layer_id: int,
        mode: str,
        hf_token: str,
        use_lora: bool,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: list[str],
        lora_dropout: float,
    ):
        super(MemoryLlama, self).__init__()

        llama = LlamaForCausalLM.from_pretrained(
            llama_hf_path,
            token=hf_token,
        )
        if not use_lora:
            for p in llama.parameters():
                p.requires_grad = False

        self.llama = llama
        self.config = llama.config

        neural_memory_config["hidden_size"] = llama.config.hidden_size
        self.neural_memory_config = neural_memory_config

        self.neural_memory = NeuralMemory(**neural_memory_config)
        self.memory_layer_id = memory_layer_id

        memory_llama = inject_memory_module(  # memory-augmented llama
            self.llama,
            self.neural_memory,
            layer_id=memory_layer_id,
            mode=mode,
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
            memory_llama.print_trainable_parameters()

        self.model = memory_llama

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        """
        Returns a list of all parameters that should be trained.
        This includes all parameters of the NeuralMemory module and,
        if LoRA is used, the trainable LoRA parameters from the base model.
        """
        gate_params = list(self.neural_memory.gate_parameters)
        llm_params = [p for p in self.model.parameters() if p.requires_grad]
        all_params = list({id(p): p for p in gate_params + llm_params}.values())
        return all_params

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
        mode: str,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.neural_memory = neural_memory
        self.mode = mode

        print(f"{Fore.BLUE}Memory Llama mode: {mode}")

        assert mode in [
            "embedding",
            "attention",
        ], f"Expected mode to be either 'embedding' or 'attention', got {mode}"

    @staticmethod
    def _assert_equal_dim(x: torch.Tensor, y: torch.Tensor) -> None:
        assert (
            x.shape == y.shape
        ), f"Memory module output shape: {x.shape}, expected {y.shape}"

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if self.mode == "embedding":
            mal_output = self.neural_memory(hidden_states)
            with torch.set_grad_enabled(self.training):
                llama_output = self.original_layer(
                    mal_output, attention_mask=attention_mask, **kwargs
                )
                attn_output = llama_output[0]
            self._assert_equal_dim(mal_output, attn_output)
            return llama_output

        if self.mode == "attention":
            with torch.set_grad_enabled(self.training):
                llama_output = self.original_layer(
                    hidden_states, attention_mask=attention_mask, **kwargs
                )
            attn_output = llama_output[0]
            mal_output = self.neural_memory(attn_output)
            self._assert_equal_dim(mal_output, attn_output)

            # replace the attention output with the memory augmented attention
            return (mal_output,) + llama_output[1:]


def inject_memory_module(
    llama: LlamaForCausalLM,
    memory_module: NeuralMemory,
    layer_id: Union[int, list],
    mode: str,
) -> LlamaForCausalLM:
    if not isinstance(layer_id, list):
        layer_id = [layer_id]

    for id in layer_id:
        original_layer = llama.model.layers[id]
        llama.model.layers[id] = LlamaMemoryAsLayer(original_layer, memory_module, mode)

    return llama
