import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from typing import Union, List
from colorama import Fore, Style
from hercules import NeuralMemory


class MemoryLlama(nn.Module):
    def __init__(
        self,
        llama_hf_path: str,
        neural_memory_config: dict,
        memory_layer_id: int,
        hf_token: str,
        trainable_blocks: Union[None, bool, List[int]] = None,
    ):
        super(MemoryLlama, self).__init__()

        llama = LlamaForCausalLM.from_pretrained(
            llama_hf_path,
            token=hf_token,
        )
        self.config = llama.config
        self.trainable_blocks = trainable_blocks

        self.llama = self._unfreeze_layers(llama)  # pre-trained llama

        neural_memory_config["hidden_size"] = llama.config.hidden_size
        self.neural_memory_config = neural_memory_config

        self.neural_memory = NeuralMemory(**neural_memory_config)
        self.memory_layer_id = memory_layer_id

        self.model = inject_memory_module(  # memory-augmented llama
            self.llama,
            self.neural_memory,
            layer_id=memory_layer_id,
        )

    def _unfreeze_layers(self, llama: LlamaForCausalLM) -> LlamaForCausalLM:
        """
        Unfreezes Llama layers based on the config:

        Options:
            ```
            - "all" (str): unfreezes all layers
            - None or False (bool): unfreezes no layer
            - List[int]: unfreezes specified layers
            ```
        """
        if self.trainable_blocks == "all":
            print(f"{Fore.BLUE}{Style.BRIGHT}--- All Llama blocks are trainable. ---")
            return llama

        # by default keep embedding and norm layers frozen
        for param in llama.model.embed_tokens.parameters():
            param.requires_grad = False

        for param in llama.model.norm.parameters():
            param.requires_grad = False

        for param in llama.model.layers.parameters():
            param.requires_grad = False

        if self.trainable_blocks is None or not self.trainable_blocks:
            print(f"{Fore.BLUE}{Style.BRIGHT}--- All Llama blocks are frozen. ---")
            n_trainable = sum(p.numel() for p in llama.parameters() if p.requires_grad)
            assert (
                n_trainable == 0
            ), f"Expected zero trainable parameters, got {n_trainable}"
            return llama

        if isinstance(self.trainable_blocks, int):
            self.trainable_blocks = list([self.trainable_blocks])

        print(
            f"{Fore.BLUE}{Style.BRIGHT}Unfreezing Llama blocks: {self.trainable_blocks} ..."
        )
        for idx in self.trainable_blocks:
            for param in llama.model.layers[idx].parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in llama.parameters() if p.requires_grad)
        assert (
            n_trainable != 0
        ), f"Expected number of trainable parameters to be positive, got {n_trainable}"

        return llama

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        """
        Returns a list of parameters containing trainable Llama
        parameters and the gate parameters of the memory module.
        """
        trainable_blocks = self.trainable_blocks
        if trainable_blocks is not None:
            llama_blocks = self.llama.model.layers
            trainable_llama_params = []
            for idx in trainable_blocks:
                trainable_llama_params.extend(llama_blocks[idx].parameters())

            all_params = self.neural_memory.gate_parameters + trainable_llama_params
            seen = set()
            param_list = []
            for p in all_params:
                if id(p) not in seen:
                    param_list.append(p)
                    seen.add(id(p))
        else:
            param_list = self.neural_memory.gate_parameters

        return param_list

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
        with torch.no_grads():
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
