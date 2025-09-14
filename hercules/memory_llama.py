import torch.nn as nn
import torch
from transformers import LlamaForCausalLM
from typing import List
from colorama import Style, Fore
from peft import LoraConfig, get_peft_model, TaskType
from hercules import NeuralMemory
from omegaconf.listconfig import ListConfig
from peft import PeftModel
from dataclasses import dataclass
from torch.nn.modules.container import ModuleList
from torch.nn.utils import parameters_to_vector
import os


@dataclass
class MemoryStatistics:
    layer: int
    gate_bias: float
    gate_mean: float
    gate_std: float
    gate_max: float
    gate_min: float
    mac_proj_memory_w_norm: float
    per_token_avg_memory_contrib_ratio: float
    global_avg_memory_contrib_ratio: float


TRAINABLE_LAYERS_PER_ARCH = {
    "context": {
        "include": ["mac_projection", "gate_projection", "memory_module"],
        "exclude": ["memory_network"],
    },
    "layer": {
        "include": ["memory_module"],
        "exclude": ["memory_network"],
    },
}


class MemoryLlama(nn.Module):
    def __init__(
        self,
        llama_hf_path: str,
        neural_memory_config: dict,
        memory_layer_id: int,
        memory_arch: str,
        hf_token: str,
        track_memory_statistics: bool,
        use_lora: bool,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        lora_target_modules: list[str] | None = None,
        lora_dropout: float | None = None,
    ):
        super(MemoryLlama, self).__init__()
        self.use_lora = use_lora

        print(f"{Fore.BLUE}{Style.BRIGHT}MemoryLlama setup:")
        llama = LlamaForCausalLM.from_pretrained(llama_hf_path, token=hf_token)
        if not use_lora:
            for param in llama.parameters():
                param.requires_grad = False

            n_trainable_params = sum(
                p.numel() for p in llama.parameters() if p.requires_grad
            )
            assert (
                n_trainable_params == 0
            ), f"""`use_lora` is false but Llama has {n_trainable_params}
            trainable parameters, expected zero."""
            print(
                f"{Fore.BLUE}Frozen Llama, number of trainable params: {n_trainable_params}"
            )

        self.config = llama.config

        self.neural_memory_config = neural_memory_config
        self.memory_arch = memory_arch

        if not isinstance(memory_layer_id, ListConfig):
            memory_layer_id = [memory_layer_id]
        self.memory_layer_ids = memory_layer_id

        self.memory_modules = nn.ModuleList(
            [
                NeuralMemory(
                    hidden_size=llama.config.hidden_size, **neural_memory_config
                )
                for _ in self.memory_layer_ids
            ]
        )

        n_memory_params = 0
        for memory_module in self.memory_modules:
            n_memory_params += sum(p.numel() for p in memory_module.parameters())
        self.n_memory_params = n_memory_params

        print(f"{Fore.BLUE}Injecting Memory as a {memory_arch.capitalize()} layers ...")
        memory_llama = inject_memory_modules(
            llama=llama,
            memory_modules=self.memory_modules,
            layer_ids=self.memory_layer_ids,
            track_memory_statistics=track_memory_statistics,
            arch=memory_arch,
        )

        if use_lora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=list(lora_target_modules),
                modules_to_save=TRAINABLE_LAYERS_PER_ARCH[memory_arch][
                    "include"
                ].extend(["memory_network"]),
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            print(f"{Fore.BLUE}Applying LoRA config ...")
            memory_llama = get_peft_model(memory_llama, lora_config)

        memory_llama = self.set_trainable_layers(
            model=memory_llama, layer_dict=TRAINABLE_LAYERS_PER_ARCH[memory_arch]
        )

        self.model = memory_llama

    def forward(self, input_ids, attention_mask, labels, **kwargs) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, input_ids, attention_mask, **kwargs) -> torch.Tensor:
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    def reset_memory(self):
        for mod in self.memory_modules:
            mod.reset_memory()

    def set_trainable_layers(
        self, model: LlamaForCausalLM, layer_dict: dict[str, list[str]]
    ) -> LlamaForCausalLM:
        """
        Set requires_grad=True for params whose name contains something in `include`,
        and requires_grad=False if in `exclude`.\\
        LoRA adapters are already handled by PEFT (requires_grad=True), this method
        is used to re-enable custom memory modules.
        """
        include = layer_dict["include"] or []
        exclude = layer_dict["exclude"] or []

        for n, p in model.named_parameters():
            if any(inc in n for inc in include):
                p.requires_grad = True

            if any(ex in n for ex in exclude):
                p.requires_grad = False

        return model

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            model = super().__getattribute__("model")
            return getattr(model, attr)

    @property
    def get_llama_model(self) -> LlamaForCausalLM:
        """Returns the base llama model without the LoRA wrapper."""
        if self.use_lora:
            return self.model.get_base_model()
        else:
            return self.model

    @property
    def layers(self) -> ModuleList:
        """
        MemoryLlama has the following model wrapper hierarchy:
        - <class '__main__.MemoryLlama'>
        - <class 'peft.peft_model.PeftModelForCausalLM'>
        - <class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>
        - <class 'transformers.models.llama.modeling_llama.LlamaModel'>

        This property returns the layers of `LlamaModel`.
        """
        if self.use_lora:
            return self.model.get_base_model().model.layers
        else:
            return self.model.model.layers

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        if self.use_lora:
            self.model.save_pretrained(path, safe_serialization=True)

        for i, mem in zip(self.memory_layer_ids, self.memory_modules):
            mem_path = os.path.join(path, f"memory_module_{i}.pt")
            torch.save(mem.state_dict(), mem_path)

            layer = self.layers[i]
            torch.save(
                {
                    "mac_projection": layer.mac_projection.state_dict(),
                    "gate_projection": layer.gate_projection.state_dict(),
                },
                os.path.join(path, f"memory_projections_{i}.pt"),
            )

        torch.save(
            {
                "memory_arch": self.memory_arch,
                "neural_memory_config": self.neural_memory_config,
                "memory_layer_ids": self.memory_layer_ids,
            },
            os.path.join(path, "memoryllama_config.pt"),
        )

    @classmethod
    def load(
        cls,
        path: str,
        memory_llama_config: dict,
    ):
        use_lora = memory_llama_config.use_lora
        memory_llama_config.__delattr__("use_lora")

        cfg = torch.load(os.path.join(path, "memoryllama_config.pt"))

        memory_llama = cls(
            llama_hf_path=memory_llama_config.llama_hf_path,
            neural_memory_config=cfg["neural_memory_config"],
            memory_layer_id=cfg["memory_layer_ids"],
            memory_arch=cfg["memory_arch"],
            hf_token=memory_llama_config.hf_token,
            track_memory_statistics=True,
            use_lora=False,
        )

        if use_lora:
            memory_llama.model = PeftModel.from_pretrained(memory_llama.model, path)
            print(f"{Fore.MAGENTA}Loaded LoRA adapters")

            assert hasattr(memory_llama.model, "peft_config"), "LoRA config missing"
            assert len(memory_llama.model.peft_config) > 0, "No LoRA adapters found"

        for i, mem in zip(cfg["memory_layer_ids"], memory_llama.memory_modules):
            mem_path = os.path.join(path, f"memory_module_{i}.pt")
            mem.load_state_dict(torch.load(mem_path))
            print(f"{Fore.MAGENTA}Loaded Memory Module at layer: {i}")

            proj_path = os.path.join(path, f"memory_projections_{i}.pt")
            projections = torch.load(proj_path)

            memory_llama.layers[i].mac_projection.load_state_dict(
                projections["mac_projection"]
            )
            memory_llama.layers[i].gate_projection.load_state_dict(
                projections["gate_projection"]
            )

            gate_bias = memory_llama.layers[i].gate_projection.bias.data[0].item()
            print(f"{Fore.MAGENTA}Gate Bias: {gate_bias:.3f}")
            assert (
                gate_bias != -6.0
            ), f"Memory module is likely not properly initialised, got gate bias of {gate_bias}."

            mac_proj_memory_w = (
                memory_llama.layers[i].mac_projection.weight[:, :2048].norm().item()
            )
            print(
                f"{Fore.MAGENTA}Mac Projection Memory Weights Norm: {mac_proj_memory_w:.3f}"
            )
            assert (
                mac_proj_memory_w != 0.0
            ), f"Memory module is likely not properly initialised, got Mac Projection Memory Weights Norm of {mac_proj_memory_w}."

        return memory_llama


class LlamaMemoryAsLayer(nn.Module):
    def __init__(
        self,
        llama_layer: nn.Module,
        memory_module: NeuralMemory,
        layer_id: int,
        track_memory_statistics: None = None,
    ):
        super().__init__()

        self.llama_layer = llama_layer
        self.memory_module = memory_module
        self.layer_id = layer_id
        self.track_memory_statistics = track_memory_statistics

        self.hidden_size = self.llama_layer.self_attn.q_proj.in_features

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        norm_hidden_states = self.llama_layer.input_layernorm(hidden_states)
        memory_outputs = self.memory_module(norm_hidden_states)
        attn = self.llama_layer.self_attn(
            memory_outputs, attention_mask=attention_mask, **kwargs
        )
        attn_outputs = attn[0]
        residual_attn = attn_outputs + hidden_states
        norm_residual_attn = self.llama_layer.post_attention_layernorm(residual_attn)
        llama_outputs = self.llama_layer.mlp(norm_residual_attn) + residual_attn

        block_outputs = (llama_outputs,) + attn[1:]

        return block_outputs


class LlamaMemoryAsContext(nn.Module):
    def __init__(
        self,
        llama_layer: nn.Module,
        memory_module: NeuralMemory,
        layer_id: int,
        track_memory_statistics: bool,
    ):
        super().__init__()

        self.llama_layer = llama_layer
        self.memory_module = memory_module
        self.layer_id = layer_id
        self.track_memory_statistics = track_memory_statistics

        hidden_size = self.llama_layer.self_attn.q_proj.in_features
        memory_sequence_size = 2 * hidden_size
        self.hidden_size = hidden_size

        # projection from [attention, memory] to [attention]
        # initially, the memory components have a weight of 0 so that the behavior
        # of the llm is unaffected before training
        self.mac_projection = nn.Linear(memory_sequence_size, hidden_size, bias=False)
        with torch.no_grad():
            self.mac_projection.weight[:, hidden_size:] = torch.eye(hidden_size)
            self.mac_projection.weight[:, :hidden_size] = 0

        # we initialise the gate with zero weights and low bias so that the initial
        # behavior of the memory-augmented llama model is almost the same as the
        # original model.
        self.gate_projection = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(
            self.gate_projection.bias, -6
        )  # sigmoid(-6) = 0.0025, sigmoid(-10) = 4.5398e-05

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        context = self.memory_module.retrieve(hidden_states)

        norm_hidden_states = self.llama_layer.input_layernorm(hidden_states)
        memory_sequence = torch.cat([context, norm_hidden_states], dim=-1)
        mac_sequence = self.mac_projection(memory_sequence)

        attn = self.llama_layer.self_attn(
            mac_sequence, attention_mask=attention_mask, **kwargs
        )
        attn_outputs = attn[0]
        memory_outputs = self.memory_module(attn_outputs)

        gate = torch.sigmoid(self.gate_projection(attn_outputs))
        memory_augmented_attn = (1 - gate) * attn_outputs + gate * memory_outputs

        residual_attn = memory_augmented_attn + hidden_states
        norm_residual_attn = self.llama_layer.post_attention_layernorm(residual_attn)
        llama_outputs = self.llama_layer.mlp(norm_residual_attn) + residual_attn

        block_outputs = (llama_outputs,) + attn[1:]

        if self.track_memory_statistics:
            self.last_stats = MemoryStatistics(
                layer=self.layer_id,
                gate_bias=self.gate_projection.bias.data[0].item(),
                gate_mean=gate.mean().item(),
                gate_std=gate.std().item(),
                gate_min=gate.min().item(),
                gate_max=gate.max().item(),
                mac_proj_memory_w_norm=self.mac_projection.weight[:, : self.hidden_size]
                .norm()
                .item(),
                global_avg_memory_contrib_ratio=(
                    (
                        (memory_outputs * gate).norm()
                        / (((1 - gate) * attn_outputs).norm() + 1e-8)
                    ).item()
                ),
                per_token_avg_memory_contrib_ratio=(
                    (
                        (memory_outputs * gate).norm(dim=-1)
                        / (((1 - gate) * attn_outputs).norm(dim=-1) + 1e-8)
                    )
                    .mean()
                    .item()
                ),
            )

        return block_outputs


def inject_memory_modules(
    llama: LlamaForCausalLM,
    memory_modules: nn.ModuleList,
    layer_ids: List[int],
    track_memory_statistics: bool,
    arch: str = "context",
) -> LlamaForCausalLM:
    memory_archs = {
        "layer": LlamaMemoryAsLayer,
        "context": LlamaMemoryAsContext,
    }
    assert (
        arch in memory_archs.keys()
    ), f"Excepted `arch` to be one of {memory_archs.keys()} but got {arch}."
    memory_augmentation_block: nn.Module = memory_archs[arch]

    for layer_id, memory_module in zip(layer_ids, memory_modules):
        llama_layer = llama.model.layers[layer_id]
        llama.model.layers[layer_id] = memory_augmentation_block(
            llama_layer=llama_layer,
            memory_module=memory_module,
            layer_id=layer_id,
            track_memory_statistics=track_memory_statistics,
        )

    return llama
