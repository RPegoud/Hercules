import hydra
import torch
import torch.nn as nn
from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf
from transformers import LlamaConfig, LlamaForCausalLM

from hercules import NeuralMemory, inject_memory_module, log_config


@hydra.main(
    config_path="config",
    config_name="pre_training.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    log_config(cfg_dict)
    print(cfg_dict.keys)

    device = torch.device("mps")

    llama_config = LlamaConfig(**cfg.proxy_llama)

    llama = LlamaForCausalLM(llama_config)
    memory_module = NeuralMemory(**cfg.lmm)

    model = inject_memory_module(llama, memory_module)
    model.lm_head = nn.Linear(cfg.proxy_llama.layer_size, cfg.proxy_llama.vocab_size)


if __name__ == "__main__":
    main()
