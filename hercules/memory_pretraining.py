import hydra
import torch
import torch.nn as nn
from accelerate import Accelerator
from colorama import Fore, Style
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from hercules import MemoryLlama, log_config


@hydra.main(
    config_path="config",
    config_name="pre_training.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    log_config(cfg_dict)

    accelerator = Accelerator()
    device = accelerator.device

    cfg.memory_llama["token"] = dotenv_values(".env")["HF_TOKEN"]

    model = MemoryLlama(neural_memory_config=cfg.lmm, **cfg.memory_llama)
    model.to(device)

    print(torch.cuda.is_available())


if __name__ == "__main__":
    main()


# train_dataloader = ...
# model, train_dataloader = accelerator.prepare(model, train_dataloader)

# for batch in train_dataloader:
#     outputs = model(**batch)
#     loss = ...
#     accelerator.backward(loss)
#     optimizer.step()
#     optimizer.zero_grad()
