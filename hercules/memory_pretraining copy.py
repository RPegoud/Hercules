import datasets
import hydra
import torch
import torch.nn as nn
from accelerate import Accelerator
from colorama import Fore, Style
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from hercules import MemoryLlama, log_config
from source.babilong.babilong_utils import (
    NoiseInjectionDataset,
    SentenceSampler,
    TaskDataset,
)


@hydra.main(
    config_path="config",
    config_name="pre_training.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    log_config(cfg_dict)
    cfg.memory_llama["token"] = dotenv_values(".env")["HF_TOKEN"]

    accelerator = Accelerator()
    device = accelerator.device
    print(device)

    model = MemoryLlama(neural_memory_config=cfg.lmm, **cfg.memory_llama)
    tokenizer = AutoTokenizer.from_pretrained(cfg.memory_llama.llama_hf_path)
    model.to(device)
    print(f"{Style.BRIGHT}{Fore.RED}Using device: {device}")

    print(torch.cuda.is_available())

    train_path = "data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt"
    test_path = "data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt"

    noise_dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")

    task_dataset_train = TaskDataset(train_path)
    task_dataset_test = TaskDataset(test_path)
    noise_sampler_train = SentenceSampler(noise_dataset["train"], tokenizer=tokenizer)
    noise_sampler_test = SentenceSampler(noise_dataset["test"], tokenizer=tokenizer)

    train_dataset = NoiseInjectionDataset(
        task_dataset=task_dataset_train,
        noise_sampler=noise_sampler_train,
        tokenizer=tokenizer,
        sample_size=cfg.train.sample_size,
    )

    test_dataset = NoiseInjectionDataset(
        task_dataset=task_dataset_test,
        noise_sampler=noise_sampler_test,
        tokenizer=tokenizer,
        sample_size=cfg.train.sample_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size)
    model, train_loader = accelerator.prepare(model, train_loader)
    for batch in train_loader:
        print(batch.shape)
        break


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
