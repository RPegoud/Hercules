import hydra
import torch
import torch.nn as nn
from accelerate import Accelerator
from colorama import Fore, Style
from datasets import load_dataset
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

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
    cfg.memory_llama["token"] = dotenv_values(".env")["HF_TOKEN"]

    accelerator = Accelerator()
    device = accelerator.device
    print(device)

    model = MemoryLlama(neural_memory_config=cfg.lmm, **cfg.memory_llama)
    tokenizer = AutoTokenizer.from_pretrained(cfg.memory_llama.llama_hf_path)
    model.to(device)
    print(f"{Style.BRIGHT}{Fore.RED}Using device: {device}")

    print(torch.cuda.is_available())

    train_ds = load_dataset("RMT-team/babilong-train-5k-samples", "2k", split="qa1")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size)
    total_loss = 0

    model, train_loader = accelerator.prepare(model, train_loader)

    for epoch in tqdm(range(cfg.train.epochs)):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")

        for batch in train_loader:
            # batch = tokenizer(batch)
            inputs = batch["input"]
            questions = batch["question"]
            targets = batch["target"]

            prompts = [
                f"{i}, Question: {q}, Answer:" for i, q in zip(inputs, questions)
            ]

            batch_inputs = {"input_ids": batch, "labels": batch}

            outputs = model(**batch_inputs)
            loss = outputs.loss

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            break


if __name__ == "__main__":
    main()
