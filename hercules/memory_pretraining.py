import os
from dataclasses import dataclass
from datetime import datetime

import hydra
import torch
import tyro
from accelerate import Accelerator, DistributedDataParallelKwargs
from colorama import Fore, Style
from datasets import load_dataset
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from hercules import BabilongCollator, Logger, MemoryLlama

TASK_TO_MAX_LEN = {
    "0k": 0,
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
}


@dataclass
class RuntimeArgs:
    track: bool = False
    """Logs the experiment with Weights and Biases."""
    save: bool = False
    """Saves the memory module at the end of training."""


@hydra.main(
    config_path="config",
    config_name="pre_training.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    accelerator_kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs], log_with="wandb")
    device = accelerator.device

    logger = Logger(accelerator=accelerator)
    rt = tyro.cli(RuntimeArgs)

    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_config(cfg_dict)
    cfg.memory_llama["hf_token"] = dotenv_values(".env")["HF_TOKEN"]
    MAX_SEQ_LEN = TASK_TO_MAX_LEN[cfg.train.train_task_name]

    model = MemoryLlama(neural_memory_config=cfg.neural_memory, **cfg.memory_llama)

    # The forget gate, adaptive learning rate and momentum are trained with backward() at train time
    # The memory module in itself (resNet) is trained at inference time using a custom logic that's
    # not compatible with ``backward()``
    gate_params = [
        p
        for n, p in model.neural_memory.named_parameters()
        if not n.startswith("memory_module.") and p.requires_grad
    ]
    logger.log_memory_model(model)
    model.to(device)

    optimizer = torch.optim.AdamW(
        gate_params,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.memory_llama.llama_hf_path)
    tokenizer.pad_token = tokenizer.eos_token

    logger.log(f"Accelerator state:\n{accelerator.state}", "red", main_process=False)

    collate_fn = BabilongCollator(tokenizer, max_length=MAX_SEQ_LEN)
    train_ds = load_dataset(
        "RMT-team/babilong-train-5k-samples",
        name=cfg.train.train_task_name,
        split=cfg.train.train_split,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, collate_fn=collate_fn, num_workers=8
    )

    test_ds = load_dataset(
        "RMT-team/babilong-train-5k-samples",
        name=cfg.train.test_task_name,
        split=cfg.train.test_split,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, collate_fn=collate_fn, num_workers=8
    )

    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    if rt.log_experiment:
        accelerator.init_trackers(project_name="Hercules", config=cfg_dict)

    logger.log("Training phase:", "cyan")
    logger.log(
        f"Task: {cfg.train.train_task_name}, Split: {cfg.train.train_split}",
        "cyan",
        style="normal",
    )

    for epoch in tqdm(range(cfg.train.epochs)):
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.train.epochs}",
            disable=not accelerator.is_main_process,
        )

        for it, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            unwrapped_model = accelerator.unwrap_model(model)
            memory_loss = unwrapped_model.neural_memory.last_associative_loss
            print(f"loss: {memory_loss}=")
            gathered_memory_loss = accelerator.gather_for_metrics(memory_loss)
            print(f"gathered loss: {gathered_memory_loss}")
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            causal_loss = loss.item()

            if accelerator.is_main_process and rt.log_experiment:
                accelerator.log(
                    {
                        "train_causal_loss": causal_loss,
                        "train_associative_loss": gathered_memory_loss.mean().item(),
                        "epoch": epoch,
                        "step": it,
                    }
                )

    logger.log("Test phase:", "cyan")
    logger.log(
        f"Task: {cfg.train.test_task_name}, Split: {cfg.train.test_split}",
        "cyan",
        style="normal",
    )

    test_progress_bar = tqdm(
        test_loader,
        disable=not accelerator.is_main_process,
    )

    for it, batch in enumerate(test_progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        unwrapped_model = accelerator.unwrap_model(model)
        memory_loss = unwrapped_model.neural_memory.last_associative_loss
        gathered_memory_loss = accelerator.gather_for_metrics(memory_loss)

        causal_loss = outputs.loss.item()

        if accelerator.is_main_process and rt.log_experiment:
            accelerator.log(
                {
                    "test_causal_loss": causal_loss,
                    "test_associative_loss": gathered_memory_loss.mean().item(),
                    "epoch": epoch,
                    "step": it,
                }
            )

    if rt.log_experiment:
        accelerator.end_training()

    if rt.save_model:
        m = accelerator.unwrap_model(model)
        ts = datetime.now().strftime("%m-%d_%H-%M")
        save_dir = os.path.join("models", ts)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(m.neural_memory, os.path.join(save_dir, "neural_memory.pt"))
        logger.log(f"Saved Neural Memory Module under: {save_dir}", "green")


if __name__ == "__main__":
    main()
