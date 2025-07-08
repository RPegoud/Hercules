import os
from dataclasses import dataclass
from datetime import datetime

import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from hercules import BabilongCollator, Logger, MemoryLlama

TASK_TO_MAX_LEN = {  # Babilong task lenghts
    "0k": 0,
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
}


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

    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_config(cfg_dict)
    cfg.memory_llama["hf_token"] = dotenv_values(".env")["HF_TOKEN"]
    MAX_SEQ_LEN_TRAIN = TASK_TO_MAX_LEN[cfg.train.train_task_name]
    MAX_SEQ_LEN_TEST = TASK_TO_MAX_LEN[cfg.train.test_task_name]

    model = MemoryLlama(neural_memory_config=cfg.neural_memory, **cfg.memory_llama)

    # The forget gate, adaptive learning rate and momentum are trained with backward() at train time
    # The memory module in itself (resNet) is trained at inference time using a custom logic that's
    # not compatible with ``backward()``
    # TODO: move to nm class
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

    collate_fn_train = BabilongCollator(tokenizer, max_length=MAX_SEQ_LEN_TRAIN)
    collate_fn_test = BabilongCollator(tokenizer, max_length=MAX_SEQ_LEN_TEST)
    train_ds = load_dataset(
        "RMT-team/babilong-train-5k-samples",
        name=cfg.train.train_task_name,
        split=cfg.train.train_split,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        collate_fn=collate_fn_train,
        num_workers=8,
    )

    test_ds = load_dataset(
        "RMT-team/babilong-train-5k-samples",
        name=cfg.train.test_task_name,
        split=cfg.train.test_split,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        collate_fn=collate_fn_test,
        num_workers=8,
    )

    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    if cfg.train.log_experiment:
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
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            train_causal_loss = loss.item()

            if accelerator.is_main_process and cfg.train.log_experiment:
                accelerator.log(
                    {
                        "train_causal_loss": train_causal_loss,
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
        test_causal_loss = outputs.loss.item()

        if accelerator.is_main_process and cfg.train.log_experiment:
            accelerator.log(
                {
                    "test_causal_loss": test_causal_loss,
                    "epoch": epoch,
                    "step": it,
                }
            )

    if cfg.train.log_experiment:
        accelerator.end_training()

    if cfg.train.save_model:
        m = accelerator.unwrap_model(model)
        ts = datetime.now().strftime("%m-%d_%H-%M")
        save_dir = os.path.join("models", ts)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(m.neural_memory, os.path.join(save_dir, "neural_memory.pt"))
        logger.log(f"Saved Neural Memory Module under: {save_dir}", "green")


if __name__ == "__main__":
    main()
