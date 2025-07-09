import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

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

ALL_BABILONG_SPLITS = [
    "qa1",
    "qa2",
    "qa3",
    "qa4",
    "qa5",
    "qa6",
    "qa7",
    "qa8",
    "qa9",
    "qa10",
]
TASK_TO_MAX_LEN = {  # Babilong task lenghts
    "0k": 0,
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
}


def _get_test_splits(cfg: DictConfig) -> List[str]:
    """Determines which test splits to use based on the config."""
    if cfg.train.test_split == "all":
        return [ts for ts in ALL_BABILONG_SPLITS if ts != cfg.train.train_split]
    elif isinstance(cfg.train.test_split, str):
        return [cfg.train.test_split]
    elif cfg.train.test_split is None:  # if not eval is requested
        return []
    else:
        return list(cfg.train.test_split)


def _get_loaders(
    cfg: DictConfig, tokenizer: AutoTokenizer
) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """ """
    train_collate_fn = BabilongCollator(
        tokenizer, max_length=TASK_TO_MAX_LEN[cfg.train.train_task_name]
    )
    test_collate_fn = BabilongCollator(
        tokenizer, max_length=TASK_TO_MAX_LEN[cfg.train.test_task_name]
    )

    train_ds = load_dataset(
        "RMT-team/babilong-train-5k-samples",
        name=cfg.train.train_task_name,
        split=cfg.train.train_split,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        collate_fn=train_collate_fn,
        num_workers=8,
        shuffle=True,
    )

    test_splits = _get_test_splits(cfg)
    test_loaders = {}
    if test_splits:
        full_test_ds = load_dataset(
            "RMT-team/babilong-train-5k-samples",
            name=cfg.train.test_task_name,
        )
        for split in test_splits:
            if split in full_test_ds:
                test_loaders[split] = DataLoader(
                    full_test_ds[split],
                    batch_size=cfg.train.batch_size,
                    collate_fn=test_collate_fn,
                    num_workers=8,
                )
    return train_loader, test_loaders


@hydra.main(
    config_path="config",
    config_name="pre_training.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    # --- accelerator setup ---
    accelerator_kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs], log_with="wandb")
    device = accelerator.device
    logger = Logger(accelerator=accelerator)

    # --- config setup ---
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg.memory_llama["hf_token"] = dotenv_values(".env")["HF_TOKEN"]

    if cfg.train.log_experiment:
        run_name = cfg.train.run_name if cfg.train.run_name else None
        accelerator.init_trackers(
            project_name="Hercules",
            config=cfg_dict,
            init_kwargs={"wandb": {"name": run_name}},
        )

    # --- model and tokenizer setup ---
    model = MemoryLlama(neural_memory_config=cfg.neural_memory, **cfg.memory_llama)
    model.to(device)

    optimizer = torch.optim.AdamW(
        # model.neural_memory.gate_parameters,
        [
            p
            for n, p in model.neural_memory.named_parameters()
            if not n.startswith("memory_module.") and p.requires_grad
        ],
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.memory_llama.llama_hf_path)
    tokenizer.pad_token = tokenizer.eos_token

    # --- get loaders and prepare ---
    train_loader, test_loaders = _get_loaders(cfg, tokenizer)
    model, optimizer, train_loader, *prepared_test_loaders = accelerator.prepare(
        model, optimizer, train_loader, *test_loaders.values()
    )
    test_loaders = {
        split: loader
        for split, loader in zip(test_loaders.keys(), prepared_test_loaders)
    }

    # --- log config, model and accelerator state ---
    logger.log_config(cfg_dict)
    logger.log_memory_model(model)
    logger.log(f"Accelerator state:\n{accelerator.state}", "red", main_process=False)

    logger.log("Training phase:", "cyan")
    logger.log(
        f"Task: {cfg.train.train_task_name}, Split: {cfg.train.train_split}",
        "cyan",
        style="normal",
    )
    model.train()

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

    if test_loaders:
        model.eval()
        for test_split, test_loader in test_loaders.items():
            test_progress_bar = tqdm(
                test_loader,
                disable=not accelerator.is_main_process,
            )
            logger.log(
                f"Task: {cfg.train.test_task_name}, Split: {test_split}",
                "cyan",
                style="normal",
            )
            for it, batch in enumerate(test_progress_bar):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                test_causal_loss = outputs.loss.item()

                if accelerator.is_main_process and cfg.train.log_experiment:
                    accelerator.log(
                        {
                            f"test_causal_loss_{test_split}": test_causal_loss,
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
