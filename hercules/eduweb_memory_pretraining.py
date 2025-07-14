import os
from itertools import chain

import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, default_data_collator

from hercules import (
    Logger,
    MemoryLlama,
)


@hydra.main(
    config_path="config",
    config_name="eduweb_pt.yaml",
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
    logger.set_experiment_name(cfg, cfg_dict)

    # --- model and tokenizer setup ---
    model = MemoryLlama(neural_memory_config=cfg.neural_memory, **cfg.memory_llama)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.neural_memory.gate_parameters,
        lr=cfg.experiment.learning_rate,
        weight_decay=cfg.experiment.weight_decay,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.memory_llama.llama_hf_path)
    tokenizer.pad_token = tokenizer.eos_token

    # --- setup dataset ---
    def tokenize_and_chunk(samples: dict, tokenizer: PreTrainedTokenizerBase):
        tokenized_inputs = tokenizer(samples["text"], truncation=False)
        concatenated_samples = {
            k: list(chain(*tokenized_inputs[k])) for k in tokenized_inputs.keys()
        }
        total_len = len(concatenated_samples[list(tokenized_inputs.keys())[0]])

        if total_len > cfg.experiment.max_seq_len:
            total_len = (
                total_len // cfg.experiment.max_seq_len
            ) * cfg.experiment.max_seq_len

        outputs = {
            k: [
                t[i : i + cfg.experiment.max_seq_len]
                for i in range(0, total_len, cfg.experiment.max_seq_len)
            ]
            for k, t in concatenated_samples.items()
        }

        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    raw_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )

    with accelerator.main_process_first():
        processed_ds = raw_ds.map(
            tokenize_and_chunk,
            batched=True,
            remove_colums=["text", "id", "dump", "url"],
        ).take(
            cfg.experiment.num_samples  # TODO: find the right setting
        )

        ds = processed_ds = processed_ds.train_test_split(
            test_size=cfg.experiment.test_size, seed=cfg.experiment.seed
        )

    train_loader = DataLoader(
        ds["train"],
        collate_fn=default_data_collator,
        batch_size=cfg.experiment.batch_size,
        num_workers=8,
        shuffle=True,
    )
    test_loader = DataLoader(
        ds["test"],
        collate_fn=default_data_collator,
        batch_size=cfg.experiment.batch_size,
        num_workers=8,
    )

    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    # --- log config, model and accelerator state ---
    logger.log_config(cfg_dict)
    logger.log_memory_model(model)
    logger.log(f"Accelerator state:\n{accelerator.state}", "red", main_process=False)

    logger.log("Training phase:", "cyan")
    logger.log(
        f"Task: {cfg.experiment.train_task_name}, Split: {cfg.experiment.train_splits}",
        "cyan",
        style="normal",
    )
    model.train()

    for epoch in tqdm(range(cfg.experiment.epochs)):
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.experiment.epochs}",
            disable=not accelerator.is_main_process,
        )

        for it, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            train_causal_loss = loss.item()

            if accelerator.is_main_process and cfg.experiment.log_experiment:
                accelerator.log(
                    {
                        "train_causal_loss": train_causal_loss,
                        "epoch": epoch,
                        "step": it,
                    }
                )

    logger.log("Test phase:", "cyan")
    logger.log(
        f"Task: {cfg.experiment.test_task_name}, Split: {cfg.experiment.test_splits}",
        "cyan",
        style="normal",
    )

    if test_loader:
        model.eval()
        test_progress_bar = tqdm(
            test_loader,
            disable=not accelerator.is_main_process,
        )
        for it, batch in enumerate(test_progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            test_causal_loss = outputs.loss.item()

            if accelerator.is_main_process and cfg.experiment.log_experiment:
                accelerator.log(
                    {
                        "test_causal_loss": test_causal_loss,
                        "epoch": epoch,
                        "step": it,
                    }
                )

    if cfg.experiment.log_experiment:
        accelerator.end_training()

    if cfg.experiment.save_model:
        m = accelerator.unwrap_model(model)
        save_dir = os.path.join("models/eduweb_ft/", logger.ts)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(m.neural_memory, os.path.join(save_dir, "neural_memory.pt"))
        logger.log(f"Saved Neural Memory Module under: {save_dir}", "green")


if __name__ == "__main__":
    main()
