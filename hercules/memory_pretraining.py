import os

import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from hercules import (
    Logger,
    MemoryLlama,
    get_global_split_dataloaders,
    get_specific_split_dataloaders,
)


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

    # --- get loaders and prepare ---
    if cfg.experiment.use_global_split:
        train_loader, test_loaders = get_global_split_dataloaders(cfg, tokenizer)
    else:
        train_loader, test_loaders = get_specific_split_dataloaders(cfg, tokenizer)

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
            outputs = model(**batch)
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

    if test_loaders:
        model.eval()
        for test_split, test_loader in test_loaders.items():
            test_progress_bar = tqdm(
                test_loader,
                disable=not accelerator.is_main_process,
            )
            logger.log(
                f"Task: {cfg.experiment.test_task_name}, Split: {test_split}",
                "cyan",
                style="normal",
            )
            for it, batch in enumerate(test_progress_bar):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                test_causal_loss = outputs.loss.item()

                if accelerator.is_main_process and cfg.experiment.log_experiment:
                    accelerator.log(
                        {
                            f"test_causal_loss_{test_split}": test_causal_loss,
                            "epoch": epoch,
                            "step": it,
                        }
                    )

    if cfg.experiment.log_experiment:
        accelerator.end_training()

    if cfg.experiment.save_model:
        m = accelerator.unwrap_model(model)
        save_dir = os.path.join("models", logger.ts)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(m.neural_memory, os.path.join(save_dir, "neural_memory.pt"))
        logger.log(f"Saved Neural Memory Module under: {save_dir}", "green")


if __name__ == "__main__":
    main()
