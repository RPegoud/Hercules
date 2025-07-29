import os
import time
from typing import Dict, Tuple

import bitsandbytes as bnb
import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from hercules import (
    Logger,
    MemoryLlama,
    get_eduweb_dataloader,
    get_specific_split_bl_dataloaders,
)


def _setup(
    cfg: DictConfig,
) -> Tuple[
    MemoryLlama,
    PreTrainedTokenizerBase,
    torch.optim.Optimizer,
    DataLoader,
    DataLoader,
    torch.device,
    Accelerator,
    Logger,
]:

    # --- accelerator setup ---
    accelerator_kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs], log_with="wandb")
    device = accelerator.device
    logger = Logger(accelerator=accelerator)

    # --- config setup ---
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg.memory_llama["hf_token"] = dotenv_values(".env")["HF_TOKEN"]
    steps_per_epoch = int(
        cfg.experiment.num_train_samples // cfg.experiment.eduweb_batch_size
    )
    cfg.experiment["steps_per_epoch"] = steps_per_epoch
    logger.set_experiment_name(cfg, cfg_dict)

    # --- model and tokenizer setup ---
    model = MemoryLlama(neural_memory_config=cfg.neural_memory, **cfg.memory_llama)

    optimizer = bnb.optim.Adam8bit(
        model.neural_memory.gate_parameters,
        lr=cfg.experiment.learning_rate,
        weight_decay=cfg.experiment.weight_decay,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.memory_llama.llama_hf_path)
    tokenizer.pad_token = tokenizer.eos_token

    # --- setup dataset ---
    train_loader = get_eduweb_dataloader(cfg, tokenizer, accelerator)
    test_loaders = get_specific_split_bl_dataloaders(
        cfg,
        tokenizer,
        test_only=True,
        return_prompts_for_generation=cfg.experiment.eval_with_generate,
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # --- log config, model and accelerator state ---
    logger.log_config(cfg_dict)
    logger.log_memory_model(model)
    time.sleep(2)
    logger.log(f"Accelerator state:\n{accelerator.state}", "red", main_process=False)

    print(torch.cuda.memory_summary(device=device))

    return (
        model,
        tokenizer,
        optimizer,
        train_loader,
        test_loaders,
        device,
        accelerator,
        logger,
    )


def _train_one_epoch(
    model: MemoryLlama,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loader: DataLoader,
    cfg: DictConfig,
    accelerator: Accelerator,
):
    model.train()

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{cfg.experiment.epochs}",
        total=cfg.experiment.steps_per_epoch,
        disable=not accelerator.is_main_process,
    )
    for it, batch in enumerate(progress_bar):
        batch = {k: v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

        train_causal_loss = loss.item()

        if accelerator.is_main_process and cfg.experiment.log_experiment:
            accelerator.log(
                {
                    "eduweb_causal_loss": train_causal_loss,
                    "epoch": epoch,
                    "step": it,
                }
            )


def _evaluate(
    model: MemoryLlama,
    tokenizer: PreTrainedTokenizerBase,
    test_loaders: Dict[str, DataLoader],
    cfg: DictConfig,
    accelerator: Accelerator,
    logger: Logger,
):

    if test_loaders:
        model.eval()
        for test_split, test_loader in test_loaders.items():
            num_correct = 0
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
                batch = {
                    k: v if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                test_causal_loss = outputs.loss.item()

                if cfg.experiment.eval_with_generate:
                    unwrapped_model = accelerator.unwrap_model(model)
                    prompt_ids = batch["prompt_input_ids"]
                    prompt_attention_mask = batch["prompt_attention_mask"]

                    generated_ids = unwrapped_model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_attention_mask,
                        max_new_tokens=cfg.experiment.max_gen_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                    )

                    newly_generated_ids = generated_ids[
                        :, -cfg.experiment.max_gen_tokens :
                    ]

                    generated_texts = tokenizer.batch_decode(
                        newly_generated_ids, skip_special_tokens=True
                    )

                    target_texts = batch["target_text"]

                    for gen_text, target_text in zip(generated_texts, target_texts):
                        if target_text in gen_text:
                            num_correct += 1
                    accuracy = num_correct / ((it + 1) * cfg.experiment.batch_size)

                else:
                    accuracy = None

                metrics_to_log = {
                    f"accuracy_{test_split}": accuracy,
                    f"test_causal_loss_{test_split}": test_causal_loss,
                    "step": it,
                }
                accelerator.log(metrics_to_log)


@hydra.main(
    config_path="../config",
    config_name="eduweb_pt.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    (
        model,
        tokenizer,
        optimizer,
        train_loader,
        test_loaders,
        accelerator,
        logger,
    ) = _setup(cfg)

    # --- Training ---
    logger.log("--- Starting training phase ---", "cyan")
    logger.log(
        f"Task: {cfg.experiment.train_task_name}, Split: {cfg.experiment.train_splits}",
        "cyan",
        style="normal",
    )

    for epoch in tqdm(range(cfg.experiment.epochs)):
        _train_one_epoch(model, optimizer, epoch, train_loader, cfg, accelerator)

    # --- Test ---
    logger.log("--- Starting test phase ---", "cyan")
    logger.log(
        f"Task: {cfg.experiment.test_task_name}, Split: {cfg.experiment.test_splits}",
        "cyan",
        style="normal",
    )
    _evaluate(model, tokenizer, test_loaders, cfg, accelerator, logger)

    if cfg.experiment.log_experiment:
        accelerator.end_training()

    if cfg.experiment.save_model:
        m = accelerator.unwrap_model(model)
        save_dir = os.path.join(f"models/{cfg.experiment.name}/", logger.ts)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(m.neural_memory, os.path.join(save_dir, "neural_memory.pt"))
        logger.log(f"Saved Neural Memory Module under: {save_dir}", "green")


if __name__ == "__main__":
    main()
