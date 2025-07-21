import os
import time
from typing import Dict, Tuple

import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerBase

from hercules import (
    Logger,
    get_eduweb_dataloader,
    get_specific_split_bl_dataloaders,
)


def _setup(
    cfg: DictConfig,
) -> Tuple[
    LlamaForCausalLM,
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
    model = LlamaForCausalLM.from_pretrained(
        cfg.memory_llama.llama_hf_path,
        token=cfg.memory_llama.hf_token,
    )
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(
        f"Llama Trainable Parameters:\n{n_trainable_params}", "blue", main_process=True
    )
    assert n_trainable_params == 0
    tokenizer = AutoTokenizer.from_pretrained(cfg.memory_llama.llama_hf_path)
    tokenizer.pad_token = tokenizer.eos_token

    # --- setup dataset ---
    eduweb_loader = get_eduweb_dataloader(cfg, tokenizer, accelerator)
    bl_loaders = get_specific_split_bl_dataloaders(
        cfg,
        tokenizer,
        test_only=True,
        return_prompts_and_targets=cfg.experiment.eval_with_generate,
    )

    model, eduweb_loader, *prepared_test_loaders = accelerator.prepare(
        model, eduweb_loader, *bl_loaders.values()
    )
    bl_loaders = {
        split: loader for split, loader in zip(bl_loaders.keys(), prepared_test_loaders)
    }

    # --- log config, model and accelerator state ---
    logger.log_config(cfg_dict)
    logger.log_memory_model(model)
    time.sleep(2)
    logger.log(f"Accelerator state:\n{accelerator.state}", "red", main_process=False)

    return (
        model,
        tokenizer,
        eduweb_loader,
        bl_loaders,
        device,
        accelerator,
        logger,
    )


def _evaluate(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    eduweb_loader: DataLoader,
    bl_loaders: Dict[str, DataLoader],
    cfg: DictConfig,
    accelerator: Accelerator,
    device: torch.device,
    logger: Logger,
):
    model.eval()

    for test_split, test_loader in bl_loaders.items():
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
        it = 0

        for batch in test_progress_bar:
            it += 1
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
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

                newly_generated_ids = generated_ids[:, -cfg.experiment.max_gen_tokens :]

                generated_texts = tokenizer.batch_decode(
                    newly_generated_ids, skip_special_tokens=True
                )

                target_texts = batch["target_text"]

                for gen_text, target_text in zip(generated_texts, target_texts):
                    if target_text in gen_text:
                        num_correct += 1
                accuracy = num_correct / ((it + 1) * cfg.experiment.batch_size)
                metrics_to_log = {
                    f"accuracy_{test_split}": accuracy,
                    f"test_causal_loss_{test_split}": test_causal_loss,
                    "step": it,
                }
                accelerator.log(metrics_to_log)

    test_progress_bar = tqdm(
        eduweb_loader,
        disable=not accelerator.is_main_process,
    )
    logger.log("Task: EduWeb", "cyan", style="normal")
    it = 0

    for batch in test_progress_bar:
        it += 1
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        test_causal_loss = outputs.loss.item()
        if accelerator.is_main_process and cfg.experiment.log_experiment:
            accelerator.log({"test_causal_loss_eduweb": test_causal_loss, "it": it})


@hydra.main(
    config_path="../config",
    config_name="baseline.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    (
        model,
        tokenizer,
        eduweb_loader,
        bl_loaders,
        device,
        accelerator,
        logger,
    ) = _setup(cfg)

    # --- Test ---
    logger.log("--- Starting test phase ---", "cyan")
    logger.log(
        f"Task: {cfg.experiment.test_task_name}, Split: {cfg.experiment.test_splits}",
        "cyan",
        style="normal",
    )
    _evaluate(
        model, tokenizer, eduweb_loader, bl_loaders, cfg, accelerator, device, logger
    )

    if cfg.experiment.log_experiment:
        accelerator.end_training()


if __name__ == "__main__":
    main()
