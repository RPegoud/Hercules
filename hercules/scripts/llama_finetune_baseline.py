import os
import time
from typing import Dict, Tuple
import bitsandbytes as bnb
import hydra
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerBase
from dataclasses import dataclass
import json
from peft import LoraConfig, get_peft_model, TaskType


from hercules import (
    Logger,
    MemoryLlama,
    get_eduweb_dataloader,
    get_specific_split_bl_dataloaders,
)


@dataclass
class TrainState:
    model: MemoryLlama
    optimizer: torch.optim.Optimizer
    scheduler: LRScheduler
    epoch: int
    global_step: int
    lowest_val_loss: float | None = None


def setup(
    cfg: DictConfig,
) -> Tuple[
    TrainState,
    PreTrainedTokenizerBase,
    DataLoader,
    DataLoader,
    DataLoader,
    Accelerator,
    Logger,
]:
    torch.manual_seed(cfg.experiment.seed)

    # --- wandb variables ---
    env_vars = dotenv_values(".env")
    os.environ["WANDB_API_KEY"] = env_vars["WANDB_TOKEN"]
    os.environ["WANDB_ENTITY"] = env_vars["WANDB_ENTITY"]
    os.environ["WANDB_PROJECT"] = env_vars["WANDB_PROJECT"]

    # --- accelerator setup ---
    accelerator_kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(
        kwargs_handlers=[accelerator_kwargs],
        log_with="wandb",
        project_config=ProjectConfiguration(project_dir=f"checkpoints/llama_finetune/"),
    )
    logger = Logger(accelerator=accelerator)

    # --- config setup ---
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg.llama["hf_token"] = env_vars["HF_TOKEN"]
    steps_per_epoch = int(
        cfg.experiment.num_train_samples // cfg.experiment.eduweb_train_batch_size
    )

    cfg.experiment["steps_per_epoch"] = steps_per_epoch
    logger.set_experiment_name(cfg, cfg_dict)
    cfg.experiment["ckpt_dir"] = os.path.join(
        accelerator.project_dir, logger.ts, "best_loss"
    )

    # --- model, optimizer, scheduler and tokenizer setup ---
    model = LlamaForCausalLM.from_pretrained(
        cfg.llama.llama_hf_path,
        token=cfg.llama.hf_token,
    )
    lora_config = LoraConfig(
        r=cfg.lora.lora_rank,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=list(cfg.lora.lora_target_modules),
        lora_dropout=cfg.lora.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    logger.log("Applying LoRA config ...", "blue")
    model = get_peft_model(model, lora_config)

    optimizer = bnb.optim.Adam8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.experiment.learning_rate,
        weight_decay=cfg.experiment.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * cfg.experiment.epochs
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.llama.llama_hf_path)
    tokenizer.pad_token = tokenizer.eos_token

    # --- dataset setup ---
    train_loader, val_loader = get_eduweb_dataloader(cfg, tokenizer, accelerator)
    test_loaders = get_specific_split_bl_dataloaders(
        cfg,
        tokenizer,
        test_only=True,
        return_prompts_for_generation=cfg.experiment.eval_with_generate,
    )

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    # --- log config, model and accelerator state ---
    logger.log_config(cfg_dict)
    logger.log("Llama Finetune", "blue")
    model.print_trainable_parameters()

    time.sleep(2)
    logger.log(f"Accelerator state:\n{accelerator.state}", "red", main_process=False)

    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=0,
        lowest_val_loss=None,
    )

    return (
        train_state,
        tokenizer,
        train_loader,
        val_loader,
        test_loaders,
        accelerator,
        logger,
    )


def train_val_one_epoch(
    state: TrainState,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    accelerator: Accelerator,
    logger: Logger,
) -> TrainState:

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {state.epoch+1}/{cfg.experiment.epochs}",
        total=cfg.experiment.steps_per_epoch,
        disable=not accelerator.is_main_process,
    )
    for it, batch in enumerate(progress_bar):
        state.model.train()
        state.global_step = it * state.epoch
        batch = {k: v for k, v in batch.items()}
        outputs = state.model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        state.optimizer.step()
        state.optimizer.zero_grad()
        state.scheduler.step()

        train_causal_loss = loss.item()

        # log training loss and learning rate
        if accelerator.is_main_process and cfg.experiment.log_experiment:
            accelerator.log(
                {
                    "eduweb_causal_loss": train_causal_loss,
                    "learning_rate": state.scheduler.get_lr()[0],
                    "epoch": state.epoch,
                    "step": state.global_step,
                }
            )

        # best validation loss checkpoints
        if (it + 1) % (
            cfg.experiment.num_train_samples // cfg.experiment.num_eval
        ) == 0:
            logger.log("--- Starting validation phase ---", "yellow")
            avg_val_loss = validate(state, val_loader, cfg, accelerator)
            logger.log(f"Average validation loss: {avg_val_loss:.3f}", "yellow")

            if state.lowest_val_loss is None or avg_val_loss < state.lowest_val_loss:
                state.lowest_val_loss = avg_val_loss
                state.model.save_pretrained(
                    cfg.experiment.ckpt_dir, safe_serialization=True
                )
                with open(
                    os.path.join(cfg.experiment.ckpt_dir, "loss_value.txt"), "w"
                ) as f:
                    f.write(f"Best loss: {state.lowest_val_loss}")

    return state


def validate(
    state: TrainState,
    val_loader: DataLoader,
    cfg: DictConfig,
    accelerator: Accelerator,
) -> float:
    state.model.eval()
    val_causal_loss = 0

    progress_bar = tqdm(
        val_loader,
        total=cfg.experiment.num_val_samples // cfg.experiment.eduweb_val_batch_size,
        disable=not accelerator.is_main_process,
    )
    for it, batch in enumerate(progress_bar):
        batch = {k: v for k, v in batch.items()}
        outputs = state.model(**batch)
        loss = outputs.loss
        val_causal_loss += loss.item()

    avg_val_causal_loss = val_causal_loss / (it + 1)
    if accelerator.is_main_process and cfg.experiment.log_experiment:
        accelerator.log(
            {
                "eduweb_val_causal_loss": avg_val_causal_loss,
                "epoch": state.epoch,
                "step": state.global_step,
            }
        )

    return avg_val_causal_loss


def evaluate(
    model: MemoryLlama,
    tokenizer: PreTrainedTokenizerBase,
    test_loaders: Dict[str, DataLoader],
    cfg: DictConfig,
    accelerator: Accelerator,
    logger: Logger,
):
    model.eval()
    accuracies = {}

    for test_split, test_loader in test_loaders.items():
        accuracies[test_split] = 0

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
                k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
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
                newly_generated_ids, skip_special_tokens=False
            )

            target_texts = batch["target_text"]

            for gen_text, target_text in zip(generated_texts, target_texts):
                if target_text in gen_text:
                    accuracies[test_split] += 1

    for split in accuracies.keys():
        accuracies[split] /= (it + 1) * cfg.experiment.babilong_test_batch_size
        logger.log(f"Split {test_split} accuracy: {accuracies[split]}", "yellow")

    os.makedirs("results/llama_finetune_baseline", exist_ok=True)
    with open(
        os.path.join("results/llama_finetune_baseline", f"{logger.ts}.json"), "w"
    ) as f:
        json.dump(accuracies, f, indent=4)

    accelerator.log({f"test/": {k: float(v) for k, v in accuracies.items()}})


@hydra.main(
    config_path="../config",
    config_name="finetune_baseline.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    (
        state,
        tokenizer,
        train_loader,
        val_loader,
        test_loaders,
        accelerator,
        logger,
    ) = setup(cfg)

    # --- Training ---
    logger.log("--- Starting training phase ---", "cyan")
    logger.log(
        f"Task: {cfg.experiment.train_task_name}, Split: {cfg.experiment.train_splits}",
        "cyan",
        style="normal",
    )

    for epoch in tqdm(range(cfg.experiment.epochs)):
        state.epoch = epoch
        state = train_val_one_epoch(
            state,
            train_loader,
            val_loader,
            cfg,
            accelerator,
            logger,
        )

    train_loader, val_loader, state.optimizer = accelerator.free_memory(
        train_loader, val_loader, state.optimizer
    )

    # --- Test ---
    logger.log("--- Starting test phase ---", "cyan")
    logger.log(
        f"Task: {cfg.experiment.test_task_name}, Split: {cfg.experiment.test_splits}",
        "cyan",
        style="normal",
    )
    evaluate(state.model, tokenizer, test_loaders, cfg, accelerator, logger)

    if cfg.experiment.log_experiment:
        accelerator.end_training()


if __name__ == "__main__":
    main()
