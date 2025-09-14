import os
import time
from typing import Dict, Tuple
import bitsandbytes as bnb
import hydra
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, ConstantLR
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from dataclasses import dataclass
import json
from hercules import (
    Logger,
    MemoryLlama,
    get_eduweb_dataloader,
    get_specific_split_bl_dataloaders,
)
from torch.optim import Optimizer
from colorama import Style, Fore


def get_scheduler(optimizer: Optimizer, cfg: DictConfig) -> LRScheduler:
    if cfg.experiment.scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=cfg.experiment.steps_per_epoch * cfg.experiment.epochs
        )
    else:
        return ConstantLR(optimizer, factor=1, total_iters=0)


@dataclass
class TrainState:
    model: MemoryLlama
    optimizer: torch.optim.Optimizer
    scheduler: LRScheduler
    epoch: int
    global_step: int
    lowest_val_loss: float | None = None
    recent_ckpt_idx: int = 0


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
        gradient_accumulation_steps=cfg.experiment.gradient_accumulation_steps,
        project_config=ProjectConfiguration(
            project_dir=f"checkpoints/memory_llama",
            automatic_checkpoint_naming=True,
        ),
    )
    logger = Logger(accelerator=accelerator)

    # --- config setup ---
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg.memory_llama["hf_token"] = env_vars["HF_TOKEN"]
    steps_per_epoch = int(
        cfg.experiment.num_train_samples // cfg.experiment.eduweb_train_batch_size
    )

    cfg.experiment["steps_per_epoch"] = steps_per_epoch
    logger.set_experiment_name(cfg, cfg_dict)
    cfg.experiment["ckpt_dir"] = os.path.join(
        accelerator.project_dir, logger.ts, "best_loss"
    )
    if not cfg.experiment.resume_from_step:
        cfg.experiment.resume_from_step = 0
    else:
        logger.log(
            f"Resuming training from step: {cfg.experiment.resume_from_step}", "yellow"
        )

    # --- model and tokenizer setup ---
    if cfg.experiment.resume_from_checkpoint:
        logger.log(
            f"Loading checkpoint state from {cfg.experiment.checkpoint_name}", "blue"
        )
        model = MemoryLlama.load(
            path=cfg.experiment.checkpoint_name,
            memory_llama_config=cfg.memory_llama,
        )
        # TODO: modify to enable load and train (currently only load and test)
        optimizer, scheduler = None, None

    else:
        model = MemoryLlama(
            neural_memory_config=cfg.neural_memory, **cfg.memory_llama, **cfg.lora
        )

        optimizer = bnb.optim.Adam8bit(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.experiment.learning_rate,
            weight_decay=cfg.experiment.weight_decay,
        )
        scheduler = get_scheduler(optimizer, cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.memory_llama.llama_hf_path)
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
    logger.log_memory_model(model)
    time.sleep(2)
    logger.log(f"Accelerator state:\n{accelerator.state}", "red", main_process=False)

    logger.log(
        f"Checkpoint directory: {cfg.experiment.ckpt_dir}",
        "magenta",
    )

    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=0,
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
        # manually skip samples until the resume step
        if it < cfg.experiment.resume_from_step:
            pass

        else:
            with accelerator.accumulate(state.model):
                state.model.train()
                state.optimizer.zero_grad()
                state.global_step = (it + 1) * (state.epoch + 1)

                batch = {k: v for k, v in batch.items()}
                outputs = state.model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                state.optimizer.step()

            train_causal_loss = loss.item()

            logger.log_metrics(
                model=state.model,
                train_causal_loss=train_causal_loss,
                state=state,
                cfg=cfg,
            )

            # regular ceckpoint
            logger.save_regular_checkpoint(
                train_causal_loss=train_causal_loss, state=state, cfg=cfg
            )

            # best validation loss checkpoints
            if (state.global_step) % (
                cfg.experiment.num_train_samples // cfg.experiment.num_eval
            ) == 0:
                logger.log("--- Starting validation phase ---", "yellow")
                avg_val_loss = validate(state, val_loader, cfg, accelerator)
                logger.log_val_loss_and_save(
                    avg_val_loss=avg_val_loss, state=state, cfg=cfg
                )

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


@torch.no_grad()
def generate_long_context(
    model: MemoryLlama,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    cfg: DictConfig,
):
    """
    Handles generation for prompts that are longer than the model's context window.
    """

    def _generate(inputs, attn_mask):
        return model.generate(
            input_ids=inputs,
            attention_mask=attn_mask,
            max_new_tokens=cfg.experiment.max_gen_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    context_window = model.config.hidden_size
    seq_len = input_ids.shape[1]

    # If the prompt fits in the context window, use the standard generate method
    if seq_len <= context_window:
        return _generate(inputs=input_ids, attn_mask=attention_mask)

    id_chunks = torch.split(input_ids, context_window, dim=1)
    label_chunks = torch.split(attention_mask, context_window, dim=1)
    attn_mask_chunks = torch.split(labels, context_window, dim=1)

    # process all chunks except the last one
    for i in range(len(id_chunks) - 1):
        model(
            input_ids=id_chunks[i],
            attention_mask=attn_mask_chunks[i],
            labels=label_chunks[i],
        )

    # only call generate on the last chunk
    return _generate(inputs=id_chunks[-1], attn_mask=attn_mask_chunks[-1])


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
    memory_stats = []

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
            unwrapped_model.reset_memory()

            generated_ids = generate_long_context(
                unwrapped_model,
                tokenizer,
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                labels=batch["labels"],
                cfg=cfg,
            )

            logs = logger.log_memory_metrics(
                model=model,
                logs={"step": it + 1},
            )
            accelerator.log(logs)

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
        logger.log(f"Split {split} accuracy: {accuracies[split]}", "yellow")

    os.makedirs(f"results/memory_llama_{cfg.memory_llama.memory_arch}", exist_ok=True)
    with open(
        os.path.join(
            f"results/memory_llama_{cfg.memory_llama.memory_arch}", f"{logger.ts}.json"
        ),
        "w",
    ) as f:
        json.dump(accuracies, f, indent=4)

    accelerator.log({f"test/": {k: float(v) for k, v in accuracies.items()}})


@hydra.main(
    config_path="../config",
    config_name="memory_llama.yaml",
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

    load_and_resume = (
        cfg.experiment.resume_from_checkpoint and cfg.experiment.resume_from_step
    )
    if load_and_resume or not cfg.experiment.resume_from_checkpoint:
        # --- Training ---
        if not cfg.experiment.resume_from_step:
            logger.log("--- Starting training phase ---", "cyan")
        else:
            logger.log(
                f"--- Resuming training from step: {cfg.experiment.resume_from_step} ---",
                "cyan",
            )

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
    test_loaders = accelerator.prepare(test_loaders)

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
