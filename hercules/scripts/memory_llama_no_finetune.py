import os
import time
from typing import Dict, Tuple
import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerBase
import json
from hercules import (
    MemoryLlama,
    Logger,
    get_specific_split_bl_dataloaders,
)


def setup(
    cfg: DictConfig,
) -> Tuple[
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
        project_config=ProjectConfiguration(
            project_dir="results/memory_llama_no_finetune"
        ),
    )
    logger = Logger(accelerator=accelerator)

    # --- config setup ---
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg.memory_llama["hf_token"] = env_vars["HF_TOKEN"]

    logger.set_experiment_name(cfg, cfg_dict)

    # --- model and tokenizer setup ---
    model = MemoryLlama(
        neural_memory_config=cfg.neural_memory, **cfg.memory_llama, **cfg.lora
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.memory_llama.llama_hf_path)
    tokenizer.pad_token = tokenizer.eos_token

    # --- dataset setup ---
    test_loaders = get_specific_split_bl_dataloaders(
        cfg,
        tokenizer,
        test_only=True,
        return_prompts_for_generation=cfg.experiment.eval_with_generate,
    )

    model = accelerator.prepare(model)

    # --- log config, model and accelerator state ---
    logger.log_config(cfg_dict)
    time.sleep(2)
    logger.log(f"Accelerator state:\n{accelerator.state}", "red", main_process=False)

    return (model, tokenizer, test_loaders, accelerator, logger)


def evaluate(
    model: LlamaForCausalLM,
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

    os.makedirs("results/memory_llama_no_finetune", exist_ok=True)
    with open(
        os.path.join("results/memory_llama_no_finetune", f"{logger.ts}.json"), "w"
    ) as f:
        json.dump(accuracies, f, indent=4)

    accelerator.log({f"test/": {k: float(v) for k, v in accuracies.items()}})


@hydra.main(
    config_path="../config",
    config_name="memory_llama_no_finetune.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    model, tokenizer, test_loaders, accelerator, logger = setup(cfg)

    logger.log("--- Starting test phase ---", "cyan")
    evaluate(model, tokenizer, test_loaders, cfg, accelerator, logger)

    if cfg.experiment.log_experiment:
        accelerator.end_training()


if __name__ == "__main__":
    main()
