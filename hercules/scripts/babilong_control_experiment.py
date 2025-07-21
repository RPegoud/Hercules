import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from colorama import Fore, Style
from datasets import load_dataset
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from hercules import BabilongCollator, Logger

TASK_TO_MAX_LEN = {
    "0k": 0,
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
}


@hydra.main(
    config_path="../config",
    config_name="pre_training.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    # ignores the frozen parameters (i.e. llama params)
    accelerator_kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs], log_with="wandb")
    device = accelerator.device

    logger = Logger(accelerator=accelerator)

    logger.log("Launching Control Experiment", "yellow", main_process=True)
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_config(cfg_dict, main_process=True)
    cfg.memory_llama["hf_token"] = dotenv_values(".env")["HF_TOKEN"]
    MAX_SEQ_LEN = TASK_TO_MAX_LEN[cfg.train.task_name]

    # Load and freeze llama
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

    accelerator.init_trackers(project_name="Hercules", config=cfg_dict)

    model.to(device)

    print(
        f"{Style.BRIGHT}{Fore.RED}Accelerator state:\n{accelerator.state}{Style.RESET_ALL}"
    )

    collate_fn = BabilongCollator(tokenizer, max_length=MAX_SEQ_LEN)
    train_ds = load_dataset(
        "RMT-team/babilong-train-5k-samples",
        name=cfg.train.task_name,
        split=cfg.train.split,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        collate_fn=collate_fn,
        num_workers=8,
    )

    model, train_loader = accelerator.prepare(model, train_loader)

    for epoch in tqdm(range(cfg.train.epochs)):
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.train.epochs}",
            disable=not accelerator.is_main_process,
        )

        for it, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            causal_loss = outputs.loss.item()

            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "causal_loss": causal_loss,
                        "epoch": epoch,
                        "step": it,
                    }
                )

    accelerator.end_training()


if __name__ == "__main__":
    main()
