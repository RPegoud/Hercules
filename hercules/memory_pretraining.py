import hydra
from accelerate import Accelerator
from colorama import Fore, Style
from datasets import load_dataset
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from hercules import BabilongCollator, MemoryLlama, log_config

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
    config_path="config",
    config_name="pre_training.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    log_config(cfg_dict)
    cfg.memory_llama["hf_token"] = dotenv_values(".env")["HF_TOKEN"]
    MAX_SEQ_LEN = TASK_TO_MAX_LEN[cfg.train.task_name]

    model = MemoryLlama(neural_memory_config=cfg.lmm, **cfg.memory_llama)
    tokenizer = AutoTokenizer.from_pretrained(cfg.memory_llama.llama_hf_path)
    tokenizer.pad_token = tokenizer.eos_token

    accelerator = Accelerator()
    device = accelerator.device

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
        num_workers=cfg.train.batch_size,
    )

    llm_losses, memory_losses = [], []

    # initialize model before calling prepare
    for b in train_loader:
        b = b.to(device)
        model(**b)
        break

    model, train_loader = accelerator.prepare(model, train_loader)

    for epoch in tqdm(range(cfg.train.epochs)):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        epoch_llm_loss, epoch_memory_loss = 0, 0

        for batch in train_loader:
            batch = batch.to(device)  # keys: ["input_ids", "attention_mask", "labels"]
            outputs = model(**batch)

            unwrapped_model = accelerator.unwrap_model(model)
            memory_loss = unwrapped_model.neural_memory.associative_loss
            memory_losses.append(memory_loss)
            epoch_memory_loss += memory_loss

            llm_loss = outputs.loss.item()
            llm_losses.append(llm_loss)
            epoch_llm_loss += llm_loss

        progress_bar.set_postfix({"llm loss": llm_loss, "memory loss": llm_loss})
        break


if __name__ == "__main__":
    main()
