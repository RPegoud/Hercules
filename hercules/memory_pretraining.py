import json

import hydra
import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
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

    model = MemoryLlama(neural_memory_config=cfg.neural_memory, **cfg.memory_llama)
    optimizer = torch.optim.AdamW(
        model.neural_memory.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.memory_llama.llama_hf_path)
    tokenizer.pad_token = tokenizer.eos_token

    # ignores the frozen parameters (i.e. llama params)
    accelerator_kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
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
        num_workers=8,
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    losses = {"causal": {}, "associative": {}}

    for epoch in tqdm(range(cfg.train.epochs)):
        for batch in tqdm(train_loader):
            epoch_associative_losses = []
            epoch_causal_losses = []

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(output_hidden_states=True, **batch)

            # unwrap the model to access the loss function
            unwrapped_model = accelerator.unwrap_model(model)
            attn_outputs_for_loss = outputs.hidden_states[
                unwrapped_model.memory_layer_id
            ]
            memory_loss = unwrapped_model.neural_memory.get_associative_memory_loss(
                attn_outputs_for_loss
            )

            accelerator.backward(memory_loss)

            optimizer.step()
            optimizer.zero_grad()

            causal_loss = outputs.loss

            gathered_causal_losses = accelerator.gather_for_metrics(causal_loss)
            gathered_memory_losses = accelerator.gather_for_metrics(memory_loss)

            epoch_causal_losses.append(gathered_causal_losses.mean().item())
            epoch_associative_losses.append(gathered_memory_losses.mean().item())

        if accelerator.is_main_process:
            avg_causal_loss = sum(epoch_causal_losses) / len(epoch_causal_losses)
            avg_associative_loss = sum(epoch_associative_losses) / len(
                epoch_associative_losses
            )

            losses["causal"][epoch] = avg_causal_loss
            losses["associative"][epoch] = avg_associative_loss

            loss_msg = f"Causal loss: {avg_causal_loss:.4e}, Associative loss: {avg_associative_loss:.4e}"
            print(
                f"{Style.BRIGHT}{Fore.YELLOW}Epoch {epoch}: {loss_msg}{Style.RESET_ALL}"
            )

    # if accelerator.is_main_process:
    #     print("--> Saving final model...")
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     torch.save(unwrapped_model.neural_memory.state_dict(), "../final_model.pth")
    #     print("--> Done!")

    print(losses)


if __name__ == "__main__":
    main()
