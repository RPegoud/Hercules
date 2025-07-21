from typing import Dict

import torch
from accelerate import Accelerator
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator


def tokenize_fn(
    examples, tokenizer: PreTrainedTokenizerBase, cfg: DictConfig
) -> Dict[str, torch.Tensor]:
    tokens = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=cfg.experiment.max_seq_len,
    )

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # mask out padding tokens in labels
    labels = []
    for ids, mask in zip(input_ids, attention_mask):
        lbl = ids.copy()
        lbl = [tok if m == 1 else -100 for tok, m in zip(lbl, mask)]
        labels.append(lbl)

    tokens["labels"] = labels
    return tokens


def get_eduweb_dataloader(
    cfg: DictConfig, tokenizer: PreTrainedTokenizerBase, accelerator: Accelerator
) -> DataLoader:
    with accelerator.main_process_first():
        raw_dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        processed_dataset = (
            raw_dataset.shuffle(seed=cfg.experiment.seed, buffer_size=10_000)
            .map(
                tokenize_fn,
                fn_kwargs={"tokenizer": tokenizer, "cfg": cfg},
                batched=True,
                remove_columns=["text", "id", "dump", "url"],
            )
            .map(lambda x: {"labels": x["input_ids"]}, batched=True)
        )
        processed_dataset = processed_dataset.take(
            int(cfg.experiment.num_train_samples)
        )
        processed_dataset = processed_dataset.with_format("torch")

    return DataLoader(
        processed_dataset,
        batch_size=cfg.experiment.eduweb_batch_size,
        collate_fn=default_data_collator,
        num_workers=4,
    )
