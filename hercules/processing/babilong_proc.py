from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, random_split
from transformers import AutoTokenizer, PreTrainedTokenizerBase

ALL_BABILONG_SPLITS = [
    "qa1",
    "qa2",
    "qa3",
    "qa4",
    "qa5",
    "qa6",
    "qa7",
    "qa8",
    "qa9",
    "qa10",
]
TASK_TO_MAX_LEN = {  # Babilong task lenghts
    "0k": 0,
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
}


class BabilongCollator:
    """Handles the preprocessing of Babilong sequences."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        return_prompts_for_generation: bool = False,
    ) -> None:
        self.max_length = max_length
        self.return_prompts_for_generation = return_prompts_for_generation
        self.IGNORE_INDEX = -100  # this token is ignored by the loss

        self.train_tokenizer = tokenizer  # right padding for training

        self.gen_tokenizer = deepcopy(tokenizer)  # left padding for generation
        self.gen_tokenizer.padding_side = "left"

    def __call__(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Returns batch of input_ids, masked labels and attention masks."""
        inputs = [sample["input"] for sample in batch]
        questions = [sample["question"] for sample in batch]
        targets = [sample["target"] for sample in batch]

        # combine inputs, questions and targets into prompts
        prompts_only = [
            f"{i}, Question: {q}, Answer:" for i, q in zip(inputs, questions)
        ]
        full_texts = [
            f"{i}, Question: {q}, Answer: {t}"
            for i, q, t in zip(inputs, questions, targets)
        ]

        tokenized_prompts = self.train_tokenizer(
            prompts_only, padding=False, truncation=False
        )
        model_inputs = self.train_tokenizer(
            full_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = model_inputs.input_ids.clone()

        # create labels by masking all the text and question tokens
        for i in range(len(labels)):
            prompt_len = len(tokenized_prompts.input_ids[i])
            labels[i, :prompt_len] = self.IGNORE_INDEX

        # mask the padding tokens
        labels[labels == self.train_tokenizer.pad_token_id] = self.IGNORE_INDEX

        model_inputs["labels"] = labels

        if self.return_prompts_for_generation:
            prompts_for_generation = self.gen_tokenizer(
                prompts_only,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            model_inputs["prompt_input_ids"] = prompts_for_generation["input_ids"]
            model_inputs["prompt_attention_mask"] = prompts_for_generation[
                "attention_mask"
            ]
            model_inputs["target_text"] = targets
        return model_inputs


def _get_split_list(split_config: Union[str, List[str]]) -> List[str]:
    """Helper to parse a dataset split config into a list of strings."""
    if isinstance(split_config, str):
        return [split_config]
    if split_config is None:
        return []
    return list(split_config)


def get_test_splits(cfg: DictConfig) -> List[str]:
    """Determines which test splits to use based on the config."""
    train_splits = _get_split_list(cfg.experiment.train_splits)
    test_config = cfg.experiment.test_splits

    if test_config == "all":
        return ALL_BABILONG_SPLITS
    if test_config == "remaining":
        return [split for split in ALL_BABILONG_SPLITS if split not in train_splits]
    return _get_split_list(test_config)


def get_specific_split_bl_dataloaders(
    cfg: DictConfig,
    tokenizer: AutoTokenizer,
    return_prompts_for_generation: bool,
    test_only: bool = False,
) -> Tuple[DataLoader, Dict[str, DataLoader]] | Dict[str, DataLoader]:
    """
    Creates and returns the train and test dataloaders for specific
    train and test splits.
    If `test_only` is set to `True`, only returns a dictionary of test sets.
    """
    train_collate_fn = BabilongCollator(
        tokenizer, max_length=TASK_TO_MAX_LEN[cfg.experiment.train_task_name]
    )
    test_collate_fn = BabilongCollator(
        tokenizer,
        max_length=TASK_TO_MAX_LEN[cfg.experiment.test_task_name],
        return_prompts_for_generation=return_prompts_for_generation,
    )

    train_splits = _get_split_list(cfg.experiment.train_splits)
    if test_only:
        test_splits = get_test_splits(cfg)
        test_loaders = {}
        if test_splits:
            full_test_ds_dict = load_dataset(
                "RMT-team/babilong-train-5k-samples", name=cfg.experiment.test_task_name
            )
            for split in test_splits:
                if split in full_test_ds_dict:
                    test_loaders[split] = DataLoader(
                        full_test_ds_dict[split],
                        batch_size=cfg.experiment.batch_size,
                        collate_fn=test_collate_fn,
                        num_workers=8,
                    )
        return test_loaders

    if not train_splits:
        raise ValueError(
            "cfg.experiment.train_splits must be configured with at least one split."
        )

    list_of_train_datasets = [
        load_dataset(
            "RMT-team/babilong-train-5k-samples",
            name=cfg.experiment.train_task_name,
            split=split,
        )
        for split in train_splits
    ]
    train_ds = ConcatDataset(list_of_train_datasets)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.experiment.batch_size,
        collate_fn=train_collate_fn,
        num_workers=8,
        shuffle=True,
    )

    return train_loader, test_loaders


def get_global_split_bl_dataloaders(
    cfg: DictConfig,
    tokenizer: AutoTokenizer,
    return_prompts_for_generation: bool,
) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """
    Creates and returns the train and test dataloaders using all the splits for training
    with `cfg.experiment.global_split_test_size`% held out for testing.
    """
    train_collate_fn = BabilongCollator(
        tokenizer, max_length=TASK_TO_MAX_LEN[cfg.experiment.train_task_name]
    )
    test_collate_fn = BabilongCollator(
        tokenizer,
        max_length=TASK_TO_MAX_LEN[cfg.experiment.test_task_name],
        return_prompts_for_generation=return_prompts_for_generation,
    )

    full_dataset_dict = load_dataset(
        "RMT-team/babilong-train-5k-samples", name=cfg.experiment.train_task_name
    )

    # merge all splits in a single dataset
    all_qa_datasets = [
        full_dataset_dict[split]
        for split in ALL_BABILONG_SPLITS
        if split in full_dataset_dict
    ]
    combined_ds = ConcatDataset(all_qa_datasets)
    train_ds, test_ds = random_split(
        combined_ds,
        [
            1 - cfg.experiment.global_split_test_size,
            cfg.experiment.global_split_test_size,
        ],
        generator=torch.Generator().manual_seed(cfg.experiment.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.experiment.batch_size,
        collate_fn=train_collate_fn,
        num_workers=8,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.experiment.batch_size,
        collate_fn=test_collate_fn,
        num_workers=8,
    )

    return train_loader, {"global_test": test_loader}
