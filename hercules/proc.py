from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase


class BabilongCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.IGNORE_INDEX = -100  # this token is ignored by the loss

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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

        prompts_tokenized = self.tokenizer(
            prompts_only, padding=False, truncation=False
        )
        model_inputs = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = model_inputs.input_ids.clone()

        # create labels by masking all the text and question tokens
        for i in range(len(labels)):
            prompt_len = len(prompts_tokenized.input_ids[i])
            labels[i, :prompt_len] = self.IGNORE_INDEX

        # mask the padding tokens
        labels[labels == self.tokenizer.pad_token_id] = self.IGNORE_INDEX

        model_inputs["labels"] = labels

        return model_inputs
