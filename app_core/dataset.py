from __future__ import annotations

from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import TrainConfig
from .prompts import create_prompt_builder, PromptPieces


class TextPairDataset(Dataset):
    """
    Dataset that:
      - pulls (input_text, output_text) rows from a DataFrame
      - uses a PromptBuilder (based on cfg + model_name) to build prompt/target text
      - tokenizes and returns input_ids, attention_mask, labels
    """

    def __init__(self, df: pd.DataFrame, tokenizer, cfg: TrainConfig):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.df = df.reset_index(drop=True)
        self.prompt_builder = create_prompt_builder(cfg)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        src = str(row["input_text"])
        tgt = str(row["output_text"])

        pieces: PromptPieces = self.prompt_builder.build(
            src=src,
            tgt=tgt,
            eos_token=self.tokenizer.eos_token,
        )
        prompt_str = pieces.prompt_str
        target_str = pieces.target_str

        # 1) Tokenize prompt and target separately
        prompt_tokens = self.tokenizer(
            prompt_str,
            add_special_tokens=True,  # Adds BOS <s> if defined
            truncation=True,
            max_length=self.cfg.max_source_length,
            padding=False,
            return_tensors="pt",
        )
        target_tokens = self.tokenizer(
            target_str,
            add_special_tokens=False,  # EOS added manually in target_str
            truncation=True,
            max_length=self.cfg.max_target_length,
            padding=False,
            return_tensors="pt",
        )

        prompt_ids = prompt_tokens["input_ids"][0]
        prompt_attn = prompt_tokens["attention_mask"][0]

        target_ids = target_tokens["input_ids"][0]
        target_attn = target_tokens["attention_mask"][0]

        # Handle case where target is truncated to empty
        if target_ids.size(0) == 0:
            target_ids = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            target_attn = torch.tensor([1], dtype=torch.long)

        # 2) Concatenate: [prompt][target]
        input_ids = torch.cat([prompt_ids, target_ids], dim=0)
        attention_mask = torch.cat([prompt_attn, target_attn], dim=0)

        # 3) Labels: ignore prompt, learn only on target
        ignore_index = -100
        labels_prompt = torch.full_like(prompt_ids, ignore_index)
        labels_target = target_ids.clone()
        labels = torch.cat([labels_prompt, labels_target], dim=0)

        # 4) Truncate if combined length is too long
        max_len = self.cfg.max_source_length + self.cfg.max_target_length
        if input_ids.size(0) > max_len:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            labels = labels[:max_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
