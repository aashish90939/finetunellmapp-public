from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from .prompts import create_prompt_builder


def split_dataframe(
    df,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split indices of a DataFrame into train/val/test.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    n_total = len(df)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def auto_select_batch_size(cfg, model, tokenizer, df):
    """
    Optional helper (was _auto_select_batch_size in your file).
    Kept for reference; not used when cfg.auto_batch_search == False.

    Uses no-grad inference to test batch sizes -> underestimates training memory.
    """
    if not torch.cuda.is_available():
        return 1

    device = torch.device("cuda")
    dummy_row = df.iloc[0]
    src = str(dummy_row["input_text"])
    tgt = str(dummy_row["output_text"])

    prompt_builder = create_prompt_builder(cfg)
    pieces = prompt_builder.build(src=src, tgt=tgt, eos_token=tokenizer.eos_token)
    prompt = pieces.prompt_str + pieces.target_str

    for bs in cfg.auto_batch_candidates:
        try:
            tokenized = tokenizer(
                [prompt] * bs,
                max_length=cfg.max_source_length + cfg.max_target_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            model.to(device)
            with torch.no_grad():
                _ = model(**tokenized)
            torch.cuda.empty_cache()
            print(f"[train] Auto batch size selected: {bs}")
            return bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[train] Batch size {bs} OOM, trying smaller...")
                torch.cuda.empty_cache()
            else:
                raise

    print("[train] Fallback batch size 1")
    return 1
