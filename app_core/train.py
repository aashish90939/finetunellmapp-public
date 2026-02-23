from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorForSeq2Seq,
)

from .config import TrainConfig
from .model import get_peft_model_for_training
from .viz import analyze_dataset, save_training_history_csv, plot_learning_curve
from .dataset import TextPairDataset
from .callbacks import PlotLearningCurveCallback, BestAdapterSaverCallback
from .data_utils import split_dataframe


def main():
    cfg = TrainConfig()

    # Directories
    output_dir = Path(cfg.output_dir)        # model stuff only
    data_dir = Path(cfg.data_path).parent    # all data-related CSVs
    analysis_dir = Path("analysis")          # stats & plots

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Save config only in outputs/
    cfg.save(output_dir / "config.json")

    # Seeds
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("[train] Loading data...")
    df = pd.read_csv(cfg.data_path)
    if cfg.max_train_samples is not None:
        df = df.sample(
            n=min(cfg.max_train_samples, len(df)),
            random_state=cfg.seed,
        )

    if "input_text" not in df.columns or "output_text" not in df.columns:
        raise ValueError("CSV must contain 'input_text' and 'output_text' columns")

    # Dataset analysis -> data/
    dataset_analysis_path = data_dir / "dataset_analysis.csv"
    analyze_dataset(df, dataset_analysis_path)

    # Train/val/test split -> data/splits.json
    train_idx, val_idx, test_idx = split_dataframe(df, seed=cfg.seed)
    splits = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }
    splits_path = data_dir / "splits.json"
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    print(
        f"[train] train / val / test sizes: "
        f"{len(df_train)} / {len(df_val)} / {len(test_idx)}"
    )

    print("[train] Loading tokenizer and model...")
    model, tokenizer = get_peft_model_for_training(cfg)

    # Use a safe, small batch size (like your working notebook)
    if cfg.per_device_train_batch_size is not None:
        per_device_train_batch_size = cfg.per_device_train_batch_size
    else:
        per_device_train_batch_size = 1

    train_dataset = TextPairDataset(df_train, tokenizer, cfg)
    eval_dataset = TextPairDataset(df_val, tokenizer, cfg)

    # Trainer checkpoints live under outputs/checkpoints (model-related)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg.num_train_epochs,

        # Batches
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_accumulation_steps=1,

        # Optimizer / scheduler
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        optim="adamw_torch_fused",

        # Logging / saving
        logging_steps=cfg.logging_steps,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        report_to=[],  # no wandb / tensorboard

        # Precision (bnb takes care of 4bit compute dtype)
        fp16=False,
        bf16=False,

        # Gradient checkpointing (model already has it enabled)
        gradient_checkpointing=True,

        # Best model selection inside Trainer
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    plot_callback = PlotLearningCurveCallback(analysis_dir=analysis_dir)
    best_adapter_callback = BestAdapterSaverCallback(
        output_dir=output_dir,
        metric_name="eval_loss",
        greater_is_better=False,  # lower loss is better
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=cfg.max_source_length + cfg.max_target_length,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[plot_callback, best_adapter_callback],
    )

    print("[train] Starting training...")
    train_result = trainer.train()

    # After a clean finish, also save Trainer's final model to outputs/adapter_final
    final_adapter_dir = output_dir / "adapter_final"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = trainer.model
    if hasattr(model_to_save, "module"):
        model_to_save = model_to_save.module
    print(f"[train] Saving final adapter (Trainer model) to: {final_adapter_dir}")
    model_to_save.save_pretrained(str(final_adapter_dir))

    # Save tokenizer in outputs/
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("[train] Evaluating on validation set (final)...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Save final history + plot (again) into analysis/
    history_csv = analysis_dir / "training_history.csv"
    save_training_history_csv(trainer.state.log_history, history_csv)
    plot_learning_curve(history_csv, analysis_dir)

    print(f"[train] Done.")
    print(f"[train] Model-related outputs in: {output_dir}")
    print(f"[train]  - Best-so-far adapter: {output_dir / 'adapter'}")
    print(f"[train]  - Final Trainer adapter: {final_adapter_dir}")
    print(f"[train] Data-related CSV in: {data_dir}")
    print(f"[train]  - dataset_analysis.csv, splits.json")
    print(f"[train] Training stats & plots in: {analysis_dir}")


if __name__ == "__main__":
    main()
