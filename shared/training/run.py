from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorForSeq2Seq,
)

from app_core.callbacks import PlotLearningCurveCallback, BestAdapterSaverCallback
from app_core.config import TrainConfig
from app_core.data_utils import split_dataframe
from app_core.dataset import TextPairDataset
from app_core.model import get_peft_model_for_training
from app_core.viz import analyze_dataset, save_training_history_csv, plot_learning_curve


def _prepare_run_dirs(cfg: TrainConfig, run_id: str) -> Dict[str, Path]:
    """
    Create run-scoped directories so multiple runs do not overwrite each other.
    """
    base_output = Path(cfg.output_dir) / run_id
    base_output.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        "output": base_output,  # model/adapters/tokenizer/checkpoints
        "checkpoints": base_output / "checkpoints",
        "analysis": base_output / "analysis",
        "data": base_output / "data",
    }
    for p in run_dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return run_dirs


def run_training(
    cfg: TrainConfig,
    dataset_path: str | Path,
    run_id: str | None = None,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Programmatic entry point for launching a training run.

    Returns a summary dict with key paths and metrics to present in a UI.
    """
    run_id = run_id or f"run-{int(time.time())}"
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Copy to run-local data folder so each run is reproducible
    run_dirs = _prepare_run_dirs(cfg, run_id)
    run_dataset_path = run_dirs["data"] / "dataset.csv"
    shutil.copy(dataset_path, run_dataset_path)

    # Save config snapshot inside the run folder
    cfg_snapshot = TrainConfig(**asdict(cfg))
    cfg_snapshot.output_dir = str(run_dirs["output"])
    cfg_snapshot.save(run_dirs["output"] / "config.json")

    # Simple file logger for this run
    log_path = run_dirs["output"] / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
        force=True,
    )

    # Seeds
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"[run {run_id}] Loading data from {run_dataset_path}...")
    df = pd.read_csv(run_dataset_path)
    if cfg.max_train_samples is not None:
        df = df.sample(
            n=min(cfg.max_train_samples, len(df)),
            random_state=cfg.seed,
        )

    if "input_text" not in df.columns or "output_text" not in df.columns:
        raise ValueError("CSV must contain 'input_text' and 'output_text' columns")

    # Dataset analysis -> run/analysis
    dataset_analysis_path = run_dirs["analysis"] / "dataset_analysis.csv"
    analyze_dataset(df, dataset_analysis_path)

    # Train/val/test split -> run/data/splits.json
    train_idx, val_idx, test_idx = split_dataframe(df, seed=cfg.seed)
    splits = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }
    splits_path = run_dirs["data"] / "splits.json"
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    print(
        f"[run {run_id}] train / val / test sizes: "
        f"{len(df_train)} / {len(df_val)} / {len(test_idx)}"
    )

    print(f"[run {run_id}] Loading tokenizer and model...")
    model, tokenizer = get_peft_model_for_training(cfg_snapshot)

    per_device_train_batch_size = (
        cfg_snapshot.per_device_train_batch_size
        if cfg_snapshot.per_device_train_batch_size is not None
        else 1
    )

    train_dataset = TextPairDataset(df_train, tokenizer, cfg_snapshot)
    eval_dataset = TextPairDataset(df_val, tokenizer, cfg_snapshot)

    training_args = TrainingArguments(
        output_dir=str(run_dirs["checkpoints"]),
        num_train_epochs=cfg_snapshot.num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=cfg_snapshot.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg_snapshot.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        learning_rate=cfg_snapshot.learning_rate,
        weight_decay=cfg_snapshot.weight_decay,
        warmup_ratio=cfg_snapshot.warmup_ratio,
        optim="adamw_torch_fused",
        logging_steps=cfg_snapshot.logging_steps,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg_snapshot.save_total_limit,
        report_to=[],
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    plot_callback = PlotLearningCurveCallback(analysis_dir=run_dirs["analysis"])
    best_adapter_callback = BestAdapterSaverCallback(
        output_dir=run_dirs["output"],
        metric_name="eval_loss",
        greater_is_better=False,
    )

    class ProgressCallback(PlotLearningCurveCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if progress_cb and logs:
                payload = {
                    "step": int(state.global_step),
                    "epoch": float(state.epoch) if state.epoch is not None else None,
                    "logs": logs,
                }
                progress_cb(payload)
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(json.dumps(payload) + "\n")
            return super().on_log(args, state, control, logs=logs, **kwargs)

    progress_callback = ProgressCallback(analysis_dir=run_dirs["analysis"])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=cfg_snapshot.max_source_length + cfg_snapshot.max_target_length,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[plot_callback, best_adapter_callback, progress_callback],
    )

    print(f"[run {run_id}] Starting training...")
    train_result = trainer.train()

    final_adapter_dir = run_dirs["output"] / "adapter_final"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = trainer.model
    if hasattr(model_to_save, "module"):
        model_to_save = model_to_save.module
    print(f"[run {run_id}] Saving final adapter to: {final_adapter_dir}")
    model_to_save.save_pretrained(str(final_adapter_dir))

    tokenizer.save_pretrained(str(run_dirs["output"] / "tokenizer"))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print(f"[run {run_id}] Evaluating on validation set (final)...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    history_csv = run_dirs["analysis"] / "training_history.csv"
    save_training_history_csv(trainer.state.log_history, history_csv)
    plot_learning_curve(history_csv, run_dirs["analysis"])

    return {
        "run_id": run_id,
        "output_dir": str(run_dirs["output"]),
        "analysis_dir": str(run_dirs["analysis"]),
        "dataset_path": str(run_dataset_path),
        "splits_path": str(splits_path),
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(test_idx),
        "train_metrics": metrics,
        "eval_metrics": eval_metrics,
        "best_adapter": str(Path(run_dirs["output"]) / "adapter"),
        "final_adapter": str(final_adapter_dir),
    }
