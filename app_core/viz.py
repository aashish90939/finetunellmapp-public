from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class LogRecord:
    step: int
    loss: float | None = None
    eval_loss: float | None = None


def save_training_history_csv(log_history: list[dict], csv_path: str | Path) -> None:
    """
    Saves the full trainer log history to a CSV file.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter Trainer log_history to a flat CSV
    # Ensure fieldnames are consistent, even if some logs are missing keys
    all_keys = set()
    for rec in log_history:
        all_keys.update(rec.keys())
    
    # Make sure 'step' is a fieldname even if history is empty
    if not all_keys:
        all_keys = {"step", "loss", "eval_loss"}
        
    fieldnames = sorted(list(all_keys))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in log_history:
            writer.writerow(rec)


def plot_learning_curve(csv_path: str | Path, output_dir: str | Path) -> None:
    """
    Plots both training and validation loss from the training history CSV
    on a single graph and saves it as 'learning_curve.png'.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the single, consistently named output file
    plot_path = output_dir / "learning_curve.png"

    if not csv_path.exists():
        print(f"[viz] No history CSV found at {csv_path}. Skipping plot.")
        return

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"[viz] History CSV at {csv_path} is empty. Skipping plot.")
        return

    if "step" not in df.columns:
        print("[viz] No 'step' column found in history CSV. Skipping plot.")
        return

    plt.figure(figsize=(12, 6))
    
    # Plot Training Loss
    if "loss" in df.columns:
        df_train_loss = df.dropna(subset=["loss"])
        if not df_train_loss.empty:
            plt.plot(
                df_train_loss["step"], 
                df_train_loss["loss"], 
                label="Training Loss", 
                marker=".", 
                alpha=0.6
            )

    # Plot Evaluation Loss
    if "eval_loss" in df.columns:
        df_eval_loss = df.dropna(subset=["eval_loss"])
        if not df_eval_loss.empty:
            plt.plot(
                df_eval_loss["step"], 
                df_eval_loss["eval_loss"], 
                label="Validation Loss", 
                marker="o", 
                linestyle="--",
                markersize=8
            )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Over Time")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    
    # Save the plot, overwriting the previous one
    plt.savefig(plot_path)
    plt.close()


def analyze_dataset(df: pd.DataFrame, output_csv: str | Path) -> None:
    """
    Simple dataset analysis: length stats of input / output texts.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if "input_text" not in df.columns or "output_text" not in df.columns:
        raise ValueError("CSV must contain 'input_text' and 'output_text' columns")

    stats = {}

    for col in ["input_text", "output_text"]:
        lengths = df[col].astype(str).str.len()
        stats[f"{col}_count"] = len(lengths)
        stats[f"{col}_min"] = lengths.min()
        stats[f"{col}_max"] = lengths.max()
        stats[f"{col}_mean"] = lengths.mean()
        stats[f"{col}_std"] = lengths.std()

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in stats.items():
            writer.writerow([k, v])