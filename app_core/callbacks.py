from __future__ import annotations

from pathlib import Path
from typing import Dict

from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)

from .viz import save_training_history_csv, plot_learning_curve


class PlotLearningCurveCallback(TrainerCallback):
    """
    Saves the training history and plots the learning curve
    after each evaluation, into the analysis/ folder.
    """

    def __init__(self, analysis_dir: str | Path):
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.analysis_dir / "training_history.csv"

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(
            f"[Callback] Evaluation complete at step {state.global_step}. "
            f"Updating learning curve..."
        )

        # Save full log history (train + eval) to CSV inside analysis/
        save_training_history_csv(state.log_history, self.csv_path)

        # Re-plot learning curve from CSV inside analysis/
        plot_learning_curve(self.csv_path, self.analysis_dir)

        return control


class BestAdapterSaverCallback(TrainerCallback):
    """
    Tracks the best evaluation metric and saves the LoRA adapter to
    <output_dir>/adapter whenever a new best is found.

    This way, you can stop training at any time and still have
    the best-so-far adapter on disk.
    """

    def __init__(
        self,
        output_dir: str | Path,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_score: float | None = None

        # We always save the best adapter here:
        self.best_dir = self.output_dir / "adapter"
        self.best_dir.mkdir(parents=True, exist_ok=True)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        metrics: Dict[str, float] | None = kwargs.get("metrics")
        if metrics is None:
            return control

        if self.metric_name not in metrics:
            return control

        current = metrics[self.metric_name]

        # Decide if the new score is better
        if self.best_score is None:
            is_better = True
        else:
            if self.greater_is_better:
                is_better = current > self.best_score
            else:
                is_better = current < self.best_score

        if not is_better:
            return control

        self.best_score = current

        model = kwargs.get("model", None)
        if model is None:
            return control

        # unwrap if wrapped in DDP / DataParallel
        if hasattr(model, "module"):
            model = model.module

        print(
            f"[Callback] New best {self.metric_name}={current:.4f}. "
            f"Saving adapter to {self.best_dir}"
        )
        model.save_pretrained(str(self.best_dir))

        return control
