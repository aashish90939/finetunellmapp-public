from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from app_core.config import TrainConfig
from shared.training import run_training

# Simple in-memory registry; replace with a DB in production
RUNS: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
PLAIN_MODELS: Dict[str, Dict[str, Any]] = {}


def _merge_config(overrides: Dict[str, Any]) -> TrainConfig:
    base = asdict(TrainConfig())
    merged = {**base, **overrides}
    return TrainConfig(**merged)


def _run_job(run_id: str, cfg: TrainConfig, dataset_path: Path) -> None:
    def progress_cb(payload: Dict[str, Any]) -> None:
        with _lock:
            if run_id in RUNS:
                RUNS[run_id]["progress"] = payload

    try:
        result = run_training(cfg, dataset_path=dataset_path, run_id=run_id, progress_cb=progress_cb)
        with _lock:
            RUNS[run_id]["status"] = "succeeded"
            RUNS[run_id]["result"] = result
    except Exception as exc:  # noqa: BLE001
        with _lock:
            RUNS[run_id]["status"] = "failed"
            RUNS[run_id]["detail"] = str(exc)


def launch_run(run_id: str, dataset_path: Path, config_overrides: Dict[str, Any]) -> str:
    cfg = _merge_config(config_overrides)
    with _lock:
        RUNS[run_id] = {"status": "running", "dataset_path": str(dataset_path)}
    thread = threading.Thread(
        target=_run_job,
        kwargs={"run_id": run_id, "cfg": cfg, "dataset_path": dataset_path},
        daemon=True,
    )
    thread.start()
    return run_id


def list_runs() -> Dict[str, Dict[str, Any]]:
    with _lock:
        return {**PLAIN_MODELS, **RUNS}


def get_run(run_id: str) -> Dict[str, Any]:
    with _lock:
        return RUNS.get(run_id) or PLAIN_MODELS.get(run_id) or {"status": "unknown"}


def register_plain_model(run_id: str, model_name: str, config_overrides: Dict[str, Any] | None = None) -> str:
    """
    Register a plain (non-finetuned) model for inference only.
    """
    overrides = config_overrides or {}
    cfg = _merge_config({**overrides, "model_name": model_name})
    with _lock:
        PLAIN_MODELS[run_id] = {
            "status": "succeeded",
            "result": {
                "run_id": run_id,
                "output_dir": model_name,  # reuse field; inference service will treat as path/id
                "plain_model": True,
                "model_name": model_name,
                "config": asdict(cfg),
            },
        }
    return run_id
