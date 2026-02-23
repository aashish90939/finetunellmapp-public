from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.models.schemas import RunCreateRequest, RunStatus
from backend.services.training_service import launch_run, get_run, list_runs, register_plain_model

router = APIRouter()


@router.get("", response_model=list[RunStatus])
def get_runs() -> list[RunStatus]:
    runs = list_runs()
    return [
        RunStatus(run_id=rid, status=meta.get("status"), detail=meta.get("detail"), result=meta.get("result"), progress=meta.get("progress"))
        for rid, meta in runs.items()
    ]


@router.post("", response_model=RunStatus)
def create_run(req: RunCreateRequest) -> RunStatus:
    run_id = req.run_id or uuid.uuid4().hex[:8]
    # If dataset_path is empty and config_overrides contain a model_name, treat as plain model
    if not req.dataset_path:
        model_name = req.config_overrides.get("model_name")
        if not model_name:
            raise HTTPException(status_code=400, detail="Provide model_name for plain inference run")
        register_plain_model(run_id, model_name, req.config_overrides)
        meta = get_run(run_id)
        return RunStatus(run_id=run_id, status=meta.get("status"), detail=meta.get("detail"))

    dataset_path = Path(req.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {dataset_path}")

    launch_run(run_id, dataset_path, req.config_overrides)
    meta = get_run(run_id)
    return RunStatus(run_id=run_id, status=meta.get("status"), detail=meta.get("detail"), progress=meta.get("progress"))


@router.get("/{run_id}", response_model=RunStatus)
def get_run_status(run_id: str) -> RunStatus:
    meta = get_run(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunStatus(run_id=run_id, status=meta.get("status"), detail=meta.get("detail"), result=meta.get("result"), progress=meta.get("progress"))


@router.get("/{run_id}/log")
def get_run_log(run_id: str, max_bytes: int = 4000) -> dict:
    meta = get_run(run_id)
    if not meta or not meta.get("result"):
        raise HTTPException(status_code=404, detail="Run not found")
    output_dir = meta["result"].get("output_dir")
    if not output_dir:
        return {"log": ""}
    log_path = Path(output_dir) / "train.log"
    if not log_path.exists():
        return {"log": ""}
    data = log_path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return {"log": data.decode("utf-8", errors="ignore")}
