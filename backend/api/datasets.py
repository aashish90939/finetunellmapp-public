from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException

from backend.models.schemas import DatasetUploadResponse
from backend.services.dataset_service import save_upload

router = APIRouter()


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    """
    Accept a CSV upload and store it for future runs.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        df = pd.read_csv(tmp_path)
    except Exception as exc:  # noqa: BLE001
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    dataset_id, dest = save_upload(tmp_path)
    tmp_path.unlink(missing_ok=True)
    return DatasetUploadResponse(
        dataset_id=dataset_id,
        path=str(dest),
        rows=len(df),
    )
