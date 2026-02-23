from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ConfigDefaults(BaseModel):
    defaults: Dict[str, Any]


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    path: str
    rows: int


class RunCreateRequest(BaseModel):
    dataset_path: str = Field(
        "",
        description="Path to a CSV with input_text/output_text columns. Leave blank for plain model inference.",
    )
    run_id: Optional[str] = Field(None, description="Optional run identifier")
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Override TrainConfig fields for this run",
    )


class RunStatus(BaseModel):
    run_id: str
    status: str
    detail: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    progress: Optional[Dict[str, Any]] = None


class InferenceRequest(BaseModel):
    run_id: str
    input_text: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class InferenceResponse(BaseModel):
    output: str
