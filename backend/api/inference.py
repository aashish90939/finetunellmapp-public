from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.models.schemas import InferenceRequest, InferenceResponse
from backend.services.inference_service import generate
from backend.services.training_service import get_run

router = APIRouter()


@router.post("", response_model=InferenceResponse)
def run_inference(req: InferenceRequest) -> InferenceResponse:
    meta = get_run(req.run_id)
    if not meta or meta.get("status") != "succeeded":
        raise HTTPException(status_code=400, detail="Run not finished or not found")

    result = meta.get("result") or {}
    output_dir = result.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=400, detail="Run has no output_dir recorded")

    output = generate(
        run_id=req.run_id,
        run_output_dir=Path(output_dir),
        input_text=req.input_text,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        cfg_overrides=result.get("config"),
        plain_model=bool(result.get("plain_model")),
    )
    return InferenceResponse(output=output)
