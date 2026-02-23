from __future__ import annotations

from fastapi import APIRouter

from app_core.config import TrainConfig
from backend.models.schemas import ConfigDefaults

router = APIRouter()


@router.get("/defaults", response_model=ConfigDefaults)
def get_defaults() -> ConfigDefaults:
    """
    Returns the current TrainConfig defaults so the UI can pre-fill forms.
    """
    return ConfigDefaults(defaults=TrainConfig().__dict__)
