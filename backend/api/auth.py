from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services import auth_service

router = APIRouter()


class TokenPayload(BaseModel):
    token: str


@router.post("/token")
def set_token(payload: TokenPayload) -> dict:
    auth_service.set_token(payload.token)
    return {"ok": True}


@router.get("/token")
def get_token() -> dict:
    token = auth_service.get_token()
    return {"token_set": bool(token)}
