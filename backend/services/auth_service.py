from __future__ import annotations

import os
from typing import Optional

_token: Optional[str] = None


def set_token(token: str) -> None:
    """
    Store the HF token in-memory and set process env so transformers uses it.
    """
    global _token
    _token = token.strip()
    os.environ["HF_TOKEN"] = _token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = _token


def get_token() -> Optional[str]:
    """
    Return the currently stored token (if any).
    """
    return _token
