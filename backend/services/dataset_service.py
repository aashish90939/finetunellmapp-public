from __future__ import annotations

import shutil
import uuid
from pathlib import Path


DATASET_ROOT = Path("data/uploads")


def save_upload(file_path: Path) -> tuple[str, Path]:
    """
    Save an uploaded dataset to a local, versioned path.
    """
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_id = uuid.uuid4().hex[:10]
    dest = DATASET_ROOT / f"{dataset_id}.csv"
    shutil.copy(file_path, dest)
    return dataset_id, dest
