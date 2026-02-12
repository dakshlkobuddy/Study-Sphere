import os
from pathlib import Path

import torch
from dotenv import load_dotenv


def load_project_env(working_dir: str):
    working_path = Path(working_dir).resolve()
    parent_path = working_path.parent
    load_dotenv(parent_path / ".env")
    load_dotenv(working_path / ".env", override=True)
    return str(parent_path)


def resolve_device(device: str | None = None) -> str:
    preferred = (device or os.getenv("DEVICE", "cpu")).strip().lower()
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if preferred == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return "cpu"
    return preferred


def get_unified_vector_db_path(parent_dir: str) -> str:
    return str(Path(parent_dir) / "vector_db" / "class_12_unified_vector_db")

