"""Đọc cấu hình params.yaml và chuẩn hoá đường dẫn theo thư mục gốc project."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.getenv("SPAMGUARD_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "configs" / "params.yaml"


@lru_cache(maxsize=1)
def load_params(path: str | None = None) -> dict[str, Any]:
    """Load params.yaml (cache lại vì được gọi ở nhiều module)."""
    params_path = Path(path) if path else DEFAULT_PARAMS_PATH
    if not params_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {params_path}")
    with params_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(relative_path: str) -> Path:
    """Đường dẫn trong params là tương đối với PROJECT_ROOT."""
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_labels() -> list[str]:
    return list(load_params()["labels"])
