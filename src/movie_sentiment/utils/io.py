from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    return path
