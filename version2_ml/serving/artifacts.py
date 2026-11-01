"""Location and loading of the exported serving artifacts (config.json, model.joblib)."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"


def model_dir() -> Path:
    return Path(os.environ.get("IPD_MODEL_DIR", DEFAULT_MODEL_DIR))


@lru_cache(maxsize=1)
def load_config() -> dict:
    path = model_dir() / "config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run version2_ml/serving/train_export.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))
