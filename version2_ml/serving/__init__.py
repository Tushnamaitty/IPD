"""V2 serving package: inference core for the A7 XGBoost / F1_detail_history model.

The saved model pickles classes from the top-level research modules
(model_framework, preprocessing) and the inference path reuses the A4/A7
feature code unchanged, so those source directories are put on sys.path the
same way the A7 runner does.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parents[1]
for _entry in (
    _BASE,
    _BASE / "experiments" / "common",
    _BASE / "experiments" / "A4" / "src",
    _BASE / "experiments" / "A7" / "src",
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))
