"""Model selection for the serving tests.

- NFHS5_PATH set: use the real exported model (version2_ml/serving/models, or
  $IPD_MODEL_DIR) and enable the golden test.
- NFHS5_PATH unset: build a synthetic fixture model into a temporary directory.

IPD_MODEL_DIR must be set before any serving module is imported, because
serving.schemas builds its enums from config.json at import time.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

SERVING = Path(__file__).resolve().parents[1]
if str(SERVING.parent) not in sys.path:
    sys.path.insert(0, str(SERVING.parent))

REAL_MODEL = bool(os.environ.get("NFHS5_PATH"))
_FIXTURE_DIR = None

if REAL_MODEL:
    os.environ.setdefault("IPD_MODEL_DIR", str(SERVING / "models"))
    if not (Path(os.environ["IPD_MODEL_DIR"]) / "model.joblib").exists():
        raise RuntimeError("NFHS5_PATH is set but no exported model found; run train_export.py first")
else:
    import serving  # noqa: F401  (research paths)
    from fixture_model import build_fixture_model

    _FIXTURE_DIR = Path(tempfile.mkdtemp(prefix="ipd_fixture_model_"))
    build_fixture_model(_FIXTURE_DIR)
    os.environ["IPD_MODEL_DIR"] = str(_FIXTURE_DIR)


def pytest_report_header(config):
    mode = "REAL exported model" if REAL_MODEL else "synthetic fixture model"
    return f"ipd serving tests: {mode} at {os.environ['IPD_MODEL_DIR']}"


def pytest_sessionfinish(session, exitstatus):
    if _FIXTURE_DIR is not None:
        shutil.rmtree(_FIXTURE_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from serving.api import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def config():
    from serving.artifacts import load_config

    return load_config()


@pytest.fixture
def valid_body():
    return {
        "maternal_age_at_index_birth": 28,
        "education_years": 10,
        "wealth_index": 3,
        "birth_order": 2,
        "twin_order": "single",
        "facility_detail": "public: govt./munic. hospital",
        "residence": "urban",
        "religion": "1",
        "is_first_birth": False,
        "previous_csection": "no",
    }
