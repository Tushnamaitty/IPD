"""Golden reproduction (real model only) and monotone direction of previous C-section."""
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from conftest import REAL_MODEL
from serving.artifacts import model_dir
from serving.mapping import to_frame
from serving.predictor import get_predictor
from serving.schemas import PredictRequest


@pytest.mark.skipif(not REAL_MODEL, reason="golden test needs NFHS5_PATH and the real exported model")
def test_golden_validation_metrics():
    import os

    import yaml

    from serving import train_export as te

    metrics = json.loads((model_dir() / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(te._A7 / "config.yaml").read_text(encoding="utf-8"))
    recorded = json.loads(te.RECORDED_SUMMARY.read_text(encoding="utf-8"))
    data = te.prepare_data(Path(os.environ["NFHS5_PATH"]), config, recorded, seed=42)
    probabilities = get_predictor().predict_frame(data["valid_frame"])
    assert round(roc_auc_score(data["y_valid"], probabilities), 4) == round(metrics["roc_auc"], 4)
    assert round(average_precision_score(data["y_valid"], probabilities), 4) == round(metrics["average_precision"], 4)


def _profiles(config, n=20, seed=7):
    rng = np.random.default_rng(seed)
    vocab = config["vocabulary"]
    for _ in range(n):
        yield {
            "maternal_age_at_index_birth": int(rng.integers(18, 45)),
            "education_years": int(rng.integers(0, 21)),
            "wealth_index": int(rng.integers(1, 6)),
            "birth_order": int(rng.integers(2, 7)),
            "twin_order": "single" if rng.random() < 0.9 else "multiple",
            "facility_detail": str(rng.choice(vocab["facility_detail"])),
            "residence": str(rng.choice(vocab["residence"])),
            "religion": str(rng.choice(vocab["religion"])),
            "is_first_birth": False,
        }


def test_previous_csection_yes_raises_probability(config):
    predictor = get_predictor()
    wins = 0
    for profile in _profiles(config):
        p_yes, p_no = (
            float(predictor.predict_frame(to_frame(PredictRequest(**profile, previous_csection=answer)))[0])
            for answer in ("yes", "no")
        )
        wins += p_yes > p_no
    print(f"\nprevious_csection yes > no in {wins}/20 profiles")
    assert wins >= 19
