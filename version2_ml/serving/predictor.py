"""Load the exported model once and score requests with calibrated probabilities."""
from __future__ import annotations

import hashlib
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .artifacts import load_config, model_dir
from .mapping import to_frame
from .schemas import PredictRequest

from a7_calibration import apply_calibrator  # noqa: E402

POSITIVE_LABEL = "C-section likely"
NEGATIVE_LABEL = "Vaginal delivery likely"


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Predictor:
    def __init__(self, config: dict, bundle: dict):
        if list(bundle["feature_order"]) != list(config["feature_order"]):
            raise RuntimeError("model.joblib feature order differs from config.json")
        if bundle["calibration_method"] != config["calibration_method"]:
            raise RuntimeError("model.joblib calibration method differs from config.json")
        self.config = config
        self.classifier = bundle["classifier"]
        self.calibrator = bundle["calibrator"]
        self.calibration_method = bundle["calibration_method"]
        self.model_version = config["model_version"]
        self.threshold = float(config["thresholds"][config["thresholds"]["default"]])
        self.band_low_upper = float(config["risk_bands"]["low_upper"])
        self.band_high_lower = float(config["risk_bands"]["high_lower"])

    @classmethod
    def load(cls) -> "Predictor":
        config = load_config()
        path = model_dir() / "model.joblib"
        actual = _sha256(path)
        if actual != config["model_sha256"]:
            raise RuntimeError(
                f"model.joblib sha256 {actual[:12]}... does not match config.json "
                f"{config['model_sha256'][:12]}...; re-run train_export.py"
            )
        return cls(config, joblib.load(path))

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        # Serving rows are unlabelled out-of-sample data; the classifier guard only
        # needs a non-fit partition label.
        base = self.classifier.predict_proba(frame, partition="validation")
        return apply_calibrator(self.calibration_method, self.calibrator, base)

    def band(self, probability: float) -> str:
        if probability < self.band_low_upper:
            return "Low"
        if probability < self.band_high_lower:
            return "Medium"
        return "High"

    def predict(self, request: PredictRequest) -> dict:
        frame = to_frame(request)
        probability = float(self.predict_frame(frame)[0])
        return {
            "probability": round(probability, 3),
            "risk_band": self.band(probability),
            "threshold_used": self.threshold,
            "predicted_label": POSITIVE_LABEL if probability >= self.threshold else NEGATIVE_LABEL,
            "model_version": self.model_version,
            "frame": frame,
        }


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor.load()
