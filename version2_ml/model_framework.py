"""IPD V2 Phase 3 Step 5: train-only binary classification and aggregate evaluation.

No raw records are exported. This module is for development/testing and does
not establish inferential or independence validity for the final test set.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
try:
    from .preprocessing import TrainOnlyPreprocessor
except ImportError:
    from preprocessing import TrainOnlyPreprocessor


def make_estimator(name: str, *, seed: int = 42):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1500, random_state=seed, solver="liblinear")
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install xgboost to use this optional model") from exc
        return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                             eval_metric="logloss", random_state=seed, n_jobs=1)
    if name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("Install catboost to use this optional model") from exc
        return CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05,
                                  verbose=False, random_seed=seed, thread_count=1)
    raise ValueError("Unknown algorithm")


def _binary(y):
    vals = pd.to_numeric(pd.Series(y), errors="coerce")
    if vals.isna().any() or not vals.isin([0, 1]).all():
        raise ValueError("Binary outcome must contain nonmissing 0/1 only")
    return vals.to_numpy(dtype=int)


def _weights(weights, size):
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != size or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Weights must be finite, strictly positive and aligned")
    return w


class AuthorizedClassifier:
    def __init__(self, feature_manifest, algorithm="logistic_regression", seed=42):
        self.preprocessor = TrainOnlyPreprocessor(feature_manifest)
        self.estimator = make_estimator(algorithm, seed=seed)
        self.is_fitted = False

    def fit(self, frame, y, *, partition: str, sample_weight=None):
        if partition != "train":
            raise ValueError("LEAKAGE: only train partition is authorized for fit")
        labels = _binary(y)
        if len(frame) != len(labels):
            raise ValueError("Feature/outcome length mismatch")
        if len(np.unique(labels)) < 2:
            raise ValueError("Training data must have both outcome classes")
        w = _weights(sample_weight, len(labels))
        x = self.preprocessor.fit_transform_train(frame)
        self.estimator = clone(self.estimator)
        if w is None:
            self.estimator.fit(x, labels)
        else:
            self.estimator.fit(x, labels, sample_weight=w)
        self.is_fitted = True
        return self

    def predict_proba(self, frame, *, partition: str):
        if partition not in {"train", "validation", "test"}:
            raise ValueError("Specify an authorized partition label")
        if not self.is_fitted:
            raise ValueError("Fit on train before prediction")
        x = self.preprocessor.transform(frame)
        pred = np.asarray(self.estimator.predict_proba(x)[:, 1], dtype=float)
        if not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
            raise ValueError("Invalid predicted probabilities")
        return pred


def evaluate_binary(y_true, probabilities, *, sample_weight=None, threshold=None):
    """Aggregated point metrics; no bootstrap CI, no survey design claim.

    Calibration slope/intercept deliberately deferred until a prespecified,
    reviewed method is implemented. Outputs are intended for exploratory use.
    """
    y = _binary(y_true)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) != len(y) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities or row alignment")
    if len(np.unique(y)) < 2:
        raise ValueError("Both outcome classes needed for AUROC")
    w = _weights(sample_weight, len(y))
    result = {
        "n": int(len(y)), "positive_n": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "average_precision": float(average_precision_score(y, p, sample_weight=w)),
        "brier_score": float(brier_score_loss(y, p, sample_weight=w)),
        "weighted_outcome_prevalence": float(np.average(y, weights=w)),
        "calibration_intercept": None, "calibration_slope": None,
        "inference_validated": False,
        "warning": "Point estimates only; no design-valid intervals; confirmatory final-test independence unverified.",
    }
    if threshold is not None:
        t = float(threshold)
        if not 0 < t < 1:
            raise ValueError("Threshold must be strictly between 0 and 1")
        predicted = p >= t
        tp = float(np.sum((y == 1) * predicted * (w if w is not None else 1)))
        fp = float(np.sum((y == 0) * predicted * (w if w is not None else 1)))
        fn = float(np.sum((y == 1) * (~predicted) * (w if w is not None else 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result.update(threshold=t, precision=precision, recall=recall,
                      f1=(2 * precision * recall / (precision + recall)
                          if precision + recall else 0.0))
    return result
