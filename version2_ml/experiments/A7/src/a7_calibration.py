"""A7 calibration utilities.

Platt scaling is fitted on the log-odds of the base model's probabilities using
the held-out grouped calibration subset of the training pool; isotonic
regression is fitted directly on probabilities. Calibration labels are never
evaluation labels. Selection between uncalibrated / Platt / isotonic uses
validation Brier only and is recorded.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
METHODS = ("uncalibrated", "platt", "isotonic")
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 4)


def to_logit(probabilities) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def fit_calibrator(method: str, probabilities, y):
    if method == "uncalibrated":
        return None
    p = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y, dtype=int)
    if len(p) != len(labels):
        raise ValueError("Calibration length mismatch")
    if len(np.unique(labels)) < 2:
        raise ValueError("Calibration subset must contain both outcome classes")
    if method == "platt":
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(to_logit(p).reshape(-1, 1), labels)
        return model
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(p, labels)
        return model
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(method: str, calibrator, probabilities) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if method == "uncalibrated":
        out = p
    elif method == "platt":
        out = calibrator.predict_proba(to_logit(p).reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        out = calibrator.predict(p)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    out = np.clip(np.asarray(out, dtype=float), 0.0, 1.0)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite calibrated probabilities")
    return out


def calibration_intercept_slope(y_true, probabilities) -> tuple[float, float]:
    labels = np.asarray(y_true, dtype=int)
    if len(np.unique(labels)) < 2:
        raise ValueError("Calibration slope requires both outcome classes")
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(to_logit(probabilities).reshape(-1, 1), labels)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def _confusion(y, predicted):
    tp = float(np.sum((y == 1) & predicted))
    fp = float(np.sum((y == 0) & predicted))
    fn = float(np.sum((y == 1) & ~predicted))
    tn = float(np.sum((y == 0) & ~predicted))
    return tp, fp, fn, tn


def f1_at_threshold(y, probabilities, threshold: float) -> float:
    predicted = np.asarray(probabilities, dtype=float) >= float(threshold)
    tp, fp, fn, _ = _confusion(np.asarray(y, dtype=int), predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def threshold_f1_optimal(y, probabilities) -> tuple[float, float]:
    best = (None, -1.0)
    for threshold in THRESHOLD_GRID:
        score = f1_at_threshold(y, probabilities, threshold)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def threshold_youden(y, probabilities) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    best = (None, -1.0)
    for threshold in THRESHOLD_GRID:
        predicted = p >= threshold
        tp, fp, fn, tn = _confusion(y, predicted)
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        score = sensitivity + specificity - 1.0
        if score > best[1]:
            best = (float(threshold), float(score))
    if best[0] is None:
        raise ValueError("Youden threshold not identifiable")
    return best[0]


def threshold_for_recall(y, probabilities, min_recall: float = 0.8):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    candidate = None
    for threshold in THRESHOLD_GRID:
        predicted = p >= threshold
        tp, fp, fn, _ = _confusion(y, predicted)
        recall = tp / (tp + fn) if tp + fn else 0.0
        if recall >= min_recall:
            candidate = float(threshold)
    return candidate


def reliability_points(y_true, probabilities, n_bins: int = 10):
    from sklearn.calibration import calibration_curve

    fraction, mean_predicted = calibration_curve(
        np.asarray(y_true, dtype=int),
        np.asarray(probabilities, dtype=float),
        n_bins=n_bins,
        strategy="quantile",
    )
    return mean_predicted, fraction
