"""Shared aggregate evaluation helpers: point metrics, calibration, threshold.

Wraps model_framework.evaluate_binary and adds logistic calibration
intercept (slope fixed at 1, logit offset) and slope. Point estimates only;
no confidence intervals (survey-design estimator NOT approved).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "version2_ml") not in sys.path:
    sys.path.insert(0, str(ROOT / "version2_ml"))
from model_framework import evaluate_binary  # noqa: E402

_EPS = 1e-12


def _newton(X, y, w, offset, iterations=100):
    beta = np.zeros(X.shape[1])
    for _ in range(iterations):
        mu = expit(X @ beta + offset)
        grad = X.T @ (w * (y - mu))
        hess = X.T @ (X * (w * mu * (1 - mu))[:, None])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            return beta
    return None


def calibration_intercept_slope(y, p, sample_weight=None) -> dict:
    """Logistic recalibration diagnostics; None when not identifiable/convergent."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("Unequal or non-1D calibration inputs")
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    lp = logit(np.clip(p, _EPS, 1 - _EPS))
    out = {"calibration_intercept": None, "calibration_slope": None}
    if len(np.unique(y)) < 2 or np.std(lp) < 1e-12:
        return out
    a = _newton(np.ones((len(y), 1)), y, w, lp)
    b = _newton(np.column_stack([np.ones_like(lp), lp]), y, w, np.zeros_like(lp))
    if a is not None:
        out["calibration_intercept"] = float(a[0])
    if b is not None:
        out["calibration_slope"] = float(b[1])
    return out


def select_threshold_max_f1(y, p) -> float:
    """Choose the F1-maximizing threshold. Call ONLY on permitted validation rows."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError("Unequal threshold-selection arrays")
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, _EPS), 0)
    best = int(np.argmax(f1[:-1]))
    return float(np.clip(thresholds[best], 1e-9, 1 - 1e-9))


def binary_metrics(y, p, *, threshold=None, sample_weight=None) -> dict:
    result = evaluate_binary(y, p, sample_weight=sample_weight, threshold=threshold)
    result.update(calibration_intercept_slope(y, p, sample_weight))
    return result
