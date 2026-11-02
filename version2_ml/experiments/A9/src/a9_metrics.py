"""A9 test metrics, PSU cluster bootstrap, curves and calibration bins (PROTOCOL.md 7-8).

Only aggregates leave this module: metric values, replicate metric values,
curve points on fixed grids and suppressed bin summaries.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from a7_calibration import calibration_intercept_slope
from exp_metrics import binary_metrics
from exp_provenance import suppress_small_cells

METRICS = (
    "roc_auc", "average_precision", "brier_score", "calibration_intercept", "calibration_slope",
    "precision_f1opt", "recall_f1opt", "f1_f1opt", "precision_rec80", "recall_rec80", "f1_rec80",
)
GRID = np.linspace(0.0, 1.0, 101)


def _at(y, p, threshold):
    predicted = p >= threshold
    tp = float(np.sum((y == 1) & predicted))
    fp = float(np.sum((y == 0) & predicted))
    fn = float(np.sum((y == 1) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def test_metrics(y, p, thresholds: dict) -> dict:
    """All PROTOCOL section-7 metrics; NaN everywhere if only one class is present."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return {key: np.nan for key in METRICS}
    point = binary_metrics(y, p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        intercept, slope = calibration_intercept_slope(y, p)
    out = {"roc_auc": float(point["roc_auc"]), "average_precision": float(point["average_precision"]),
           "brier_score": float(point["brier_score"]), "calibration_intercept": float(intercept),
           "calibration_slope": float(slope)}
    for suffix, key in (("f1opt", "f1_optimal"), ("rec80", "recall_80")):
        precision, recall, f1 = _at(y, p, thresholds[key])
        out.update({f"precision_{suffix}": precision, f"recall_{suffix}": recall, f"f1_{suffix}": f1})
    return out


def psu_bootstrap(y, probabilities: dict, psu, thresholds: dict, *, replicates: int, seed: int) -> dict:
    """Paired PSU cluster bootstrap: the same draws for every model.

    Returns {model: {metric: array(replicates)}}; NaN where a replicate had a
    single outcome class.
    """
    y = np.asarray(y, dtype=int)
    psu = np.asarray(psu)
    codes, inverse = np.unique(psu, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(codes)))
    ends = np.append(starts[1:], len(order))
    rows_of = [order[s:e] for s, e in zip(starts, ends)]
    rng = np.random.default_rng(seed)
    out = {name: {key: np.empty(replicates) for key in METRICS} for name in probabilities}
    for b in range(replicates):
        draw = rng.integers(0, len(codes), len(codes))
        idx = np.concatenate([rows_of[k] for k in draw])
        for name, p in probabilities.items():
            values = test_metrics(y[idx], np.asarray(p)[idx], thresholds[name])
            for key in METRICS:
                out[name][key][b] = values[key]
    return out


def percentile_ci(values) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return np.nan, np.nan, 0
    low, high = np.percentile(valid, [2.5, 97.5])
    return float(low), float(high), int(len(valid))


def curves(y, p) -> dict:
    """ROC and PR curves interpolated on a fixed 101-point grid (aggregate only)."""
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    interp_precision = [float(np.max(precision[recall >= r])) if np.any(recall >= r) else 0.0 for r in GRID]
    return {"grid": GRID.round(2).tolist(), "tpr_at_fpr": np.interp(GRID, fpr, tpr).tolist(),
            "precision_at_recall": interp_precision}


def calibration_bins(y, p, *, n_bins: int, min_cell: int) -> pd.DataFrame:
    """Quantile bins of calibrated probability; small cells suppressed (births, events, non-events)."""
    frame = pd.DataFrame({"y": np.asarray(y, dtype=int), "p": np.asarray(p, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False) + 1
    table = frame.groupby("bin").agg(n_births=("y", "size"), n_events=("y", "sum"),
                                     mean_predicted=("p", "mean"), observed_rate=("y", "mean")).reset_index()
    return suppress_small_cells(table, n_col="n_births", event_col="n_events", min_n=min_cell,
                                min_events=min_cell, value_cols=("mean_predicted", "observed_rate"))


def weighted_metrics(y, p, weights) -> dict:
    point = binary_metrics(y, p, sample_weight=np.asarray(weights, dtype=float))
    return {"roc_auc": float(point["roc_auc"]), "average_precision": float(point["average_precision"]),
            "brier_score": float(point["brier_score"]),
            "weighted_prevalence": float(point["weighted_outcome_prevalence"])}
