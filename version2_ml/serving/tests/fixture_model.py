"""Synthetic stand-in for the exported model, used when NFHS5_PATH is not set.

Builds a small model with the same pipeline as train_export.py
(make_core_features + facility_detail + fit_candidate + Platt) on synthetic
rows that cover exactly the vocabulary the real model was trained on, and
writes model.joblib / config.json / metrics.json in the real format. No NFHS
data is read. The synthetic outcome makes an observed previous C-section and
private facilities raise risk, so behavioural tests have a known direction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from serving import train_export as te

from a4_history import (  # noqa: E402
    FIRST_RECORDED_LIVE_BIRTH,
    HAS_ANY_FALSE,
    HAS_ANY_TRUE,
    HAS_ANY_UNKNOWN,
    NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
    OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
    QUALITY_AMBIGUOUS,
    QUALITY_OK,
    UNKNOWN_HISTORY,
    UNOBSERVED_HISTORY,
)

N_ROWS = 4000
SEED = 42
# m15 codes present in the real fit pool (20 and 30 never occur there).
M15_CODES = (21, 22, 23, 24, 25, 26, 27, 31, 32, 33, 96)
RELIGION_CODES = (1, 2, 3, 4, 5, 6, 8, 9, 96)
TWIN_CODES = (0, 1, 2, 3)
# Value labels as read from the IABR7EFL.DTA metadata by train_export._dta_value_labels.
DTA_LABELS = {
    "religion": {"1": "hindu", "2": "muslim", "3": "christian", "4": "sikh",
                 "5": "buddhist / neo-buddhist", "6": "jain", "7": "jewish",
                 "8": "parsi / zoroastrian", "9": "no religion", "96": "other"},
    "twin_order": {"0": "single birth", "1": "1st of multiple", "2": "2nd of multiple",
                   "3": "3rd of multiple", "4": "4th of multiple", "5": "5th of multiple"},
    "wealth_index": {"1": "poorest", "2": "poorer", "3": "middle", "4": "richer", "5": "richest"},
}
HISTORY_EFFECT = {
    FIRST_RECORDED_LIVE_BIRTH: 0.3,
    OBSERVED_PREVIOUS_INSTITUTIONAL_CS: 3.0,
    NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS: -1.0,
    UNOBSERVED_HISTORY: -0.5,
    UNKNOWN_HISTORY: 0.0,
}


def _synthetic(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    bord = np.where(rng.random(N_ROWS) < 0.4, 1, rng.integers(2, 9, N_ROWS))
    raw = pd.DataFrame({
        "b3": rng.integers(15 * 12, 50 * 12, N_ROWS),
        "v011": 0,
        "v133": rng.integers(0, 21, N_ROWS),
        "v190": rng.integers(1, 6, N_ROWS),
        "bord": bord,
        "m15": rng.choice(M15_CODES, N_ROWS),
        "v025": rng.integers(1, 3, N_ROWS),
        "v130": rng.choice(RELIGION_CODES, N_ROWS),
        "b0": rng.choice(TWIN_CODES, N_ROWS, p=(0.85, 0.06, 0.06, 0.03)),
    })
    higher = rng.choice(
        [OBSERVED_PREVIOUS_INSTITUTIONAL_CS, NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
         UNOBSERVED_HISTORY, UNKNOWN_HISTORY],
        N_ROWS, p=(0.15, 0.35, 0.47, 0.03),
    )
    level = np.where(bord == 1, FIRST_RECORDED_LIVE_BIRTH, higher)
    has_any = np.select(
        [np.isin(level, [OBSERVED_PREVIOUS_INSTITUTIONAL_CS, NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS]),
         level == UNKNOWN_HISTORY],
        [HAS_ANY_TRUE, HAS_ANY_UNKNOWN], default=HAS_ANY_FALSE,
    )
    quality = np.where(level == UNKNOWN_HISTORY, QUALITY_AMBIGUOUS, QUALITY_OK)
    history = pd.DataFrame({
        "history_four_level": level,
        "has_any_observed_earlier_institutional_birth": has_any,
        "history_quality_flag": quality,
    }, dtype="string")
    core = te.make_core_features(raw, wave="NFHS-5")
    frame = te.run_a7._tier_frame(core, te.facility_detail(raw["m15"]), history, te.TIER)
    logit = (
        -1.6
        + 1.4 * (core["facility_type"] == "private").to_numpy()
        + pd.Series(level).map(HISTORY_EFFECT).to_numpy()
        + 0.15 * (raw["v190"].to_numpy() - 3)
    )
    y = (rng.random(N_ROWS) < 1 / (1 + np.exp(-logit))).astype(int)
    return frame, y


def build_fixture_model(out_dir: Path) -> Path:
    rng = np.random.default_rng(SEED)
    frame, y = _synthetic(rng)
    order = rng.permutation(N_ROWS)
    fit, calib, valid = np.split(order, [int(0.6 * N_ROWS), int(0.8 * N_ROWS)])
    params = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8,
              "colsample_bytree": 0.8, "eval_metric": "logloss", "tree_method": "hist",
              "scale_pos_weight": te.scale_pos_weight(y[fit])}
    fit_frame = frame.iloc[fit].reset_index(drop=True)
    classifier = te.fit_candidate(te.MODEL, te.TIER, params, fit_frame, y[fit], SEED)
    calibrator = te.fit_calibrator(
        te.CALIBRATION_METHOD, te.predict_candidate(classifier, frame.iloc[calib].reset_index(drop=True), "train"), y[calib]
    )
    p_valid = te.apply_calibrator(
        te.CALIBRATION_METHOD, calibrator,
        te.predict_candidate(classifier, frame.iloc[valid].reset_index(drop=True), "validation"),
    )
    metrics = te.evaluate_binary(y[valid], p_valid, threshold=0.5)
    threshold_f1, _ = te.threshold_f1_optimal(y[valid], p_valid)
    threshold_rec = te.threshold_for_recall(y[valid], p_valid, 0.8)
    cut_low, cut_high = (float(v) for v in np.percentile(p_valid, te.RISK_BAND_PERCENTILES))

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump({"classifier": classifier, "calibrator": calibrator,
                 "calibration_method": te.CALIBRATION_METHOD, "feature_order": list(te.FEATURE_ORDER)}, model_path)
    vocabulary = te._encoder_vocabulary(classifier)
    labels = te._display_labels(vocabulary, DTA_LABELS)
    labels["wealth_index"] = DTA_LABELS["wealth_index"]
    version = f"{te.MODEL_VERSION_PREFIX}-fixture"
    config = {
        "model_version": version,
        "model": te.MODEL,
        "feature_tier": te.TIER,
        "calibration_method": te.CALIBRATION_METHOD,
        "params": params,
        "feature_order": list(te.FEATURE_ORDER),
        "numerical_features": list(te.NUMERICAL),
        "categorical_features": list(te.CATEGORICAL),
        "thresholds": {"f1_optimal": threshold_f1, "recall_80": threshold_rec, "default": "f1_optimal"},
        "risk_bands": {"basis": "fixture", "low_upper": cut_low, "high_lower": cut_high,
                       "labels": ["Low", "Medium", "High"]},
        "vocabulary": vocabulary,
        "display_labels": labels,
        "numeric_ranges": te._numeric_ranges(fit_frame),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "provenance": {"source": "synthetic fixture (tests only)", "test_partition_scored": False},
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({
        "model_version": version,
        "partition": "synthetic validation",
        "n_births": int(metrics["n"]),
        "prevalence": float(metrics["weighted_outcome_prevalence"]),
        "roc_auc": float(metrics["roc_auc"]),
        "average_precision": float(metrics["average_precision"]),
        "brier_score": float(metrics["brier_score"]),
    }, indent=2), encoding="utf-8")
    return out_dir
