"""A9 model registry: V2, V2_LR, F1_core, V1_spec (PROTOCOL.md section 5).

Each model is fitted on a seed's fit pool and Platt-calibrated on the
PSU-grouped carve, with the A7 recipe. Fitting and threshold selection never
see test rows; test probabilities are produced only by ``predict_test``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from a7_calibration import apply_calibrator, fit_calibrator, threshold_f1_optimal, threshold_for_recall
from a7_candidates import estimator_factory, scale_pos_weight, tier_manifest
from a8_arms import arm_frame, arm_manifest
from a9_v1 import F0_MANIFEST
from model_framework import AuthorizedClassifier

FINAL_ARM = "H1"
MODEL_ORDER = ("V2", "V2_LR", "F1_core", "V1_spec")
FEATURE_LABELS = {
    "V2": "F1_detail_history + state_ut (FINAL_TIER, 13)",
    "V2_LR": "F1_detail_history + state_ut (FINAL_TIER, 13)",
    "F1_core": "8 core features (F1)",
    "V1_spec": "V1 F0_historical_36 (incl. BMI, ANC)",
}


def model_frames(base: pd.DataFrame, state_labels, f0: pd.DataFrame | None) -> dict:
    """Feature frame and manifest per model, all row-aligned with the cohort."""
    final = arm_frame(base, state_labels, None, FINAL_ARM)
    f1 = tier_manifest("F1")
    frames = {
        "V2": (final, arm_manifest(FINAL_ARM)),
        "V2_LR": (final, arm_manifest(FINAL_ARM)),
        "F1_core": (base[list(f1.numerical) + list(f1.categorical)], f1),
    }
    if f0 is not None:
        if len(f0) != len(base):
            raise ValueError("F0 frame is not row-aligned with the cohort")
        frames["V1_spec"] = (f0, F0_MANIFEST)
    return frames


def fit_model(name: str, frame: pd.DataFrame, manifest, algorithm: str, params: dict, *, fit_positions,
              calib_positions, y, seed: int) -> dict:
    y = np.asarray(y, dtype=int)
    y_fit = y[fit_positions]
    run_params = dict(params)
    if algorithm == "xgboost":
        run_params["scale_pos_weight"] = scale_pos_weight(y_fit)
    classifier = AuthorizedClassifier(manifest, algorithm=algorithm, seed=seed)
    classifier.estimator = estimator_factory(algorithm, run_params, seed)
    classifier.fit(frame.iloc[fit_positions].reset_index(drop=True), y_fit, partition="train")
    p_calib = classifier.predict_proba(frame.iloc[calib_positions].reset_index(drop=True), partition="train")
    calibrator = fit_calibrator("platt", p_calib, y[calib_positions])
    return {"name": name, "classifier": classifier, "calibrator": calibrator, "params": run_params}


def predict(model: dict, frame: pd.DataFrame, positions, partition: str) -> np.ndarray:
    base = model["classifier"].predict_proba(frame.iloc[positions].reset_index(drop=True), partition=partition)
    return apply_calibrator("platt", model["calibrator"], base)


def validation_thresholds(y_valid, p_valid) -> dict:
    f1_threshold, f1_value = threshold_f1_optimal(y_valid, p_valid)
    recall_threshold = threshold_for_recall(y_valid, p_valid, 0.8)
    if recall_threshold is None:
        raise RuntimeError("Recall >= 0.80 not attainable on validation")
    return {"f1_optimal": float(f1_threshold), "f1_optimal_validation_f1": float(f1_value),
            "recall_80": float(recall_threshold)}


def fit_seed(frames: dict, specs: dict, params: dict, *, fit_positions, calib_positions, valid_positions,
             y, seed: int) -> dict:
    """Fit every model, calibrate, and fix thresholds on validation. No test rows used."""
    y = np.asarray(y, dtype=int)
    out = {}
    for name in MODEL_ORDER:
        if name not in frames:
            continue
        frame, manifest = frames[name]
        spec = specs[name]
        model = fit_model(name, frame, manifest, spec["algorithm"], params[spec["params"]],
                          fit_positions=fit_positions, calib_positions=calib_positions, y=y, seed=seed)
        p_valid = predict(model, frame, valid_positions, "validation")
        model["thresholds"] = validation_thresholds(y[valid_positions], p_valid)
        model["p_valid"] = p_valid
        out[name] = model
    return out


def predict_test(models: dict, frames: dict, test_positions) -> dict:
    """The only place test probabilities are produced (guarded by the caller)."""
    return {name: predict(model, frames[name][0], test_positions, "test") for name, model in models.items()}
