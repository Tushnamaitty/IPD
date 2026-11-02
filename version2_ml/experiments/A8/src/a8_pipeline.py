"""A8 per-seed matched evaluation: every arm and model on identical rows.

For one mother-grouped split: fit pool / calibration carve / validation rows
are fixed once; the state x sector encoder is fitted on fit-pool labels; each
(model, arm) is fitted on the fit pool, Platt-calibrated on the carve (A7
recipe) and scored on validation only. Nothing here touches test rows: the
caller passes only fit, calibration and validation positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from a7_calibration import apply_calibrator, calibration_intercept_slope, fit_calibrator, threshold_f1_optimal
from a7_candidates import scale_pos_weight
from a8_arms import ARM_ORDER, arm_frame, encode_state_sector, fit_arm, te_key_frame
from exp_metrics import binary_metrics

CALIBRATION_METHOD = "platt"
METRIC_KEYS = ("roc_auc", "average_precision", "brier_score", "calibration_intercept",
               "calibration_slope", "threshold", "precision", "recall", "f1")


def _metrics(y_valid, p_valid) -> dict:
    threshold, best_f1 = threshold_f1_optimal(y_valid, p_valid)
    point = binary_metrics(y_valid, p_valid, threshold=threshold)
    intercept, slope = calibration_intercept_slope(y_valid, p_valid)
    return {
        "n_births": int(point["n"]),
        "n_events": int(point["positive_n"]),
        "prevalence": float(point["weighted_outcome_prevalence"]),
        "roc_auc": float(point["roc_auc"]),
        "average_precision": float(point["average_precision"]),
        "brier_score": float(point["brier_score"]),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "threshold": float(threshold),
        "precision": float(point["precision"]),
        "recall": float(point["recall"]),
        "f1": float(best_f1),
    }


def run_seed(base: pd.DataFrame, state_labels, state_codes, y, groups, *, fit_positions,
             calib_positions, valid_positions, models, params: dict, te_config: dict,
             estimator_seed: int, arms=ARM_ORDER) -> dict:
    """Fit and evaluate every (model, arm) for one split.

    ``base`` is the F1_detail_history frame for all cohort rows; the positional
    arrays select this split's fit pool, calibration carve and validation rows.
    Returns metric rows, in-memory validation probabilities, fitted
    classifiers/calibrators and the encoder digest. Nothing is written.
    """
    fit_positions, calib_positions, valid_positions = (
        np.asarray(p, dtype=int) for p in (fit_positions, calib_positions, valid_positions)
    )
    for a, b in ((fit_positions, calib_positions), (fit_positions, valid_positions), (calib_positions, valid_positions)):
        if np.intersect1d(a, b).size:
            raise ValueError("Fit, calibration and validation positions overlap")
    rows = np.concatenate([fit_positions, calib_positions, valid_positions])
    n_fit, n_calib = len(fit_positions), len(calib_positions)
    fit_mask = np.zeros(len(rows), dtype=bool)
    fit_mask[:n_fit] = True
    calib_slice, valid_slice = slice(n_fit, n_fit + n_calib), slice(n_fit + n_calib, None)

    y_rows = np.asarray(y, dtype=int)[rows]
    groups_rows = np.asarray(groups)[rows]
    sub = base.iloc[rows].reset_index(drop=True)
    states = np.asarray(state_labels)[rows]
    keys = te_key_frame(np.asarray(state_codes)[rows], sub["facility_type"], sub.index)
    encoder, te_values = encode_state_sector(
        keys, y_rows, groups_rows, fit_mask, alpha=te_config["alpha"], n_min=te_config["n_min"],
        n_splits=te_config["inner_folds"], seed=te_config["inner_seed"],
    )
    y_fit, y_calib, y_valid = y_rows[:n_fit], y_rows[calib_slice], y_rows[valid_slice]

    results = {"rows": [], "probabilities": {}, "classifiers": {}, "calibrators": {},
               "frames": {}, "te_digest": encoder.full_table.digest(), "te_values": te_values}
    for model in models:
        run_params = dict(params[model])
        if model in ("xgboost", "catboost"):
            run_params["scale_pos_weight"] = scale_pos_weight(y_fit)
        for arm in arms:
            frame = arm_frame(sub, states, te_values, arm)
            classifier = fit_arm(model, arm, run_params, frame.iloc[:n_fit].reset_index(drop=True),
                                 y_fit, estimator_seed)
            p_calib_base = classifier.predict_proba(frame.iloc[calib_slice].reset_index(drop=True), partition="train")
            calibrator = fit_calibrator(CALIBRATION_METHOD, p_calib_base, y_calib)
            valid_frame = frame.iloc[valid_slice].reset_index(drop=True)
            p_valid_base = classifier.predict_proba(valid_frame, partition="validation")
            p_valid = apply_calibrator(CALIBRATION_METHOD, calibrator, p_valid_base)
            results["rows"].append({"model": model, "arm": arm, **_metrics(y_valid, p_valid)})
            results["probabilities"][(model, arm)] = {"base": p_valid_base, "calibrated": p_valid}
            results["classifiers"][(model, arm)] = classifier
            results["calibrators"][(model, arm)] = calibrator
            results["frames"][arm] = valid_frame
    results["n_mothers_valid"] = int(pd.Series(groups_rows[valid_slice]).nunique())
    return results


def paired_deltas(metrics: pd.DataFrame, *, comparator: str = "H0") -> pd.DataFrame:
    """Arm minus comparator for every (seed, model) on identical validation rows."""
    required = {"seed", "model", "arm", "valid_digest", *METRIC_KEYS}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics frame missing columns: {sorted(missing)}")
    out = []
    for (seed, model), block in metrics.groupby(["seed", "model"], sort=True):
        if block["valid_digest"].nunique() != 1:
            raise ValueError(f"MATCHED-ROW VIOLATION: seed {seed}, {model}")
        reference = block.loc[block["arm"] == comparator]
        if len(reference) != 1:
            raise ValueError(f"Comparator {comparator} missing for seed {seed}, {model}")
        reference = reference.iloc[0]
        for row in block.itertuples(index=False):
            if row.arm == comparator:
                continue
            out.append({
                "seed": int(seed), "model": model, "arm": row.arm, "comparator": comparator,
                **{f"delta_{key}": float(getattr(row, key) - reference[key]) for key in METRIC_KEYS},
            })
    return pd.DataFrame(out)
