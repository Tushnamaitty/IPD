"""A7 candidate registry, grouped-CV tuning and fitting.

Candidates share identical training/validation rows and folds. Tuning runs
entirely inside the training pool with mother-grouped StratifiedGroupKFold;
evaluation labels are never an input to any selection function here.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from a4_facility_detail import DETAIL_FEATURE
from a4_history import HAS_ANY_FEATURE, HISTORY_FEATURE, QUALITY_FEATURE
from feature_builder import CORE_NAMES
from model_framework import AuthorizedClassifier
from preprocessing import FeatureManifest

F1_NUMERICAL = CORE_NAMES[:4]
F1_CATEGORICAL = CORE_NAMES[4:]
HISTORY_FEATURE_NAMES = (HISTORY_FEATURE, HAS_ANY_FEATURE, QUALITY_FEATURE)

TIERS = {
    "F1": {
        "manifest_tier": "F1",
        "numerical": F1_NUMERICAL,
        "categorical": F1_CATEGORICAL,
    },
    "F1_detail_history": {
        "manifest_tier": "F2",
        "numerical": F1_NUMERICAL,
        "categorical": F1_CATEGORICAL + (DETAIL_FEATURE,) + HISTORY_FEATURE_NAMES,
    },
}
MODELS = ("logistic_regression", "xgboost", "catboost")


def tier_manifest(tier: str) -> FeatureManifest:
    spec = TIERS[tier]
    return FeatureManifest(spec["numerical"], spec["categorical"], spec["manifest_tier"])


def build_tier_frame(core: pd.DataFrame, detail, history_feats, tier: str) -> pd.DataFrame:
    spec = TIERS[tier]
    frame = core.copy()
    if DETAIL_FEATURE in spec["categorical"]:
        frame[DETAIL_FEATURE] = detail.to_numpy()
    for column in HISTORY_FEATURE_NAMES:
        if column in spec["categorical"]:
            frame[column] = history_feats[column].to_numpy()
    missing = sorted(set(spec["categorical"]) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Tier frame missing columns: {missing}")
    return frame


def scale_pos_weight(y_train) -> float:
    labels = np.asarray(y_train, dtype=int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Training partition lost an outcome class")
    return float(negatives) / float(positives)


def carve_calibration_partition(
    frame: pd.DataFrame,
    *,
    group_col: str = "caseid",
    target_col: str = "m17",
    calibration_fraction: float = 0.2,
    seed: int = 4208,
) -> pd.Series:
    """Deterministic mother-exclusive two-way carve: fit vs calibration rows.

    Mirrors the grouped hash-bucket scheme of the frozen split so calibrator
    fitting never touches validation/test rows.
    """
    if not 0 < calibration_fraction < 1:
        raise ValueError("Calibration fraction must be between 0 and 1")
    group = frame[group_col].astype("string")
    if group.isna().any() or group.str.strip().eq("").any():
        raise ValueError("Missing or blank group identifier")
    target = pd.to_numeric(frame[target_col], errors="coerce")
    if not target.isin([0, 1]).all():
        raise ValueError("Target must contain only nonmissing 0/1 values")
    stats = pd.DataFrame(
        {"group": group.to_numpy(), "y": target.astype(int).to_numpy()}
    )
    aggregated = stats.groupby("group", sort=True)["y"].agg(["mean", "size"])
    if len(aggregated) < 2:
        raise ValueError("At least two distinct mothers required for a carve")

    def h(token):
        digest = hashlib.sha256(f"{seed}|{token}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    aggregated["bucket"] = (aggregated["mean"] >= 0.5).astype(int)
    assignments = {}
    for _, subset in aggregated.groupby("bucket"):
        order = sorted(subset.index.tolist(), key=h)
        n_calibration = int(round(calibration_fraction * len(order)))
        for key in order[:n_calibration]:
            assignments[str(key)] = "calibration"
        for key in order[n_calibration:]:
            assignments[str(key)] = "fit"
    result = group.map(assignments)
    if result.isna().any():
        raise RuntimeError("Unassigned group in calibration carve")
    audit = pd.DataFrame({"group": group.to_numpy(), "part": result.to_numpy()})
    if audit.groupby("group")["part"].nunique().gt(1).any():
        raise ValueError("LEAKAGE: a mother appears in both carve partitions")
    labels = target.to_numpy()
    for label in ("fit", "calibration"):
        mask = result.eq(label).to_numpy()
        positives = int(labels[mask].sum())
        if mask.sum() == 0 or positives == 0 or (int(mask.sum()) - positives) == 0:
            raise ValueError(f"{label}: empty carve partition or outcome class")
    return pd.Series(result.to_numpy(), index=frame.index, name="calibration_carve")


def psu_overlap_summary(train_psu, valid_psu) -> dict:
    train = pd.Series(train_psu).dropna().astype("string")
    valid = pd.Series(valid_psu).dropna().astype("string")
    train_set = set(train.unique())
    valid_set = set(valid.unique())
    shared = train_set & valid_set
    share = float(valid.isin(shared).mean()) if len(valid) else 0.0
    return {
        "n_train_psus": len(train_set),
        "n_valid_psus": len(valid_set),
        "n_shared_psus": len(shared),
        "share_validation_rows_in_shared_psus": round(share, 6),
    }


def estimator_factory(model_name: str, params: dict, seed: int):
    params = dict(params)
    if model_name == "logistic_regression":
        params.update({"random_state": seed})
        return LogisticRegression(**params)
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        params.update({"random_state": seed, "n_jobs": 1, "verbosity": 0})
        return XGBClassifier(**params)
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        params.update(
            {
                "random_seed": seed,
                "thread_count": 1,
                "verbose": False,
                "allow_writing_files": False,
            }
        )
        return CatBoostClassifier(**params)
    raise ValueError(f"Unknown model: {model_name}")


def fit_candidate(
    model_name: str,
    tier: str,
    params: dict,
    train_frame: pd.DataFrame,
    y_train,
    seed: int,
) -> AuthorizedClassifier:
    manifest = tier_manifest(tier)
    classifier = AuthorizedClassifier(manifest, algorithm=model_name, seed=seed)
    classifier.estimator = estimator_factory(model_name, params, seed)
    classifier.fit(train_frame, y_train, partition="train")
    return classifier


def predict_candidate(classifier: AuthorizedClassifier, frame: pd.DataFrame, partition: str):
    return classifier.predict_proba(frame, partition=partition)


def _assert_disjoint(groups_train, groups_valid):
    overlap = set(np.asarray(groups_train).tolist()) & set(np.asarray(groups_valid).tolist())
    if overlap:
        raise RuntimeError(
            f"Group leakage inside CV fold: {len(overlap)} shared group(s)"
        )


def tune_candidate(
    model_name: str,
    tier: str,
    train_frame: pd.DataFrame,
    y_train,
    groups,
    *,
    folds: int,
    seed: int,
    grid: list[dict],
) -> dict:
    labels = np.asarray(y_train, dtype=int)
    group_values = np.asarray(groups)
    if len(train_frame) != len(labels) or len(group_values) != len(labels):
        raise ValueError("Tuning inputs are not row-aligned")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = []
    for params in grid:
        fold_scores = []
        for fold_index, (train_idx, valid_idx) in enumerate(
            splitter.split(train_frame, labels, group_values)
        ):
            _assert_disjoint(group_values[train_idx], group_values[valid_idx])
            fold_labels = labels[train_idx]
            valid_labels = labels[valid_idx]
            if len(np.unique(fold_labels)) < 2 or len(np.unique(valid_labels)) < 2:
                raise RuntimeError("A tuning fold lost an outcome class")
            run_params = dict(params)
            if model_name == "xgboost":
                run_params["scale_pos_weight"] = scale_pos_weight(fold_labels)
            elif model_name == "catboost":
                run_params["scale_pos_weight"] = scale_pos_weight(fold_labels)
            classifier = fit_candidate(
                model_name,
                tier,
                run_params,
                train_frame.iloc[train_idx].reset_index(drop=True),
                fold_labels,
                seed + fold_index,
            )
            probabilities = predict_candidate(
                classifier, train_frame.iloc[valid_idx].reset_index(drop=True), "validation"
            )
            fold_scores.append(float(average_precision_score(valid_labels, probabilities)))
        results.append(
            {
                "params": dict(params),
                "fold_average_precision": [round(score, 6) for score in fold_scores],
                "mean_average_precision": round(float(np.mean(fold_scores)), 6),
            }
        )
    best = max(results, key=lambda row: row["mean_average_precision"])
    return {"model": model_name, "tier": tier, "results": results, "best": best["params"]}


def fold_models(
    model_name: str,
    tier: str,
    params: dict,
    train_frame: pd.DataFrame,
    y_train,
    groups,
    *,
    folds: int,
    seed: int,
) -> list[AuthorizedClassifier]:
    labels = np.asarray(y_train, dtype=int)
    group_values = np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    models = []
    for fold_index, (train_idx, valid_idx) in enumerate(
        splitter.split(train_frame, labels, group_values)
    ):
        _assert_disjoint(group_values[train_idx], group_values[valid_idx])
        fold_labels = labels[train_idx]
        run_params = dict(params)
        if model_name in ("xgboost", "catboost"):
            run_params["scale_pos_weight"] = scale_pos_weight(fold_labels)
        classifier = fit_candidate(
            model_name,
            tier,
            run_params,
            train_frame.iloc[train_idx].reset_index(drop=True),
            fold_labels,
            seed + fold_index,
        )
        models.append(classifier)
    return models
