"""A7 CLI runner: matched multi-model comparison, calibration, SHAP, robustness.

One evaluation contract on the frozen C1 cohort and provisional split:
- candidates (LR/XGB/CatBoost on F1 and F1+facility-detail+history) share
  identical fit-pool rows and validation keys;
- tuning runs only inside the training pool with mother-grouped CV;
- calibrators are fitted only on a grouped carve of the training pool;
- the development test partition is scored exactly once (cached guard) and is
  labelled development-test because of documented V1 exposure;
- SHAP family ranks and subgroup robustness are reported as point estimates
  without design-based intervals; positive counts below 10 are suppressed.

Never writes row-level identifiers; local input paths are never committed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, roc_curve

_HERE = Path(__file__).resolve()
_BASE = _HERE.parents[3]
for _entry in (
    _BASE,
    _BASE / "experiments" / "common",
    _BASE / "experiments" / "A4" / "src",
    _BASE / "experiments" / "A1" / "src",
    str(_HERE.parent),
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from a1_state_context import map_states
from a4_facility_detail import facility_detail
from a4_history import build_history_features
from a7_calibration import (
    METHODS,
    apply_calibrator,
    calibration_intercept_slope,
    f1_at_threshold,
    fit_calibrator,
    reliability_points,
    threshold_f1_optimal,
    threshold_for_recall,
    threshold_youden,
)
from a7_candidates import (
    MODELS,
    carve_calibration_partition,
    estimator_factory,
    fit_candidate,
    fold_models,
    predict_candidate,
    psu_overlap_summary,
    scale_pos_weight,
    tier_manifest,
    tune_candidate,
)
from a7_shap import rank_stability, shap_family_table
from aggregate_schema import (
    UNCERTAINTY_STATUS,
    build_row,
    suppress_small_count,
    write_rows,
)
from checksums import membership_digest
from data_access import (
    C1_COHORT_RULE,
    FROZEN_SPLIT_HASH,
    FROZEN_TRAIN_DIGEST,
    FROZEN_VALID_DIGEST,
    assert_frozen_c1,
    build_c1_cohort,
    read_dta_subset,
    repo_state,
    sha256_file,
    software_versions,
)
from feature_builder import F0_BIN, F0_CAT, F0_NAMES, F0_NUM, make_core_features
from manifests import feature_manifest_hash, split_manifest_hash
from model_framework import AuthorizedClassifier, evaluate_binary
from preprocessing import FeatureManifest
from split_contracts import audit_partition, grouped_partition

TIER_ORDER = ("F1", "F1_detail_history")
DESIGN_VALIDATION = "matched_candidates_shared_fit_pool_and_validation_keys"
DESIGN_TEST = "development_test_single_scoring_v1_exposure_disclosed"
DESIGN_F0 = "f0_reconciled_historical36_a7_reference_params"
SUPPRESSION = 10
MIN_SUBGROUP_ROWS = 200
MIN_SUBGROUP_CLASS_ROWS = 30
HISTORY_COLUMNS = (
    "history_four_level",
    "has_any_observed_earlier_institutional_birth",
    "history_quality_flag",
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="A7 matched evaluation runner")
    parser.add_argument("--nfhs5-path", required=True)
    parser.add_argument("--config", default=str(_HERE.parents[1] / "config.yaml"))
    parser.add_argument("--out-dir", default=str(_HERE.parents[1] / "outputs"))
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory holding a7_tuning.json and a7_test_scored.json "
             "(default: <out-dir>/private)",
    )
    parser.add_argument(
        "--allow-test-score",
        action="store_true",
        help="Permit scoring the development test partition when no test cache exists",
    )
    parser.add_argument("--f0-parquet", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--force-retune", action="store_true")
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = (
        "split", "models", "tuning", "candidates", "evaluation",
        "disclosure", "expected_cohort", "shap", "robustness",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config missing sections: {missing}")
    if tuple(config["candidates"]["tiers"]) != TIER_ORDER:
        raise ValueError("Config candidate tiers differ from implemented tiers")
    if tuple(config["candidates"]["models"]) != MODELS:
        raise ValueError("Config candidate models differ from implemented models")
    return config


def _key_tokens(cohort: pd.DataFrame) -> np.ndarray:
    return (
        cohort["caseid"].astype("string")
        + "|"
        + cohort["bidx"].map(lambda value: str(int(value))).astype("string")
    ).to_numpy()


def _masked_counts(mapping: dict, threshold: int = SUPPRESSION) -> dict:
    return {
        str(key): suppress_small_count(int(value), threshold=threshold)
        for key, value in mapping.items()
    }


def _metrics_payload(metrics: dict, intercept, slope) -> dict:
    return {
        "n": int(metrics["n"]),
        "positive_n": int(metrics["positive_n"]),
        "prevalence": float(metrics["weighted_outcome_prevalence"]),
        "roc_auc": float(metrics["roc_auc"]),
        "average_precision": float(metrics["average_precision"]),
        "brier_score": float(metrics["brier_score"]),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "threshold": float(metrics["threshold"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }


def _schema_row(
    *, config, repo, versions_string, design, fold, model, tier, metrics,
    n_mothers, seed, feature_hash, split_hash, cohort_rule=C1_COHORT_RULE,
):
    return build_row(
        experiment_id="A7",
        wave="NFHS-5",
        cohort_rule=cohort_rule,
        evaluation_design=design,
        split_id=config["split"]["split_id"],
        fold=fold,
        source_domain="NFHS-5",
        target_domain="NFHS-5",
        model=model,
        feature_tier=tier,
        baseline_id="not_applicable",
        n_births=int(metrics["n"]),
        n_mothers=n_mothers,
        n_events=int(metrics["positive_n"]),
        prevalence=float(metrics["prevalence"]),
        roc_auc=float(metrics["roc_auc"]),
        average_precision=float(metrics["average_precision"]),
        brier_score=float(metrics["brier_score"]),
        calibration_intercept=float(metrics["calibration_intercept"]),
        calibration_slope=float(metrics["calibration_slope"]),
        threshold=float(metrics["threshold"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        weighted_descriptive_flag="unweighted_primary_analysis",
        uncertainty_status=UNCERTAINTY_STATUS,
        repo_commit=repo["repo_commit"],
        seed=seed,
        software_versions=versions_string,
        feature_manifest_hash=feature_hash,
        split_manifest_hash=split_hash,
    )


def _catboost_version() -> str:
    try:
        import catboost

        return catboost.__version__
    except ImportError:
        return "not_installed"


def _tier_frame(core, detail, history_feats, tier) -> pd.DataFrame:
    frame = core.copy()
    spec = tier_manifest(tier)
    if "facility_detail" in spec.categorical:
        frame["facility_detail"] = detail.to_numpy()
    for column in HISTORY_COLUMNS:
        if column in spec.categorical:
            frame[column] = history_feats[column].to_numpy()
    missing = sorted(set(spec.categorical + spec.numerical) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{tier}: tier frame missing columns {missing}")
    return frame


def _builder_id(tier: str) -> str:
    builder = "version2_ml/feature_builder.py::make_core_features"
    if tier == "F1_detail_history":
        builder += " + version2_ml/experiments/A4/src/a4_facility_detail.py::facility_detail"
        builder += " + version2_ml/experiments/A4/src/a4_history.py::build_history_features"
    return builder


def _support_ok(n_rows: int, n_events: int) -> bool:
    return (
        n_rows >= MIN_SUBGROUP_ROWS
        and n_events >= MIN_SUBGROUP_CLASS_ROWS
        and (n_rows - n_events) >= MIN_SUBGROUP_CLASS_ROWS
    )


def _mother_map(partition, cohort):
    frame = (
        pd.DataFrame(
            {
                "caseid": cohort["caseid"].astype("string").to_numpy(),
                "partition": partition.to_numpy(),
            }
        )
    )
    if frame["caseid"].duplicated().any():
        assignments = frame.drop_duplicates()
        if assignments["caseid"].duplicated().any():
            raise RuntimeError("Inconsistent mother-level partition assignment")
        frame = assignments
    return frame.drop_duplicates().set_index("caseid")["partition"]


def _inherit_partition(partition, cohort, subset_cohort):
    mother_map = _mother_map(partition, cohort)
    return pd.Series(
        mother_map.loc[subset_cohort["caseid"].astype("string")].to_numpy(),
        index=subset_cohort.index,
        name="partition",
    )


def _tuning_stage(config, f1_fit_frame, y_fit, groups_fit, private_dir, force):
    cache = private_dir / "a7_tuning.json"
    if cache.exists() and not force:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        payload["source"] = "cache"
        return payload
    tuning = {}
    for model in MODELS:
        tuning[model] = tune_candidate(
            model, "F1", f1_fit_frame, y_fit, groups_fit,
            folds=int(config["tuning"]["folds"]),
            seed=int(config["tuning"]["seed"]),
            grid=config["tuning"][f"{model}_grid"],
        )
    payload = {"source": "computed", "models": tuning}
    cache.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def _candidate_stage(config, tuning, frame_by_tier, fit_positions, calib_positions,
                     valid_positions, y_fit, y_calib, y_valid, seed):
    results = {}
    threshold = float(config["thresholds"]["primary"])
    order = {method: index for index, method in enumerate(METHODS)}
    for model in MODELS:
        params = tuning["models"][model]["best"]
        for tier in TIER_ORDER:
            frame = frame_by_tier[tier]
            run_params = dict(params)
            if model in ("xgboost", "catboost"):
                run_params["scale_pos_weight"] = scale_pos_weight(y_fit)
            classifier = fit_candidate(
                model, tier, run_params,
                frame.iloc[fit_positions].reset_index(drop=True), y_fit, seed,
            )
            p_valid_base = predict_candidate(
                classifier, frame.iloc[valid_positions].reset_index(drop=True), "validation"
            )
            p_calib_base = predict_candidate(
                classifier, frame.iloc[calib_positions].reset_index(drop=True), "train"
            )
            calibrators = {
                method: fit_calibrator(method, p_calib_base, y_calib)
                for method in METHODS
            }
            methods = {}
            for method in METHODS:
                p_valid = apply_calibrator(method, calibrators[method], p_valid_base)
                metrics = evaluate_binary(y_valid, p_valid, threshold=threshold)
                intercept, slope = calibration_intercept_slope(y_valid, p_valid)
                threshold_f1, best_f1 = threshold_f1_optimal(y_valid, p_valid)
                threshold_you = threshold_youden(y_valid, p_valid)
                threshold_rec = threshold_for_recall(y_valid, p_valid, 0.8)
                methods[method] = {
                    "metrics": _metrics_payload(metrics, intercept, slope),
                    "probabilities": p_valid,
                    "thresholds": {
                        "f1_optimal": threshold_f1,
                        "f1_optimal_value": best_f1,
                        "youden": threshold_you,
                        "youden_f1": f1_at_threshold(y_valid, p_valid, threshold_you),
                        "recall_ge_0.80": threshold_rec,
                        "recall_ge_0.80_f1": (
                            f1_at_threshold(y_valid, p_valid, threshold_rec)
                            if threshold_rec is not None else None
                        ),
                    },
                }
            selected_method = min(
                METHODS,
                key=lambda method: (methods[method]["metrics"]["brier_score"], order[method]),
            )
            results[f"{model}__{tier}"] = {
                "model": model,
                "tier": tier,
                "params": run_params,
                "classifier": classifier,
                "calibrators": calibrators,
                "methods": methods,
                "selected_method": selected_method,
            }
    return results


def _test_stage(private_dir, candidates, frame_by_tier, test_positions, y_test, seed, repo,
                versions, allow_test_score=False):
    cache = private_dir / "a7_test_scored.json"
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        payload["source"] = "cache_no_rescore"
        payload.setdefault("metrics", {})
        return payload
    if not allow_test_score:
        raise RuntimeError(
            "Refusing to score the development test partition: no cached "
            f"{cache.name} in {private_dir}. Point --cache-dir at the recorded "
            "cache directory, or pass --allow-test-score to acknowledge a new "
            "development-test scoring."
        )
    metrics = {}
    for key, payload in candidates.items():
        frame = frame_by_tier[payload["tier"]]
        p_test_base = predict_candidate(
            payload["classifier"], frame.iloc[test_positions].reset_index(drop=True), "test"
        )
        entry = {}
        for method in METHODS:
            probabilities = apply_calibrator(method, payload["calibrators"][method], p_test_base)
            result = evaluate_binary(y_test, probabilities, threshold=0.5)
            intercept, slope = calibration_intercept_slope(y_test, probabilities)
            entry[method] = _metrics_payload(result, intercept, slope)
        metrics[key] = entry
    return {
        "source": "computed",
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": repo["repo_commit"],
        "software_versions": versions,
        "seed": seed,
        "metrics": metrics,
        "note": "development-test scored once; V1 exposure disclosed; not an independent test",
    }


def _f0_stage(config, f0_parquet, cohort, keys, fit_positions, valid_positions,
              test_positions, y_fit, y_valid, y_test, seed, test_payload):
    block = {"status": "not_run_no_parquet_flag", "reconciliation": {}}
    if not f0_parquet:
        return block
    parquet_path = Path(f0_parquet)
    if not parquet_path.exists():
        block["status"] = "not_run_parquet_missing"
        return block
    pq = pd.read_parquet(parquet_path, columns=list(F0_NAMES) + ["respondent_id", "birth_index", "csection", "delivery_place_code"])
    pq_keys = (
        pq["respondent_id"].astype("string")
        + "|"
        + pq["birth_index"].map(lambda value: str(int(value))).astype("string")
    )
    reconciliation = {
        "parquet_rows": int(len(pq)),
        "cohort_rows": int(len(cohort)),
        "matched_keys": int(pq_keys.isin(keys).sum()),
        "cohort_only_keys": int(len(set(keys) - set(pq_keys))),
        "parquet_only_keys": int(len(set(pq_keys) - set(keys))),
        "duplicate_parquet_keys": int(pq_keys.duplicated().sum()),
    }
    exact = (
        reconciliation["parquet_rows"] == len(cohort)
        and reconciliation["matched_keys"] == len(cohort)
        and reconciliation["cohort_only_keys"] == 0
        and reconciliation["parquet_only_keys"] == 0
        and reconciliation["duplicate_parquet_keys"] == 0
    )
    if not exact:
        block.update({"status": "not_run_reconciliation_failed", "reconciliation": reconciliation})
        return block
    aligned = pq.set_index(pq_keys).loc[keys]
    outcome_mismatch = int(
        (pd.to_numeric(aligned["csection"], errors="coerce").to_numpy() != cohort["m17"].to_numpy()).sum()
    )
    facility_mismatch = int(
        (pd.to_numeric(aligned["delivery_place_code"], errors="coerce").to_numpy() != cohort["m15"].to_numpy()).sum()
    )
    reconciliation["outcome_mismatch"] = outcome_mismatch
    reconciliation["facility_mismatch"] = facility_mismatch
    block["reconciliation"] = reconciliation
    if outcome_mismatch or facility_mismatch:
        block["status"] = "not_run_reconciliation_failed"
        return block
    f0_frame = aligned[list(F0_NAMES)].reset_index(drop=True)
    manifest = FeatureManifest(F0_NUM + F0_BIN, F0_CAT, "F0")
    params = dict(config["models"]["xgboost"])
    params["scale_pos_weight"] = scale_pos_weight(y_fit)
    classifier = AuthorizedClassifier(manifest, algorithm="xgboost", seed=seed)
    classifier.estimator = estimator_factory("xgboost", params, seed)
    classifier.fit(f0_frame.iloc[fit_positions].reset_index(drop=True), y_fit, partition="train")
    p_valid = classifier.predict_proba(
        f0_frame.iloc[valid_positions].reset_index(drop=True), partition="validation"
    )
    metrics = evaluate_binary(y_valid, p_valid, threshold=0.5)
    intercept, slope = calibration_intercept_slope(y_valid, p_valid)
    block["validation"] = _metrics_payload(metrics, intercept, slope)
    key = "xgboost__F0_historical"
    if test_payload["source"] == "computed":
        p_test = classifier.predict_proba(
            f0_frame.iloc[test_positions].reset_index(drop=True), partition="test"
        )
        test_metrics = evaluate_binary(y_test, p_test, threshold=0.5)
        test_intercept, test_slope = calibration_intercept_slope(y_test, p_test)
        test_payload.setdefault("f0_metrics", {})[key] = _metrics_payload(
            test_metrics, test_intercept, test_slope
        )
        block["test"] = test_payload["f0_metrics"][key]
    else:
        block["test"] = test_payload.get("f0_metrics", {}).get(key, "not_applicable")
    block["status"] = "reconciled_and_fitted"
    block["feature_manifest_hash"] = feature_manifest_hash(
        tier="F0",
        numerical=list(F0_NUM + F0_BIN),
        categorical=list(F0_CAT),
        builder_id="version2_ml/feature_builder.py historical F0 36 features via processed parquet",
    )
    return block


def _shap_stage(config, candidates, frame_by_tier, valid_positions, y_valid,
                fit_positions, y_fit, groups_fit):
    payload = candidates["xgboost__F1_detail_history"]
    classifier = payload["classifier"]
    enh_valid = frame_by_tier["F1_detail_history"].iloc[valid_positions].reset_index(drop=True)
    sample_size = int(config["shap"]["sample_size"])
    shap_seed = int(config["shap"]["seed"])
    overall = shap_family_table(classifier, enh_valid, sample_size=sample_size, seed=shap_seed)
    folds = fold_models(
        "xgboost", "F1_detail_history", dict(payload["params"]),
        frame_by_tier["F1_detail_history"].iloc[fit_positions].reset_index(drop=True),
        y_fit, groups_fit,
        folds=int(config["tuning"]["folds"]), seed=int(config["tuning"]["seed"]),
    )
    fold_tables = [
        shap_family_table(model, enh_valid, sample_size=sample_size, seed=shap_seed)
        for model in folds
    ]
    stability = rank_stability([overall] + fold_tables)
    subgroup_tables = {}
    subgroup_support = {}
    residence = enh_valid["residence"].astype("string")
    label_series = pd.Series(y_valid)
    for name, mask in (
        ("residence_urban", residence.eq("urban").to_numpy()),
        ("residence_rural", residence.eq("rural").to_numpy()),
    ):
        n_rows = int(mask.sum())
        n_events = int(label_series[mask].sum())
        supported = _support_ok(n_rows, n_events)
        subgroup_support[name] = {
            "n_rows": n_rows,
            "n_events": suppress_small_count(n_events),
            "supported": supported,
        }
        if supported:
            table = shap_family_table(
                classifier, enh_valid.loc[mask].reset_index(drop=True),
                sample_size=sample_size, seed=shap_seed,
            )
            subgroup_tables[name] = table
            subgroup_support[name]["min_pairwise_spearman_vs_overall"] = rank_stability(
                [overall, table]
            )["min_pairwise_spearman"]
        else:
            subgroup_support[name]["min_pairwise_spearman_vs_overall"] = "not_run_support_below_threshold"
    return {
        "status": "computed",
        "explainer_note": overall["explainer"],
        "overall": overall,
        "fold_tables": fold_tables,
        "stability": stability,
        "subgroup_tables": subgroup_tables,
        "subgroup_support": subgroup_support,
    }


def _robustness_stage(config, cohort, keys, frame_by_tier, partition, train_mask,
                      valid_mask, y_all, candidates, seed):
    rows = []
    repeated = []
    for repeat_seed in config["robustness"]["repeated_split_seeds"]:
        repeat_partition = grouped_partition(
            cohort,
            group_col=config["split"]["group_col"],
            target_col=config["split"]["target_col"],
            fractions=tuple(config["split"]["fractions"]),
            seed=int(repeat_seed),
        )
        audit_partition(cohort, repeat_partition)
        r_train = repeat_partition.eq("train").to_numpy()
        r_valid = repeat_partition.eq("validation").to_numpy()
        y_r_train = y_all[r_train]
        y_r_valid = y_all[r_valid]
        digest = membership_digest(keys[r_train], seed=int(repeat_seed))
        repeat_row = {"seed": int(repeat_seed), "train_digest": digest}
        for tier in config["robustness"]["repeated_split_tiers"]:
            frame = frame_by_tier[tier]
            params = dict(config["models"]["xgboost"])
            params["scale_pos_weight"] = scale_pos_weight(y_r_train)
            classifier = fit_candidate(
                "xgboost", tier, params,
                frame[r_train].reset_index(drop=True), y_r_train, int(repeat_seed),
            )
            probabilities = predict_candidate(
                classifier, frame[r_valid].reset_index(drop=True), "validation"
            )
            metrics = evaluate_binary(y_r_valid, probabilities, threshold=0.5)
            repeat_row[f"{tier}_average_precision"] = round(float(metrics["average_precision"]), 6)
            rows.append({
                "check": "repeated_grouped_split",
                "detail": f"seed={repeat_seed}",
                "model": "xgboost",
                "tier": tier,
                "n_births": int(metrics["n"]),
                "n_events": suppress_small_count(metrics["positive_n"]),
                "roc_auc": round(float(metrics["roc_auc"]), 6),
                "average_precision": round(float(metrics["average_precision"]), 6),
                "brier_score": round(float(metrics["brier_score"]), 6),
                "note": f"train_digest={digest[:16]}",
            })
        repeated.append(repeat_row)

    sensitivity = _sensitivity_m15_96(config, cohort, partition, y_all, seed)
    rows.extend(sensitivity["rows"])
    recent = _most_recent_birth_check(config, cohort, partition, y_all, seed)
    rows.extend(recent["rows"])
    subgroups = _subgroup_metrics(config, cohort, frame_by_tier, valid_mask, y_all, candidates)
    rows.extend(subgroups["rows"])
    psu = _psu_check(cohort, train_mask, valid_mask)
    rows.append({
        "check": "psu_overlap",
        "detail": "v021 sampled PSU codes (never exported)",
        "model": "not_applicable",
        "tier": "not_applicable",
        "n_births": psu["n_shared_psus"],
        "n_events": "not_applicable",
        "roc_auc": "not_applicable",
        "average_precision": "not_applicable",
        "brier_score": "not_applicable",
        "note": json.dumps(psu, sort_keys=True),
    })
    return {
        "status": "computed",
        "repeated_splits": repeated,
        "rows": rows,
        "sensitivity": sensitivity["summary"],
        "most_recent_birth": recent["summary"],
        "subgroups": subgroups["summary"],
        "psu_overlap": psu,
    }


def _sensitivity_m15_96(config, cohort, partition, y_all, seed):
    subset = cohort[cohort["m15"] != 96].reset_index(drop=True)
    subset_core = make_core_features(subset, wave="NFHS-5")
    subset_partition = _inherit_partition(partition, cohort, subset)
    audit_partition(subset, subset_partition)
    train_mask = subset_partition.eq("train").to_numpy()
    valid_mask = subset_partition.eq("validation").to_numpy()
    y_sub = subset["m17"].to_numpy()
    params = dict(config["models"]["xgboost"])
    params["scale_pos_weight"] = scale_pos_weight(y_sub[train_mask])
    classifier = fit_candidate(
        "xgboost", "F1", params,
        subset_core[train_mask].reset_index(drop=True), y_sub[train_mask], seed,
    )
    probabilities = predict_candidate(
        classifier, subset_core[valid_mask].reset_index(drop=True), "validation"
    )
    metrics = evaluate_binary(y_sub[valid_mask], probabilities, threshold=0.5)
    row = {
        "check": "m15_96_excluded_partition_inherited",
        "detail": "consistent exclusion from index and history",
        "model": "xgboost",
        "tier": "F1",
        "n_births": int(metrics["n"]),
        "n_events": suppress_small_count(metrics["positive_n"]),
        "roc_auc": round(float(metrics["roc_auc"]), 6),
        "average_precision": round(float(metrics["average_precision"]), 6),
        "brier_score": round(float(metrics["brier_score"]), 6),
        "note": f"excluded_rows={int((cohort['m15'] == 96).sum())}",
    }
    summary = {
        "n_births": int(len(subset)),
        "n_events": int(subset["m17"].sum()),
        "excluded_rows": int((cohort["m15"] == 96).sum()),
        "roc_auc": round(float(metrics["roc_auc"]), 6),
        "average_precision": round(float(metrics["average_precision"]), 6),
        "brier_score": round(float(metrics["brier_score"]), 6),
    }
    return {"rows": [row], "summary": summary}


def _most_recent_birth_check(config, cohort, partition, y_all, seed):
    ordered = cohort.sort_values(["caseid", "b3", "bidx"], kind="stable")
    recent_index = np.sort(ordered.groupby("caseid", sort=True).tail(1).index.to_numpy())
    subset = cohort.iloc[recent_index].reset_index(drop=True)
    subset_core = make_core_features(subset, wave="NFHS-5")
    subset_partition = _inherit_partition(partition, cohort, subset)
    audit_partition(subset, subset_partition)
    train_mask = subset_partition.eq("train").to_numpy()
    valid_mask = subset_partition.eq("validation").to_numpy()
    y_sub = subset["m17"].to_numpy()
    params = dict(config["models"]["xgboost"])
    params["scale_pos_weight"] = scale_pos_weight(y_sub[train_mask])
    classifier = fit_candidate(
        "xgboost", "F1", params,
        subset_core[train_mask].reset_index(drop=True), y_sub[train_mask], seed,
    )
    probabilities = predict_candidate(
        classifier, subset_core[valid_mask].reset_index(drop=True), "validation"
    )
    metrics = evaluate_binary(y_sub[valid_mask], probabilities, threshold=0.5)
    row = {
        "check": "most_recent_birth_per_mother",
        "detail": "latest b3/bidx by calendar order",
        "model": "xgboost",
        "tier": "F1",
        "n_births": int(metrics["n"]),
        "n_events": suppress_small_count(metrics["positive_n"]),
        "roc_auc": round(float(metrics["roc_auc"]), 6),
        "average_precision": round(float(metrics["average_precision"]), 6),
        "brier_score": round(float(metrics["brier_score"]), 6),
        "note": f"subset_births={int(len(subset))}",
    }
    summary = {
        "n_births": int(len(subset)),
        "n_events": int(subset["m17"].sum()),
        "roc_auc": round(float(metrics["roc_auc"]), 6),
        "average_precision": round(float(metrics["average_precision"]), 6),
        "brier_score": round(float(metrics["brier_score"]), 6),
    }
    return {"rows": [row], "summary": summary}


def _subgroup_metrics(config, cohort, frame_by_tier, valid_mask, y_all, candidates):
    payload = candidates["xgboost__F1_detail_history"]
    probabilities = payload["methods"][payload["selected_method"]]["probabilities"]
    y_valid = y_all[valid_mask]
    valid_core = frame_by_tier["F1"].loc[valid_mask]
    facility = valid_core["facility_type"].astype("string")
    residence = valid_core["residence"].astype("string")
    observed_facility = set(facility.dropna().unique())
    if not observed_facility <= {"public", "private", "other"}:
        raise RuntimeError(f"Unexpected facility_type categories: {sorted(observed_facility)}")
    rows = []
    summary = {}
    masks = {
        "facility_type_public": facility.eq("public").to_numpy(),
        "facility_type_private": facility.eq("private").to_numpy(),
        "facility_type_other": facility.eq("other").to_numpy(),
        "residence_urban": residence.eq("urban").to_numpy(),
        "residence_rural": residence.eq("rural").to_numpy(),
    }
    for name, mask in masks.items():
        n_rows = int(mask.sum())
        n_events = int(y_valid[mask].sum())
        supported = _support_ok(n_rows, n_events)
        entry = {"n_rows": n_rows, "n_events": suppress_small_count(n_events), "supported": supported}
        if supported:
            metrics = evaluate_binary(y_valid[mask], probabilities[mask], threshold=0.5)
            intercept, slope = calibration_intercept_slope(y_valid[mask], probabilities[mask])
            entry.update({
                "roc_auc": round(float(metrics["roc_auc"]), 6),
                "average_precision": round(float(metrics["average_precision"]), 6),
                "brier_score": round(float(metrics["brier_score"]), 6),
                "calibration_slope": round(slope, 6),
            })
            rows.append({
                "check": "subgroup", "detail": name, "model": "xgboost",
                "tier": "F1_detail_history", "n_births": n_rows,
                "n_events": suppress_small_count(n_events),
                "roc_auc": round(float(metrics["roc_auc"]), 6),
                "average_precision": round(float(metrics["average_precision"]), 6),
                "brier_score": round(float(metrics["brier_score"]), 6),
                "note": f"calibration_slope={round(slope, 4)}",
            })
        else:
            rows.append({
                "check": "subgroup", "detail": name, "model": "xgboost",
                "tier": "F1_detail_history", "n_births": n_rows,
                "n_events": suppress_small_count(n_events),
                "roc_auc": "not_run_support_below_threshold",
                "average_precision": "not_run_support_below_threshold",
                "brier_score": "not_run_support_below_threshold",
                "note": "minimum 200 rows and 30 rows per class required",
            })
        summary[name] = entry
    state_labels = map_states(cohort["v024"]).to_numpy()
    states_valid = state_labels[valid_mask]
    qualifying = []
    state_entries = {}
    for state in sorted(set(states_valid)):
        mask = states_valid == state
        n_rows = int(mask.sum())
        n_events = int(y_valid[mask].sum())
        supported = _support_ok(n_rows, n_events)
        state_entries[str(state)] = {
            "n_rows": n_rows,
            "n_events": suppress_small_count(n_events),
            "supported": supported,
        }
        if supported:
            metrics = evaluate_binary(y_valid[mask], probabilities[mask], threshold=0.5)
            qualifying.append({
                "state": str(state),
                "n_rows": n_rows,
                "n_events": n_events,
                "roc_auc": float(metrics["roc_auc"]),
                "average_precision": float(metrics["average_precision"]),
            })
    summary["states"] = state_entries
    if qualifying:
        qualifying_set = {item["state"] for item in qualifying}
        pooled_mask = np.array([str(name) in qualifying_set for name in states_valid])
        metrics = evaluate_binary(y_valid[pooled_mask], probabilities[pooled_mask], threshold=0.5)
        intercept, slope = calibration_intercept_slope(y_valid[pooled_mask], probabilities[pooled_mask])
        summary["states_pooled"] = {
            "n_qualifying_states": len(qualifying),
            "n_rows": int(pooled_mask.sum()),
            "n_events": suppress_small_count(int(y_valid[pooled_mask].sum())),
            "roc_auc": round(float(metrics["roc_auc"]), 6),
            "average_precision": round(float(metrics["average_precision"]), 6),
            "brier_score": round(float(metrics["brier_score"]), 6),
            "calibration_slope": round(slope, 6),
            "state_roc_auc_min": round(min(item["roc_auc"] for item in qualifying), 6),
            "state_roc_auc_max": round(max(item["roc_auc"] for item in qualifying), 6),
        }
        rows.append({
            "check": "subgroup_states_pooled",
            "detail": f"{len(qualifying)} supported states",
            "model": "xgboost",
            "tier": "F1_detail_history",
            "n_births": int(pooled_mask.sum()),
            "n_events": suppress_small_count(int(y_valid[pooled_mask].sum())),
            "roc_auc": round(float(metrics["roc_auc"]), 6),
            "average_precision": round(float(metrics["average_precision"]), 6),
            "brier_score": round(float(metrics["brier_score"]), 6),
            "note": (
                f"calibration_slope={round(slope, 4)} state_auc_range="
                f"({summary['states_pooled']['state_roc_auc_min']},"
                f"{summary['states_pooled']['state_roc_auc_max']})"
            ),
        })
    else:
        summary["states_pooled"] = {"n_qualifying_states": 0, "note": "no state met support"}
    return {"rows": rows, "summary": summary}


def _psu_check(cohort, train_mask, valid_mask):
    frame = pd.DataFrame({
        "caseid": cohort["caseid"].astype("string").to_numpy(),
        "psu": pd.to_numeric(cohort["v021"], errors="coerce").to_numpy(),
    })
    train_psu = frame.loc[train_mask].groupby("caseid", sort=True)["psu"].first()
    valid_psu = frame.loc[valid_mask].groupby("caseid", sort=True)["psu"].first()
    return psu_overlap_summary(train_psu, valid_psu)


def _write_outputs(state: dict) -> None:
    config = state["config"]
    out_dir = state["out_dir"]
    private_dir = state["private_dir"]
    candidates = state["candidates"]
    test_payload = state["test_payload"]
    f0_block = state["f0_block"]
    repo = state["repo"]
    versions = state["versions"]
    versions_string = json.dumps(versions, sort_keys=True, separators=(",", ":"))
    split_hash = state["split_hash"]
    feature_hashes = state["feature_hashes"]

    if test_payload["source"] == "computed":
        (private_dir / "a7_test_scored.json").write_text(
            json.dumps(test_payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    rows = []
    for key in sorted(candidates):
        payload = candidates[key]
        for method in METHODS:
            metrics = payload["methods"][method]["metrics"]
            rows.append(_schema_row(
                config=config, repo=repo, versions_string=versions_string,
                design=DESIGN_VALIDATION, fold="validation",
                model=payload["model"], tier=payload["tier"], metrics=metrics,
                n_mothers=state["valid_mothers"], seed=state["seed"],
                feature_hash=feature_hashes[payload["tier"]], split_hash=split_hash,
            ))
    for key in sorted(candidates):
        payload = candidates[key]
        method = payload["selected_method"]
        metrics = test_payload["metrics"][key][method]
        rows.append(_schema_row(
            config=config, repo=repo, versions_string=versions_string,
            design=DESIGN_TEST, fold="test_development_only",
            model=payload["model"], tier=payload["tier"], metrics=metrics,
            n_mothers=state["test_mothers"], seed=state["seed"],
            feature_hash=feature_hashes[payload["tier"]], split_hash=split_hash,
        ))
    if f0_block.get("status") == "reconciled_and_fitted":
        rows.append(_schema_row(
            config=config, repo=repo, versions_string=versions_string,
            design=DESIGN_F0, fold="validation", model="xgboost",
            tier="F0_historical_36", metrics=f0_block["validation"],
            n_mothers=state["valid_mothers"], seed=state["seed"],
            feature_hash=f0_block["feature_manifest_hash"], split_hash=split_hash,
        ))
        rows.append(_schema_row(
            config=config, repo=repo, versions_string=versions_string,
            design=DESIGN_F0, fold="test_development_only", model="xgboost",
            tier="F0_historical_36", metrics=f0_block["test"],
            n_mothers=state["test_mothers"], seed=state["seed"],
            feature_hash=f0_block["feature_manifest_hash"], split_hash=split_hash,
        ))
    write_rows(rows, out_dir / "aggregate_model_comparison.csv")

    calibration_rows = []
    for key in sorted(candidates):
        payload = candidates[key]
        for method in METHODS:
            entry = payload["methods"][method]
            metrics = entry["metrics"]
            thresholds = entry["thresholds"]
            calibration_rows.append({
                "candidate": key,
                "model": payload["model"],
                "tier": payload["tier"],
                "method": method,
                "selected": method == payload["selected_method"],
                "brier_score": round(metrics["brier_score"], 6),
                "calibration_intercept": round(metrics["calibration_intercept"], 6),
                "calibration_slope": round(metrics["calibration_slope"], 6),
                "threshold_primary": 0.5,
                "precision_at_primary": round(metrics["precision"], 6),
                "recall_at_primary": round(metrics["recall"], 6),
                "f1_at_primary": round(metrics["f1"], 6),
                "threshold_f1_optimal": thresholds["f1_optimal"],
                "f1_at_optimal_threshold": round(thresholds["f1_optimal_value"], 6),
                "threshold_youden": thresholds["youden"],
                "f1_at_youden_threshold": round(thresholds["youden_f1"], 6),
                "threshold_recall_ge_0_80": thresholds["recall_ge_0.80"]
                if thresholds["recall_ge_0.80"] is not None else "not_attainable",
                "f1_at_recall_threshold": (
                    round(thresholds["recall_ge_0.80_f1"], 6)
                    if thresholds["recall_ge_0.80_f1"] is not None else "not_attainable"
                ),
            })
    pd.DataFrame(calibration_rows).to_csv(out_dir / "calibration_summary.csv", index=False)

    tuning_rows = []
    tuning = state["tuning"]
    for model, outcome in sorted(tuning["models"].items()):
        for item in outcome["results"]:
            tuning_rows.append({
                "model": model,
                "tier": outcome["tier"],
                "params": json.dumps(item["params"], sort_keys=True),
                "fold_average_precision": ";".join(str(value) for value in item["fold_average_precision"]),
                "mean_average_precision": item["mean_average_precision"],
                "selected": item["params"] == outcome["best"],
            })
    pd.DataFrame(tuning_rows).to_csv(out_dir / "a7_tuning_summary.csv", index=False)

    robustness = state["robustness"]
    if robustness.get("status") == "computed":
        pd.DataFrame(robustness["rows"]).to_csv(out_dir / "a7_robustness.csv", index=False)

    shap_block = state["shap_block"]
    if shap_block.get("status") == "computed":
        shap_rows = []
        tables = [("overall", shap_block["overall"])]
        tables += [(f"fold_{index + 1}", table) for index, table in enumerate(shap_block["fold_tables"])]
        tables += [(name, table) for name, table in sorted(shap_block["subgroup_tables"].items())]
        for name, table in tables:
            for row in table["families"]:
                shap_rows.append({
                    "table": name,
                    "n_sample_rows": table["n_sample_rows"],
                    "n_encoded_features": table["n_encoded_features"],
                    "family": row["family"],
                    "rank": row["rank"],
                    "mean_abs_shap_margin": row["mean_abs_shap_margin"],
                    "share_of_total": row["share_of_total"],
                })
        pd.DataFrame(shap_rows).to_csv(out_dir / "shap_feature_family_summary.csv", index=False)
        stability_rows = [{
            "comparison": "overall_vs_fold_models",
            "n_models": shap_block["stability"]["n_models"],
            "min_pairwise_spearman": shap_block["stability"]["min_pairwise_spearman"],
            "pairwise_spearman": ";".join(str(value) for value in shap_block["stability"]["pairwise_spearman"]),
        }]
        for name, support in sorted(shap_block["subgroup_support"].items()):
            stability_rows.append({
                "comparison": f"overall_vs_{name}",
                "n_models": 2,
                "min_pairwise_spearman": support.get("min_pairwise_spearman_vs_overall", "not_applicable"),
                "pairwise_spearman": "not_applicable",
            })
        pd.DataFrame(stability_rows).to_csv(out_dir / "a7_shap_stability.csv", index=False)

    reconciliation_rows = [{"check": "f0_status", "detail": f0_block.get("status"), "value": "not_applicable"}]
    for key, value in sorted(f0_block.get("reconciliation", {}).items()):
        reconciliation_rows.append({"check": "f0_reconciliation", "detail": key, "value": value})
    if robustness.get("status") == "computed":
        for key, value in sorted(robustness["psu_overlap"].items()):
            reconciliation_rows.append({"check": "psu_overlap_v021", "detail": key, "value": value})
    pd.DataFrame(reconciliation_rows).to_csv(out_dir / "a7_reconciliation.csv", index=False)

    _fig_roc_pr(state, out_dir / "fig_a7_roc_pr.png")
    _fig_calibration(state, out_dir / "fig_a7_calibration.png")
    if shap_block.get("status") == "computed":
        _fig_shap(state, out_dir / "fig_a7_shap_stability.png")
    _write_methods(state, out_dir / "a7_methods_results.md")
    _write_summary(state)


def _fig_roc_pr(state, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    y_valid = state["y_valid"]
    for key in sorted(state["candidates"]):
        payload = state["candidates"][key]
        probabilities = payload["methods"][payload["selected_method"]]["probabilities"]
        fpr, tpr, _ = roc_curve(y_valid, probabilities)
        precision, recall, _ = precision_recall_curve(y_valid, probabilities)
        label = f"{key}"
        axes[0].plot(fpr, tpr, linewidth=1.2, label=label)
        axes[1].plot(recall, precision, linewidth=1.2, label=label)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("A7 validation ROC (selected calibration)")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("A7 validation precision-recall (selected calibration)")
    for axis in axes:
        axis.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_calibration(state, path):
    y_valid = state["y_valid"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, model in zip(axes, MODELS):
        f1_key = f"{model}__F1"
        payload = state["candidates"][f1_key]
        for method, line_style in (("uncalibrated", "--"), (payload["selected_method"], "-")):
            probabilities = payload["methods"][method]["probabilities"]
            mean_predicted, fraction = reliability_points(y_valid, probabilities)
            axis.plot(mean_predicted, fraction, line_style, linewidth=1.3, label=method)
        axis.plot([0, 1], [0, 1], color="gray", linewidth=0.8)
        axis.set_title(f"{model} (F1)")
        axis.set_xlabel("mean predicted probability")
        axis.set_ylabel("observed event rate")
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_shap(state, path):
    shap_block = state["shap_block"]
    tables = [("overall", shap_block["overall"])]
    tables += [(f"fold_{index + 1}", table) for index, table in enumerate(shap_block["fold_tables"])]
    tables += [(name, table) for name, table in sorted(shap_block["subgroup_tables"].items())]
    families = [row["family"] for row in shap_block["overall"]["families"]]
    fig, axis = plt.subplots(figsize=(9, max(4.5, 0.35 * len(families))))
    x_positions = np.arange(len(tables))
    for family in families:
        ranks = []
        for _, table in tables:
            lookup = {row["family"]: row["rank"] for row in table["families"]}
            ranks.append(lookup.get(family, np.nan))
        axis.plot(x_positions, ranks, marker="o", markersize=3, linewidth=1.0, label=family)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([name for name, _ in tables], rotation=30, ha="right", fontsize=7)
    axis.invert_yaxis()
    axis.set_ylabel("SHAP family rank (1 = largest mean |value|)")
    axis.set_title(
        "A7 SHAP rank stability; min pairwise Spearman="
        f"{shap_block['stability']['min_pairwise_spearman']}"
    )
    axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_methods(state, path):
    config = state["config"]
    lines = []
    lines.append("# A7 methods and results (development, point estimates only)\n")
    lines.append(
        "Generated by `version2_ml/experiments/A7/src/run_a7.py` from the frozen C1 "
        "cohort on the provisional mother-grouped split; the development test "
        "partition was scored once and is not an independent test (V1 exposure). "
        "Positive counts below 10 are suppressed; no confidence intervals are "
        "reported.\n"
    )
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines.append(f"- generated_at_utc: {generated}")
    lines.append(f"- repo_commit: {state['repo']['repo_commit']}")
    lines.append(f"- branch: {state['repo']['branch']}")
    lines.append(f"- split_manifest_hash: {state['split_hash']}")
    lines.append(f"- input_sha256: {state['input_hash']}")
    lines.append("")
    lines.append("## Candidate validation metrics (selected calibration method)\n")
    lines.append("| candidate | selected method | ROC-AUC | AP | Brier | intercept | slope |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for key in sorted(state["candidates"]):
        payload = state["candidates"][key]
        metrics = payload["methods"][payload["selected_method"]]["metrics"]
        lines.append(
            f"| {key} | {payload['selected_method']} | {metrics['roc_auc']:.4f} | "
            f"{metrics['average_precision']:.4f} | {metrics['brier_score']:.4f} | "
            f"{metrics['calibration_intercept']:.4f} | {metrics['calibration_slope']:.4f} |"
        )
    lines.append("")
    lines.append("## All calibration variants (validation Brier)\n")
    lines.append("| candidate | method | Brier | selected |")
    lines.append("| --- | --- | --- | --- |")
    for key in sorted(state["candidates"]):
        payload = state["candidates"][key]
        for method in METHODS:
            metrics = payload["methods"][method]["metrics"]
            flag = "yes" if method == payload["selected_method"] else ""
            lines.append(f"| {key} | {method} | {metrics['brier_score']:.4f} | {flag} |")
    lines.append("")
    lines.append("## Development-test snapshot (single scoring)\n")
    lines.append("| candidate | selected method | ROC-AUC | AP | Brier |")
    lines.append("| --- | --- | --- | --- | --- |")
    for key in sorted(state["candidates"]):
        payload = state["candidates"][key]
        metrics = state["test_payload"]["metrics"][key][payload["selected_method"]]
        lines.append(
            f"| {key} | {payload['selected_method']} | {metrics['roc_auc']:.4f} | "
            f"{metrics['average_precision']:.4f} | {metrics['brier_score']:.4f} |"
        )
    lines.append("")
    f0_block = state["f0_block"]
    lines.append("## F0 reconciliation status\n")
    lines.append(f"- status: {f0_block.get('status')}")
    for key, value in sorted(f0_block.get("reconciliation", {}).items()):
        lines.append(f"- {key}: {value}")
    lines.append("")
    shap_block = state["shap_block"]
    if shap_block.get("status") == "computed":
        lines.append("## SHAP family ranking (validation sample, margin space)\n")
        lines.append("| rank | family | mean_abs_shap_margin | share |")
        lines.append("| --- | --- | --- | --- |")
        for row in shap_block["overall"]["families"]:
            lines.append(
                f"| {row['rank']} | {row['family']} | {row['mean_abs_shap_margin']:.4f} | "
                f"{row['share_of_total']:.4f} |"
            )
        lines.append("")
        lines.append(
            f"- min pairwise Spearman across fold models: "
            f"{shap_block['stability']['min_pairwise_spearman']}"
        )
        for name, support in sorted(shap_block["subgroup_support"].items()):
            lines.append(
                f"- {name}: rows={support['n_rows']}, events={support['n_events']}, "
                f"supported={support['supported']}, rho_vs_overall="
                f"{support.get('min_pairwise_spearman_vs_overall', 'not_applicable')}"
            )
        lines.append("")
    robustness = state["robustness"]
    lines.append("## Robustness\n")
    if robustness.get("status") == "computed":
        for row in robustness["rows"]:
            detail = row["detail"]
            lines.append(
                f"- {row['check']} ({detail}): AP={row['average_precision']}, "
                f"ROC-AUC={row['roc_auc']}, Brier={row['brier_score']}"
            )
        psu = robustness["psu_overlap"]
        lines.append(
            f"- PSU v021 overlap: shared={psu['n_shared_psus']}, "
            f"share_validation_rows_in_shared_psus={psu['share_validation_rows_in_shared_psus']}"
        )
    else:
        lines.append(f"- status: {robustness.get('status')}")
    lines.append("")
    lines.append("## Tuning selection (grouped CV inside fit pool)\n")
    for model, outcome in sorted(state["tuning"]["models"].items()):
        lines.append(f"- {model}: {json.dumps(outcome['best'], sort_keys=True)}")
    lines.append("")
    lines.append("## Disclosure\n")
    lines.append(
        "- All metrics are unweighted point estimates on development partitions; "
        "no design-based intervals; no causal claims."
    )
    lines.append(
        "- The development test was scored exactly once by the runner cache guard; "
        "V1 test exposure prevents independent-test interpretation."
    )
    lines.append(
        "- SHAP values are predictive attributions in margin space, aggregated to "
        "source families via the fitted encoder, not causal effects."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _masked_history_diagnostics(history_diag: dict) -> dict:
    masked = dict(history_diag)
    for key in ("level_counts", "has_any_counts", "quality_counts"):
        if key in masked:
            masked[key] = _masked_counts(masked[key])
    return masked


def _write_summary(state) -> None:
    summary = {
        "experiment_id": "A7",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": state["repo"],
        "software_versions": state["versions"],
        "input_sha256": state["input_hash"],
        "split_manifest_hash": state["split_hash"],
        "frozen_digests": {
            "train": FROZEN_TRAIN_DIGEST,
            "validation": FROZEN_VALID_DIGEST,
            "fit_pool": state["fit_digest"],
            "calibration_pool": state["calib_digest"],
            "test": state["test_digest"],
        },
        "cohort": state["cohort_summary"],
        "partitions": {
            "fit_rows": int(len(state["fit_positions"])),
            "calibration_rows": int(len(state["calib_positions"])),
            "validation_rows": int(len(state["valid_positions"])),
            "test_rows": int(len(state["test_positions"])),
        },
        "tuning": {
            "source": state["tuning"]["source"],
            "selected_params": {
                model: outcome["best"] for model, outcome in state["tuning"]["models"].items()
            },
        },
        "candidate_selection": {
            key: {
                "model": payload["model"],
                "tier": payload["tier"],
                "selected_method": payload["selected_method"],
                "validation_metrics": payload["methods"][payload["selected_method"]]["metrics"],
                "thresholds": payload["methods"][payload["selected_method"]]["thresholds"],
            }
            for key, payload in sorted(state["candidates"].items())
        },
        "test_scoring": {
            "source": state["test_payload"]["source"],
            "scored_at": state["test_payload"].get("scored_at", "from_cache"),
            "note": state["test_payload"]["note"],
        },
        "f0": {
            "status": state["f0_block"].get("status"),
            "reconciliation": state["f0_block"].get("reconciliation", {}),
            "validation": state["f0_block"].get("validation"),
        },
        "shap": {
            "status": state["shap_block"].get("status"),
            "top_families": [
                row["family"] for row in state["shap_block"].get("overall", {}).get("families", [])[:6]
            ],
            "min_pairwise_spearman": state["shap_block"].get("stability", {}).get("min_pairwise_spearman"),
            "subgroup_support": state["shap_block"].get("subgroup_support", {}),
        },
        "robustness": {
            "status": state["robustness"].get("status"),
            "repeated_splits": state["robustness"].get("repeated_splits", []),
            "sensitivity": state["robustness"].get("sensitivity", {}),
            "most_recent_birth": state["robustness"].get("most_recent_birth", {}),
            "psu_overlap": state["robustness"].get("psu_overlap", {}),
        },
        "suppressed_counts": {
            "m15_96_index_rows": suppress_small_count(int((state["cohort"]["m15"] == 96).sum())),
            "history_diagnostics": _masked_history_diagnostics(state["history_diag"]),
        },
        "warnings": state["warnings"],
        "disclosure": {
            "suppression_threshold": SUPPRESSION,
            "uncertainty_status": UNCERTAINTY_STATUS,
            "test_is_independent": False,
            "test_exposure_note": "V1 inspected the same NFHS-5 outcomes; development-test only.",
        },
    }
    state["summary"] = summary
    (state["out_dir"] / "a7_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    config = _load_config(Path(args.config))
    seed = int(args.seed) if args.seed is not None else int(config["split"]["seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    private_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)

    nfhs5_path = Path(args.nfhs5_path)
    if not nfhs5_path.exists():
        raise FileNotFoundError(f"NFHS-5 file not found: {nfhs5_path}")
    versions = software_versions(extra={"catboost": _catboost_version()})
    repo = repo_state()
    input_hash = sha256_file(nfhs5_path)

    raw = read_dta_subset(nfhs5_path, extra_columns=("v021",))
    cohort, cohort_summary = build_c1_cohort(raw)
    warnings = assert_frozen_c1(cohort_summary)
    expected = config["expected_cohort"]
    for key in ("n_births", "n_mothers", "n_events"):
        if int(expected[key]) != int(cohort_summary[key]):
            raise RuntimeError(f"Config expects {key}={expected[key]} but observed {cohort_summary[key]}")

    core = make_core_features(cohort, wave="NFHS-5")
    detail = facility_detail(cohort["m15"])
    history_feats, history_diag = build_history_features(raw, cohort, include_other_unknown_sector=True)
    observed_prior_cs = history_diag["level_counts"].get(
        "observed_previous_institutional_cs", 0
    )
    if observed_prior_cs != 6276:
        raise RuntimeError(
            f"History reproduction failure: {observed_prior_cs} vs audit 6276"
        )

    partition = grouped_partition(
        cohort,
        group_col=config["split"]["group_col"],
        target_col=config["split"]["target_col"],
        fractions=tuple(config["split"]["fractions"]),
        seed=seed,
    )
    audit_partition(cohort, partition)
    split_hash = split_manifest_hash(
        split_id=config["split"]["split_id"],
        seed=seed,
        fractions=config["split"]["fractions"],
        assignments=zip(cohort["caseid"].tolist(), partition.tolist()),
    )
    if split_hash != FROZEN_SPLIT_HASH:
        raise RuntimeError("Provisional split hash differs from the frozen A1 split")

    keys = _key_tokens(cohort)
    train_mask = partition.eq("train").to_numpy()
    valid_mask = partition.eq("validation").to_numpy()
    test_mask = partition.eq("test").to_numpy()
    if membership_digest(keys[train_mask], seed=seed) != FROZEN_TRAIN_DIGEST:
        raise RuntimeError("Train membership digest differs from the frozen A1 digest")
    if membership_digest(keys[valid_mask], seed=seed) != FROZEN_VALID_DIGEST:
        raise RuntimeError("Validation membership digest differs from the frozen A1 digest")

    train_positions = np.flatnonzero(train_mask)
    valid_positions = np.flatnonzero(valid_mask)
    test_positions = np.flatnonzero(test_mask)
    train_cohort = cohort.iloc[train_positions].reset_index(drop=True)
    carve = carve_calibration_partition(
        train_cohort,
        calibration_fraction=float(config["candidates"]["calibration"]["subset_fraction"]),
        seed=int(config["candidates"]["calibration"]["seed"]),
    )
    fit_positions = train_positions[carve.eq("fit").to_numpy()]
    calib_positions = train_positions[carve.eq("calibration").to_numpy()]
    y_all = cohort["m17"].to_numpy()
    y_fit = y_all[fit_positions]
    y_calib = y_all[calib_positions]
    y_valid = y_all[valid_positions]
    y_test = y_all[test_positions]
    groups_fit = cohort["caseid"].astype("string").to_numpy()[fit_positions]
    fit_digest = membership_digest(keys[fit_positions], seed=seed)
    calib_digest = membership_digest(keys[calib_positions], seed=seed)
    test_digest = membership_digest(keys[test_positions], seed=seed)

    f1_frame = _tier_frame(core, detail, history_feats, "F1")
    enh_frame = _tier_frame(core, detail, history_feats, "F1_detail_history")
    frame_by_tier = {"F1": f1_frame, "F1_detail_history": enh_frame}
    feature_hashes = {
        tier: feature_manifest_hash(
            tier=tier,
            numerical=list(tier_manifest(tier).numerical),
            categorical=list(tier_manifest(tier).categorical),
            builder_id=_builder_id(tier),
        )
        for tier in TIER_ORDER
    }

    tuning = _tuning_stage(
        config,
        f1_frame.iloc[fit_positions].reset_index(drop=True),
        y_fit, groups_fit, private_dir, args.force_retune,
    )
    candidates = _candidate_stage(
        config, tuning, frame_by_tier, fit_positions, calib_positions, valid_positions,
        y_fit, y_calib, y_valid, seed,
    )
    test_payload = _test_stage(
        private_dir, candidates, frame_by_tier, test_positions, y_test, seed, repo, versions,
        allow_test_score=args.allow_test_score,
    )
    f0_block = _f0_stage(
        config, args.f0_parquet, cohort, keys, fit_positions, valid_positions,
        test_positions, y_fit, y_valid, y_test, seed, test_payload,
    )
    shap_block = {"status": "skipped_by_flag"}
    if not args.skip_shap:
        shap_block = _shap_stage(
            config, candidates, frame_by_tier, valid_positions, y_valid,
            fit_positions, y_fit, groups_fit,
        )
    robustness = {"status": "skipped_by_flag"}
    if not args.skip_robustness:
        robustness = _robustness_stage(
            config, cohort, keys, frame_by_tier, partition, train_mask, valid_mask,
            y_all, candidates, seed,
        )

    _write_outputs({
        "config": config,
        "out_dir": out_dir,
        "private_dir": private_dir,
        "cohort": cohort,
        "cohort_summary": cohort_summary,
        "warnings": warnings,
        "repo": repo,
        "versions": versions,
        "input_hash": input_hash,
        "seed": seed,
        "split_hash": split_hash,
        "fit_positions": fit_positions,
        "calib_positions": calib_positions,
        "valid_positions": valid_positions,
        "test_positions": test_positions,
        "fit_digest": fit_digest,
        "calib_digest": calib_digest,
        "test_digest": test_digest,
        "valid_mothers": int(cohort.loc[valid_mask, "caseid"].nunique()),
        "test_mothers": int(cohort.loc[test_mask, "caseid"].nunique()),
        "y_valid": y_valid,
        "candidates": candidates,
        "test_payload": test_payload,
        "f0_block": f0_block,
        "shap_block": shap_block,
        "robustness": robustness,
        "tuning": tuning,
        "feature_hashes": feature_hashes,
        "history_diag": history_diag,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
