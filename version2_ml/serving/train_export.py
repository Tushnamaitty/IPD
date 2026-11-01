"""V2 serving export: refit the A7 XGBoost / F1_detail_history model and save it.

Packaging only, not research. Follows the exact path of
`version2_ml/experiments/A7/src/run_a7.py::main` up to the candidate stage,
for the single shipped candidate:
- frozen C1 cohort (asserted 201,311 births / 158,429 mothers / 44,742 events);
- frozen mother-grouped split (split manifest hash and train/validation
  membership digests asserted against the frozen A1 values; fit-pool and
  calibration-pool digests asserted against the A7 run summary);
- 20 percent mother-grouped calibration carve of the training pool (seed 4208);
- XGBoost fitted on the fit pool with the A7-selected parameters and
  scale_pos_weight = fit-pool negatives / positives;
- Platt calibrator fitted on the calibration carve;
- metrics, thresholds and risk bands on calibrated validation probabilities.

The development test partition is never indexed, predicted or scored here;
A7 already scored it once. Nothing row-level is written: model.joblib holds
fitted objects only, and the other outputs are aggregates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_BASE = _HERE.parents[1]
_A7 = _BASE / "experiments" / "A7"
for _entry in (
    _BASE,
    _BASE / "experiments" / "common",
    _BASE / "experiments" / "A4" / "src",
    _BASE / "experiments" / "A1" / "src",
    _A7 / "src",
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import run_a7  # noqa: E402  (sets matplotlib Agg backend and A7 import paths)
import matplotlib.pyplot as plt  # noqa: E402

from a4_facility_detail import FACILITY_DETAIL_LABELS, facility_detail  # noqa: E402
from a4_history import (  # noqa: E402
    FIRST_RECORDED_LIVE_BIRTH,
    NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
    OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
    UNKNOWN_HISTORY,
    UNOBSERVED_HISTORY,
    build_history_features,
)
from a7_calibration import (  # noqa: E402
    apply_calibrator,
    calibration_intercept_slope,
    f1_at_threshold,
    fit_calibrator,
    threshold_f1_optimal,
    threshold_for_recall,
)
from a7_candidates import (  # noqa: E402
    TIERS,
    carve_calibration_partition,
    fit_candidate,
    predict_candidate,
    scale_pos_weight,
)
from a7_shap import shap_family_table  # noqa: E402
from checksums import membership_digest  # noqa: E402
from data_access import (  # noqa: E402
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
from feature_builder import make_core_features  # noqa: E402
from manifests import split_manifest_hash  # noqa: E402
from model_framework import evaluate_binary  # noqa: E402
from split_contracts import audit_partition, grouped_partition  # noqa: E402

MODEL = "xgboost"
TIER = "F1_detail_history"
CALIBRATION_METHOD = "platt"
MODEL_VERSION_PREFIX = "v2.0-a7-f1dh-xgb-platt"
FEATURE_ORDER = tuple(TIERS[TIER]["numerical"]) + tuple(TIERS[TIER]["categorical"])
NUMERICAL = tuple(TIERS[TIER]["numerical"])
CATEGORICAL = tuple(TIERS[TIER]["categorical"])
RISK_BAND_PERCENTILES = (33.3, 66.7)
EXPECTED_HISTORY_OBSERVED_CS = 6276

# Recorded A7 values (validation, xgboost__F1_detail_history, Platt) and the
# packaging acceptance bands from the V2 implementation brief.
RECORDED_SUMMARY = _A7 / "outputs" / "a7_run_summary.json"
ROC_AUC_TOLERANCE = 0.003
AP_TOLERANCE = 0.005
SLOPE_RANGE = (0.95, 1.05)
F1_THRESHOLD_RANGE = (0.25, 0.35)

# Display labels. facility_type / residence follow feature_builder.make_core_features;
# religion, twin_order and wealth_index codes are labelled from the NFHS-5
# IABR7EFL.DTA value metadata at export time (see _dta_value_labels).
FACILITY_TYPE_LABELS = {
    "public": "Public sector facility",
    "private": "Private sector facility (incl. NGO/trust)",
    "other": "Other / unknown sector",
}
RESIDENCE_LABELS = {"urban": "Urban", "rural": "Rural"}
HISTORY_LEVEL_LABELS = {
    FIRST_RECORDED_LIVE_BIRTH: "First recorded live birth",
    OBSERVED_PREVIOUS_INSTITUTIONAL_CS: "Earlier institutional birth was a C-section",
    NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS: "Earlier institutional birth(s), none a C-section",
    UNOBSERVED_HISTORY: "Higher order, no earlier institutional birth observed",
    UNKNOWN_HISTORY: "Unknown / ambiguous history",
}
_DTA_LABEL_SOURCES = {"religion": "v130", "twin_order": "b0", "wealth_index": "v190"}


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Export the A7 serving model")
    parser.add_argument(
        "--nfhs5-path",
        default=os.environ.get("NFHS5_PATH"),
        help="Local NFHS-5 IABR7EFL.DTA (default: $env:NFHS5_PATH)",
    )
    parser.add_argument("--out-dir", default=str(_HERE.parent / "models"))
    parser.add_argument("--config", default=str(_A7 / "config.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args(argv)
    if not args.nfhs5_path:
        parser.error("--nfhs5-path is required when NFHS5_PATH is not set")
    return args


def _dta_value_labels(path: Path) -> dict[str, dict[str, str]]:
    import pyreadstat

    _, meta = pyreadstat.read_dta(
        str(path), metadataonly=True, usecols=list(_DTA_LABEL_SOURCES.values())
    )
    labels = {}
    for feature, variable in _DTA_LABEL_SOURCES.items():
        label_set = meta.variable_to_label.get(variable)
        values = meta.value_labels.get(label_set, {}) if label_set else {}
        labels[feature] = {str(int(code)): str(text) for code, text in values.items()}
    return labels


def _encoder_vocabulary(classifier) -> dict[str, list[str]]:
    # Same source a7_shap.family_map reads: the fitted one-hot encoder categories.
    pipeline = classifier.preprocessor.pipeline
    manifest = classifier.preprocessor.manifest
    encoder = pipeline.named_transformers_["categorical"].named_steps["onehot"]
    return {
        column: [str(value) for value in values]
        for column, values in zip(manifest.categorical, encoder.categories_)
    }


def _display_labels(vocabulary, dta_labels) -> dict[str, dict[str, str]]:
    sources = {
        "facility_detail": {label: label for label in FACILITY_DETAIL_LABELS.values()},
        "facility_type": FACILITY_TYPE_LABELS,
        "residence": RESIDENCE_LABELS,
        "religion": dta_labels["religion"],
        "twin_order": dta_labels["twin_order"],
        "history_four_level": HISTORY_LEVEL_LABELS,
    }
    labels = {}
    for column, mapping in sources.items():
        missing = [value for value in vocabulary[column] if value not in mapping]
        if missing:
            raise RuntimeError(f"{column}: encoder categories without a label: {missing}")
        labels[column] = {value: mapping[value] for value in vocabulary[column]}
    return labels


def _numeric_ranges(fit_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges = {}
    for column in NUMERICAL:
        values = pd.to_numeric(fit_frame[column], errors="coerce")
        ranges[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "n_missing_fit_pool": int(values.isna().sum()),
        }
    return ranges


def _check(results, name, recorded, reproduced, passed, rule):
    results.append({
        "metric": name,
        "recorded_a7": recorded,
        "reproduced": reproduced,
        "rule": rule,
        "pass": bool(passed),
    })


def _print_checks(results) -> None:
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    header = f"{'metric':<34}{'recorded A7':>13}{'reproduced':>13}  {'rule':<26}result"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['metric']:<34}{fmt(row['recorded_a7']):>13}{fmt(row['reproduced']):>13}  "
            f"{row['rule']:<26}{'PASS' if row['pass'] else 'FAIL'}"
        )


def _fig_shap(table: dict, path: Path) -> None:
    rows = list(reversed(table["families"]))
    families = [row["family"] for row in rows]
    shares = [row["share_of_total"] for row in rows]
    surface, ink, ink_secondary, grid, bar = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6"
    fig, axis = plt.subplots(figsize=(9, 5.4), facecolor=surface)
    axis.set_facecolor(surface)
    positions = np.arange(len(families))
    axis.barh(positions, shares, height=0.55, color=bar, edgecolor=surface, linewidth=2)
    for position, share in zip(positions, shares):
        axis.text(share + 0.004, position, f"{share:.1%}", va="center", fontsize=9, color=ink_secondary)
    axis.set_yticks(positions)
    axis.set_yticklabels(families, fontsize=9, color=ink)
    axis.set_xlim(0, max(shares) * 1.15)
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.tick_params(axis="x", colors=ink_secondary, labelsize=8)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color=grid, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(grid)
    axis.set_xlabel("share of total mean |SHAP| (log-odds margin)", fontsize=9, color=ink_secondary)
    fig.suptitle(
        "What drives the V2 C-section risk score", x=0.01, ha="left",
        fontsize=12, color=ink,
    )
    fig.text(
        0.01, 0.01,
        f"Validation sample n={table['n_sample_rows']:,}; predictive attributions, not causal effects.",
        fontsize=7.5, color=ink_secondary,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200, facecolor=surface)
    plt.close(fig)


def prepare_data(nfhs5_path: Path, config: dict, recorded: dict, seed: int) -> dict:
    """Frozen cohort, split, calibration carve and tier frames, exactly as run_a7.main.

    Returns fit, calibration and validation rows only; test positions are never built.
    """
    print("hashing input ...", flush=True)
    input_hash = sha256_file(nfhs5_path)
    if input_hash != recorded["input_sha256"]:
        raise RuntimeError("NFHS-5 input sha256 differs from the A7 recording")

    print("reading NFHS-5 subset ...", flush=True)
    raw = read_dta_subset(nfhs5_path, extra_columns=("v021",))
    cohort, cohort_summary = build_c1_cohort(raw)
    warnings = assert_frozen_c1(cohort_summary)
    for key in ("n_births", "n_mothers", "n_events"):
        if int(config["expected_cohort"][key]) != int(cohort_summary[key]):
            raise RuntimeError(
                f"Config expects {key}={config['expected_cohort'][key]} but observed {cohort_summary[key]}"
            )

    core = make_core_features(cohort, wave="NFHS-5")
    detail = facility_detail(cohort["m15"])
    history_feats, history_diag = build_history_features(raw, cohort, include_other_unknown_sector=True)
    observed_prior_cs = history_diag["level_counts"].get(OBSERVED_PREVIOUS_INSTITUTIONAL_CS, 0)
    if observed_prior_cs != EXPECTED_HISTORY_OBSERVED_CS:
        raise RuntimeError(
            f"History reproduction failure: {observed_prior_cs} vs audit {EXPECTED_HISTORY_OBSERVED_CS}"
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

    keys = run_a7._key_tokens(cohort)
    train_mask = partition.eq("train").to_numpy()
    valid_mask = partition.eq("validation").to_numpy()
    if membership_digest(keys[train_mask], seed=seed) != FROZEN_TRAIN_DIGEST:
        raise RuntimeError("Train membership digest differs from the frozen A1 digest")
    if membership_digest(keys[valid_mask], seed=seed) != FROZEN_VALID_DIGEST:
        raise RuntimeError("Validation membership digest differs from the frozen A1 digest")

    train_positions = np.flatnonzero(train_mask)
    valid_positions = np.flatnonzero(valid_mask)
    train_cohort = cohort.iloc[train_positions].reset_index(drop=True)
    carve = carve_calibration_partition(
        train_cohort,
        calibration_fraction=float(config["candidates"]["calibration"]["subset_fraction"]),
        seed=int(config["candidates"]["calibration"]["seed"]),
    )
    fit_positions = train_positions[carve.eq("fit").to_numpy()]
    calib_positions = train_positions[carve.eq("calibration").to_numpy()]
    if membership_digest(keys[fit_positions], seed=seed) != recorded["frozen_digests"]["fit_pool"]:
        raise RuntimeError("Fit-pool membership digest differs from the A7 recording")
    if membership_digest(keys[calib_positions], seed=seed) != recorded["frozen_digests"]["calibration_pool"]:
        raise RuntimeError("Calibration-pool membership digest differs from the A7 recording")
    y_all = cohort["m17"].to_numpy()
    y_fit = y_all[fit_positions]
    y_calib = y_all[calib_positions]
    y_valid = y_all[valid_positions]

    frame = run_a7._tier_frame(core, detail, history_feats, TIER)
    fit_frame = frame.iloc[fit_positions].reset_index(drop=True)
    calib_frame = frame.iloc[calib_positions].reset_index(drop=True)
    valid_frame = frame.iloc[valid_positions].reset_index(drop=True)

    return {
        "input_hash": input_hash,
        "cohort_summary": cohort_summary,
        "warnings": warnings,
        "split_hash": split_hash,
        "fit_positions": fit_positions,
        "calib_positions": calib_positions,
        "y_fit": y_fit,
        "y_calib": y_calib,
        "y_valid": y_valid,
        "fit_frame": fit_frame,
        "calib_frame": calib_frame,
        "valid_frame": valid_frame,
    }


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    config = run_a7._load_config(Path(args.config))
    seed = int(args.seed)
    if seed != int(config["split"]["seed"]):
        raise RuntimeError(f"--seed {seed} differs from the frozen split seed {config['split']['seed']}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nfhs5_path = Path(args.nfhs5_path)
    if not nfhs5_path.exists():
        raise FileNotFoundError(f"NFHS-5 file not found: {nfhs5_path}")
    recorded = json.loads(RECORDED_SUMMARY.read_text(encoding="utf-8"))
    recorded_candidate = recorded["candidate_selection"][f"{MODEL}__{TIER}"]
    if recorded_candidate["selected_method"] != CALIBRATION_METHOD:
        raise RuntimeError("A7 did not select Platt calibration for the shipped candidate")

    repo = repo_state()
    versions = software_versions()
    data = prepare_data(nfhs5_path, config, recorded, seed)
    input_hash, warnings, split_hash = data["input_hash"], data["warnings"], data["split_hash"]
    fit_positions, calib_positions = data["fit_positions"], data["calib_positions"]
    y_fit, y_calib, y_valid = data["y_fit"], data["y_calib"], data["y_valid"]
    fit_frame, calib_frame, valid_frame = data["fit_frame"], data["calib_frame"], data["valid_frame"]

    params = dict(recorded["tuning"]["selected_params"][MODEL])
    params["scale_pos_weight"] = scale_pos_weight(y_fit)
    print(f"fitting {MODEL} / {TIER} on {len(fit_positions):,} fit-pool rows ...", flush=True)
    classifier = fit_candidate(MODEL, TIER, params, fit_frame, y_fit, seed)
    p_calib_base = predict_candidate(classifier, calib_frame, "train")
    calibrator = fit_calibrator(CALIBRATION_METHOD, p_calib_base, y_calib)
    p_valid_base = predict_candidate(classifier, valid_frame, "validation")
    p_valid = apply_calibrator(CALIBRATION_METHOD, calibrator, p_valid_base)

    metrics = evaluate_binary(y_valid, p_valid, threshold=float(config["thresholds"]["primary"]))
    intercept, slope = calibration_intercept_slope(y_valid, p_valid)
    threshold_f1, best_f1 = threshold_f1_optimal(y_valid, p_valid)
    threshold_rec = threshold_for_recall(y_valid, p_valid, 0.8)
    if threshold_rec is None:
        raise RuntimeError("Recall >= 0.80 is not attainable on validation")
    validation = run_a7._metrics_payload(metrics, intercept, slope)
    operating_points = {}
    for name, value in (("f1_optimal", threshold_f1), ("recall_80", threshold_rec)):
        point = evaluate_binary(y_valid, p_valid, threshold=value)
        operating_points[name] = {
            "threshold": value,
            "precision": float(point["precision"]),
            "recall": float(point["recall"]),
            "f1": float(f1_at_threshold(y_valid, p_valid, value)),
        }
    cut_low, cut_high = (float(value) for value in np.percentile(p_valid, RISK_BAND_PERCENTILES))

    rec_metrics = recorded_candidate["validation_metrics"]
    rec_thresholds = recorded_candidate["thresholds"]
    checks = []
    _check(checks, "validation ROC-AUC", rec_metrics["roc_auc"], validation["roc_auc"],
           abs(validation["roc_auc"] - 0.8298) <= ROC_AUC_TOLERANCE, "within 0.003 of 0.8298")
    _check(checks, "validation PR-AUC (AP)", rec_metrics["average_precision"], validation["average_precision"],
           abs(validation["average_precision"] - 0.6113) <= AP_TOLERANCE, "within 0.005 of 0.6113")
    _check(checks, "calibration slope", rec_metrics["calibration_slope"], validation["calibration_slope"],
           SLOPE_RANGE[0] <= validation["calibration_slope"] <= SLOPE_RANGE[1], "in [0.95, 1.05]")
    _check(checks, "F1-optimal threshold", rec_thresholds["f1_optimal"], threshold_f1,
           F1_THRESHOLD_RANGE[0] <= threshold_f1 <= F1_THRESHOLD_RANGE[1], "in [0.25, 0.35]")
    _check(checks, "Brier (calibrated)", rec_metrics["brier_score"], validation["brier_score"],
           True, "reported")
    _check(checks, "calibration intercept", rec_metrics["calibration_intercept"],
           validation["calibration_intercept"], True, "reported")
    _check(checks, "F1 at F1-optimal threshold", rec_thresholds["f1_optimal_value"], best_f1,
           True, "reported")
    _check(checks, "recall>=0.80 threshold", rec_thresholds["recall_ge_0.80"], threshold_rec,
           True, "reported")
    _check(checks, "validation n", rec_metrics["n"], validation["n"],
           rec_metrics["n"] == validation["n"], "equal")
    _check(checks, "validation prevalence", rec_metrics["prevalence"], validation["prevalence"],
           abs(rec_metrics["prevalence"] - validation["prevalence"]) < 1e-12, "equal")

    shap_table = None
    if not args.skip_shap:
        print("computing SHAP on validation sample ...", flush=True)
        shap_table = shap_family_table(
            classifier, valid_frame,
            sample_size=int(config["shap"]["sample_size"]),
            seed=int(config["shap"]["seed"]),
        )
        recorded_top = recorded["shap"]["top_families"]
        reproduced_top = [row["family"] for row in shap_table["families"][: len(recorded_top)]]
        _check(checks, "SHAP top-6 family order", "recorded", "same" if reproduced_top == recorded_top else "differs",
               reproduced_top == recorded_top, "reported")
        checks[-1]["pass"] = True  # informational only; not an acceptance criterion

    _print_checks(checks)
    failed = [row["metric"] for row in checks if not row["pass"]]
    if failed:
        print(f"\nACCEPTANCE CHECK FAILED: {failed}; nothing saved.")
        return 1

    vocabulary = _encoder_vocabulary(classifier)
    dta_labels = _dta_value_labels(nfhs5_path)
    labels = _display_labels(vocabulary, dta_labels)
    labels["wealth_index"] = dta_labels["wealth_index"]

    model_path = out_dir / "model.joblib"
    joblib.dump(
        {
            "classifier": classifier,
            "calibrator": calibrator,
            "calibration_method": CALIBRATION_METHOD,
            "feature_order": list(FEATURE_ORDER),
        },
        model_path,
    )
    model_sha = sha256_file(model_path)
    short_sha = repo["repo_commit"][:7]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    config_payload = {
        "model_version": f"{MODEL_VERSION_PREFIX}-{short_sha}",
        "generated_at_utc": generated,
        "repo": repo,
        "software_versions": versions,
        "model": MODEL,
        "feature_tier": TIER,
        "calibration_method": CALIBRATION_METHOD,
        "params": params,
        "feature_order": list(FEATURE_ORDER),
        "numerical_features": list(NUMERICAL),
        "categorical_features": list(CATEGORICAL),
        "thresholds": {
            "f1_optimal": threshold_f1,
            "recall_80": threshold_rec,
            "default": "f1_optimal",
        },
        "risk_bands": {
            "basis": "33.3rd and 66.7th percentiles of calibrated validation probabilities",
            "low_upper": cut_low,
            "high_lower": cut_high,
            "labels": ["Low", "Medium", "High"],
        },
        "vocabulary": vocabulary,
        "display_labels": labels,
        "numeric_ranges": _numeric_ranges(fit_frame),
        "model_sha256": model_sha,
        "provenance": {
            "source": "version2_ml/experiments/A7 (xgboost__F1_detail_history, Platt)",
            "input_sha256": input_hash,
            "split_manifest_hash": split_hash,
            "fit_rows": int(len(fit_positions)),
            "calibration_rows": int(len(calib_positions)),
            "test_partition_scored": False,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

    metrics_payload = {
        "model_version": config_payload["model_version"],
        "generated_at_utc": generated,
        "partition": "validation",
        "n_births": validation["n"],
        "positive_n": validation["positive_n"],
        "prevalence": validation["prevalence"],
        "roc_auc": validation["roc_auc"],
        "average_precision": validation["average_precision"],
        "brier_score": validation["brier_score"],
        "calibration_intercept": validation["calibration_intercept"],
        "calibration_slope": validation["calibration_slope"],
        "at_threshold_0_50": {
            key: validation[key] for key in ("precision", "recall", "f1")
        },
        "operating_points": operating_points,
        "acceptance_checks": checks,
        "cohort_warnings": warnings,
        "disclosure": "Unweighted point estimates on the development validation partition; no design-based intervals.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    if shap_table is not None:
        top = pd.DataFrame(shap_table["families"][:10])[["rank", "family", "mean_abs_shap_margin", "share_of_total"]]
        top.to_csv(out_dir / "shap_top10.csv", index=False)
        _fig_shap(shap_table, out_dir / "shap_summary.png")

    print(f"\nsaved to {out_dir}: model.joblib (sha256 {model_sha[:12]}...), config.json, metrics.json"
          + (", shap_summary.png, shap_top10.csv" if shap_table is not None else ""))
    print(f"model_version: {config_payload['model_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
