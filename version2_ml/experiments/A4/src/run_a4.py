"""A4 CLI runner: observed-history builder, facility detail and matched ablations.

Runs on the frozen C1 NFHS-5 cohort with the identical provisional
mother-grouped split used by A1 (split hash and membership digests are asserted
against the frozen A1 values). Outputs are aggregate-only and disclosure
suppressed; the timeline and all row-level history records stay in memory.
The test partition is never scored here (single frozen A7 evaluation).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_HERE = Path(__file__).resolve()
_VERSION2_ML = _HERE.parents[3]
_COMMON_DIR = _VERSION2_ML / "experiments" / "common"
for _entry in (str(_VERSION2_ML), str(_COMMON_DIR), str(_HERE.parent)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from xgboost import XGBClassifier

from a4_facility_detail import (
    DETAIL_FEATURE,
    LABEL_SOURCE,
    crosswalk_rows,
    facility_detail,
)
from a4_history import (
    FIRST_RECORDED_LIVE_BIRTH,
    HAS_ANY_FEATURE,
    HISTORY_FEATURE,
    OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
    QUALITY_FEATURE,
    build_history_features,
    history_support_counts,
)
from aggregate_schema import (
    UNCERTAINTY_STATUS,
    build_row,
    suppress_small_count,
    write_rows,
)
from checksums import assert_matched_membership, membership_digest
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
from feature_builder import CORE_NAMES, make_core_features
from manifests import feature_manifest_hash, split_manifest_hash
from model_framework import AuthorizedClassifier, evaluate_binary
from preprocessing import FeatureManifest
from split_contracts import audit_partition, grouped_partition

F1_NUMERICAL = CORE_NAMES[:4]
F1_CATEGORICAL = CORE_NAMES[4:]
HISTORY_FEATURE_NAMES = (HISTORY_FEATURE, HAS_ANY_FEATURE, QUALITY_FEATURE)

ARM_SPECS = (
    {
        "tier": "F1",
        "manifest_tier": "F1",
        "categorical": F1_CATEGORICAL,
        "uses_detail": False,
        "uses_history": False,
        "drop_broad_facility": False,
        "label": "F1 (eight-core, broad facility_type)",
    },
    {
        "tier": "F1_detail",
        "manifest_tier": "F2",
        "categorical": F1_CATEGORICAL + (DETAIL_FEATURE,),
        "uses_detail": True,
        "uses_history": False,
        "drop_broad_facility": False,
        "label": "F1 + detailed m15 facility",
    },
    {
        "tier": "F1_detail_replace",
        "manifest_tier": "F2",
        "categorical": (F1_CATEGORICAL[1:] + (DETAIL_FEATURE,)),
        "uses_detail": True,
        "uses_history": False,
        "drop_broad_facility": True,
        "label": "F1 with detailed m15 facility replacing broad facility_type",
    },
    {
        "tier": "F1_history",
        "manifest_tier": "F2",
        "categorical": F1_CATEGORICAL + HISTORY_FEATURE_NAMES,
        "uses_detail": False,
        "uses_history": True,
        "drop_broad_facility": False,
        "label": "F1 + observed delivery history",
    },
    {
        "tier": "F1_detail_history",
        "manifest_tier": "F2",
        "categorical": F1_CATEGORICAL + (DETAIL_FEATURE,) + HISTORY_FEATURE_NAMES,
        "uses_detail": True,
        "uses_history": True,
        "drop_broad_facility": False,
        "label": "F1 + detailed m15 facility + observed delivery history",
    },
)

EXPECTED_FEASIBILITY = {
    FIRST_RECORDED_LIVE_BIRTH: 82607,
    OBSERVED_PREVIOUS_INSTITUTIONAL_CS: 6276,
}

_MODEL_LABELS = {"xgboost": "XGBoost", "logistic_regression": "Logistic regression"}
_TIER_COLORS = {
    "F1": "#1f77b4",
    "F1_detail": "#d62728",
    "F1_detail_replace": "#2ca02c",
    "F1_history": "#9467bd",
    "F1_detail_history": "#ff7f0e",
}
_FOOTNOTE = (
    "Development-validation point estimates only; no intervals; provisional "
    "mother-grouped split; test partition not scored."
)
COHORT_RULE_SENSITIVITY = "c1_m15_institutional_b19_0_59_m17_0_1_excl_96"
DESIGN_MAIN = "matched_ablation_validation_only_include_m15_96"
DESIGN_SENSITIVITY = "matched_ablation_validation_only_m15_96_excluded_sensitivity"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="A4 observed-history and facility-detail ablation (aggregate-only)"
    )
    parser.add_argument(
        "--nfhs5-path",
        required=True,
        help="Authorized local NFHS-5 birth recode file (.DTA); never committed",
    )
    parser.add_argument("--config", default=str(_HERE.parents[1] / "config.yaml"))
    parser.add_argument("--out-dir", default=str(_HERE.parents[1] / "outputs"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = (
        "split",
        "models",
        "evaluation",
        "disclosure",
        "expected_cohort",
        "feasibility",
        "arms",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config missing sections: {missing}")
    config_tiers = {str(tier) for tier in config["arms"]}
    spec_tiers = {spec["tier"] for spec in ARM_SPECS}
    if config_tiers != spec_tiers:
        raise ValueError(
            f"Config arms {sorted(config_tiers)} differ from implemented arms "
            f"{sorted(spec_tiers)}"
        )
    return config


def _arm_builder_id(spec: dict) -> str:
    builder = "version2_ml/feature_builder.py::make_core_features"
    if spec["uses_detail"]:
        builder += " + version2_ml/experiments/A4/src/a4_facility_detail.py::facility_detail"
    if spec["uses_history"]:
        builder += " + version2_ml/experiments/A4/src/a4_history.py::build_history_features"
    if spec["drop_broad_facility"]:
        builder += " (broad facility_type replaced)"
    return builder


def _arm_frame(core, detail, history_feats, spec) -> pd.DataFrame:
    frame = core.copy()
    if spec["drop_broad_facility"]:
        frame = frame.drop(columns=["facility_type"])
    if DETAIL_FEATURE in spec["categorical"]:
        if detail is None:
            raise RuntimeError("Facility detail requested but not built")
        frame[DETAIL_FEATURE] = detail.to_numpy()
    if HISTORY_FEATURE in spec["categorical"]:
        if history_feats is None:
            raise RuntimeError("History features requested but not built")
        for column in HISTORY_FEATURE_NAMES:
            frame[column] = history_feats[column].to_numpy()
    expected = set(spec["categorical"]) | set(F1_NUMERICAL)
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise RuntimeError(f"Arm frame missing required columns: {missing}")
    return frame


def _estimator_for(model_name: str, config: dict, seed: int, y_train: np.ndarray):
    if model_name == "xgboost":
        params = dict(config["models"]["xgboost"])
        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)
        if positives == 0 or negatives == 0:
            raise ValueError("Training partition lost an outcome class")
        params["scale_pos_weight"] = float(negatives) / float(positives)
        params.update({"random_state": seed, "n_jobs": 1, "verbosity": 0})
        return XGBClassifier(**params), params
    params = dict(config["models"]["logistic_regression"])
    params.update({"random_state": seed})
    estimator = LogisticRegression(**params)
    recorded = dict(params)
    recorded["penalty"] = "l2 (library default; deprecated explicit argument omitted)"
    return estimator, recorded


def _fit_arm(estimator, manifest, train_frame, y_train, valid_frame, seed):
    classifier = AuthorizedClassifier(manifest, algorithm="logistic_regression", seed=seed)
    classifier.estimator = estimator
    classifier.fit(train_frame, y_train, partition="train")
    return classifier.predict_proba(valid_frame, partition="validation")


def _masked_counts(mapping: dict, threshold: int) -> dict:
    return {
        str(key): suppress_small_count(int(value), threshold=threshold)
        for key, value in mapping.items()
    }


def _run_block(
    block_name: str,
    cohort: pd.DataFrame,
    core: pd.DataFrame,
    detail: pd.Series | None,
    history_feats: pd.DataFrame | None,
    partition,
    split_hash: str,
    config: dict,
    seed: int,
    suppression: int,
):
    train_mask = partition.eq("train").to_numpy()
    valid_mask = partition.eq("validation").to_numpy()
    keys = (
        cohort["caseid"].astype("string")
        + "|"
        + cohort["bidx"].map(lambda value: str(int(value))).astype("string")
    ).to_numpy()
    train_digest = membership_digest(keys[train_mask], seed=seed)
    valid_digest = membership_digest(keys[valid_mask], seed=seed)

    y_train = cohort.loc[train_mask, "m17"].to_numpy()
    y_valid = cohort.loc[valid_mask, "m17"].to_numpy()
    train_mothers = int(cohort.loc[train_mask, "caseid"].nunique())
    valid_mothers = int(cohort.loc[valid_mask, "caseid"].nunique())

    predictions = {}
    metrics_by_model = {}
    model_params = {}
    frames = {}
    manifests = {}
    feature_hashes = {}
    for spec in ARM_SPECS:
        frame = _arm_frame(core, detail, history_feats, spec)
        frames[spec["tier"]] = frame
        manifests[spec["tier"]] = FeatureManifest(
            F1_NUMERICAL, spec["categorical"], spec["manifest_tier"]
        )
        feature_hashes[spec["tier"]] = feature_manifest_hash(
            tier=spec["tier"],
            numerical=F1_NUMERICAL,
            categorical=spec["categorical"],
            builder_id=_arm_builder_id(spec),
        )
        if not frame.index.equals(core.index):
            raise RuntimeError(f"{spec['tier']}: paired arms must share row order")

    for model_name in ("xgboost", "logistic_regression"):
        metrics_by_model[model_name] = {}
        for spec in ARM_SPECS:
            estimator, params = _estimator_for(model_name, config, seed, y_train)
            model_params[model_name] = params
            arm_digest = membership_digest(keys[train_mask], seed=seed)
            assert_matched_membership(train_digest, arm_digest)
            frame = frames[spec["tier"]]
            probabilities = _fit_arm(
                estimator,
                manifests[spec["tier"]],
                frame.loc[train_mask],
                y_train,
                frame.loc[valid_mask],
                seed,
            )
            predictions[(model_name, spec["tier"])] = probabilities
            metrics_by_model[model_name][spec["tier"]] = evaluate_binary(
                y_valid, probabilities, threshold=float(config["evaluation"]["threshold"])
            )

    rows = []
    versions_string = json.dumps(
        software_versions(), sort_keys=True, separators=(",", ":")
    )
    repo = repo_state()
    cohort_rule = C1_COHORT_RULE if block_name == "main" else COHORT_RULE_SENSITIVITY
    design = DESIGN_MAIN if block_name == "main" else DESIGN_SENSITIVITY
    for model_name in metrics_by_model:
        for spec in ARM_SPECS:
            metrics = metrics_by_model[model_name][spec["tier"]]
            rows.append(
                build_row(
                    experiment_id="A4",
                    wave="NFHS-5",
                    cohort_rule=cohort_rule,
                    evaluation_design=design,
                    split_id=config["split"]["split_id"],
                    fold="validation",
                    source_domain="NFHS-5",
                    target_domain="NFHS-5",
                    model=model_name,
                    feature_tier=spec["tier"],
                    baseline_id="F1_reference",
                    n_births=int(metrics["n"]),
                    n_mothers=valid_mothers,
                    n_events=int(metrics["positive_n"]),
                    prevalence=float(metrics["weighted_outcome_prevalence"]),
                    roc_auc=float(metrics["roc_auc"]),
                    average_precision=float(metrics["average_precision"]),
                    brier_score=float(metrics["brier_score"]),
                    threshold=float(config["evaluation"]["threshold"]),
                    precision=float(metrics["precision"]),
                    recall=float(metrics["recall"]),
                    f1=float(metrics["f1"]),
                    weighted_descriptive_flag="unweighted_primary_analysis",
                    uncertainty_status=UNCERTAINTY_STATUS,
                    repo_commit=repo["repo_commit"],
                    seed=seed,
                    software_versions=versions_string,
                    feature_manifest_hash=feature_hashes[spec["tier"]],
                    split_manifest_hash=split_hash,
                )
            )

    block = {
        "block": block_name,
        "rows": rows,
        "metrics_by_model": metrics_by_model,
        "predictions": predictions,
        "model_params": model_params,
        "feature_hashes": feature_hashes,
        "feature_names": {tier: list(frame.columns) for tier, frame in frames.items()},
        "train_digest": train_digest,
        "valid_digest": valid_digest,
        "train_mothers": train_mothers,
        "valid_mothers": valid_mothers,
        "y_valid": y_valid,
        "y_train": y_train,
        "train_mask": train_mask,
        "valid_mask": valid_mask,
        "suppression": suppression,
    }
    return block


def _write_crosswalk(out_dir: Path) -> Path:
    rows = crosswalk_rows()
    path = out_dir / "a4_facility_crosswalk.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "wave",
                "m15_code",
                "broad_class",
                "detail_label",
                "notes",
                "verification_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_history_support(out_dir: Path, support_rows, validation_counts, threshold: int) -> Path:
    path = out_dir / "a4_history_support.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "value",
                "n_births",
                "n_positive",
                "n_negative",
                "n_validation",
            ],
        )
        writer.writeheader()
        for row in support_rows:
            writer.writerow(
                {
                    "group": row["group"],
                    "value": row["value"],
                    "n_births": suppress_small_count(row["n_births"], threshold=threshold),
                    "n_positive": suppress_small_count(row["n_positive"], threshold=threshold),
                    "n_negative": suppress_small_count(row["n_negative"], threshold=threshold),
                    "n_validation": suppress_small_count(
                        validation_counts.get((row["group"], row["value"]), 0),
                        threshold=threshold,
                    ),
                }
            )
    return path


def _write_roc_pr_figure(out_dir: Path, block: dict, suffix: str) -> Path:
    y_valid = block["y_valid"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for (model_name, tier), probabilities in sorted(block["predictions"].items()):
        color = _TIER_COLORS[tier]
        style = "-" if model_name == "xgboost" else "--"
        label = f"{_MODEL_LABELS[model_name]}: {tier}"
        fpr, tpr, _ = roc_curve(y_valid, probabilities)
        axes[0].plot(fpr, tpr, color=color, linestyle=style, label=label, linewidth=1.6)
        precision, recall, _ = precision_recall_curve(y_valid, probabilities)
        axes[1].plot(recall, precision, color=color, linestyle=style, label=label, linewidth=1.6)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC (development validation)")
    prevalence = float(np.mean(y_valid))
    axes[1].axhline(prevalence, color="gray", linestyle="--", linewidth=1,
                    label=f"Prevalence = {prevalence:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision-Recall (development validation)")
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
    figure.text(0.01, 0.01, _FOOTNOTE, fontsize=7, color="dimgray")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    path = out_dir / f"fig_a4_roc_pr{suffix}.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _write_history_distribution_figure(
    out_dir: Path, history_feats: pd.DataFrame, y_valid: np.ndarray, threshold: int
) -> Path:
    levels = history_feats[HISTORY_FEATURE]
    validation = pd.DataFrame(
        {"level": levels.to_numpy(), "y": y_valid}
    )
    levels_order = [level for level in levels.value_counts().index]
    supported = []
    for level in levels_order:
        subset = validation[validation["level"] == level]
        if len(subset) and int(subset["y"].sum()) >= threshold and int((subset["y"] == 0).sum()) >= threshold:
            supported.append(level)
    groups = ["cesarean", "non-cesarean"]
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    width = 0.38
    positions = np.arange(len(supported))
    for index, (outcome, group_label) in enumerate(((1, groups[0]), (0, groups[1]))):
        shares = []
        for level in supported:
            subset = validation[(validation["level"] == level) & (validation["y"] == outcome)]
            group_total = int((validation["y"] == outcome).sum())
            shares.append(len(subset) / group_total if group_total else 0.0)
        axis.bar(positions + (index - 0.5) * width, shares, width, label=group_label)
    axis.set_xticks(positions)
    axis.set_xticklabels([str(level) for level in supported], rotation=20, ha="right", fontsize=8)
    axis.set(ylabel="Share within outcome group", title="Observed history levels by validation outcome")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    figure.text(0.01, 0.01,
                _FOOTNOTE + " Levels with a validation cell below 10 are omitted.",
                fontsize=7, color="dimgray")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    path = out_dir / "fig_a4_history_distribution.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _write_methods(
    out_dir: Path,
    context: dict,
    main_block: dict,
    sensitivity_block: dict,
    history_diagnostics: dict,
    facility_source: str,
    crosswalk_path: Path,
    support_path: Path,
) -> Path:
    lines = []
    lines.append("# A4 - Observed delivery history and facility detail: methods and results (auto-generated)")
    lines.append("")
    lines.append(
        f"Generated: {context['timestamp']} | Commit: {context['repo_commit']} "
        f"({context['branch']}) | Seed: {context['seed']}"
    )
    lines.append("")
    lines.append("## Methods (as executed)")
    lines.append("")
    cohort = context["cohort"]
    lines.append(
        f"- Frozen cohort C1: {cohort['n_births']} births / {cohort['n_mothers']} "
        f"mothers ({cohort['n_events']} events); m15 = 96 rows: {cohort['m15_96_rows']} "
        "(included per D2)."
    )
    lines.append(
        f"- Provisional mother-grouped split (70/15/15, seed {context['seed']}); "
        f"split hash `{context['split_hash']}` equals the frozen A1 split, and "
        f"train/validation membership digests equal the A1 digests "
        f"(train `{context['train_digest']}`, validation `{context['valid_digest']}`)."
    )
    lines.append(
        "- Observed-history timeline: all survey-window roster births (b19 0-59) of "
        "each mother, sorted by (caseid, b3); home deliveries stay in the timeline as "
        "observed noninstitutional records; births outside the window are never used. "
        "Only strictly earlier calendar months contribute (same-month twins never "
        "count as previous deliveries); the outcome of the index birth or later "
        "births is never used."
    )
    lines.append(
        "- Four-category history (index births only): "
        "first_recorded_live_birth (bord==1); observed_previous_institutional_cs "
        "(strictly earlier institutional birth with m17==1); "
        "higher_order_no_observed_previous_institutional_cs (strictly earlier "
        "institutional birth(s), none with m17==1); "
        "higher_order_earlier_institutional_history_unobserved (no strictly earlier "
        "institutional record; includes earlier noninstitutional and out-of-recall "
        "births). Missing or ambiguous records yield unknown_history plus "
        "history_quality_flag tokens; they are never treated as negatives."
    )
    lines.append(
        f"- Facility detail labels are verbatim NFHS-5 metadata: {facility_source}; "
        f"crosswalk `{crosswalk_path.name}` (NFHS-4 pending local verification)."
    )
    lines.append(
        "- Ablation arms (identical rows and order across arms): F1; F1 + detailed "
        "m15 facility; F1 with detailed facility replacing broad facility_type; "
        "F1 + observed history; F1 + both. Models: XGBoost and logistic regression "
        "(same configurations as A1)."
    )
    lines.append(
        "- m15 != 96 sensitivity: 96 rows removed consistently from the index cohort "
        "and the history timeline, with the mother partition inherited from the main "
        "split; threshold fixed at 0.50."
    )
    lines.append(
        "- Point estimates only; no design-based intervals or significance claims; "
        "cells below 10 suppressed."
    )
    lines.append("")
    lines.append("## Feasibility cross-checks (Phase-0 audit reproduction)")
    lines.append("")
    for key, expected in context["feasibility"].items():
        observed = history_diagnostics["level_counts"].get(key, 0)
        lines.append(f"- {key}: observed {observed}, audit expected {expected}, match={observed == expected}.")
    lines.append("")
    lines.append("## Results (development validation, point estimates)")
    lines.append("")
    lines.append(
        "| model | tier | n_births | n_events | ROC-AUC | average precision | Brier | "
        "precision@0.50 | recall@0.50 | F1@0.50 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for model_name in main_block["metrics_by_model"]:
        for spec in ARM_SPECS:
            metrics = main_block["metrics_by_model"][model_name][spec["tier"]]
            lines.append(
                f"| {_MODEL_LABELS[model_name]} | {spec['tier']} | {metrics['n']} | "
                f"{metrics['positive_n']} | {metrics['roc_auc']:.4f} | "
                f"{metrics['average_precision']:.4f} | {metrics['brier_score']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |"
            )
    lines.append("")
    for model_name, tiers in main_block["metrics_by_model"].items():
        base = tiers["F1"]
        for spec in ARM_SPECS:
            if spec["tier"] == "F1":
                continue
            candidate = tiers[spec["tier"]]
            lines.append(
                f"- {_MODEL_LABELS[model_name]} {spec['tier']} minus F1 (point estimates): "
                f"ROC-AUC {candidate['roc_auc'] - base['roc_auc']:+.4f}, average precision "
                f"{candidate['average_precision'] - base['average_precision']:+.4f}, Brier "
                f"{candidate['brier_score'] - base['brier_score']:+.4f}."
            )
    lines.append("")
    lines.append("### m15 != 96 sensitivity")
    lines.append("")
    lines.append(
        "| model | tier | n_births | n_events | ROC-AUC | average precision | Brier |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for model_name in sensitivity_block["metrics_by_model"]:
        for spec in ARM_SPECS:
            metrics = sensitivity_block["metrics_by_model"][model_name][spec["tier"]]
            lines.append(
                f"| {_MODEL_LABELS[model_name]} | {spec['tier']} | {metrics['n']} | "
                f"{metrics['positive_n']} | {metrics['roc_auc']:.4f} | "
                f"{metrics['average_precision']:.4f} | {metrics['brier_score']:.4f} |"
            )
    lines.append("")
    lines.append("## Disclosure-safe history availability")
    lines.append("")
    lines.append(f"- See `{support_path.name}` for support by level, has-any flag and quality flag (<10 suppressed).")
    lines.append(
        f"- Quality flags overall: {json.dumps(context['quality_counts'], sort_keys=True)}."
    )
    lines.append("")
    lines.append("## Interpretation and limitations (as executed)")
    lines.append("")
    lines.append(
        "- Only observed history within survey recall is captured; a missing earlier "
        "institutional record is not evidence of no lifetime prior cesarean."
    )
    lines.append(
        "- Facility type is observed at delivery; historical selection effects are "
        "possible and no claim about necessity of cesareans is made."
    )
    lines.append(
        "- The survey-window definition follows the frozen C1 cohort (b19 0-59); "
        "birth dates (b3) are used only for ordering."
    )
    lines.append(
        "- NFHS-4 facility-detail harmonization is not implemented because the file "
        "is not available in the authorized environment."
    )
    lines.append(
        "- Point estimates only; no intervals or significance; V1 exposure caveat "
        "applies to any later A7 test scoring."
    )
    lines.append("")
    path = out_dir / "a4_methods_results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    args = parse_args(argv)
    config = _load_config(Path(args.config))
    seed = int(args.seed if args.seed is not None else config["split"]["seed"])
    suppression = int(config["disclosure"]["suppression_threshold"])
    data_path = Path(args.nfhs5_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.is_file():
        raise FileNotFoundError("--nfhs5-path does not exist or is not a file")

    raw = read_dta_subset(data_path)
    cohort, cohort_summary = build_c1_cohort(raw)
    warnings = assert_frozen_c1(cohort_summary)
    for key, expected in config["expected_cohort"].items():
        if int(cohort_summary[key]) != int(expected):
            raise RuntimeError(f"C1 cohort {key} mismatch vs config: {cohort_summary[key]} != {expected}")
    for key, expected in config["feasibility"].items():
        if int(expected) != int(EXPECTED_FEASIBILITY[key]):
            raise RuntimeError(f"Feasibility expectation for {key} differs from the frozen audit value")

    core = make_core_features(cohort, wave="NFHS-5")
    detail = facility_detail(cohort["m15"])
    history_feats, history_diagnostics = build_history_features(raw, cohort)
    observed_first = history_diagnostics["level_counts"].get(FIRST_RECORDED_LIVE_BIRTH, 0)
    observed_prior_cs = history_diagnostics["level_counts"].get(
        OBSERVED_PREVIOUS_INSTITUTIONAL_CS, 0
    )
    if observed_first != EXPECTED_FEASIBILITY[FIRST_RECORDED_LIVE_BIRTH]:
        raise RuntimeError(
            "History builder reproduction failure for first recorded live births: "
            f"{observed_first} vs audit {EXPECTED_FEASIBILITY[FIRST_RECORDED_LIVE_BIRTH]}"
        )
    if observed_prior_cs != EXPECTED_FEASIBILITY[OBSERVED_PREVIOUS_INSTITUTIONAL_CS]:
        raise RuntimeError(
            "History builder reproduction failure for observed previous "
            f"institutional CS: {observed_prior_cs} vs audit "
            f"{EXPECTED_FEASIBILITY[OBSERVED_PREVIOUS_INSTITUTIONAL_CS]}"
        )

    partition = grouped_partition(
        cohort,
        group_col=config["split"]["group_col"],
        target_col=config["split"]["target_col"],
        fractions=tuple(config["split"]["fractions"]),
        seed=seed,
    )
    split_audit = audit_partition(
        cohort,
        partition,
        group_col=config["split"]["group_col"],
        target_col=config["split"]["target_col"],
    )
    split_hash = split_manifest_hash(
        split_id=config["split"]["split_id"],
        seed=seed,
        fractions=config["split"]["fractions"],
        assignments=zip(cohort["caseid"].tolist(), partition.tolist()),
    )
    if split_hash != FROZEN_SPLIT_HASH:
        raise RuntimeError(
            "Provisional split hash differs from the frozen A1 split; matched "
            "comparison requires the identical partition"
        )

    main_block = _run_block(
        "main", cohort, core, detail, history_feats, partition, split_hash,
        config, seed, suppression,
    )
    if main_block["train_digest"] != FROZEN_TRAIN_DIGEST:
        raise RuntimeError("Train membership digest differs from the frozen A1 digest")
    if main_block["valid_digest"] != FROZEN_VALID_DIGEST:
        raise RuntimeError("Validation membership digest differs from the frozen A1 digest")

    sensitivity_cohort = cohort[cohort["m15"] != 96].reset_index(drop=True)
    sensitivity_core = make_core_features(sensitivity_cohort, wave="NFHS-5")
    sensitivity_detail = facility_detail(sensitivity_cohort["m15"])
    sensitivity_history, sensitivity_history_diag = build_history_features(
        raw, sensitivity_cohort, include_other_unknown_sector=False
    )
    mother_assignments = (
        pd.DataFrame(
            {
                "caseid": cohort["caseid"].astype("string").to_numpy(),
                "partition": partition.to_numpy(),
            }
        )
        .drop_duplicates()
    )
    if mother_assignments["caseid"].duplicated().any():
        raise RuntimeError("Inconsistent mother-level partition assignment")
    mother_map = mother_assignments.set_index("caseid")["partition"]
    sensitivity_partition = pd.Series(
        mother_map.loc[sensitivity_cohort["caseid"].astype("string")].to_numpy(),
        index=sensitivity_cohort.index,
        name="partition",
    )
    sensitivity_audit = audit_partition(
        sensitivity_cohort,
        sensitivity_partition,
        group_col=config["split"]["group_col"],
        target_col=config["split"]["target_col"],
    )
    sensitivity_hash = split_manifest_hash(
        split_id=config["split"]["split_id"] + "_m15_96_excluded_inherited",
        seed=seed,
        fractions=config["split"]["fractions"],
        assignments=zip(
            sensitivity_cohort["caseid"].tolist(), sensitivity_partition.tolist()
        ),
    )
    sensitivity_block = _run_block(
        "sensitivity", sensitivity_cohort, sensitivity_core, sensitivity_detail,
        sensitivity_history, sensitivity_partition, sensitivity_hash, config, seed,
        suppression,
    )

    v008 = pd.to_numeric(cohort["v008"], errors="coerce")
    outside_recode_window = int((pd.to_numeric(cohort["b3"], errors="coerce") < v008 - 59).sum())

    support_rows = history_support_counts(history_feats, cohort["m17"].to_numpy())
    valid_mask = main_block["valid_mask"]
    validation_counts = {}
    for group, column in (
        (HISTORY_FEATURE, HISTORY_FEATURE),
        (HAS_ANY_FEATURE, HAS_ANY_FEATURE),
        (QUALITY_FEATURE, QUALITY_FEATURE),
    ):
        counts = (
            history_feats.loc[valid_mask, column].astype("string").value_counts().to_dict()
        )
        for value, count in counts.items():
            validation_counts[(group, str(value))] = int(count)

    ablation_path = write_rows(
        main_block["rows"], out_dir / "a4_matched_ablations.csv"
    )
    sensitivity_path = write_rows(
        sensitivity_block["rows"], out_dir / "a4_sensitivity_m15_96.csv"
    )
    crosswalk_path = _write_crosswalk(out_dir)
    support_path = _write_history_support(
        out_dir, support_rows, validation_counts, suppression
    )
    roc_path = _write_roc_pr_figure(out_dir, main_block, "")
    sensitivity_roc_path = _write_roc_pr_figure(out_dir, sensitivity_block, "_sensitivity")
    distribution_path = _write_history_distribution_figure(
        out_dir,
        history_feats.loc[main_block["valid_mask"]],
        main_block["y_valid"],
        suppression,
    )

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    repo = repo_state()
    versions = software_versions()

    summary = {
        "experiment_id": "A4",
        "status": "COMPLETED_LOCAL_AGGREGATE_ONLY",
        "timestamp_utc": timestamp,
        "repo_commit": repo["repo_commit"],
        "branch": repo["branch"],
        "wave": "NFHS-5",
        "cohort_rule": C1_COHORT_RULE,
        "input_file": {
            "name": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
        "cohort": cohort_summary,
        "history": {
            "diagnostics": {
                key: value
                for key, value in history_diagnostics.items()
                if key
                not in ("level_counts", "has_any_counts", "quality_counts")
            },
            "level_counts_masked": _masked_counts(
                history_diagnostics["level_counts"], suppression
            ),
            "has_any_counts_masked": _masked_counts(
                history_diagnostics["has_any_counts"], suppression
            ),
            "quality_counts_masked": _masked_counts(
                history_diagnostics["quality_counts"], suppression
            ),
            "feasibility": {
                "expected": EXPECTED_FEASIBILITY,
                "observed_before_masking": {
                    FIRST_RECORDED_LIVE_BIRTH: observed_first,
                    OBSERVED_PREVIOUS_INSTITUTIONAL_CS: observed_prior_cs,
                },
                "match": True,
            },
            "sensitivity_excluded_level_counts_masked": _masked_counts(
                sensitivity_history_diag["level_counts"], suppression
            ),
        },
        "facility": {
            "label_source": LABEL_SOURCE,
            "crosswalk_file": crosswalk_path.name,
        },
        "data_quality": {
            "index_births_by_b3_before_recode_window": suppress_small_count(
                outside_recode_window, threshold=suppression
            )
        },
        "split": {
            "split_id": config["split"]["split_id"],
            "fractions": list(config["split"]["fractions"]),
            "seed": seed,
            "audit": split_audit,
            "split_manifest_hash": split_hash,
            "frozen_match": True,
            "sensitivity": {
                "inheritance": "mother_partition_inherited_from_main_split",
                "audit": sensitivity_audit,
                "split_manifest_hash_inherited_assignments": sensitivity_hash,
                "parent_split_manifest_hash": split_hash,
            },
        },
        "membership_digests": {
            "train_rows": main_block["train_digest"],
            "validation_rows": main_block["valid_digest"],
            "frozen_a1_train_rows": FROZEN_TRAIN_DIGEST,
            "frozen_a1_validation_rows": FROZEN_VALID_DIGEST,
            "train_mothers": main_block["train_mothers"],
            "valid_mothers": main_block["valid_mothers"],
            "sensitivity_train_rows": sensitivity_block["train_digest"],
            "sensitivity_validation_rows": sensitivity_block["valid_digest"],
        },
        "arms": {
            spec["tier"]: {
                "label": spec["label"],
                "numerical": list(F1_NUMERICAL),
                "categorical": list(spec["categorical"]),
                "feature_manifest_hash": main_block["feature_hashes"][spec["tier"]],
                "builder_id": _arm_builder_id(spec),
            }
            for spec in ARM_SPECS
        },
        "models": main_block["model_params"],
        "metrics": {
            model_name: {
                tier: {
                    key: (int(value) if isinstance(value, (np.integer,)) else
                          float(value) if isinstance(value, (np.floating,)) else value)
                    for key, value in metrics.items()
                }
                for tier, metrics in tiers.items()
            }
            for model_name, tiers in main_block["metrics_by_model"].items()
        },
        "sensitivity_metrics": {
            model_name: {
                tier: {
                    key: (int(value) if isinstance(value, (np.integer,)) else
                          float(value) if isinstance(value, (np.floating,)) else value)
                    for key, value in metrics.items()
                }
                for tier, metrics in tiers.items()
            }
            for model_name, tiers in sensitivity_block["metrics_by_model"].items()
        },
        "evaluation": {
            "partition": "validation",
            "threshold": float(config["evaluation"]["threshold"]),
            "threshold_basis": config["evaluation"]["threshold_basis"],
            "test_partition": config["evaluation"]["test_partition"],
        },
        "software_versions": versions,
        "warnings": warnings,
        "limitations": [
            "Only observed history within survey recall; missing records are explicit unknowns, not negatives.",
            "Partial institutional history; facility observed at delivery; possible historical selection effects.",
            "No inference about which cesareans were necessary.",
            "Survey-window definition follows frozen C1 (b19 0-59); b3 used only for ordering.",
            "NFHS-4 facility-detail harmonization pending local label verification.",
            "Point estimates only; no intervals or significance claims.",
            "Split fractions provisional until the A7 freeze; test partition not scored.",
        ],
        "outputs": [
            ablation_path.name,
            sensitivity_path.name,
            crosswalk_path.name,
            support_path.name,
            roc_path.name,
            sensitivity_roc_path.name,
            distribution_path.name,
        ],
    }
    summary_path = out_dir / "a4_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    methods_path = _write_methods(
        out_dir,
        {
            "timestamp": timestamp,
            "repo_commit": repo["repo_commit"],
            "branch": repo["branch"],
            "seed": seed,
            "cohort": cohort_summary,
            "split_hash": split_hash,
            "train_digest": main_block["train_digest"],
            "valid_digest": main_block["valid_digest"],
            "feasibility": EXPECTED_FEASIBILITY,
            "quality_counts": _masked_counts(
                history_diagnostics["quality_counts"], suppression
            ),
        },
        main_block,
        sensitivity_block,
        history_diagnostics,
        LABEL_SOURCE,
        crosswalk_path,
        support_path,
    )

    print(
        f"A4 completed. Cohort: {cohort_summary['n_births']} births / "
        f"{cohort_summary['n_mothers']} mothers; events {cohort_summary['n_events']}."
    )
    print(
        "History feasibility reproduced: first births "
        f"{observed_first}, observed previous institutional CS {observed_prior_cs}."
    )
    print(
        f"Split hash matches frozen A1 split: {split_hash == FROZEN_SPLIT_HASH}; "
        "digests match: True."
    )
    for model_name in main_block["metrics_by_model"]:
        for spec in ARM_SPECS:
            metrics = main_block["metrics_by_model"][model_name][spec["tier"]]
            print(
                f"  {model_name} {spec['tier']}: ROC-AUC {metrics['roc_auc']:.4f}, "
                f"AP {metrics['average_precision']:.4f}, Brier {metrics['brier_score']:.4f}"
            )
    print(f"Outputs written to {out_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
