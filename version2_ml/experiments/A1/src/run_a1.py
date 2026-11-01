"""A1 CLI runner: matched F1 vs F1+state ablation on the frozen NFHS-5 cohort.

Outputs are aggregate-only and disclosure-suppressed; no row-level values,
identifiers or predictions are written anywhere. The test partition is never
scored here (deferred to the single frozen A7 evaluation per the rework plan).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
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
import sklearn
import xgboost
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from xgboost import XGBClassifier

from a1_state_context import (
    F1_CATEGORICAL,
    F1_NUMERICAL,
    INTERACTION_FEATURE,
    STATE_FEATURE,
    assert_external_context_specs,
    assert_unique_birth_keys,
    build_state_frame,
    map_states,
    support_screen,
)
from aggregate_schema import (
    UNCERTAINTY_STATUS,
    build_row,
    suppress_small_count,
    write_rows,
)
from checksums import assert_matched_membership, membership_digest
from feature_builder import make_core_features
from manifests import feature_manifest_hash, split_manifest_hash
from model_framework import AuthorizedClassifier, evaluate_binary
from preprocessing import FeatureManifest
from split_contracts import audit_partition, grouped_partition

READ_COLUMNS = (
    "caseid",
    "bidx",
    "b3",
    "b19",
    "v008",
    "v011",
    "m15",
    "m17",
    "v024",
    "v025",
    "v130",
    "v133",
    "v190",
    "bord",
    "b0",
)
INSTITUTIONAL = set(range(20, 28)) | set(range(30, 34)) | {96}
EXPECTED_BIRTHS = 201311
EXPECTED_MOTHERS = 158429
COHORT_RULE = "c1_m15_institutional_b19_0_59_m17_0_1"

_MODEL_LABELS = {"xgboost": "XGBoost", "logistic_regression": "Logistic regression"}
_TIER_LABELS = {
    "F1": "F1 (eight-core)",
    "F2": "F2 (F1 + state)",
    "F2_interaction": "F2 + state x residence",
}
_COLORS = {
    ("xgboost", "F1"): "#1f77b4",
    ("xgboost", "F2"): "#d62728",
    ("xgboost", "F2_interaction"): "#9467bd",
    ("logistic_regression", "F1"): "#2ca02c",
    ("logistic_regression", "F2"): "#ff7f0e",
    ("logistic_regression", "F2_interaction"): "#8c564b",
}
_FOOTNOTE = (
    "Development-validation point estimates only; no intervals; provisional "
    "mother-grouped split; test partition not scored."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="A1 state/area context matched ablation (aggregate-only outputs)"
    )
    parser.add_argument(
        "--nfhs5-path",
        required=True,
        help="Authorized local NFHS-5 birth recode file (.DTA); never committed",
    )
    parser.add_argument(
        "--config", default=str(_HERE.parents[1] / "config.yaml")
    )
    parser.add_argument(
        "--out-dir", default=str(_HERE.parents[1] / "outputs")
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = (
        "split",
        "models",
        "features",
        "support_gate",
        "evaluation",
        "disclosure",
        "expected_cohort",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config missing sections: {missing}")
    return config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _software_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _repo_state() -> dict:
    repo_root = _VERSION2_ML.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"repo_commit": commit, "branch": branch}
    except Exception:
        return {"repo_commit": "unavailable", "branch": "unavailable"}


def _read_frame(path: Path) -> pd.DataFrame:
    import pyreadstat

    frame, _ = pyreadstat.read_dta(
        str(path), usecols=list(READ_COLUMNS), apply_value_formats=False
    )
    return frame


def _build_cohort(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    facility = pd.to_numeric(raw["m15"], errors="coerce")
    outcome = pd.to_numeric(raw["m17"], errors="coerce")
    months = pd.to_numeric(raw["b19"], errors="coerce")
    mask = (
        facility.isin(sorted(INSTITUTIONAL))
        & months.between(0, 59)
        & outcome.isin([0, 1])
    )
    cohort = raw.loc[mask].reset_index(drop=True)
    cohort["m15"] = pd.to_numeric(cohort["m15"], errors="coerce")
    cohort["m17"] = pd.to_numeric(cohort["m17"], errors="coerce").astype(int)
    state_codes = pd.to_numeric(cohort["v024"], errors="coerce")
    summary = {
        "n_births": int(len(cohort)),
        "n_mothers": int(cohort["caseid"].astype("string").nunique()),
        "n_events": int(cohort["m17"].sum()),
        "n_controls": int((cohort["m17"] == 0).sum()),
        "m15_96_rows": int((cohort["m15"] == 96).sum()),
        "states_observed": int(state_codes.nunique()),
        "missing_state": int(state_codes.isna().sum()),
        "duplicate_birth_keys": int(cohort[["caseid", "bidx"]].duplicated().sum()),
        "missing_caseid": int(cohort["caseid"].isna().sum()),
        "missing_bidx": int(cohort["bidx"].isna().sum()),
    }
    return cohort, summary


def _assert_frozen_cohort(summary: dict) -> None:
    if summary["n_births"] != EXPECTED_BIRTHS or summary["n_mothers"] != EXPECTED_MOTHERS:
        raise RuntimeError(
            "Frozen C1 cohort mismatch: observed "
            f"{summary['n_births']} births / {summary['n_mothers']} mothers vs "
            f"expected {EXPECTED_BIRTHS} / {EXPECTED_MOTHERS}; re-audit the input."
        )
    if summary["duplicate_birth_keys"] or summary["missing_caseid"] or summary["missing_bidx"]:
        raise RuntimeError("Cohort integrity failure; re-audit the input.")
    if summary["missing_state"]:
        raise RuntimeError(
            "Missing v024 values are incompatible with the consistent state mapping."
        )


def _manifest_for(tier: str) -> FeatureManifest:
    if tier == "F1":
        return FeatureManifest(F1_NUMERICAL, F1_CATEGORICAL, "F1")
    return FeatureManifest(
        F1_NUMERICAL, F1_CATEGORICAL + (STATE_FEATURE,), "F2"
    )


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


def _public_metrics(metrics: dict) -> dict:
    public = {}
    for key, value in metrics.items():
        if value is None:
            public[key] = "not_applicable"
        elif isinstance(value, (np.integer,)):
            public[key] = int(value)
        elif isinstance(value, (np.floating,)):
            public[key] = float(value)
        else:
            public[key] = value
    return public


def _masked_missingness(frame: pd.DataFrame, threshold: int) -> dict:
    return {
        str(column): suppress_small_count(
            int(frame[column].isna().sum()), threshold=threshold
        )
        for column in frame.columns
    }


def _masked_cells(cells, threshold: int):
    masked = []
    for cell in cells:
        masked.append(
            {
                "cell": cell["cell"],
                "n_rows": suppress_small_count(cell["n_rows"], threshold=threshold),
                "n_positive": suppress_small_count(cell["n_positive"], threshold=threshold),
                "n_negative": suppress_small_count(cell["n_negative"], threshold=threshold),
                "passes": cell["passes"],
            }
        )
    return masked


def _write_state_support_csv(out_dir: Path, state_support: dict, validation_counts: dict, threshold: int):
    states = sorted({cell["cell"] for cell in state_support["cells"]} | set(validation_counts))
    by_state = {cell["cell"]: cell for cell in state_support["cells"]}
    rows = []
    for state in states:
        cell = by_state.get(
            state, {"n_rows": 0, "n_positive": 0, "n_negative": 0, "passes": False}
        )
        rows.append(
            {
                "state_ut": state,
                "n_train": suppress_small_count(cell["n_rows"], threshold=threshold),
                "n_train_positive": suppress_small_count(cell["n_positive"], threshold=threshold),
                "n_train_negative": suppress_small_count(cell["n_negative"], threshold=threshold),
                "n_validation": suppress_small_count(validation_counts.get(state, 0), threshold=threshold),
                "training_support_gate_passed": cell["passes"],
            }
        )
    path = out_dir / "a1_support_table.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "state_ut",
                "n_train",
                "n_train_positive",
                "n_train_negative",
                "n_validation",
                "training_support_gate_passed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_interaction_gate_csv(out_dir: Path, interaction_support: dict, threshold: int):
    path = out_dir / "a1_interaction_gate.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cell",
                "n_train_rows",
                "n_train_positive",
                "n_train_negative",
                "passes",
            ],
        )
        writer.writeheader()
        for cell in interaction_support["cells"]:
            writer.writerow(
                {
                    "cell": cell["cell"],
                    "n_train_rows": suppress_small_count(cell["n_rows"], threshold=threshold),
                    "n_train_positive": suppress_small_count(cell["n_positive"], threshold=threshold),
                    "n_train_negative": suppress_small_count(cell["n_negative"], threshold=threshold),
                    "passes": cell["passes"],
                }
            )
    return path


def _write_roc_pr_figure(out_dir: Path, y_valid: np.ndarray, predictions: dict):
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for (model_name, tier), probabilities in predictions.items():
        color = _COLORS[(model_name, tier)]
        label = f"{_MODEL_LABELS[model_name]}: {_TIER_LABELS[tier]}"
        fpr, tpr, _ = roc_curve(y_valid, probabilities)
        axes[0].plot(fpr, tpr, color=color, label=label, linewidth=1.8)
        precision, recall, _ = precision_recall_curve(y_valid, probabilities)
        axes[1].plot(recall, precision, color=color, label=label, linewidth=1.8)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC (development validation)",
    )
    prevalence = float(np.mean(y_valid))
    axes[1].axhline(prevalence, color="gray", linestyle="--", linewidth=1, label=f"Prevalence = {prevalence:.3f}")
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall (development validation)",
    )
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    figure.text(0.01, 0.01, _FOOTNOTE, fontsize=7, color="dimgray")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    path = out_dir / "fig_a1_roc_pr.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _write_calibration_figure(out_dir: Path, y_valid: np.ndarray, predictions: dict):
    model_order = sorted({model for model, _ in predictions})
    figure, axes = plt.subplots(
        1, len(model_order), figsize=(5.6 * len(model_order), 4.6), squeeze=False
    )
    for axis, model_name in zip(axes[0], model_order):
        axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        for (name, tier), probabilities in predictions.items():
            if name != model_name:
                continue
            fraction, mean_predicted = calibration_curve(
                y_valid, probabilities, n_bins=10, strategy="quantile"
            )
            axis.plot(
                mean_predicted,
                fraction,
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=_COLORS[(name, tier)],
                label=_TIER_LABELS[tier],
            )
        axis.set(
            xlabel="Mean predicted probability",
            ylabel="Observed cesarean proportion",
            title=f"{_MODEL_LABELS[model_name]} calibration",
        )
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    figure.text(0.01, 0.01, _FOOTNOTE, fontsize=7, color="dimgray")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    path = out_dir / "fig_a1_calibration.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _write_methods(
    out_dir: Path,
    context: dict,
    metrics_by_model: dict,
    interaction_ran: bool,
    interaction_support: dict,
    threshold: float,
) -> Path:
    lines = []
    lines.append("# A1 - State/area context: methods and results (auto-generated)")
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
        f"mothers ({cohort['n_events']} events); m15 = 96 rows: {cohort['m15_96_rows']}; "
        f"states observed: {cohort['states_observed']}; missing v024: {cohort['missing_state']}."
    )
    lines.append(
        f"- Mother-grouped provisional split (70/15/15, seed {context['seed']}), split "
        f"manifest hash `{context['split_manifest_hash']}`; group-exclusive audit passed; "
        "evaluation on the validation partition only; test partition NOT RUN (deferred "
        "to the single frozen A7 evaluation)."
    )
    lines.append(
        "- Features: F1 = eight-core conservative set ("
        + ", ".join(context["F1_features"])
        + "); F2 = F1 + `state_ut` (36 verified NFHS-5 states/UTs plus explicit "
        "unknown_state token), one-hot fitted on training rows only with "
        "handle_unknown=ignore."
    )
    lines.append(
        "- No external area descriptor with validated year/geography/coverage "
        "provenance was available, so the handoff fallback applies: categorical state "
        "and residence (residence is already inside F1). No contextual enrichment is claimed."
    )
    for model_name, params in context["model_params"].items():
        lines.append(f"- {_MODEL_LABELS[model_name]} params: `{json.dumps(params, sort_keys=True)}`.")
    lines.append(
        f"- Matched arms: identical training/validation rows in identical order, proven "
        f"by row-membership digests (train `{context['train_digest']}`, validation "
        f"`{context['valid_digest']}`); threshold fixed at {threshold:.2f} "
        f"({context['threshold_basis']})."
    )
    lines.append(
        "- Point estimates only: no design-based intervals, no significance or "
        "confirmatory claims; small cells (<10) suppressed."
    )
    lines.append("")
    lines.append("## Results (development validation, point estimates)")
    lines.append("")
    lines.append(
        "| model | tier | n_births | n_mothers | n_events | prevalence | ROC-AUC | "
        "average precision | Brier | precision@0.50 | recall@0.50 | F1@0.50 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model_name in metrics_by_model:
        for tier, metrics in metrics_by_model[model_name].items():
            lines.append(
                f"| {_MODEL_LABELS[model_name]} | {_TIER_LABELS[tier]} | {metrics['n']} | "
                f"{context['valid_mothers']} | {metrics['positive_n']} | "
                f"{metrics['weighted_outcome_prevalence']:.4f} | {metrics['roc_auc']:.4f} | "
                f"{metrics['average_precision']:.4f} | {metrics['brier_score']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |"
            )
    lines.append("")
    for model_name, tiers in metrics_by_model.items():
        if "F1" in tiers and "F2" in tiers:
            base, state = tiers["F1"], tiers["F2"]
            lines.append(
                f"- {_MODEL_LABELS[model_name]} F2 minus F1 (point estimates): "
                f"ROC-AUC {state['roc_auc'] - base['roc_auc']:+.4f}, average precision "
                f"{state['average_precision'] - base['average_precision']:+.4f}, Brier "
                f"{state['brier_score'] - base['brier_score']:+.4f}."
            )
    if interaction_ran:
        lines.append(
            "- State x residence interaction: RUN (training-row support screen passed)."
        )
    else:
        lines.append(
            "- State x residence interaction: NOT RUN - D9 support screen failed on "
            f"training rows ({interaction_support['reason']}); the reported fallback is "
            "F1 vs F1+state as above."
        )
    lines.append("")
    lines.append("## Interpretation and limitations (as executed)")
    lines.append("")
    lines.append(
        "- The comparison is predictive, not ecological-causal; state identity may "
        "encode local response prevalence."
    )
    lines.append(
        "- Within-state generalization only; unseen-state validity is outside A1 (A6 scope)."
    )
    lines.append(
        "- Split fractions remain provisional until the A7 freeze; the cohort rule C1 "
        "is frozen; m15 = 96 births are included per the recorded D2 decision."
    )
    lines.append(
        "- V1's held-out test was already examined in the original project; even a "
        "future single test scoring is a development-test, not independent validation."
    )
    lines.append(
        "- Statewise metrics are not reported; the support table shows training support "
        "flags only, with counts below 10 suppressed."
    )
    lines.append("")
    path = out_dir / "a1_methods_results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    args = parse_args(argv)
    config = _load_config(Path(args.config))
    seed = int(args.seed if args.seed is not None else config["split"]["seed"])
    threshold = float(config["evaluation"]["threshold"])
    min_rows = int(config["support_gate"]["min_training_rows"])
    min_class = int(config["support_gate"]["min_class_rows"])
    suppression = int(config["disclosure"]["suppression_threshold"])
    data_path = Path(args.nfhs5_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.is_file():
        raise FileNotFoundError("--nfhs5-path does not exist or is not a file")

    external = assert_external_context_specs(config.get("external_area_context", []))
    if external["n_specs"]:
        raise ValueError(
            "External area descriptors are declared but not implemented in A1; "
            "remove them or implement documented adapters with provenance."
        )

    warnings = []
    cohort, cohort_summary = _build_cohort(_read_frame(data_path))
    _assert_frozen_cohort(cohort_summary)
    assert_unique_birth_keys(cohort)
    if cohort_summary["m15_96_rows"] != 517:
        warnings.append("m15=96 count differs from the Phase-0 verified 517.")
    if cohort_summary["states_observed"] != 36:
        warnings.append("Observed state count differs from the verified 36.")

    core = make_core_features(cohort, wave="NFHS-5")
    states = map_states(cohort["v024"])
    state_frame = build_state_frame(core, states, include_interaction=False)
    interaction_frame = build_state_frame(core, states, include_interaction=True)

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

    train_mask = partition.eq("train").to_numpy()
    valid_mask = partition.eq("validation").to_numpy()
    test_mask = partition.eq("test").to_numpy()
    bidx_tokens = cohort["bidx"].map(lambda value: str(int(value)))
    keys = (cohort["caseid"].astype("string") + "|" + bidx_tokens.astype("string")).to_numpy()
    train_digest = membership_digest(keys[train_mask], seed=seed)
    valid_digest = membership_digest(keys[valid_mask], seed=seed)

    y_train = cohort.loc[train_mask, "m17"].to_numpy()
    y_valid = cohort.loc[valid_mask, "m17"].to_numpy()
    valid_mothers = int(cohort.loc[valid_mask, "caseid"].nunique())
    train_mothers = int(cohort.loc[train_mask, "caseid"].nunique())

    state_support = support_screen(
        state_frame.loc[train_mask, STATE_FEATURE].to_numpy(),
        y_train,
        min_rows=min_rows,
        min_class=min_class,
    )
    interaction_support = support_screen(
        interaction_frame.loc[train_mask, INTERACTION_FEATURE].to_numpy(),
        y_train,
        min_rows=min_rows,
        min_class=min_class,
    )
    low_support_states = sum(1 for cell in state_support["cells"] if not cell["passes"])
    if low_support_states:
        warnings.append(
            f"{low_support_states} state cells fail the >= {min_rows}-row training support screen."
        )

    version_info = _software_versions()
    versions_string = json.dumps(version_info, sort_keys=True, separators=(",", ":"))
    repo_state = _repo_state()
    feature_hashes = {
        "F1": feature_manifest_hash(
            tier="F1",
            numerical=F1_NUMERICAL,
            categorical=F1_CATEGORICAL,
            builder_id="version2_ml/feature_builder.py::make_core_features",
        ),
        "F2": feature_manifest_hash(
            tier="F2",
            numerical=F1_NUMERICAL,
            categorical=F1_CATEGORICAL + (STATE_FEATURE,),
            builder_id="version2_ml/experiments/A1/src/a1_state_context.py::build_state_frame",
        ),
        "F2_interaction": feature_manifest_hash(
            tier="F2",
            numerical=F1_NUMERICAL,
            categorical=F1_CATEGORICAL + (STATE_FEATURE, INTERACTION_FEATURE),
            builder_id="version2_ml/experiments/A1/src/a1_state_context.py::build_state_frame",
            extra={"baseline_id": "F1_reference"},
        ),
    }
    for tier, frame in (("F1", core), ("F2", state_frame), ("F2_interaction", interaction_frame)):
        if not frame.index.equals(core.index):
            raise RuntimeError(f"{tier}: paired arms must share identical row order")

    predictions = {}
    metrics_by_model = {}
    model_params = {}
    for model_name in ("xgboost", "logistic_regression"):
        estimator, params = _estimator_for(model_name, config, seed, y_train)
        model_params[model_name] = params
        metrics_by_model[model_name] = {}
        for tier, frame in (("F1", core), ("F2", state_frame)):
            arm_train_digest = membership_digest(keys[train_mask], seed=seed)
            assert_matched_membership(train_digest, arm_train_digest)
            probabilities = _fit_arm(
                estimator,
                _manifest_for(tier),
                frame.loc[train_mask],
                y_train,
                frame.loc[valid_mask],
                seed,
            )
            predictions[(model_name, tier)] = probabilities
            metrics_by_model[model_name][tier] = evaluate_binary(
                y_valid, probabilities, threshold=threshold
            )

    interaction_ran = bool(interaction_support["gate_passed"])
    if interaction_ran:
        for model_name in ("xgboost", "logistic_regression"):
            estimator, _ = _estimator_for(model_name, config, seed, y_train)
            probabilities = _fit_arm(
                estimator,
                FeatureManifest(
                    F1_NUMERICAL,
                    F1_CATEGORICAL + (STATE_FEATURE, INTERACTION_FEATURE),
                    "F2",
                ),
                interaction_frame.loc[train_mask],
                y_train,
                interaction_frame.loc[valid_mask],
                seed,
            )
            predictions[(model_name, "F2_interaction")] = probabilities
            metrics_by_model[model_name]["F2_interaction"] = evaluate_binary(
                y_valid, probabilities, threshold=threshold
            )

    rows = []
    for model_name in metrics_by_model:
        for tier, metrics in metrics_by_model[model_name].items():
            rows.append(
                build_row(
                    experiment_id="A1",
                    wave="NFHS-5",
                    cohort_rule=COHORT_RULE,
                    evaluation_design="matched_ablation_validation_only",
                    split_id=config["split"]["split_id"],
                    fold="validation",
                    source_domain="NFHS-5",
                    target_domain="NFHS-5",
                    model=model_name,
                    feature_tier=tier,
                    baseline_id="F1_reference",
                    n_births=int(metrics["n"]),
                    n_mothers=valid_mothers,
                    n_events=int(metrics["positive_n"]),
                    prevalence=float(metrics["weighted_outcome_prevalence"]),
                    roc_auc=float(metrics["roc_auc"]),
                    average_precision=float(metrics["average_precision"]),
                    brier_score=float(metrics["brier_score"]),
                    threshold=threshold,
                    precision=float(metrics["precision"]),
                    recall=float(metrics["recall"]),
                    f1=float(metrics["f1"]),
                    weighted_descriptive_flag="unweighted_primary_analysis",
                    uncertainty_status=UNCERTAINTY_STATUS,
                    repo_commit=repo_state["repo_commit"],
                    seed=seed,
                    software_versions=versions_string,
                    feature_manifest_hash=feature_hashes[tier],
                    split_manifest_hash=split_hash,
                )
            )
    if not interaction_ran:
        for model_name in ("xgboost", "logistic_regression"):
            rows.append(
                build_row(
                    experiment_id="A1",
                    wave="NFHS-5",
                    cohort_rule=COHORT_RULE,
                    evaluation_design="not_run_support_screen_failed",
                    split_id=config["split"]["split_id"],
                    fold="validation",
                    source_domain="NFHS-5",
                    target_domain="NFHS-5",
                    model=model_name,
                    feature_tier="F2_interaction",
                    baseline_id="F1_reference",
                    weighted_descriptive_flag="unweighted_primary_analysis",
                    uncertainty_status=UNCERTAINTY_STATUS,
                    repo_commit=repo_state["repo_commit"],
                    seed=seed,
                    software_versions=versions_string,
                    feature_manifest_hash=feature_hashes["F2_interaction"],
                    split_manifest_hash=split_hash,
                )
            )

    ablation_path = write_rows(rows, out_dir / "a1_matched_ablation.csv")
    validation_counts = (
        state_frame.loc[valid_mask, STATE_FEATURE].value_counts().to_dict()
    )
    support_path = _write_state_support_csv(
        out_dir, state_support, validation_counts, suppression
    )
    gate_path = _write_interaction_gate_csv(out_dir, interaction_support, suppression)
    roc_path = _write_roc_pr_figure(out_dir, y_valid, predictions)
    calibration_path = _write_calibration_figure(out_dir, y_valid, predictions)

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {
        "experiment_id": "A1",
        "status": "COMPLETED_LOCAL_AGGREGATE_ONLY",
        "timestamp_utc": timestamp,
        "repo_commit": repo_state["repo_commit"],
        "branch": repo_state["branch"],
        "wave": "NFHS-5",
        "cohort_rule": COHORT_RULE,
        "input_file": {
            "name": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": _sha256_file(data_path),
        },
        "cohort": cohort_summary,
        "feature_missingness": {
            "F1": _masked_missingness(core, suppression),
            "F2": _masked_missingness(state_frame, suppression),
        },
        "split": {
            "split_id": config["split"]["split_id"],
            "fractions": list(config["split"]["fractions"]),
            "seed": seed,
            "audit": split_audit,
            "split_manifest_hash": split_hash,
            "status": config["split"]["status"],
        },
        "membership_digests": {
            "train_rows": train_digest,
            "validation_rows": valid_digest,
            "train_mothers": train_mothers,
        },
        "features": {
            "F1": {"names": list(core.columns), "manifest_hash": feature_hashes["F1"]},
            "F2": {"names": list(state_frame.columns), "manifest_hash": feature_hashes["F2"]},
            "F2_interaction": {
                "names": list(interaction_frame.columns),
                "manifest_hash": feature_hashes["F2_interaction"],
                "ran": interaction_ran,
                "gate": {
                    "min_rows": interaction_support["min_rows"],
                    "min_class": interaction_support["min_class"],
                    "gate_passed": interaction_support["gate_passed"],
                    "reason": interaction_support["reason"],
                    "n_cells": interaction_support["n_cells"],
                    "cells": _masked_cells(interaction_support["cells"], suppression),
                },
            },
            "external_context": external,
        },
        "support": {
            "min_rows": state_support["min_rows"],
            "min_class": state_support["min_class"],
            "n_cells": state_support["n_cells"],
            "n_failing_cells": low_support_states,
            "cells": _masked_cells(state_support["cells"], suppression),
        },
        "models": model_params,
        "metrics": {
            model_name: {
                tier: _public_metrics(metrics)
                for tier, metrics in tiers.items()
            }
            for model_name, tiers in metrics_by_model.items()
        },
        "evaluation": {
            "partition": "validation",
            "threshold": threshold,
            "threshold_basis": config["evaluation"]["threshold_basis"],
            "test_partition": config["evaluation"]["test_partition"],
        },
        "software_versions": version_info,
        "warnings": warnings,
        "limitations": [
            "Point estimates only; no design-valid intervals; no significance claims.",
            "Within-state generalization only; unseen-state validation is A6 scope.",
            "State identity may encode local response prevalence; not ecological-causal.",
            "Split fractions provisional until the A7 freeze.",
            "No external area descriptor validated; categorical state + residence fallback.",
            "V1 exposure: the original held-out test was already examined historically.",
        ],
        "outputs": [
            ablation_path.name,
            support_path.name,
            gate_path.name,
            roc_path.name,
            calibration_path.name,
        ],
    }
    summary_path = out_dir / "a1_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    methods_path = _write_methods(
        out_dir,
        {
            "timestamp": timestamp,
            "repo_commit": repo_state["repo_commit"],
            "branch": repo_state["branch"],
            "seed": seed,
            "cohort": cohort_summary,
            "split_manifest_hash": split_hash,
            "F1_features": list(core.columns),
            "model_params": model_params,
            "train_digest": train_digest,
            "valid_digest": valid_digest,
            "threshold_basis": config["evaluation"]["threshold_basis"],
            "valid_mothers": valid_mothers,
        },
        metrics_by_model,
        interaction_ran,
        interaction_support,
        threshold,
    )

    print(f"A1 completed. Cohort: {cohort_summary['n_births']} births / "
          f"{cohort_summary['n_mothers']} mothers; events {cohort_summary['n_events']}.")
    print(f"Split: train {split_audit['summary']['train']['rows']} rows / "
          f"{split_audit['summary']['train']['mothers']} mothers; validation "
          f"{split_audit['summary']['validation']['rows']} rows; test partition NOT RUN.")
    print(f"Interaction gate passed: {interaction_ran} "
          f"({interaction_support['reason']})." if not interaction_ran
          else "Interaction gate passed: True.")
    for model_name in metrics_by_model:
        for tier, metrics in metrics_by_model[model_name].items():
            print(f"  {model_name} {tier}: ROC-AUC {metrics['roc_auc']:.4f}, "
                  f"AP {metrics['average_precision']:.4f}, Brier {metrics['brier_score']:.4f}")
    print(f"Outputs written to {out_dir} ({ablation_path.name}, {summary_path.name}, "
          f"{methods_path.name}, figures).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
