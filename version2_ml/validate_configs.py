#!/usr/bin/env python3
"""Validate IPD Version 2 planning YAML configurations without reading DHS microdata.

Run from repository root:
  python version2_ml/validate_configs.py
Requires: pip install pyyaml
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

CONFIG_NAMES = ("cohorts", "features", "splits", "models", "evaluation")
ROOT = Path(__file__).resolve().parent
RESTRICTED = {
    "csection", "respondent_id", "birth_key", "sample_weight",
    "primary_sampling_unit", "sample_stratum_v022", "sample_stratum_v023"
}


def validate(configs: dict) -> dict:
    """Return actionable errors/warnings. Validates *planning* contracts, not data."""
    errors, warnings = [], []
    for name in CONFIG_NAMES:
        value = configs.get(name)
        if not isinstance(value, dict):
            errors.append(f"{name}: expected YAML mapping, got {type(value).__name__}")
            continue
        if not str(value.get("schema_version", "")).startswith("0.1-"):
            errors.append(f"{name}: unsupported or absent schema_version")
        if not value.get("status"):
            errors.append(f"{name}: missing status")
    if errors:
        return {"status": "INVALID", "errors": errors, "warnings": warnings}

    co, fe, sp, mo, ev = (configs[n] for n in CONFIG_NAMES)
    if co.get("prediction_unit") != "birth_record":
        errors.append("cohorts.prediction_unit must be birth_record")
    key = co.get("group_key")
    if not isinstance(key, str) or not key.strip():
        errors.append("cohorts.group_key must be non-empty")
    if sp.get("grouping") != key:
        errors.append("splits.grouping must match cohorts.group_key")
    if co.get("outcome", {}).get("source") != "m17":
        errors.append("cohorts.outcome.source must be m17")
    if co.get("outcome", {}).get("positive_class") != 1:
        errors.append("cohorts.outcome.positive_class must be 1")
    if co.get("cohorts", {}).get("C1", {}).get("expected_n_historical_only") != 201311:
        warnings.append("C1 historical sample count differs from audited V1 cohort (201311)")
    tiers = fe.get("feature_tiers", {})
    if not isinstance(tiers, dict):
        errors.append("features.feature_tiers must be mapping")
        tiers = {}
    for tier in ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "FT"):
        if tier not in tiers:
            errors.append(f"features.feature_tiers missing {tier}")
    for tier, cfg in tiers.items():
        if not isinstance(cfg, dict):
            errors.append(f"features.feature_tiers.{tier} must be mapping")
            continue
        dep = cfg.get("depends_on")
        if dep is not None and dep not in tiers:
            errors.append(f"{tier}.depends_on references unknown tier {dep}")
    restricted = set(fe.get("restricted_columns_never_predictors", []))
    if not RESTRICTED.issubset(restricted):
        errors.append("features.restricted_columns_never_predictors missing required columns: "
                      + ", ".join(sorted(RESTRICTED - restricted)))
    if sp.get("development_cv", {}).get("n_splits", 0) < 2:
        errors.append("splits.development_cv.n_splits must be >= 2")
    if sp.get("wave_transfer", {}).get("target_labels_for_source_tuning") != "forbidden":
        errors.append("splits.wave_transfer.target_labels_for_source_tuning must be forbidden")
    if sp.get("wave_transfer", {}).get("adaptation_and_evaluation") != "disjoint":
        errors.append("splits.wave_transfer.adaptation_and_evaluation must be disjoint")
    if not sp.get("primary_nfhs5", {}).get("previous_v1_test_already_examined"):
        errors.append("Previous V1 test exposure must remain declared")
    if mo.get("preprocessing", {}).get("fit_scope") != "train_fold_only":
        errors.append("models.preprocessing.fit_scope must be train_fold_only")
    if mo.get("preprocessing", {}).get("imputation") != "train_fold_only":
        errors.append("models.preprocessing.imputation must be train_fold_only")
    if mo.get("preprocessing", {}).get("target_encoding") != "grouped_out_of_fold":
        errors.append("models.preprocessing.target_encoding must be grouped_out_of_fold")
    if mo.get("baseline", {}).get("features") != "F0":
        errors.append("models.baseline.features must be F0")
    if mo.get("model_selection", {}).get("data") != "development_only":
        errors.append("models.model_selection.data must be development_only")
    if not ev.get("primary", {}).get("paired_rows_required"):
        errors.append("evaluation.primary.paired_rows_required must be true")
    if not ev.get("primary", {}).get("confirmatory_only_if_untouched_test"):
        errors.append("evaluation.primary.confirmatory_only_if_untouched_test must be true")
    if ev.get("inference", {}).get("bootstrap_replicates_target", 0) < 100:
        errors.append("evaluation.inference.bootstrap_replicates_target too low")
    if ev.get("inference", {}).get("row_bootstrap_not_design_valid") is not True:
        errors.append("evaluation.inference.row_bootstrap_not_design_valid must be true")

    for path, val in (
        ("cohorts.outcome.coding_verified", co.get("outcome", {}).get("coding_verified")),
        ("evaluation.survey.variable_selection", ev.get("survey", {}).get("variable_selection")),
        ("evaluation.inference.replicate_weight_construction",
         ev.get("inference", {}).get("replicate_weight_construction")),
        ("evaluation.inference.singleton_strata_handling",
         ev.get("inference", {}).get("singleton_strata_handling")),
    ):
        if val is False or (isinstance(val, str) and val.startswith("PENDING")):
            warnings.append(f"{path}: not yet empirically signed off")
    if sp.get("primary_nfhs5", {}).get("proportions") is None:
        warnings.append("splits.primary_nfhs5.proportions: pending freeze")
    if ev.get("status") == "inference_implementation_NOT_VALIDATED":
        warnings.append("Survey/bootstrap inference remains UNVALIDATED")
    return {"status": "INVALID" if errors else "PLANNING_CONFIG_VALID",
            "errors": errors, "warnings": warnings,
            "note": "No claims about cohort, feature timing, splits, or inference are verified by this check."}


def load_configs(directory: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Run: python -m pip install pyyaml") from exc
    configs = {}
    for name in CONFIG_NAMES:
        path = directory / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing configuration file: {path}")
        with path.open("r", encoding="utf-8") as stream:
            configs[name] = yaml.safe_load(stream)
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "audits" / "phase3" / "outputs" / "config_validation.json")
    args = parser.parse_args()
    try:
        report = validate(load_configs(args.config_dir))
    except (OSError, ValueError) as exc:
        report = {"status": "INVALID", "errors": [str(exc)], "warnings": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.output}")
    return 0 if report["status"] == "PLANNING_CONFIG_VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
