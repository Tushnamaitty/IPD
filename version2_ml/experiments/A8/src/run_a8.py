"""A8 CLI runner: history + state context ablation that decides FINAL_TIER.

Order of operations (stops at the first failed gate, before writing outputs):
1. synthetic A8 test suite must pass;
2. NFHS-5 frozen C1 cohort via the A7 data path (input sha256, cohort counts,
   history reproduction 6276 asserted);
3. per seed (42, 7, 2026, 99): mother-grouped 70/15/15 split
   (split_contracts.grouped_partition), A7 20% calibration carve (seed 4208);
   seed 42 is asserted against the frozen split hash, the A1 train/validation
   digests and the A7 fit/calibration digests;
4. arms H0-H3 x {XGBoost, LR} on identical rows, Platt-calibrated, validation
   only; the test partition is never indexed;
5. gate: H0 XGBoost seed 42 must reproduce A7 validation ROC-AUC / AP;
6. runtime leakage checks on real data; paired deltas; predeclared decision;
   state x residence support screen; SHAP families for the selected arm;
7. aggregate outputs + MANIFEST.json.

Never writes row-level data, keys or per-birth predictions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_HERE = Path(__file__).resolve()
_A8 = _HERE.parents[1]
_BASE = _HERE.parents[3]
_REPO = _BASE.parent
for _entry in (
    _BASE,
    _BASE / "experiments" / "common",
    _BASE / "experiments" / "A1" / "src",
    _BASE / "experiments" / "A3",
    _BASE / "experiments" / "A4" / "src",
    _BASE / "experiments" / "A7" / "src",
    _HERE.parent,
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from a1_state_context import INTERACTION_SEPARATOR, map_states, support_screen  # noqa: E402
from a4_facility_detail import facility_detail  # noqa: E402
from a4_history import OBSERVED_PREVIOUS_INSTITUTIONAL_CS, build_history_features  # noqa: E402
from a7_candidates import build_tier_frame, carve_calibration_partition  # noqa: E402
from a7_shap import shap_family_table  # noqa: E402
from a8_arms import ARM_ORDER, ARMS, BASE_TIER, added_feature_count, arm_manifest  # noqa: E402
from a8_decision import decide  # noqa: E402
from a8_outputs import write_csv, write_json, write_manifest, write_text  # noqa: E402
from a8_pipeline import paired_deltas, run_seed  # noqa: E402
from aggregate_schema import UNCERTAINTY_STATUS, build_row, suppress_small_count  # noqa: E402
from checksums import membership_digest  # noqa: E402
from data_access import (  # noqa: E402
    C1_COHORT_RULE,
    FROZEN_SPLIT_HASH,
    FROZEN_TRAIN_DIGEST,
    FROZEN_VALID_DIGEST,
    assert_frozen_c1,
    build_c1_cohort,
    read_dta_subset,
    sha256_file,
    software_versions,
)
from feature_builder import make_core_features  # noqa: E402
from manifests import feature_manifest_hash, split_manifest_hash  # noqa: E402
from split_contracts import audit_partition, grouped_partition  # noqa: E402

A7_SUMMARY = _BASE / "experiments" / "A7" / "outputs" / "a7_run_summary.json"
EXPECTED_HISTORY_OBSERVED_CS = 6276
DESIGN = "matched_arms_identical_rows_platt_validation_only"
BUILDER = ("version2_ml/feature_builder.py::make_core_features + A4 facility_detail + A4 build_history_features"
           " + A1 map_states + A3 encode_partitions (fit-pool labels only)")
ARM_COLORS = {"xgboost": "#2a78d6", "logistic_regression": "#e0782b"}


class GateFailure(RuntimeError):
    pass


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="A8 history + state ablation")
    parser.add_argument("--nfhs5-path", default=os.environ.get("NFHS5_PATH"))
    parser.add_argument("--config", default=str(_A8 / "config.yaml"))
    parser.add_argument("--out-dir", default=str(_A8 / "outputs"))
    parser.add_argument("--skip-tests", action="store_true", help="development only; never for recorded runs")
    args = parser.parse_args(argv)
    if not args.nfhs5_path:
        parser.error("NFHS5_PATH is not set and --nfhs5-path was not given")
    return args


def _run_synthetic_tests() -> str:
    suite = unittest.defaultTestLoader.discover(str(_A8 / "tests"), pattern="test_*.py", top_level_dir=str(_A8 / "tests"))
    result = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0:
        raise GateFailure(f"Synthetic A8 tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
    return f"{result.testsRun} synthetic tests OK"


def _check(qa: list, name: str, ok: bool, detail: str = "") -> None:
    qa.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    if not ok:
        raise GateFailure(f"{name}: {detail}")


def _key_tokens(cohort: pd.DataFrame) -> np.ndarray:
    return (cohort["caseid"].astype("string") + "|"
            + cohort["bidx"].map(lambda value: str(int(value))).astype("string")).to_numpy()


def _model_params(config: dict, recorded: dict) -> dict:
    selected = recorded["tuning"]["selected_params"]
    params = {}
    for model in config["models"]:
        expected = dict(config["model_params"][model])
        if selected[model] != expected:
            raise GateFailure(f"{model}: A7 selected params {selected[model]} differ from A8 config {expected}")
        params[model] = dict(selected[model])
    return params


def _load_cohort(path: Path, config: dict, recorded: dict, qa: list):
    input_hash = sha256_file(path)
    _check(qa, "input_sha256_matches_a7", input_hash == recorded["input_sha256"], input_hash[:16])
    raw = read_dta_subset(path)
    cohort, summary = build_c1_cohort(raw)
    warnings = assert_frozen_c1(summary)
    for key, value in config["expected_cohort"].items():
        _check(qa, f"cohort_{key}", int(summary[key]) == int(value), f"{summary[key]} vs {value}")
    core = make_core_features(cohort, wave="NFHS-5")
    detail = facility_detail(cohort["m15"])
    history, diag = build_history_features(raw, cohort, include_other_unknown_sector=True)
    observed = diag["level_counts"].get(OBSERVED_PREVIOUS_INSTITUTIONAL_CS, 0)
    _check(qa, "history_reproduction_6276", observed == EXPECTED_HISTORY_OBSERVED_CS, str(observed))
    base = build_tier_frame(core, detail, history, BASE_TIER)
    return cohort, summary, warnings, base, input_hash


def _split(cohort, config, seed, keys, recorded, qa):
    split = config["split"]
    partition = grouped_partition(cohort, group_col=split["group_col"], target_col=split["target_col"],
                                  fractions=tuple(split["fractions"]), seed=seed)
    audit_partition(cohort, partition)
    is_frozen = seed == int(split["frozen_seed"])
    split_id = split["frozen_split_id"] if is_frozen else split["split_id_template"].format(seed=seed)
    split_hash = split_manifest_hash(split_id=split_id, seed=seed, fractions=split["fractions"],
                                     assignments=zip(cohort["caseid"].tolist(), partition.tolist()))
    train_mask = partition.eq("train").to_numpy()
    valid_mask = partition.eq("validation").to_numpy()
    train_positions, valid_positions = np.flatnonzero(train_mask), np.flatnonzero(valid_mask)
    carve_cfg = config["calibration"]["carve"]
    carve = carve_calibration_partition(cohort.iloc[train_positions].reset_index(drop=True),
                                        calibration_fraction=float(carve_cfg["subset_fraction"]),
                                        seed=int(carve_cfg["seed"]))
    fit_positions = train_positions[carve.eq("fit").to_numpy()]
    calib_positions = train_positions[carve.eq("calibration").to_numpy()]
    digests = {
        "train": membership_digest(keys[train_mask], seed=seed),
        "validation": membership_digest(keys[valid_mask], seed=seed),
        "fit_pool": membership_digest(keys[fit_positions], seed=seed),
        "calibration_pool": membership_digest(keys[calib_positions], seed=seed),
    }
    if is_frozen:
        frozen = recorded["frozen_digests"]
        _check(qa, "seed42_split_hash_frozen", split_hash == FROZEN_SPLIT_HASH, split_hash[:16])
        _check(qa, "seed42_train_digest_frozen", digests["train"] == FROZEN_TRAIN_DIGEST)
        _check(qa, "seed42_validation_digest_frozen", digests["validation"] == FROZEN_VALID_DIGEST)
        _check(qa, "seed42_fit_pool_digest_a7", digests["fit_pool"] == frozen["fit_pool"])
        _check(qa, "seed42_calibration_digest_a7", digests["calibration_pool"] == frozen["calibration_pool"])
    return {"split_id": split_id, "split_hash": split_hash, "train_mask": train_mask, "fit": fit_positions,
            "calib": calib_positions, "valid": valid_positions, "digests": digests}


def _leakage_checks(base, states, codes, y, groups, split, params, te_cfg, seed, reference, qa):
    """Real-data adversarial checks on seed 42, arm H3 (both added feature types), XGBoost."""
    rng = np.random.default_rng(seed)
    ref_te = reference["te_values"].tobytes()
    ref_probs = reference["probabilities"][("xgboost", "H3")]
    for label, held in (("validation", split["valid"]), ("calibration", split["calib"])):
        mutated = np.asarray(y).copy()
        mutated[held] = rng.permutation(mutated[held])
        mutated[held] = 1 - mutated[held]
        result = run_seed(base, states, codes, mutated, groups, fit_positions=split["fit"],
                          calib_positions=split["calib"], valid_positions=split["valid"], models=["xgboost"],
                          params=params, te_config=te_cfg, estimator_seed=seed, arms=("H3",))
        probs = result["probabilities"][("xgboost", "H3")]
        _check(qa, f"runtime_{label}_labels_mutated_encoder_identical", result["te_digest"] == reference["te_digest"])
        _check(qa, f"runtime_{label}_labels_mutated_features_identical", result["te_values"].tobytes() == ref_te)
        _check(qa, f"runtime_{label}_labels_mutated_base_predictions_identical",
               probs["base"].tobytes() == ref_probs["base"].tobytes())
        if label == "validation":
            _check(qa, "runtime_validation_labels_mutated_calibrated_predictions_identical",
                   probs["calibrated"].tobytes() == ref_probs["calibrated"].tobytes())


def _schema_rows(metrics, repo_commit, versions_string, feature_hashes):
    rows = []
    for row in metrics.itertuples(index=False):
        rows.append(build_row(
            experiment_id="A8", wave="NFHS-5", cohort_rule=C1_COHORT_RULE, evaluation_design=DESIGN,
            split_id=row.split_id, fold="validation", source_domain="NFHS-5", target_domain="NFHS-5",
            model=row.model, feature_tier=f"{row.arm}_{row.tier_name}", baseline_id="H0",
            n_births=int(row.n_births), n_mothers=int(row.n_mothers),
            n_events=suppress_small_count(int(row.n_events)), prevalence=float(row.prevalence),
            roc_auc=float(row.roc_auc), average_precision=float(row.average_precision),
            brier_score=float(row.brier_score), calibration_intercept=float(row.calibration_intercept),
            calibration_slope=float(row.calibration_slope), threshold=float(row.threshold),
            precision=float(row.precision), recall=float(row.recall), f1=float(row.f1),
            weighted_descriptive_flag="unweighted_primary_analysis", uncertainty_status=UNCERTAINTY_STATUS,
            repo_commit=repo_commit, seed=int(row.seed), software_versions=versions_string,
            feature_manifest_hash=feature_hashes[row.arm], split_manifest_hash=row.split_hash,
        ))
    return rows


def _support_rows(screen: dict, min_cell: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "cell": cell["cell"],
        "n_train_rows": suppress_small_count(cell["n_rows"], threshold=min_cell),
        "n_train_positive": suppress_small_count(cell["n_positive"], threshold=min_cell),
        "n_train_negative": suppress_small_count(cell["n_negative"], threshold=min_cell),
        "passes": cell["passes"],
    } for cell in screen["cells"]])


def _fig_deltas(deltas: pd.DataFrame, rule: dict, path: Path) -> None:
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    arms = [arm for arm in ARM_ORDER if arm != "H0"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=surface)
    panels = (("delta_roc_auc", "Δ ROC-AUC vs H0", rule["min_delta_roc_auc"]),
              ("delta_average_precision", "Δ average precision vs H0", rule["min_delta_average_precision"]))
    for axis, (column, title, threshold) in zip(axes, panels):
        axis.set_facecolor(surface)
        for offset, model in ((-0.12, "xgboost"), (0.12, "logistic_regression")):
            block = deltas[deltas["model"] == model]
            for index, arm in enumerate(arms):
                values = block.loc[block["arm"] == arm, column].to_numpy()
                jitter = np.linspace(-0.04, 0.04, len(values)) if len(values) > 1 else np.zeros(len(values))
                axis.scatter(index + offset + jitter, values, s=36, color=ARM_COLORS[model],
                             edgecolor=surface, linewidth=1, zorder=3,
                             label=("XGBoost (decision model)" if model == "xgboost" else "Logistic regression")
                             if index == 0 else None)
        axis.axhline(0, color=muted, linewidth=0.8)
        axis.axhline(threshold, color=muted, linewidth=0.8, linestyle="--")
        axis.text(len(arms) - 0.5, threshold, f" rule ≥ {threshold:+.3f}", va="bottom", ha="right",
                  fontsize=8, color=muted)
        axis.set_xticks(range(len(arms)))
        axis.set_xticklabels(arms, color=ink)
        axis.set_title(title, fontsize=10, color=ink, loc="left")
        axis.grid(axis="y", color=grid, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(colors=muted, labelsize=8)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("A8: paired deltas vs F1_detail_history, one point per split seed (validation)",
                 x=0.01, ha="left", fontsize=11, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def _methods(state: dict) -> str:
    config, decision, metrics = state["config"], state["decision"], state["metrics"]
    lines = ["# A8 methods and results: history + state context (validation only)", ""]
    lines += [
        "Generated by `version2_ml/experiments/A8/src/run_a8.py`. Aggregate point estimates on the "
        "development validation partition of each split; the test partition was never indexed. "
        "No confidence intervals; unweighted.", "",
        f"- generated_at_utc: {state['generated']}",
        f"- repo_commit: {state['git_commit']}",
        f"- input_sha256: {state['input_hash']}",
        f"- seeds: {config['split']['seeds']} (seed 42 = frozen A1/A7 split)", "",
        "## Design", "",
        "- Arms: H0 = F1_detail_history (A7); H1 = H0 + `state_ut` (A1 labels); H2 = H0 + `te_state_sector` "
        "(A3 grouped OOF target encoding, alpha 5, n_min 30, 3 inner mother-grouped folds); H3 = H0 + both.",
        "- Per seed: mother-grouped 70/15/15 split; A7 20% mother-grouped calibration carve (seed 4208) of the "
        "training pool; models fitted on the fit pool; Platt fitted on the carve; metrics on validation.",
        "- Target encoder fitted on fit-pool labels only (OOF for fit rows); calibration and validation rows use "
        "the 100% fit-pool table. Calibration and validation labels never reach the encoder.",
        "- Fixed models (no tuning): XGBoost and logistic regression with A7's selected parameters; XGBoost "
        "scale_pos_weight = fit-pool negatives/positives; estimator seed = split seed.",
        "- Threshold: A7 F1-optimal grid on calibrated validation probabilities.", "",
        "## Predeclared decision rule", "",
        f"- {config['decision_rule']['adopt_arm_if']} (model: {config['decision_rule']['model']}).",
        f"- {config['decision_rule']['multiple_pass']}; ties -> {config['decision_rule']['tie_break']}.",
        f"- {config['decision_rule']['none_pass']}.", "",
        "## Decision", "",
        f"- **FINAL_TIER = `{decision['selected_tier']}`** (arm {decision['selected_arm']}).",
        f"- Arms passing on all seeds: {decision['passed_arms'] or 'none'}.", "",
        "| arm | seed | Δ ROC-AUC | Δ AP | passes |", "| --- | --- | --- | --- | --- |",
    ]
    for arm, seeds in decision["per_seed_deltas"].items():
        for seed, item in seeds.items():
            lines.append(f"| {arm} | {seed} | {item['delta_roc_auc']:+.4f} | "
                         f"{item['delta_average_precision']:+.4f} | {'yes' if item['passes'] else 'no'} |")
    lines += ["", "## Validation metrics (calibrated)", "",
              "| seed | model | arm | ROC-AUC | AP | Brier | intercept | slope | thr | F1 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in metrics.sort_values(["seed", "model", "arm"]).itertuples(index=False):
        lines.append(f"| {row.seed} | {row.model} | {row.arm} | {row.roc_auc:.4f} | {row.average_precision:.4f} | "
                     f"{row.brier_score:.4f} | {row.calibration_intercept:+.3f} | {row.calibration_slope:.3f} | "
                     f"{row.threshold:.2f} | {row.f1:.4f} |")
    screen = state["support"]
    lines += ["", "## State x residence support screen (A1 D9 gate, seed-42 training rows)", "",
              f"- min rows {screen['min_rows']}, min per class {screen['min_class']}: "
              f"{screen['n_cells']} cells, gate {'PASSED' if screen['gate_passed'] else 'FAILED'} "
              f"({screen['reason']}). Cell table: `a8_support_screen.csv` (counts < 10 suppressed).", ""]
    shap_block = state["shap"]
    lines += [f"## SHAP family shares: arm {decision['selected_arm']}, XGBoost, seed 42", "",
              f"- {shap_block['explainer']}; n = {shap_block['n_sample_rows']} validation rows (seed 4209).", "",
              "| rank | family | mean abs SHAP (margin) | share |", "| --- | --- | --- | --- |"]
    for row in shap_block["families"]:
        lines.append(f"| {row['rank']} | {row['family']} | {row['mean_abs_shap_margin']:.4f} | "
                     f"{row['share_of_total']:.4f} |")
    lines += ["", "## Quality and leakage checks", "", "| check | status | detail |", "| --- | --- | --- |"]
    for item in state["qa"]:
        lines.append(f"| {item['check']} | {item['status']} | {item['detail']} |")
    lines += ["", "## Disclosure", "",
              "- Aggregate outputs only; counts below 10 suppressed; no keys, rows or per-birth predictions.",
              "- Point estimates without design-based intervals; the decision uses the predeclared rule only."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    recorded = json.loads(A7_SUMMARY.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    qa: list = []
    try:
        if not args.skip_tests:
            _check(qa, "synthetic_a8_tests", True, _run_synthetic_tests())
        nfhs5 = Path(args.nfhs5_path)
        if not nfhs5.exists():
            raise GateFailure(f"NFHS-5 file not found: {nfhs5}")
        params = _model_params(config, recorded)
        print("loading NFHS-5 cohort ...", flush=True)
        cohort, summary, warnings, base, input_hash = _load_cohort(nfhs5, config, recorded, qa)
        states = map_states(cohort["v024"]).to_numpy()
        codes = cohort["v024"].to_numpy()
        y = cohort["m17"].to_numpy(dtype=int)
        groups = cohort["caseid"].astype("string").to_numpy()
        keys = _key_tokens(cohort)
        te_cfg = config["target_encoding"]
        gate = config["reproduction_gate"]

        rows, split_hashes, seed42 = [], {}, None
        for seed in config["split"]["seeds"]:
            seed = int(seed)
            split = _split(cohort, config, seed, keys, recorded, qa)
            split_hashes[seed] = split["split_hash"]
            print(f"seed {seed}: fit {len(split['fit']):,} / calibration {len(split['calib']):,} / "
                  f"validation {len(split['valid']):,}", flush=True)
            result = run_seed(base, states, codes, y, groups, fit_positions=split["fit"],
                              calib_positions=split["calib"], valid_positions=split["valid"],
                              models=config["models"], params=params, te_config=te_cfg, estimator_seed=seed)
            for row in result["rows"]:
                rows.append({**row, "seed": seed, "split_id": split["split_id"], "split_hash": split["split_hash"],
                             "valid_digest": split["digests"]["validation"], "n_mothers": result["n_mothers_valid"],
                             "tier_name": config["decision_rule"]["tier_names"][row["arm"]]})
            if seed == int(config["split"]["frozen_seed"]):
                h0 = next(r for r in result["rows"] if r["model"] == "xgboost" and r["arm"] == "H0")
                for metric in ("roc_auc", "average_precision"):
                    target, tol = gate[metric]["target"], gate[metric]["tolerance"]
                    _check(qa, f"h0_xgboost_seed42_{metric}_reproduces_a7", abs(h0[metric] - target) <= tol,
                           f"{h0[metric]:.4f} vs {target} ± {tol}")
                print("running real-data leakage checks ...", flush=True)
                _leakage_checks(base, states, codes, y, groups, split, params, te_cfg, seed, result, qa)
                seed42 = {"split": split, "result": result}

        metrics = pd.DataFrame(rows)
        deltas = paired_deltas(metrics)
        decision = decide(deltas, config["decision_rule"], config["split"]["seeds"])

        screen_cfg = config["support_screen"]
        train_mask = seed42["split"]["train_mask"]
        cells = pd.Series(states[train_mask]).astype("string") + INTERACTION_SEPARATOR + \
            base.loc[train_mask, "residence"].astype("string").to_numpy()
        support = support_screen(cells.to_numpy(), y[train_mask], min_rows=int(screen_cfg["min_training_rows"]),
                                 min_class=int(screen_cfg["min_class_rows"]))
        selected = decision["selected_arm"]
        print(f"SHAP for arm {selected} ...", flush=True)
        shap_block = shap_family_table(seed42["result"]["classifiers"][("xgboost", selected)],
                                       seed42["result"]["frames"][selected],
                                       sample_size=int(config["shap"]["sample_size"]), seed=int(config["shap"]["seed"]))
    except GateFailure as exc:
        print(f"\nSTOPPED: {exc}\nNo outputs written.")
        for item in qa:
            print(f"  {item['status']:4s} {item['check']} {item['detail']}")
        return 1

    from exp_provenance import git_info

    git = git_info(_REPO)
    versions = software_versions()
    versions_string = json.dumps(versions, sort_keys=True, separators=(",", ":"))
    feature_hashes = {
        arm: feature_manifest_hash(tier=arm, numerical=list(arm_manifest(arm).numerical),
                                   categorical=list(arm_manifest(arm).categorical), builder_id=BUILDER,
                                   extra={"te": te_cfg} if ARMS[arm]["numerical"] else None)
        for arm in ARM_ORDER
    }
    min_cell = int(config["disclosure"]["min_cell"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*"):
        if stale.is_file():
            stale.unlink()
    write_csv(pd.DataFrame(_schema_rows(metrics, git["commit"], versions_string, feature_hashes)),
              out_dir / "a8_ablation.csv")
    write_csv(deltas.round(6), out_dir / "a8_paired_deltas.csv")
    write_json(decision, out_dir / "a8_decision.json")
    write_csv(_support_rows(support, min_cell), out_dir / "a8_support_screen.csv")
    write_csv(pd.DataFrame(shap_block["families"])[["rank", "family", "mean_abs_shap_margin", "share_of_total"]],
              out_dir / "a8_shap_families.csv")
    _fig_deltas(deltas, config["decision_rule"], out_dir / "fig_a8_deltas.png")
    state = {"config": config, "decision": decision, "metrics": metrics, "support": support, "shap": shap_block,
             "qa": qa, "git_commit": git["commit"], "input_hash": input_hash,
             "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_text(_methods(state), out_dir / "a8_methods_results.md")
    write_manifest(out_dir, experiment_id="A8", config=config, config_path=config_path, repo_root=_REPO,
                   seeds=config["split"]["seeds"], split_manifest_hashes=split_hashes,
                   extra={"software_versions_runtime": versions, "feature_manifest_hashes": feature_hashes,
                          "membership_digests_seed42": seed42["split"]["digests"],
                          "cohort": {k: summary[k] for k in ("n_births", "n_mothers", "n_events")},
                          "cohort_warnings": warnings, "input_sha256": input_hash,
                          "estimator_seed_rule": "split seed", "calibration_carve_seed": 4208,
                          "te_inner_seed": te_cfg["inner_seed"], "shap_seed": config["shap"]["seed"],
                          "qa": qa, "test_partition": "never indexed or scored"})

    print("\nPer-seed paired deltas vs H0 (validation, XGBoost = decision model):")
    table = deltas.sort_values(["model", "arm", "seed"])
    print(table[["model", "arm", "seed", "delta_roc_auc", "delta_average_precision", "delta_brier_score"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nDecision: FINAL_TIER = {decision['selected_tier']} (arm {decision['selected_arm']}); "
          f"passed arms: {decision['passed_arms'] or 'none'}")
    print(f"outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
