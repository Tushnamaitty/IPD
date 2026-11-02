"""A9 CLI runner: final PSU-grouped evaluation with cluster-bootstrap CIs.

Follows PROTOCOL.md exactly. Before any data are read it verifies that
PROTOCOL.md and config.yaml are byte-identical to the protocol commit, which
is recorded in every output. Each seed's test partition is scored once,
behind a guard file in outputs/private/ (git-ignored, aggregates only).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_HERE = Path(__file__).resolve()
_A9 = _HERE.parents[1]
_BASE = _HERE.parents[3]
_REPO = _BASE.parent
for _entry in (
    _BASE,
    _BASE / "experiments" / "common",
    _BASE / "experiments" / "A1" / "src",
    _BASE / "experiments" / "A3",
    _BASE / "experiments" / "A4" / "src",
    _BASE / "experiments" / "A7" / "src",
    _BASE / "experiments" / "A8" / "src",
    _HERE.parent,
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from a1_state_context import map_states  # noqa: E402
from a4_facility_detail import facility_detail  # noqa: E402
from a4_history import OBSERVED_PREVIOUS_INSTITUTIONAL_CS, build_history_features  # noqa: E402
from a7_candidates import build_tier_frame, carve_calibration_partition  # noqa: E402
from a8_outputs import write_csv, write_json, write_manifest, write_text  # noqa: E402
from a9_guard import GuardBlocked, complete_guard, read_guard, start_guard  # noqa: E402
from a9_metrics import calibration_bins, curves, psu_bootstrap, test_metrics, weighted_metrics  # noqa: E402
from a9_models import FEATURE_LABELS, MODEL_ORDER, fit_seed, model_frames, predict_test  # noqa: E402
from a9_report import (  # noqa: E402
    fig_calibration, fig_forest, fig_roc_pr, paired_deltas_table, paper_table, pooled_summary_table,
    statements, test_metrics_table,
)
from a9_splits import assert_psu_nesting, overlap_counts, partition_summary, psu_stratified_partition  # noqa: E402
from a9_v1 import reconcile_f0  # noqa: E402
from aggregate_schema import suppress_small_count  # noqa: E402
from checksums import membership_digest  # noqa: E402
from data_access import assert_frozen_c1, build_c1_cohort, read_dta_subset, sha256_file, software_versions  # noqa: E402
from feature_builder import make_core_features  # noqa: E402
from manifests import split_manifest_hash  # noqa: E402

A7 = _BASE / "experiments" / "A7"
PROTOCOL_FILES = ("PROTOCOL.md", "config.yaml")


class ProtocolError(RuntimeError):
    pass


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True)


def verify_protocol() -> str:
    """Protocol commit = last commit touching PROTOCOL.md; files must be unchanged since then."""
    rel = [str((_A9 / name).relative_to(_REPO)).replace("\\", "/") for name in PROTOCOL_FILES]
    commit = _git("log", "-n1", "--format=%H", "--", rel[0]).stdout.strip()
    if not commit:
        raise ProtocolError("PROTOCOL.md has never been committed; commit it before any scoring")
    if _git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        raise ProtocolError(f"Protocol commit {commit[:7]} is not an ancestor of HEAD")
    for path in rel:
        if _git("diff", "--quiet", commit, "--", path).returncode != 0:
            raise ProtocolError(f"{path} differs from the protocol commit {commit[:7]}")
        if _git("status", "--porcelain", "--", path).stdout.strip():
            raise ProtocolError(f"{path} has uncommitted changes")
    return commit


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="A9 final evaluation")
    parser.add_argument("--nfhs5-path", default=os.environ.get("NFHS5_PATH"))
    parser.add_argument("--config", default=str(_A9 / "config.yaml"))
    parser.add_argument("--out-dir", default=str(_A9 / "outputs"))
    args = parser.parse_args(argv)
    if not args.nfhs5_path:
        parser.error("NFHS5_PATH is not set and --nfhs5-path was not given")
    return args


def _run_synthetic_tests() -> str:
    suite = unittest.defaultTestLoader.discover(str(_A9 / "tests"), pattern="test_*.py", top_level_dir=str(_A9 / "tests"))
    result = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0:
        raise ProtocolError(f"Synthetic A9 tests failed ({len(result.failures)} failures, {len(result.errors)} errors)")
    return f"{result.testsRun} synthetic tests OK"


def _params(config) -> dict:
    recorded = json.loads((A7 / "outputs" / "a7_run_summary.json").read_text(encoding="utf-8"))["tuning"]["selected_params"]
    reference = yaml.safe_load((A7 / "config.yaml").read_text(encoding="utf-8"))["models"]["xgboost"]
    reference = {k: v for k, v in reference.items() if k != "scale_pos_weight"}
    found = {"a7_selected_xgboost": recorded["xgboost"], "a7_selected_logistic_regression": recorded["logistic_regression"],
             "a7_reference_xgboost": reference}
    for key, value in found.items():
        if value != config["model_params"][key]:
            raise ProtocolError(f"{key}: A7 source {value} differs from A9 protocol {config['model_params'][key]}")
    return found


def _load(config, path: Path, qa: list) -> dict:
    recorded = json.loads((A7 / "outputs" / "a7_run_summary.json").read_text(encoding="utf-8"))
    input_hash = sha256_file(path)
    if input_hash != recorded["input_sha256"]:
        raise ProtocolError("NFHS-5 input sha256 differs from the A7 recording")
    qa.append({"check": "input_sha256_matches_a7", "status": "PASS", "detail": input_hash[:16]})
    raw = read_dta_subset(path, extra_columns=("v021", "v005"))
    cohort, summary = build_c1_cohort(raw)
    warnings = assert_frozen_c1(summary)
    for key, value in config["expected_cohort"].items():
        if int(summary[key]) != int(value):
            raise ProtocolError(f"cohort {key} {summary[key]} != {value}")
    core = make_core_features(cohort, wave="NFHS-5")
    history, diag = build_history_features(raw, cohort, include_other_unknown_sector=True)
    if diag["level_counts"].get(OBSERVED_PREVIOUS_INSTITUTIONAL_CS, 0) != 6276:
        raise ProtocolError("History reproduction failure")
    qa.append({"check": "cohort_and_history_frozen", "status": "PASS", "detail": "201311/158429/44742; 6276"})
    nesting = assert_psu_nesting(cohort, psu_col=config["split"]["group_col"], mother_col=config["split"]["mother_col"])
    qa.append({"check": "every_mother_in_exactly_one_psu", "status": "PASS", "detail": f"{nesting['n_psus']} PSUs"})
    base = build_tier_frame(core, facility_detail(cohort["m15"]), history, "F1_detail_history")
    keys = (cohort["caseid"].astype("string") + "|"
            + cohort["bidx"].map(lambda value: str(int(value))).astype("string")).to_numpy()
    return {"cohort": cohort, "summary": summary, "warnings": warnings, "base": base, "keys": keys,
            "states": map_states(cohort["v024"]).to_numpy(), "input_hash": input_hash, "nesting": nesting}


def _score_seed(seed, data, frames, config, params, guard_dir, protocol_commit) -> dict:
    cohort, keys = data["cohort"], data["keys"]
    split = config["split"]
    psu = pd.to_numeric(cohort[split["group_col"]]).astype(int).astype(str)
    partition = psu_stratified_partition(cohort, seed=seed, group_col=split["group_col"], target_col=split["target_col"],
                                         fractions=tuple(split["fractions"]), edges=tuple(split["strata_edges"]),
                                         mother_col=split["mother_col"])
    overlaps = {"psu": overlap_counts(cohort, partition, split["group_col"]),
                "mother": overlap_counts(cohort, partition, split["mother_col"])}
    if any(v for block in overlaps.values() for v in block.values()):
        raise ProtocolError(f"seed {seed}: partition overlap {overlaps}")
    split_id = split["split_id_template"].format(seed=seed)
    psu_assign = dict(zip(psu.tolist(), partition.tolist()))
    split_hash = split_manifest_hash(split_id=split_id, seed=seed, fractions=split["fractions"],
                                     assignments=psu_assign.items())
    masks = {label: partition.eq(label).to_numpy() for label in ("train", "validation", "test")}
    train = np.flatnonzero(masks["train"])
    carve_cfg = config["calibration"]["carve"]
    carve = carve_calibration_partition(cohort.iloc[train].reset_index(drop=True), group_col=carve_cfg["group_col"],
                                        calibration_fraction=float(carve_cfg["subset_fraction"]), seed=int(carve_cfg["seed"]))
    fit_pos, calib_pos = train[carve.eq("fit").to_numpy()], train[carve.eq("calibration").to_numpy()]
    if set(psu.iloc[fit_pos]) & set(psu.iloc[calib_pos]):
        raise ProtocolError(f"seed {seed}: calibration carve shares PSUs with the fit pool")
    valid_pos, test_pos = np.flatnonzero(masks["validation"]), np.flatnonzero(masks["test"])
    audit = {
        "seed": seed, "split_id": split_id, "split_manifest_hash": split_hash,
        "partitions": partition_summary(cohort, partition, psu_col=split["group_col"], mother_col=split["mother_col"]),
        "overlap": overlaps,
        "membership_digests": {label: membership_digest(keys[mask], seed=seed) for label, mask in masks.items()},
        "calibration_carve": {"fit_births": int(len(fit_pos)), "calibration_births": int(len(calib_pos)),
                              "group": carve_cfg["group_col"], "psu_overlap": 0},
    }
    y = cohort[split["target_col"]].to_numpy(dtype=int)
    print(f"seed {seed}: fitting {len(frames)} models on {len(fit_pos):,} fit rows ...", flush=True)
    models = fit_seed(frames, config["models"], params, fit_positions=fit_pos, calib_positions=calib_pos,
                      valid_positions=valid_pos, y=y, seed=seed)
    thresholds = {name: model["thresholds"] for name, model in models.items()}

    start_guard(guard_dir, seed, protocol_commit)          # from here on, this seed's test counts as scored
    print(f"seed {seed}: scoring test once ({len(test_pos):,} births) and bootstrapping ...", flush=True)
    p_test = predict_test(models, frames, test_pos)
    y_test = y[test_pos]
    psu_test = psu.to_numpy()[test_pos]
    weights = cohort["v005"].to_numpy(dtype=float)[test_pos] / float(config["weighted_sensitivity"]["scale"])
    boot = config["bootstrap"]
    replicates = psu_bootstrap(y_test, p_test, psu_test, thresholds, replicates=int(boot["replicates"]),
                               seed=int(boot["seed"]))
    min_cell = int(config["disclosure"]["min_cell"])
    results = {
        "seed": seed, "protocol_commit": protocol_commit, "audit": audit, "thresholds": thresholds,
        "test_counts": {"n_births": int(len(test_pos)), "n_psus": int(len(set(psu_test))),
                        "n_mothers": int(cohort[split["mother_col"]].iloc[test_pos].nunique()),
                        "n_events": suppress_small_count(int(y_test.sum()), threshold=min_cell),
                        "prevalence": float(y_test.mean())},
        "point": {name: test_metrics(y_test, p, thresholds[name]) for name, p in p_test.items()},
        "replicates": {name: {k: v.tolist() for k, v in reps.items()} for name, reps in replicates.items()},
        "bootstrap": {"replicates": int(boot["replicates"]), "seed": int(boot["seed"]), "unit": "v021 within test"},
        "weighted": {name: weighted_metrics(y_test, p, weights) for name, p in p_test.items()},
        "bins": {name: calibration_bins(y_test, p, n_bins=int(config["calibration_bins"]["n_bins"]),
                                        min_cell=min_cell).to_dict(orient="records") for name, p in p_test.items()},
        "curves": {name: curves(y_test, p) for name, p in p_test.items()},
    }
    complete_guard(guard_dir, seed, results)
    del p_test, y_test
    return results


def public_split_audit(protocol_commit, nesting, audits, v1_info) -> dict:
    """a9_split_audit.json: overlap counts are renamed so no key collides with restricted field names."""
    seeds = []
    for audit in audits:
        public = {k: v for k, v in audit.items() if k != "overlap"}
        public["overlap_counts"] = {"between_partitions_by_cluster": audit["overlap"]["psu"],
                                    "between_partitions_by_mother": audit["overlap"]["mother"]}
        seeds.append(public)
    return {"protocol_commit": protocol_commit, "cluster_nesting": nesting, "seeds": seeds, "v1_spec": v1_info}


def _methods(state) -> str:
    config, st = state["config"], state["statements"]
    lines = ["# A9 methods and results: final PSU-grouped evaluation", "",
             f"- protocol_commit: `{state['protocol_commit']}` (PROTOCOL.md committed before any scoring)",
             f"- repo_commit at run: `{state['git']['commit']}`",
             f"- generated_at_utc: {state['generated']}",
             f"- input_sha256: {state['input_hash']}",
             f"- FINAL_TIER: `{config['final_tier']['name']}` (A8 arm {config['final_tier']['a8_arm']}, commit {config['final_tier']['a8_commit']})",
             f"- seeds: {config['split']['seeds']}; PSU (v021) grouped 70/15/15 stratified on PSU outcome rate; "
             f"bootstrap B = {config['bootstrap']['replicates']}, seed {config['bootstrap']['seed']}, PSU resampling within test",
             f"- V1_spec: {state['v1']['status']}", "",
             "## Predeclared statements (wording fixed in PROTOCOL.md)", ""]
    for key in ("superiority", "calibration", "equivalence"):
        lines.append(f"{('1', '2', '3')[('superiority', 'calibration', 'equivalence').index(key)]}. **{st[key]['statement']}.** "
                     f"{st[key]['evidence']}.")
    lines += ["", "## Paper table (pooled = mean across 5 seeds; 95% CI from the across-seed mean of PSU bootstrap replicates)",
              "", state["paper_table"], "",
              "The V1 recorded row is the original V1 report (different split, mother-level random test set, not paired, "
              "no CI available); it is shown for orientation only.", "",
              "## Paired deltas (test, ROC-AUC and PR-AUC)", "",
              "| comparison | seed | metric | delta | 95% CI |", "| --- | --- | --- | --- | --- |"]
    d = state["deltas"]
    for row in d[d.metric.isin(["roc_auc", "average_precision"])].itertuples():
        lines.append(f"| {row.comparison} | {row.seed} | {row.metric} | {row.delta:+.4f} | [{row.ci_low:+.4f}, {row.ci_high:+.4f}] |")
    lines += ["", "## Per-seed test metrics (V2)", "",
              "| seed | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier | slope | intercept | thr F1 | F1 | thr rec80 | recall | precision |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    t = state["metrics"]
    for r in t[t.model == "V2"].itertuples():
        lines.append(f"| {r.seed} | {r.roc_auc:.4f} ({r.roc_auc_ci_low:.4f}-{r.roc_auc_ci_high:.4f}) | "
                     f"{r.average_precision:.4f} ({r.average_precision_ci_low:.4f}-{r.average_precision_ci_high:.4f}) | "
                     f"{r.brier_score:.4f} | {r.calibration_slope:.3f} | {r.calibration_intercept:+.3f} | "
                     f"{r.threshold_f1_optimal:.2f} | {r.f1_f1opt:.4f} | {r.threshold_recall_80:.2f} | {r.recall_rec80:.4f} | "
                     f"{r.precision_rec80:.4f} |")
    lines += ["", "## Split audit", "", "| seed | train births/PSUs | validation births/PSUs | test births/PSUs | test prevalence | PSU overlap | mother overlap |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for a in state["audits"]:
        p = a["partitions"]
        lines.append(f"| {a['seed']} | {p['train']['n_births']:,}/{p['train']['n_psus']:,} | "
                     f"{p['validation']['n_births']:,}/{p['validation']['n_psus']:,} | {p['test']['n_births']:,}/{p['test']['n_psus']:,} | "
                     f"{p['test']['prevalence']:.4f} | {sum(a['overlap']['psu'].values())} | {sum(a['overlap']['mother'].values())} |")
    lines += ["", "## V1_spec reconciliation", "", f"- status: {state['v1']['status']}"]
    lines += [f"- {k}: {v}" for k, v in state["v1"]["reconciliation"].items()]
    lines += ["", "## Test scoring log", ""]
    lines += [f"- seed {g['seed']}: {g['source']} (guard completed {g['completed_utc']})" for g in state["guards"]]
    lines += ["", "## Quality checks", "", "| check | status | detail |", "| --- | --- | --- |"]
    lines += [f"| {q['check']} | {q['status']} | {q['detail']} |" for q in state["qa"]]
    lines += ["", "## Disclosure and limitations", "",
              "- Unweighted primary analysis; weighted (v005) metrics in `a9_weighted_sensitivity.csv` are descriptive only.",
              "- Percentile CIs from a PSU cluster bootstrap within each test partition. They ignore the survey's "
              "stratification and are not design-based standard errors.",
              "- The NFHS-5 outcomes were inspected in V1 and in A1-A8 development work; PSU-grouped test partitions are "
              "new, but the data are not an independent external validation.",
              "- Aggregate outputs only; counts below 10 suppressed; no keys, rows or per-birth predictions written."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    guard_dir = out_dir / config["guard"]["dir"].split("/", 1)[1]
    qa = []
    try:
        protocol_commit = verify_protocol()
        qa.append({"check": "protocol_files_unchanged_since_protocol_commit", "status": "PASS", "detail": protocol_commit[:7]})
        qa.append({"check": "synthetic_a9_tests", "status": "PASS", "detail": _run_synthetic_tests()})
        params = _params(config)
        print("loading NFHS-5 cohort ...", flush=True)
        data = _load(config, Path(args.nfhs5_path), qa)
        v1_info, f0 = reconcile_f0(_REPO / config["v1_spec"]["parquet"], data["cohort"], data["keys"])
        qa.append({"check": "v1_spec_parquet_reconciled", "status": "PASS" if v1_info["status"] == "reconciled" else "NOT_RUN",
                   "detail": v1_info["status"]})
        frames = model_frames(data["base"], data["states"], f0)
        results, guards = [], []
        for seed in config["split"]["seeds"]:
            seed = int(seed)
            cached = read_guard(guard_dir, seed)
            if cached is not None:
                if cached["results"]["protocol_commit"] != protocol_commit:
                    raise ProtocolError(f"seed {seed} was scored under protocol {cached['results']['protocol_commit'][:7]}")
                print(f"seed {seed}: test already scored; reusing stored aggregates (no re-scoring)", flush=True)
                results.append(cached["results"])
                guards.append({"seed": seed, "source": "cache_no_rescore", "completed_utc": cached["completed_utc"]})
                continue
            results.append(_score_seed(seed, data, frames, config, params, guard_dir, protocol_commit))
            guards.append({"seed": seed, "source": "scored_once_this_run",
                           "completed_utc": read_guard(guard_dir, seed)["completed_utc"]})
        qa.append({"check": "zero_psu_and_mother_overlap_all_seeds", "status": "PASS",
                   "detail": "asserted per seed before fitting"})
        qa.append({"check": "guard_files_completed", "status": "PASS",
                   "detail": ", ".join(f"test_scored_{g['seed']}.json" for g in guards)})
    except (ProtocolError, GuardBlocked) as exc:
        print(f"\nSTOPPED: {exc}")
        return 1

    from exp_provenance import git_info

    git = git_info(_REPO)
    metrics = test_metrics_table(results, protocol_commit)
    deltas = paired_deltas_table(results, protocol_commit)
    pooled = pooled_summary_table(results, protocol_commit)
    st = statements(deltas, pooled, config, v1_info["status"])
    table = paper_table(pooled, config["v1_spec"]["recorded_v1"])
    bins = pd.DataFrame([{"protocol_commit": protocol_commit, "seed": r["seed"], "model": m, **row}
                         for r in results for m, rows in r["bins"].items() for row in rows])
    weighted = pd.DataFrame([{"protocol_commit": protocol_commit, "seed": r["seed"], "model": m,
                              "role": "descriptive_sensitivity_only", **vals}
                             for r in results for m, vals in r["weighted"].items()])
    audits = [r["audit"] for r in results]

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*"):
        if stale.is_file():
            stale.unlink()
    write_csv(metrics, out_dir / "a9_test_metrics.csv")
    write_csv(deltas, out_dir / "a9_paired_deltas.csv")
    write_csv(pooled, out_dir / "a9_pooled_summary.csv")
    write_csv(bins, out_dir / "a9_calibration_bins.csv")
    write_csv(weighted, out_dir / "a9_weighted_sensitivity.csv")
    write_json(public_split_audit(protocol_commit, data["nesting"], audits, v1_info), out_dir / "a9_split_audit.json")
    write_json({"protocol_commit": protocol_commit, "statements": st}, out_dir / "a9_statements.json")
    fig_roc_pr(results, out_dir / "fig_a9_roc_pr.png")
    fig_calibration(bins, out_dir / "fig_a9_calibration.png")
    fig_forest(deltas, out_dir / "fig_a9_forest.png")
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_text(_methods({"config": config, "statements": st, "paper_table": table, "deltas": deltas, "metrics": metrics,
                         "audits": audits, "v1": v1_info, "guards": guards, "qa": qa, "git": git,
                         "protocol_commit": protocol_commit, "generated": generated, "input_hash": data["input_hash"]}),
               out_dir / "a9_methods_results.md")
    write_manifest(out_dir, experiment_id="A9", config=config, config_path=config_path, repo_root=_REPO,
                   seeds=config["split"]["seeds"], split_manifest_hashes={a["seed"]: a["split_manifest_hash"] for a in audits},
                   extra={"protocol_commit": protocol_commit, "software_versions_runtime": software_versions(),
                          "input_sha256": data["input_hash"], "bootstrap": config["bootstrap"],
                          "calibration_carve_seed": config["calibration"]["carve"]["seed"],
                          "estimator_seed_rule": "split seed", "test_scoring": guards, "qa": qa,
                          "excluded": "outputs/private/ (score-once guard files): git-ignored, aggregates only"})

    print("\nPaper table (pooled across 5 PSU-grouped seeds):\n")
    print(table)
    print("\nPredeclared statements:")
    for key in ("superiority", "calibration", "equivalence"):
        print(f"  - {st[key]['statement']}. [{st[key]['evidence']}]")
    print(f"\nprotocol_commit {protocol_commit}; outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
