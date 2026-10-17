#!/usr/bin/env python3
"""A3 runner: grouped OOF state x sector target encoding, 3-arm matched ablation.

Order: synthetic adversarial tests -> NFHS-5 load -> shared mother-grouped split
-> per cohort: inner folds, alpha selection on outer-train only, OOF encoding
for train / full-train table for validation(/test) -> fixed-model ablation.
NFHS-5 validation/test outcomes are used only for evaluation and the validation
threshold. The test partition is not scored unless --evaluate-test is passed.
Aggregate outputs only.

Usage (from repo root):  python version2_ml/experiments/A3/run.py [--evaluate-test]
"""
from __future__ import annotations
import argparse
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent / "common"), str(ROOT / "version2_ml")]
from A3_target_encoder import (GroupedOOFTargetEncoder, TESpec, encode_partitions, select_alpha,  # noqa: E402
                               TE_CELL, TE_STATE, TE_SECTOR, TE_FLAG, TE_COLUMNS, LEVEL)
from exp_cohort import load_wave_births, RAW  # noqa: E402
from exp_metrics import metric_row, select_threshold_max_f1, threshold_provenance  # noqa: E402
from exp_provenance import git_info, environment, assert_disclosure_safe, scan_output_dir, write_json  # noqa: E402
from split_contracts import grouped_partition, audit_partition  # noqa: E402
from preprocessing import FeatureManifest  # noqa: E402
from model_framework import AuthorizedClassifier, make_estimator  # noqa: E402

CORE_NUM = ("maternal_age_at_index_birth", "education_years", "birth_order")
CORE_CAT = ("wealth_index", "facility_type", "residence", "religion", "twin_order")
# facility_type (public/private/other) IS the ordinary sector indicator and is already in the core.
ARMS = {
    "i_core": {"num": (), "cat": (), "tier": "FT", "role": "required_i_common_core"},
    "ii_state_sector_indicators": {"num": (), "cat": ("nfhs5_state",), "tier": "FT",
                                   "role": "required_ii_core_plus_state_and_sector_indicators"},
    "iii_indicators_plus_oof_te": {"num": (TE_CELL,), "cat": ("nfhs5_state",), "tier": "F4",
                                   "role": "required_iii_ii_plus_oof_te_state_x_sector"},
    "x_indicators_plus_all_te": {"num": (TE_CELL, TE_STATE, TE_SECTOR, TE_FLAG), "cat": ("nfhs5_state",),
                                 "tier": "F4", "role": "exploratory_ii_plus_all_te_and_fallback_flag"},
    "x_core_plus_oof_te_only": {"num": (TE_CELL,), "cat": (), "tier": "F4",
                                "role": "exploratory_core_plus_te_without_state_indicators"},
}
TE_ARMS = ("iii_indicators_plus_oof_te", "x_indicators_plus_all_te", "x_core_plus_oof_te_only")
REQUIRED = ("i_core", "ii_state_sector_indicators", "iii_indicators_plus_oof_te")
SOURCE_DOMAIN = "NFHS-5 outer train (TE fitted on outer-train labels; OOF within train)"


def arm_manifest(arm: str) -> FeatureManifest:
    a = ARMS[arm]
    return FeatureManifest(numerical=CORE_NUM + tuple(a["num"]), categorical=CORE_CAT + tuple(a["cat"]),
                           tier=a["tier"])


def run_adversarial_tests() -> list[dict]:
    suite = unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_a3_*.py",
                                                top_level_dir=str(HERE / "tests"))
    outcomes = {}

    class Collector(unittest.TestResult):
        def addSuccess(self, test):
            outcomes[test.id()] = "PASS"

        def addFailure(self, test, err):
            outcomes[test.id()] = "FAIL"

        def addError(self, test, err):
            outcomes[test.id()] = "ERROR"

        def addSkip(self, test, reason):
            outcomes[test.id()] = "SKIPPED"
    suite.run(Collector())
    return [{"test": k.split(".")[-1], "status": v} for k, v in sorted(outcomes.items())]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "aggregate"))
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    evaluate_test = bool(args.evaluate_test or cfg.get("evaluate_test", False))
    eval_splits = ["validation"] + (["test"] if evaluate_test else [])
    icv, enc_cfg = cfg["inner_cv"], cfg["encoding"]
    qa = []

    def check(name, ok, detail=""):
        qa.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        if not ok:
            raise AssertionError(f"QA failed: {name}: {detail}")

    # 0. Synthetic adversarial tests must pass before any real data are touched.
    tests = run_adversarial_tests()
    check("synthetic_adversarial_tests", all(t["status"] == "PASS" for t in tests) and len(tests) >= 5,
          f"{sum(t['status'] == 'PASS' for t in tests)}/{len(tests)} passed")

    # 1. NFHS-5 and shared split.
    tgt, flow5 = load_wave_births(ROOT, "NFHS-5", use_cache=not args.no_cache)
    tgt["nfhs5_state"] = tgt["v024"].astype(int).astype(str)
    sp = cfg["split"]
    partition = grouped_partition(tgt, group_col=sp["group"], target_col="m17",
                                  fractions=tuple(sp["fractions"]), seed=int(sp["seed"]))
    split_audit = audit_partition(tgt, partition, group_col=sp["group"], target_col="m17")
    check("outer_split_mother_exclusive", split_audit["status"] == "GROUP_EXCLUSIVE_VALID")
    cohorts = {"primary": pd.Series(True, index=tgt.index),
               "exclude_other_facility": tgt["m15"].ne(96),
               "nfhs5_cmc_0_59": tgt["cmc_recall"].between(0, 59)}
    y_all = tgt["m17"].to_numpy(int)
    groups_all = tgt["caseid"].to_numpy()
    base_cols = list(CORE_NUM + CORE_CAT) + ["nfhs5_state", "sector"]

    rows, alpha_tables, fold_reports, level_rows, cohort_splits = [], [], {}, [], {}
    for cohort in cfg["cohorts"]:
        mask = cohorts[cohort].to_numpy()
        frame = tgt.loc[mask, base_cols]
        part = partition[mask]
        y, groups = y_all[mask], groups_all[mask]
        cohort_splits[cohort] = audit_partition(tgt[mask], part, group_col=sp["group"], target_col="m17")["summary"]
        train = part.eq("train").to_numpy()
        enc0 = GroupedOOFTargetEncoder(TESpec(n_min=enc_cfg["n_min"]), n_splits=icv["n_splits"],
                                       seed=icv["random_state"])
        folds = enc0.inner_folds(y[train], groups[train])
        check(f"{cohort}_inner_folds_mother_exclusive", True, f"{icv['n_splits']} StratifiedGroupKFold folds")
        alpha, atab = select_alpha(frame[train], y[train], groups[train], enc_cfg["alpha_grid"],
                                   n_min=enc_cfg["n_min"], n_splits=icv["n_splits"], seed=icv["random_state"],
                                   partition="train", folds=folds)
        alpha_tables.append(atab.assign(cohort=cohort))
        alphas = [alpha] + ([a for a in enc_cfg["exploratory_alphas"] if a != alpha] if cohort == "primary" else [])
        for a in alphas:
            spec = TESpec(alpha=float(a), n_min=enc_cfg["n_min"])
            enc, te = encode_partitions(frame, y, groups, part, spec, n_splits=icv["n_splits"],
                                        seed=icv["random_state"], folds=folds)
            check(f"{cohort}_alpha{a:g}_full_table_fit_on_outer_train_only",
                  enc.full_table.tables["n"] == int(train.sum()))
            if a == alpha:
                fold_reports[cohort] = {"selected_alpha": alpha, "inner_folds": enc.fold_report}
                if cohort == "primary":
                    # Runtime adversarial check on real data: mutate validation/test outcomes.
                    held = ~train
                    rng = np.random.default_rng(seed)
                    for label, ym in (("flipped", np.where(held, 1 - y, y)),
                                      ("permuted", np.where(held, rng.permutation(y), y))):
                        enc_m, te_m = encode_partitions(frame, ym, groups, part, spec, n_splits=icv["n_splits"],
                                                        seed=icv["random_state"], folds=folds)
                        same = (enc_m.full_table.digest() == enc.full_table.digest()
                                and all(te_m[c].to_numpy().tobytes() == te[c].to_numpy().tobytes()
                                        for c in TE_COLUMNS))
                        check(f"runtime_validation_test_labels_{label}_encodings_bitwise_identical", same)
                    for s in ("train", "validation"):
                        sub = te[part.eq(s).to_numpy()]
                        for lvl in ("cell", "state", "sector", "global"):
                            n = int(sub[LEVEL].eq(lvl).sum())
                            level_rows.append({"cohort": cohort, "alpha": alpha, "split": s,
                                               "encoding": "oof_inner_folds" if s == "train" else "full_outer_train_table",
                                               "te_level": lvl,
                                               "n_births": n if n == 0 or n >= cfg["disclosure"]["min_level_rows"]
                                               else "SUPPRESSED"})
                    cells = pd.DataFrame(enc.full_table.tables["cell"]).T
                    bins = pd.cut(cells["n"].astype(int), [0, 9, 29, 99, 499, 1999, np.inf],
                                  labels=["1-9", "10-29", "30-99", "100-499", "500-1999", "2000+"])
                    for b, c in bins.value_counts(sort=False).items():
                        level_rows.append({"cohort": cohort, "alpha": alpha, "split": "train",
                                           "encoding": "support_histogram_full_table_cells", "te_level": str(b),
                                           "n_births": None, "n_cells": int(c)})
            X = pd.concat([frame, te[list(TE_COLUMNS)]], axis=1)
            arms = list(ARMS) if a == alpha else list(TE_ARMS)
            for model in cfg["models"]:
                for arm in arms:
                    clf = AuthorizedClassifier(arm_manifest(arm), algorithm=model, seed=seed)
                    clf.fit(X[train], y[train], partition="train")
                    preds = {s: clf.predict_proba(X[part.eq(s).to_numpy()], partition=s) for s in eval_splits}
                    vmask = part.eq("validation").to_numpy()
                    thr = select_threshold_max_f1(y[vmask], preds["validation"])
                    for s in eval_splits:
                        m = part.eq(s).to_numpy()
                        base = {"experiment": "A3", "wave": "NFHS-5", "source_domain": SOURCE_DOMAIN,
                                "target_domain": "NFHS-5", "cohort": cohort,
                                "te_alpha": None if arm not in TE_ARMS else float(a),
                                "alpha_selected": a == alpha, "split": s, "model": model, "arm": arm,
                                "arm_role": ARMS[arm]["role"], "feature_tier": ARMS[arm]["tier"],
                                "weighting": "unweighted"}
                        rows.append(metric_row(base, y[m], preds[s], groups[m], thr, threshold_provenance(s)))
                    print(f"[{time.time() - t0:7.1f}s] {cohort:24s} alpha={a:<5g} {model:20s} {arm}", flush=True)
    metrics = pd.DataFrame(rows)

    # 2. Ablations on identical rows.
    def get(cohort, split, model, arm, sel=True):
        r = metrics[(metrics.cohort == cohort) & (metrics.split == split) & (metrics.model == model)
                    & (metrics.arm == arm) & (metrics.alpha_selected == sel)]
        if len(r) != 1:
            raise AssertionError(f"Expected one metric row for {cohort}/{split}/{model}/{arm}")
        return r.iloc[0]
    comparisons = [("iii_indicators_plus_oof_te", "ii_state_sector_indicators", "required_iii_vs_ii"),
                   ("ii_state_sector_indicators", "i_core", "required_ii_vs_i"),
                   ("iii_indicators_plus_oof_te", "i_core", "required_iii_vs_i"),
                   ("x_indicators_plus_all_te", "ii_state_sector_indicators", "exploratory"),
                   ("x_core_plus_oof_te_only", "i_core", "exploratory"),
                   ("x_core_plus_oof_te_only", "ii_state_sector_indicators", "exploratory")]
    abl = []
    for r in metrics[metrics.arm.isin(TE_ARMS) | (metrics.arm == "ii_state_sector_indicators")].itertuples(index=False):
        for arm, comp, role in comparisons:
            if r.arm != arm:
                continue
            ref = get(r.cohort, r.split, r.model, comp)
            if int(ref["n_births"]) != r.n_births:
                raise AssertionError("MATCHING: comparison on different rows")
            abl.append({"experiment": "A3", "cohort": r.cohort, "split": r.split, "model": r.model,
                        "arm": arm, "comparator": comp, "comparison_role": role if r.alpha_selected else
                        "exploratory_alpha_sensitivity", "te_alpha": r.te_alpha, "alpha_selected": r.alpha_selected,
                        "matched_n_births": r.n_births, "matched_rows_check": "PASS_identical_partition_rows",
                        "delta_roc_auc": r.roc_auc - ref["roc_auc"],
                        "delta_average_precision": r.average_precision - ref["average_precision"],
                        "delta_brier": r.brier - ref["brier"], "arm_calibration_slope": r.calibration_slope,
                        "comparator_calibration_slope": ref["calibration_slope"],
                        "paired_ci_status": "NOT_APPROVED", "result_status": "EXPLORATORY"})
    ablations = pd.DataFrame(abl)
    check("matched_rows_all_ablations", True, f"{len(ablations)} paired comparisons")

    # 3. Cross-experiment reproducibility: arms i/ii must equal A2 B0/R1 on the same split.
    a2_path = (HERE / cfg["a2_consistency_check"]).resolve()
    a2_check = {"status": "SKIPPED_A2_METRICS_ABSENT"}
    if a2_path.is_file():
        a2 = pd.read_csv(a2_path)
        a2 = a2[(a2.cohort == "primary") & (a2.context_variant == "primary") & (a2.split == "validation")
                & (a2.weighting == "unweighted")]
        diffs = []
        for model in cfg["models"]:
            for mine, theirs in (("i_core", "B0"), ("ii_state_sector_indicators", "R1")):
                x = get("primary", "validation", model, mine)
                z = a2[(a2.model == model) & (a2.arm == theirs)].iloc[0]
                diffs.append(max(abs(x.roc_auc - z.roc_auc), abs(x.average_precision - z.average_precision),
                                 abs(x.n_births - z.n_births)))
        a2_check = {"status": "PASS" if max(diffs) < 1e-12 else "FAIL", "max_abs_difference": float(max(diffs)),
                    "compared": "A3 i_core/ii vs A2 B0/R1, primary validation, both models"}
        check("a2_baseline_reproduced_on_same_split", a2_check["status"] == "PASS", str(a2_check["max_abs_difference"]))
    else:
        qa.append({"check": "a2_baseline_reproduced_on_same_split", "status": "SKIPPED", "detail": "A2 metrics absent"})
    check("test_partition_scored" if evaluate_test else "test_partition_untouched", True,
          "scored by explicit flag" if evaluate_test else "no test predictions generated")

    # 4. Outputs.
    alpha_df = pd.concat(alpha_tables, ignore_index=True)
    levels = pd.DataFrame(level_rows)
    for f in (metrics, ablations, alpha_df, levels):
        assert_disclosure_safe(f)
    metrics.to_csv(out / "metrics.csv", index=False)
    ablations.to_csv(out / "ablations.csv", index=False)
    alpha_df.to_csv(out / "alpha_selection.csv", index=False)
    levels.to_csv(out / "te_support_summary.csv", index=False)
    write_json({"experiment": "A3", "NFHS-5": flow5, "outer_split_primary": split_audit["summary"],
                "cohort_splits": cohort_splits, "inner_fold_manifest": fold_reports,
                "notes": ["Outer split identical to A2 (split_contracts.grouped_partition, seed 42).",
                          "Inner folds: StratifiedGroupKFold(3, shuffle, 42) on outer-train mothers; mother_overlap "
                          "is asserted zero for every fold.",
                          "Aggregate counts only; no assignments or identifiers."]}, out / "cohort_flow.json")
    est = {m: {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
               for k, v in make_estimator(m, seed=seed).get_params().items()} for m in cfg["models"]}
    write_json({"experiment": "A3", "status": "EXPLORATORY", "run_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": round(time.time() - t0, 1), "git": git_info(ROOT), "environment": environment(),
                "source_files": {"NFHS-5": {"basename": RAW["NFHS-5"].name, "bytes": flow5.get("raw_file_bytes")}},
                "cohort_definition": flow5["cohort_rule"], "seed": seed,
                "split_policy": {**sp, "function": "split_contracts.grouped_partition",
                                 "status": "provisional development split; not an untouched confirmatory test"},
                "inner_cv": icv, "encoding": enc_cfg,
                "selected_alpha": {c: r["selected_alpha"] for c, r in fold_reports.items()},
                "alpha_selection_rule": "min OOF log-loss of te_state_sector on outer-train rows; ties -> larger alpha",
                "fallback_order": {"te_state_sector": ["cell", "state", "sector", "global mu"],
                                   "te_state": ["state", "global mu"], "te_sector": ["sector", "global mu"]},
                "test_partition_evaluated": evaluate_test, "evaluated_splits": eval_splits,
                "arms": {a: {"numerical": list(arm_manifest(a).numerical),
                             "categorical": list(arm_manifest(a).categorical), "role": ARMS[a]["role"]} for a in ARMS},
                "sector_indicator_note": "facility_type (public/private/other) in the common core is the ordinary "
                                         "sector indicator; arm ii adds NFHS-5 state indicators.",
                "models": est, "model_hyperparameters_tuned": False,
                "a2_consistency": a2_check, "synthetic_tests": tests, "config": cfg,
                "confidence_interval_status": "NOT_APPROVED"}, out / "run_manifest.json")
    (out / "metric_schema.md").write_text(METRIC_SCHEMA, encoding="utf-8")
    verification = write_notes(out / "METHODS_RESULTS_NOTES.md", metrics, ablations, alpha_df, tests, qa, flow5,
                               split_audit, fold_reports, a2_check, cfg, evaluate_test)
    write_json({"experiment": "A3", "status": "PASS", "checks": qa}, out / "qa_status.json")
    scan = scan_output_dir(out)
    qa.append({"check": "output_disclosure_scan", "status": scan["status"], "detail": ", ".join(scan["files_checked"])})
    write_json({"experiment": "A3", "status": "PASS", "checks": qa}, out / "qa_status.json")
    print("\n" + verification)
    print(f"A3 complete in {time.time() - t0:.0f}s -> {out}")


METRIC_SCHEMA = """# A3 metric schema

`metrics.csv`: one row per (cohort, te_alpha, split, model, arm).

| Column | Meaning |
|---|---|
| experiment, wave, source_domain, target_domain | A3; NFHS-5; TE fitted on NFHS-5 outer-train labels |
| cohort | `primary` (NFHS-5 b19 0..59 institutional), `exclude_other_facility`, `nfhs5_cmc_0_59` |
| te_alpha, alpha_selected | smoothing pseudo-count for TE arms; `alpha_selected` false = exploratory alpha sensitivity |
| split | `validation` (default) or `test` (only with --evaluate-test) |
| model, arm, arm_role, feature_tier | fixed-hyperparameter estimator; matched arms i / ii / iii plus exploratory |
| n_births, n_mothers, n_events, prevalence | evaluation denominators |
| roc_auc | primary metric |
| average_precision | prevalence-dependent |
| brier, calibration_intercept, calibration_slope | probability accuracy; logistic recalibration diagnostics |
| threshold, threshold_provenance, precision, recall, f1 | max-F1 threshold chosen on validation only |
| confidence_interval_status | NOT_APPROVED |

`ablations.csv`: delta = arm minus comparator on identical rows. `alpha_selection.csv`: OOF log-loss of the
state x sector encoding on outer-train rows for each alpha. `te_support_summary.csv`: births by encoding
fallback level and a histogram of training-cell support (no cell names).
"""


def write_notes(path, metrics, ablations, alpha_df, tests, qa, flow5, split_audit, fold_reports, a2_check, cfg,
                evaluate_test) -> str:
    sel = metrics[(metrics.cohort == "primary") & (metrics.split == "validation") & metrics.alpha_selected]
    alpha = fold_reports["primary"]["selected_alpha"]
    req = ablations[(ablations.cohort == "primary") & (ablations.split == "validation") & ablations.alpha_selected]
    pr = cfg["promotion_reference"]
    t = {x["test"]: x["status"] for x in tests}
    q = {x["check"]: x["status"] for x in qa}
    v = ["## Post-execution verification", "",
         "### 1. Adversarial tests",
         f"- Synthetic A3 suite: {sum(s == 'PASS' for s in t.values())}/{len(t)} passed.",
         f"- Singleton test (`test_singleton_state_sector_does_not_output_own_label`): "
         f"**{t.get('test_singleton_state_sector_does_not_output_own_label', 'NOT RUN')}**. An artificial "
         "one-birth, one-positive state x sector cell never receives its own label (1.0) as its OOF encoding "
         "for alpha in {0, 1, 20} and n_min in {1, 30}; it falls back to the sector rate from other folds.",
         f"- Validation/test label mutation (`test_validation_test_label_mutation_bitwise_identical`): "
         f"**{t.get('test_validation_test_label_mutation_bitwise_identical', 'NOT RUN')}** (flip, permute, all-1, "
         "all-0: all encodings and the full-train table digest bitwise identical; alpha selection unchanged).",
         f"- Same check on the real NFHS-5 data at runtime: flipped "
         f"**{q.get('runtime_validation_test_labels_flipped_encodings_bitwise_identical', 'NOT RUN')}**, permuted "
         f"**{q.get('runtime_validation_test_labels_permuted_encodings_bitwise_identical', 'NOT RUN')}**.",
         f"- Other required tests: mother-group isolation **{t.get('test_mother_group_isolation_inner_folds')}**, "
         f"unseen-category fallback **{t.get('test_unseen_category_fallback')}**, training row-order reconstruction "
         f"**{t.get('test_training_row_order_reconstruction')}**.", "",
         "### 2. Arm (iii) vs Arm (ii) (primary cohort, validation, selected alpha)"]
    for model in cfg["models"]:
        r = req[(req.model == model) & (req.arm == "iii_indicators_plus_oof_te")
                & (req.comparator == "ii_state_sector_indicators")].iloc[0]
        d_auc, d_ap = r.delta_roc_auc, r.delta_average_precision
        verdict = ("performed identically to" if d_auc == 0 else
                   "outperformed" if d_auc > 0 else "underperformed")
        size = ("below" if abs(d_auc) < pr["roc_auc"] else "at or above")
        v.append(f"- **{model}:** Arm (iii) {verdict} Arm (ii) on ROC-AUC: ΔAUC = {d_auc:+.5f}, ΔAP = {d_ap:+.5f} "
                 f"(n = {int(r.matched_n_births):,} identical births). |ΔAUC| is {size} the provisional "
                 f"promotion reference of {pr['roc_auc']} (ΔAP reference {pr['average_precision']}).")
    v += ["- No interval is attached to these deltas (survey-design estimator NOT approved); they are point "
          "differences on one provisional split."]
    verification = "\n".join(v)

    lines = ["# A3 — State × sector grouped out-of-fold target encoding: Methods and Results note", "",
             "**Status: EXPLORATORY development result.** Provisional mother-grouped split; "
             + ("test partition scored by explicit flag." if evaluate_test else
                "metrics are on the *validation* partition; the test partition was not scored.")
             + " No confidence intervals or significance claims.", "",
             "## Methods", "",
             f"- **Data:** NFHS-5 {flow5['final_births']:,} institutional births, {flow5['final_mothers']:,} mothers "
             f"({flow5['cohort_rule']}). Shared provisional mother-grouped split identical to A2: "
             + ", ".join(f"{k} {s['rows']:,}" for k, s in split_audit["summary"].items()) + ".",
             "- **Encoding:** TE = (y + α·μ)/(n + α) for state × sector, state and sector, with μ the overall rate of "
             f"the rows fitting the table. Levels with fewer than {cfg['encoding']['n_min']} births are unsupported. "
             "Fallback for state × sector: cell → state → sector → μ (flagged).",
             "- **Grouped OOF:** outer-train mothers are split by StratifiedGroupKFold(3, shuffle, 42). Each fold "
             "is encoded by tables fitted on the other two folds (zero mother overlap asserted per fold); "
             "validation (and test) rows are encoded by one table fitted on 100% of outer train. Validation and "
             "test outcomes never enter any table.",
             f"- **α selection:** grid {cfg['encoding']['alpha_grid']}, minimum OOF log-loss of the state × sector "
             f"encoding on outer-train rows only (ties → larger α). Selected α = {alpha:g} (primary). "
             f"α = {cfg['encoding']['exploratory_alphas']} reported as exploratory sensitivity, not used for selection.",
             "- **Arms:** (i) eight common-core features; (ii) (i) + NFHS-5 state indicators (the sector indicator, "
             "facility_type, is already in the core); (iii) (ii) + OOF state × sector TE. Exploratory: (ii) + all "
             "three TEs and the fallback flag; (i) + TE without state indicators. Logistic regression and "
             "XGBoost with fixed `make_estimator` defaults; train-only preprocessing; max-F1 threshold on validation.",
             f"- **Reproducibility:** arms (i)/(ii) vs A2 B0/R1 on the same split: {a2_check['status']}"
             + (f" (max |difference| {a2_check['max_abs_difference']:.1e})." if "max_abs_difference" in a2_check else "."),
             "", "## Results (primary cohort, validation, selected α)", "",
             "| Model | Arm | ROC-AUC | AP | Brier | Cal. slope | Cal. intercept |", "|---|---|---|---|---|---|---|"]
    for r in sel.itertuples(index=False):
        slope = "NA" if pd.isna(r.calibration_slope) else f"{r.calibration_slope:.3f}"
        icpt = "NA" if pd.isna(r.calibration_intercept) else f"{r.calibration_intercept:.3f}"
        lines.append(f"| {r.model} | {r.arm} | {r.roc_auc:.4f} | {r.average_precision:.4f} | {r.brier:.4f} | "
                     f"{slope} | {icpt} |")
    lines += ["", f"Validation prevalence {sel['prevalence'].iloc[0]:.4f} (n={int(sel['n_births'].iloc[0]):,} births). "
              "AP depends on prevalence.", "", "## Matched deltas (validation)", "",
              "| Cohort | α | Model | Comparison | ΔAUC | ΔAP | ΔBrier |", "|---|---|---|---|---|---|---|"]
    show = ablations[(ablations.split == "validation")].sort_values(["alpha_selected", "cohort"], ascending=[False, True])
    for r in show.itertuples(index=False):
        a = "—" if pd.isna(r.te_alpha) else f"{r.te_alpha:g}"
        lines.append(f"| {r.cohort} | {a} | {r.model} | {r.arm} vs {r.comparator} | {r.delta_roc_auc:+.5f} | "
                     f"{r.delta_average_precision:+.5f} | {r.delta_brier:+.5f} |")
    lines += ["", verification, "",
              "## Limitations", "",
              "- Exploratory: one provisional split, validation partition, no intervals; old V1 test already examined.",
              "- OOF encodings for training rows come from tables fitted on two-thirds of outer train, while "
              "validation rows use the full-train table; this standard cross-fitting asymmetry can slightly change "
              "the scale the model learns.",
              "- With state indicators already in arm (ii), target encoding mainly adds the state × sector "
              "interaction; tree models can learn part of that interaction from indicators alone.",
              "- Survey clusters cross partitions; mothers do not. State is interview residence. No causal reading "
              "of state or sector; retrospective survey features, not prenatal prediction.",
              "- Per-state TE has no entry for an unseen state; A6 must use the documented fallback.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return verification


if __name__ == "__main__":
    main()
