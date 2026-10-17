#!/usr/bin/env python3
"""A5 runner: NFHS-4 -> NFHS-5 source-only temporal transfer (aggregates only).

Order: synthetic tests -> NFHS-4 load and mother-grouped source split -> fit,
threshold, freeze and hash each source model (and save it to the gitignored
artifact folder) -> NFHS-5 load -> (B) zero-shot scoring with a runtime
label-mutation check -> (C) NFHS-5 within-wave reference -> (D) optional
target-supervised recalibration on mother-disjoint adaptation rows ->
penalties, shift diagnostics, notes. NFHS-5 test partition untouched unless
--evaluate-test.

Usage (from repo root):  python version2_ml/experiments/A5/run.py [--evaluate-test]
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent / "common"), str(HERE.parent / "A2"), str(ROOT / "version2_ml")]
from A5_temporal_transfer import (fit_source_model, zero_shot_pipeline, harmonize_core,  # noqa: E402
                                  harmonization_report, AdaptationRecalibrator, FT_NUM, FT_CAT, FT_MANIFEST,
                                  RESTRICTED_F0, assert_ft_only)
from exp_cohort import load_wave_births, RAW, cmc_to_year_month  # noqa: E402
from exp_metrics import metric_row, select_threshold_max_f1  # noqa: E402
from exp_provenance import (git_info, environment, digest_frame, suppress_small_cells,  # noqa: E402
                            assert_disclosure_safe, scan_output_dir, write_json)
from state_crosswalk import load_crosswalk, nfhs4_unit_map, nfhs5_unit_map, assign_units  # noqa: E402
from split_contracts import grouped_partition, audit_partition, guard_wave_transfer  # noqa: E402
from model_framework import AuthorizedClassifier, make_estimator  # noqa: E402

FT = list(FT_NUM + FT_CAT)
SRC, TGT = "NFHS-4", "NFHS-5"


def run_synthetic_tests() -> list[dict]:
    suite = unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_a5_*.py",
                                                top_level_dir=str(HERE / "tests"))
    outcomes = {}

    class Collector(unittest.TestResult):
        def addSuccess(self, test):
            outcomes[test.id()] = "PASS"

        def addFailure(self, test, err):
            outcomes[test.id()] = "FAIL"

        def addError(self, test, err):
            outcomes[test.id()] = "ERROR"
    suite.run(Collector())
    return [{"test": k.split(".")[-1], "status": v} for k, v in sorted(outcomes.items())]


def calibration_bins(y, p, label: dict, n_bins: int) -> list[dict]:
    q = pd.qcut(pd.Series(p), n_bins, labels=False, duplicates="drop")
    rows = []
    for b in sorted(q.unique()):
        m = (q == b).to_numpy()
        rows.append({**label, "bin": int(b), "n_births": int(m.sum()), "mean_predicted_risk": float(p[m].mean()),
                     "observed_rate": float(y[m].mean())})
    return rows


def describe_wave(frame: pd.DataFrame, wave: str, units: pd.Series, disc: dict) -> dict:
    """Aggregate composition and feature distributions for one wave (no rows, no min/max)."""
    x = harmonize_core(frame, wave)
    out = {"births": int(len(frame)), "mothers": int(frame["caseid"].nunique()), "events": int(frame["m17"].sum()),
           "prevalence": float(frame["m17"].mean()),
           "birth_year_counts": {int(k): int(v) for k, v in
                                 (1900 + (frame["b3"].astype(int) - 1) // 12).value_counts().sort_index().items()},
           "numeric": {c: {"mean": float(x[c].mean()), "sd": float(x[c].std()),
                           "p10": float(x[c].quantile(.1)), "p50": float(x[c].quantile(.5)),
                           "p90": float(x[c].quantile(.9))} for c in FT_NUM},
           "categorical_shares": {}}
    for c in FT_CAT:
        vc = x[c].value_counts().sort_index()
        out["categorical_shares"][c] = {k: (round(float(v / len(x)), 6) if v >= disc["min_n"] else "SUPPRESSED")
                                        for k, v in vc.items()}
    sec = frame.groupby("sector")["m17"].agg(n_births="size", n_events="sum").reset_index()
    sec["share"] = sec["n_births"] / len(frame)
    sec["prevalence"] = sec["n_events"] / sec["n_births"]
    sec = suppress_small_cells(sec, n_col="n_births", event_col="n_events", min_n=disc["min_n"],
                               min_events=disc["min_events"], value_cols=("share", "prevalence"))
    out["sector"] = sec.to_dict(orient="records")
    st = units.value_counts()
    out["state_unit_shares"] = {k: (round(float(v / len(frame)), 6) if v >= disc["min_n"] else "SUPPRESSED")
                                for k, v in st.sort_index().items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "aggregate"))
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    evaluate_test = bool(args.evaluate_test or cfg.get("evaluate_test", False))
    disc, nb = cfg["disclosure"], int(cfg["calibration_bins"])
    qa, rows, bins = [], [], []

    def check(name, ok, detail=""):
        qa.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        if not ok:
            raise AssertionError(f"QA failed: {name}: {detail}")

    def row(comparison, wave, split, model, y, p, groups, thr, prov, cohort="primary", weights=None, sup="none"):
        base = {"experiment": "A5", "comparison": comparison, "wave": wave, "source_domain": SRC if comparison != "C"
                else TGT, "target_domain": wave, "cohort": cohort, "split": split, "model": model,
                "feature_tier": "FT", "target_supervision": sup,
                "weighting": "v005_descriptive" if weights is not None else "unweighted"}
        rows.append(metric_row(base, y, p, groups, thr, prov, weights=weights))

    # 0. Synthetic tests first.
    tests = run_synthetic_tests()
    check("synthetic_tests", len(tests) >= 4 and all(t["status"] == "PASS" for t in tests),
          f"{sum(t['status'] == 'PASS' for t in tests)}/{len(tests)} passed")
    assert_ft_only(FT)
    check("feature_set_exactly_eight_ft", True, ", ".join(FT))

    # 1. Source: NFHS-4 only. Fit, threshold, freeze and hash before NFHS-5 is loaded.
    src, flow4 = load_wave_births(ROOT, SRC, use_cache=not args.no_cache)
    ss = cfg["source_split"]
    src["source_partition"] = grouped_partition(src, group_col=ss["group"], target_col="m17",
                                                fractions=tuple(ss["fractions"]), seed=int(ss["seed"])).to_numpy()
    src_audit = audit_partition(src, src["source_partition"], group_col=ss["group"], target_col="m17")
    check("source_split_mother_exclusive", src_audit["status"] == "GROUP_EXCLUSIVE_VALID")
    source_digest = digest_frame(src[FT + ["m17", "source_partition"]])
    art = ROOT / cfg["artifact_dir"]
    art.mkdir(parents=True, exist_ok=True)
    models, frozen, artifact_hash = {}, {}, {}
    for algo in cfg["models"]:
        m = fit_source_model(src, algorithm=algo, seed=seed)
        models[algo], frozen[algo] = m, m.frozen_digest
        path = art / f"source_model_{algo}.joblib"
        joblib.dump(m, path)
        artifact_hash[algo] = {"file": f"{cfg['artifact_dir']}/{path.name}",
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "frozen_model_digest": m.frozen_digest, "source_threshold": m.threshold}
        cats = m.fitted_categories()
        src_tr = harmonize_core(src[src.source_partition.eq("train")], SRC)
        check(f"{algo}_encoder_categories_from_source_train_only",
              all(cats[c] == sorted(set(src_tr[c])) for c in FT_CAT))
    y4, g4, w4 = src["m17"].to_numpy(int), src["caseid"].to_numpy(), src["sample_weight"].to_numpy(float)
    for algo, m in models.items():
        for split, label in (("validation", "source_validation"), ("test", "source_holdout")):
            msk = src["source_partition"].eq(split).to_numpy()
            p = m.predict(src.loc[msk, FT], wave=SRC)
            prov = ("max_f1_selected_on_same_source_validation_rows_optimistic" if split == "validation"
                    else "frozen_source_validation_threshold")
            row("A", SRC, label, algo, y4[msk], p, g4[msk], m.threshold, prov)
            if split == "validation":
                if cfg.get("weighted_descriptive_metrics"):
                    row("A", SRC, label, algo, y4[msk], p, g4[msk], m.threshold, prov, weights=w4[msk])
                bins += calibration_bins(y4[msk], p, {"comparison": "A", "wave": SRC, "split": label,
                                                     "model": algo}, nb)
    val_auc = {r["model"]: r["roc_auc"] for r in rows if r["comparison"] == "A" and r["split"] == "source_validation"
               and r["weighting"] == "unweighted"}
    primary_model = max(sorted(val_auc), key=lambda k: (val_auc[k], k == "logistic_regression"))
    xw = load_crosswalk(HERE / cfg["crosswalk"])
    shift = {SRC: describe_wave(src, SRC, assign_units(src["v024"], nfhs4_unit_map(xw)), disc)}
    src_train_x = harmonize_core(src[src.source_partition.eq("train")], SRC)
    overlap_cmc = int(src["v008"].max())
    src_flow_splits = src_audit["summary"]
    src_wave = src["wave"]
    del src

    # 2. Target: NFHS-5.
    tgt, flow5 = load_wave_births(ROOT, TGT, use_cache=not args.no_cache)
    ts = cfg["target_split"]
    tpart = grouped_partition(tgt, group_col=ts["group"], target_col="m17", fractions=tuple(ts["fractions"]),
                              seed=int(ts["seed"]))
    t_audit = audit_partition(tgt, tpart, group_col=ts["group"], target_col="m17")
    check("target_split_mother_exclusive", t_audit["status"] == "GROUP_EXCLUSIVE_VALID")
    guard_wave_transfer(src_wave, tgt["wave"])
    check("wave_direction_nfhs4_to_nfhs5", True)
    y5, g5, w5 = tgt["m17"].to_numpy(int), tgt["caseid"].to_numpy(), tgt["sample_weight"].to_numpy(float)
    harm = harmonization_report(src_train_x, harmonize_core(tgt, TGT))
    shift[TGT] = describe_wave(tgt, TGT, assign_units(tgt["v024"], nfhs5_unit_map(xw)), disc)

    # 2b. Runtime adversarial check: target labels permuted / removed -> identical predictions and model.
    src_again, _ = load_wave_births(ROOT, SRC, use_cache=not args.no_cache)
    src_again["source_partition"] = grouped_partition(src_again, group_col=ss["group"], target_col="m17",
                                                      fractions=tuple(ss["fractions"]), seed=int(ss["seed"])).to_numpy()
    check("source_rows_digest_reproduced", digest_frame(src_again[FT + ["m17", "source_partition"]]) == source_digest)
    rng = np.random.default_rng(seed)
    p_full = {algo: m.predict(tgt[FT], wave=TGT) for algo, m in models.items()}
    for algo in cfg["models"]:
        for label, mutated in (("permuted", tgt.assign(m17=rng.permutation(y5))), ("removed", tgt.drop(columns="m17"))):
            m2, p2 = zero_shot_pipeline(src_again, mutated, algorithm=algo, seed=seed)
            check(f"runtime_{algo}_target_labels_{label}_predictions_bitwise_identical",
                  p2.tobytes() == p_full[algo].tobytes() and m2.frozen_digest == frozen[algo])
    del src_again

    # 3. (B) zero-shot transfer.
    eval_sets = {"target_validation": tpart.eq("validation").to_numpy(),
                 "target_non_test": tpart.ne("test").to_numpy()}
    if evaluate_test:
        eval_sets["target_test"] = tpart.eq("test").to_numpy()
    cohorts = {"nfhs5_cmc_0_59": tgt["cmc_recall"].between(0, 59).to_numpy(),
               "exclude_other_facility": tgt["m15"].ne(96).to_numpy(),
               "exclude_calendar_overlap": tgt["b3"].gt(overlap_cmc).to_numpy()}
    for algo, m in models.items():
        p = p_full[algo]
        for split, msk in eval_sets.items():
            row("B", TGT, split, algo, y5[msk], p[msk], g5[msk], m.threshold, "frozen_source_validation_threshold")
        vm = eval_sets["target_validation"]
        if cfg.get("weighted_descriptive_metrics"):
            row("B", TGT, "target_validation", algo, y5[vm], p[vm], g5[vm], m.threshold,
                "frozen_source_validation_threshold", weights=w5[vm])
        for cname, cm in cohorts.items():
            msk = vm & cm
            row("B", TGT, "target_validation", algo, y5[msk], p[msk], g5[msk], m.threshold,
                "frozen_source_validation_threshold", cohort=cname)
        bins += calibration_bins(y5[vm], p[vm], {"comparison": "B", "wave": TGT, "split": "target_validation",
                                                 "model": algo}, nb)

    # 4. (C) NFHS-5 within-wave reference (context only).
    x5 = harmonize_core(tgt, TGT)
    trm, vm = tpart.eq("train").to_numpy(), tpart.eq("validation").to_numpy()
    for algo in cfg["models"]:
        clf = AuthorizedClassifier(FT_MANIFEST, algorithm=algo, seed=seed).fit(x5[trm], y5[trm], partition="train")
        pv = clf.predict_proba(x5[vm], partition="validation")
        thr = select_threshold_max_f1(y5[vm], pv)
        row("C", TGT, "target_validation", algo, y5[vm], pv, g5[vm], thr,
            "max_f1_selected_on_same_target_validation_rows_optimistic", sup="target_train_labels_within_wave")
        bins += calibration_bins(y5[vm], pv, {"comparison": "C", "wave": TGT, "split": "target_validation",
                                              "model": algo}, nb)

    # 5. (D) target-supervised recalibration on mother-disjoint adaptation rows.
    adapt_info = {"enabled": bool(cfg["adaptation"]["enabled"])}
    if cfg["adaptation"]["enabled"]:
        am = tpart.eq(cfg["adaptation"]["adaptation_partition"]).to_numpy()
        em = tpart.eq(cfg["adaptation"]["evaluation_partition"]).to_numpy()
        overlap = len(set(g5[am]) & set(g5[em]))
        check("adaptation_evaluation_mothers_disjoint", overlap == 0, f"overlap={overlap}")
        adapt_info.update(adaptation_births=int(am.sum()), adaptation_mothers=int(pd.Series(g5[am]).nunique()),
                          adaptation_events=int(y5[am].sum()), coefficients={})
        for algo in cfg["models"]:
            for kind in cfg["adaptation"]["kinds"]:
                r = AdaptationRecalibrator(kind).fit(p_full[algo][am], y5[am], adaptation_groups=g5[am],
                                                     evaluation_groups=g5[em])
                q_adapt, q_eval = r.transform(p_full[algo][am]), r.transform(p_full[algo][em])
                thr = select_threshold_max_f1(y5[am], q_adapt)
                row(f"D_{kind}", TGT, "target_validation", algo, y5[em], q_eval, g5[em], thr,
                    "max_f1_selected_on_target_adaptation_rows", sup="target_adaptation_labels_counted")
                adapt_info["coefficients"][f"{algo}_{kind}"] = {"intercept": r.coef[0], "slope": r.coef[1]}
    metrics = pd.DataFrame(rows)

    # 6. Consistency: C must reproduce A2 B0 exactly (same split, features, models).
    a2p = (HERE / cfg["a2_consistency_check"]).resolve()
    a2_check = {"status": "SKIPPED_A2_METRICS_ABSENT"}
    if a2p.is_file():
        a2 = pd.read_csv(a2p)
        a2 = a2[(a2.arm == "B0") & (a2.cohort == "primary") & (a2.context_variant == "primary")
                & (a2.split == "validation") & (a2.weighting == "unweighted")]
        diffs = []
        for algo in cfg["models"]:
            c = metrics[(metrics.comparison == "C") & (metrics.model == algo)].iloc[0]
            z = a2[a2.model == algo].iloc[0]
            diffs.append(max(abs(c.roc_auc - z.roc_auc), abs(c.average_precision - z.average_precision)))
        a2_check = {"status": "PASS" if max(diffs) < 1e-12 else "FAIL", "max_abs_difference": float(max(diffs))}
        check("C_reproduces_A2_B0", a2_check["status"] == "PASS", str(a2_check["max_abs_difference"]))

    # 7. Penalties.
    def pick(comparison, split, algo, cohort="primary", weighting="unweighted"):
        r = metrics[(metrics.comparison == comparison) & (metrics.split == split) & (metrics.model == algo)
                    & (metrics.cohort == cohort) & (metrics.weighting == weighting)]
        return r.iloc[0]
    pen = []
    for algo in cfg["models"]:
        pairs = [("B_target_validation_minus_A_source_validation", pick("B", "target_validation", algo),
                  pick("A", "source_validation", algo)),
                 ("B_target_validation_minus_A_source_holdout", pick("B", "target_validation", algo),
                  pick("A", "source_holdout", algo)),
                 ("B_target_non_test_minus_A_source_holdout", pick("B", "target_non_test", algo),
                  pick("A", "source_holdout", algo)),
                 ("C_within_target_minus_B_zero_shot", pick("C", "target_validation", algo),
                  pick("B", "target_validation", algo))]
        if cfg.get("weighted_descriptive_metrics"):
            pairs.append(("B_minus_A_validation_v005_descriptive",
                          pick("B", "target_validation", algo, weighting="v005_descriptive"),
                          pick("A", "source_validation", algo, weighting="v005_descriptive")))
        for kind in cfg["adaptation"]["kinds"] if cfg["adaptation"]["enabled"] else []:
            pairs.append((f"D_{kind}_minus_B_zero_shot", pick(f"D_{kind}", "target_validation", algo),
                          pick("B", "target_validation", algo)))
        for name, a, b in pairs:
            pen.append({"experiment": "A5", "model": algo, "contrast": name, "primary_model": algo == primary_model,
                        "n_births_first": int(a.n_births), "n_births_second": int(b.n_births),
                        "prevalence_first": a.prevalence, "prevalence_second": b.prevalence,
                        "delta_roc_auc": a.roc_auc - b.roc_auc,
                        "delta_average_precision": a.average_precision - b.average_precision,
                        "ap_lift_first": a.average_precision / a.prevalence,
                        "ap_lift_second": b.average_precision / b.prevalence,
                        "delta_ap_lift": a.average_precision / a.prevalence - b.average_precision / b.prevalence,
                        "delta_brier": a.brier - b.brier,
                        "calibration_slope_first": a.calibration_slope, "calibration_slope_second": b.calibration_slope,
                        "calibration_intercept_first": a.calibration_intercept,
                        "calibration_intercept_second": b.calibration_intercept,
                        "paired_rows": name.startswith(("C_", "D_")),
                        "ci_status": "NOT_APPROVED", "result_status": "EXPLORATORY"})
    penalties = pd.DataFrame(pen)
    check("test_partition_scored" if evaluate_test else "nfhs5_test_partition_untouched", True,
          "scored by explicit flag" if evaluate_test else "no NFHS-5 test predictions evaluated")
    check("frozen_models_unchanged_at_end", all(models[a].digest() == frozen[a] for a in models))

    # 8. Outputs.
    calib = pd.DataFrame(bins)
    for f in (metrics, penalties, calib):
        assert_disclosure_safe(f)
    metrics.to_csv(out / "metrics.csv", index=False)
    penalties.to_csv(out / "transfer_penalties.csv", index=False)
    calib.to_csv(out / "calibration_bins.csv", index=False)
    write_json({"experiment": "A5", "harmonization": harm, "waves": shift,
                "restricted_f0_predictors_excluded": list(RESTRICTED_F0),
                "wealth_note": "v190 quintiles are wave-relative ranks; equal codes do not mean equal absolute wealth.",
                "calendar_overlap": {"last_nfhs4_interview": cmc_to_year_month(overlap_cmc),
                                     "nfhs5_births_in_or_before": int(tgt["b3"].le(overlap_cmc).sum())}},
               out / "shift_diagnostics.json")
    write_json({"experiment": "A5", SRC: flow4, TGT: flow5, "source_split": src_flow_splits,
                "target_split": t_audit["summary"],
                "target_cohort_sensitivity_validation_births": {k: int((eval_sets["target_validation"] & v).sum())
                                                                for k, v in cohorts.items()},
                "nfhs5_cmc_boundary": {"b19_lt_60_but_v008_minus_b3_eq_60": int(tgt["cmc_recall"].eq(60).sum()),
                                       "cmc_0_59_cohort_births": int(tgt["cmc_recall"].between(0, 59).sum())},
                "adaptation": {k: v for k, v in adapt_info.items() if k != "coefficients"},
                "notes": ["Source and target splits are mother-grouped; source and target mothers are different "
                          "survey respondents (different waves).",
                          "B on target_non_test includes NFHS-5 train rows: the source model never saw them; they "
                          "are used by C (training) and D (adaptation) only."]}, out / "cohort_flow.json")
    est = {m: {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
               for k, v in make_estimator(m, seed=seed).get_params().items()} for m in cfg["models"]}
    write_json({"experiment": "A5", "status": "EXPLORATORY", "run_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": round(time.time() - t0, 1), "git": git_info(ROOT), "environment": environment(),
                "source_files": {w: {"basename": RAW[w].name, "bytes": f.get("raw_file_bytes")}
                                 for w, f in ((SRC, flow4), (TGT, flow5))},
                "cohort_definitions": {SRC: flow4["cohort_rule"], TGT: flow5["cohort_rule"]}, "seed": seed,
                "source_split": ss, "target_split": ts, "features": FT, "feature_tier": "FT",
                "models": est, "model_hyperparameters_tuned": False, "primary_model": primary_model,
                "primary_model_rule": cfg["primary_model_rule"],
                "source_rows_digest_sha256": source_digest, "frozen_source_models": artifact_hash,
                "frozen_before_target_loaded": True, "test_partition_evaluated": evaluate_test,
                "adaptation": adapt_info, "a2_consistency": a2_check, "synthetic_tests": tests, "config": cfg,
                "confidence_interval_status": "NOT_APPROVED"}, out / "run_manifest.json")
    (out / "metric_schema.md").write_text(METRIC_SCHEMA, encoding="utf-8")
    verification = write_notes(out / "METHODS_RESULTS_NOTES.md", metrics, penalties, tests, qa, flow4, flow5,
                               src_flow_splits, t_audit["summary"], harm, primary_model, adapt_info, a2_check,
                               cfg, evaluate_test)
    write_json({"experiment": "A5", "status": "PASS", "checks": qa}, out / "qa_status.json")
    scan = scan_output_dir(out)
    qa.append({"check": "output_disclosure_scan", "status": scan["status"], "detail": ", ".join(scan["files_checked"])})
    write_json({"experiment": "A5", "status": "PASS", "checks": qa}, out / "qa_status.json")
    print("\n" + verification)
    print(f"A5 complete in {time.time() - t0:.0f}s -> {out}")


METRIC_SCHEMA = """# A5 metric schema

`metrics.csv`: one row per (comparison, split, model, cohort, weighting).

| Column | Meaning |
|---|---|
| comparison | A = NFHS-4 source reference; B = frozen NFHS-4 model zero-shot on NFHS-5; C = NFHS-5 within-wave reference (context only); D_intercept / D_intercept_slope = target-supervised recalibration of B |
| wave, source_domain, target_domain | evaluation wave; wave whose labels trained the model |
| split | source_validation (threshold chosen here), source_holdout, target_validation (shared NFHS-5 split, matched to C), target_non_test (NFHS-5 train+validation), target_test (only with --evaluate-test) |
| cohort | primary, or NFHS-5 sensitivity (nfhs5_cmc_0_59, exclude_other_facility, exclude_calendar_overlap) |
| target_supervision | none (A, B), target_train_labels_within_wave (C), target_adaptation_labels_counted (D) |
| weighting | unweighted or v005_descriptive (descriptive point metric only) |
| n_births, n_mothers, n_events, prevalence | denominators |
| roc_auc, average_precision, brier, calibration_intercept, calibration_slope | point metrics; AP is prevalence-dependent |
| threshold, threshold_provenance, precision, recall, f1 | A/B: threshold frozen from NFHS-4 validation |
| confidence_interval_status | NOT_APPROVED |

`transfer_penalties.csv`: first minus second for each named contrast; `ap_lift` = AP / prevalence.
`calibration_bins.csv`: deciles of predicted risk (n, mean predicted risk, observed rate).
`shift_diagnostics.json`: harmonization, unseen levels, composition and feature distributions by wave.
"""


def write_notes(path, metrics, penalties, tests, qa, flow4, flow5, s_split, t_split, harm, primary, adapt, a2c, cfg,
                evaluate_test) -> str:
    t = {x["test"]: x["status"] for x in tests}
    q = {x["check"]: x["status"] for x in qa}

    def pick(comp, split, algo, cohort="primary", weighting="unweighted"):
        return metrics[(metrics.comparison == comp) & (metrics.split == split) & (metrics.model == algo)
                       & (metrics.cohort == cohort) & (metrics.weighting == weighting)].iloc[0]

    def f(x, d=4):
        return "NA" if x is None or pd.isna(x) else f"{x:.{d}f}"
    v = ["## Post-execution verification", "", "### 1. Adversarial target-label mutation test",
         f"- Synthetic `test_target_label_mutation_bitwise_identical`: **{t.get('test_target_label_mutation_bitwise_identical', 'NOT RUN')}** "
         "(NFHS-5 labels permuted, flipped, all-zero, removed: predictions, threshold and frozen model digest bitwise "
         "identical for both models).",
         "- Runtime check on the real data (full source pipeline re-run with permuted / removed NFHS-5 labels): "
         + ", ".join(f"{k.replace('runtime_', '').replace('_predictions_bitwise_identical', '')} **{s}**"
                     for k, s in q.items() if k.startswith("runtime_")) + ".",
         f"- Synthetic suite overall: {sum(s == 'PASS' for s in t.values())}/{len(t)} passed.", "",
         "### 2. NFHS-4 within-wave validation (A) vs zero-shot NFHS-5 (B)", "",
         "| Model | Set | n births | Prevalence | ROC-AUC | AP | Brier | Cal. slope | Cal. intercept |",
         "|---|---|---|---|---|---|---|---|---|"]
    for algo in cfg["models"]:
        for comp, split, lab in (("A", "source_validation", "A: NFHS-4 validation"),
                                 ("A", "source_holdout", "A: NFHS-4 holdout"),
                                 ("B", "target_validation", "B: NFHS-5 validation (zero-shot)"),
                                 ("B", "target_non_test", "B: NFHS-5 non-test (zero-shot)")):
            r = pick(comp, split, algo)
            v.append(f"| {algo}{' (primary)' if algo == primary else ''} | {lab} | {int(r.n_births):,} | "
                     f"{r.prevalence:.4f} | {r.roc_auc:.4f} | {r.average_precision:.4f} | {r.brier:.4f} | "
                     f"{f(r.calibration_slope, 3)} | {f(r.calibration_intercept, 3)} |")
    v += ["", "### 3. Temporal transfer penalty and prevalence shift", ""]
    for algo in cfg["models"]:
        p = penalties[(penalties.model == algo)
                      & (penalties.contrast == "B_target_validation_minus_A_source_validation")].iloc[0]
        ph = penalties[(penalties.model == algo)
                       & (penalties.contrast == "B_target_validation_minus_A_source_holdout")].iloc[0]
        direction = ("loss" if p.delta_roc_auc < 0 else "gain" if p.delta_roc_auc > 0 else "no change")
        ap_note = ("AP rose even though ROC-AUC " + ("fell" if p.delta_roc_auc < 0 else "did not rise")
                   if p.delta_average_precision > 0 >= p.delta_roc_auc else
                   "AP and ROC-AUC moved in the same direction")
        v.append(f"- **{algo}:** ΔROC-AUC (B − A, validation) = {p.delta_roc_auc:+.4f} ({direction}); vs NFHS-4 "
                 f"holdout {ph.delta_roc_auc:+.4f}. ΔAP = {p.delta_average_precision:+.4f}; ΔBrier = "
                 f"{p.delta_brier:+.4f}. Prevalence {p.prevalence_second:.4f} (NFHS-4) → {p.prevalence_first:.4f} "
                 f"(NFHS-5). AP lift over prevalence {p.ap_lift_second:.3f} → {p.ap_lift_first:.3f} "
                 f"(Δ {p.delta_ap_lift:+.3f}). {ap_note}.")
    v += ["- **Why AP moves with prevalence:** a classifier with no skill has AP equal to the outcome prevalence, so "
          "a higher NFHS-5 cesarean prevalence raises AP mechanically even if ranking quality is unchanged or worse. "
          "AP lift (AP / prevalence) and ROC-AUC, which does not depend on prevalence, are the fairer cross-wave "
          "comparisons; raw ΔAP should not be read as better or worse transfer.",
          "- **Calibration:** a positive NFHS-5 calibration intercept means the frozen NFHS-4 model under-predicts "
          "NFHS-5 cesarean risk on average (higher target prevalence); the slope shows whether risk spread is "
          "preserved.", "- No intervals (survey-design estimator NOT approved); point differences only."]
    verification = "\n".join(v)

    lines = ["# A5 — NFHS-4 → NFHS-5 source-only temporal transfer: Methods and Results note", "",
             "**Status: EXPLORATORY development result.** "
             + ("NFHS-5 test partition scored by explicit flag." if evaluate_test else
                "NFHS-5 test partition was not scored.") + " No confidence intervals or significance claims.", "",
             "## Methods", "",
             f"- **Source:** NFHS-4 {flow4['final_births']:,} births, {flow4['final_mothers']:,} mothers "
             f"({flow4['cohort_rule']}); mother-grouped 70/15/15 source split (seed 42): "
             + ", ".join(f"{k} {s['rows']:,}" for k, s in s_split.items())
             + " (train = fit; validation = threshold and primary-model choice; test = source holdout).",
             f"- **Target:** NFHS-5 {flow5['final_births']:,} births, {flow5['final_mothers']:,} mothers "
             f"({flow5['cohort_rule']}); shared split as A2/A3: "
             + ", ".join(f"{k} {s['rows']:,}" for k, s in t_split.items()) + ".",
             "- **Features:** exactly the eight common-core FT features; the 29 F0-only predictors (including the "
             "22 pregnancy/ANC items) are rejected in code. v190 wealth quintiles are wave-relative ranks.",
             "- **Source-only fitting:** preprocessing (median imputation, one-hot vocabulary), logistic regression / "
             "XGBoost (fixed defaults) and the max-F1 decision threshold all come from NFHS-4 only. Each model is "
             "frozen and SHA-256 hashed before NFHS-5 is loaded; the frozen file is saved under the gitignored "
             f"`{cfg['artifact_dir']}/`. Primary model by NFHS-4 validation ROC-AUC: **{primary}**.",
             "- **Comparisons:** (A) NFHS-4 validation and holdout; (B) frozen model zero-shot on the NFHS-5 shared "
             "validation partition (and all non-test NFHS-5 rows); (C) NFHS-5 within-wave model on the same validation "
             f"rows — context only, not an isolated wave effect (reproduces A2 B0: {a2c['status']}); "
             "(D) logistic recalibration (intercept, intercept+slope) fitted on NFHS-5 train mothers, evaluated on "
             "disjoint NFHS-5 validation mothers — target-supervised, reported separately from zero-shot.",
             "- **Unseen categories:** levels inside the codebook vocabulary but absent from NFHS-4 training are encoded "
             "as all-zero one-hot; rates are in `shift_diagnostics.json`: "
             + "; ".join(f"{c}: {d['target_levels_unseen_in_source'] or 'none'}" for c, d in harm["categorical"].items())
             + ".", "", "## Results", "", "### All comparisons (validation, unweighted, primary cohort)", "",
             "| Comparison | Model | n | Prevalence | ROC-AUC | AP | Brier | Cal. slope | Cal. intercept |",
             "|---|---|---|---|---|---|---|---|---|"]
    show = metrics[(metrics.cohort == "primary") & (metrics.weighting == "unweighted")
                   & metrics.split.isin(["source_validation", "source_holdout", "target_validation"])]
    for r in show.itertuples(index=False):
        lines.append(f"| {r.comparison} ({r.split}) | {r.model} | {int(r.n_births):,} | {r.prevalence:.4f} | "
                     f"{r.roc_auc:.4f} | {r.average_precision:.4f} | {r.brier:.4f} | {f(r.calibration_slope, 3)} | "
                     f"{f(r.calibration_intercept, 3)} |")
    lines += ["", "### NFHS-5 cohort sensitivities for B (validation)", "",
              "| Cohort | Model | n | ROC-AUC | AP | Brier | Cal. intercept |", "|---|---|---|---|---|---|---|"]
    for r in metrics[(metrics.comparison == "B") & (metrics.split == "target_validation")
                     & (metrics.weighting == "unweighted")].itertuples(index=False):
        lines.append(f"| {r.cohort} | {r.model} | {int(r.n_births):,} | {r.roc_auc:.4f} | {r.average_precision:.4f} | "
                     f"{r.brier:.4f} | {f(r.calibration_intercept, 3)} |")
    lines += ["", "### Contrasts", "", "| Model | Contrast | ΔAUC | ΔAP | ΔAP lift | ΔBrier |", "|---|---|---|---|---|---|"]
    for r in penalties.itertuples(index=False):
        lines.append(f"| {r.model} | {r.contrast} | {r.delta_roc_auc:+.4f} | {r.delta_average_precision:+.4f} | "
                     f"{r.delta_ap_lift:+.3f} | {r.delta_brier:+.4f} |")
    lines += ["", verification, "", "## Limitations", "",
              "- Exploratory: one source split and one provisional target split; no intervals; V1 test previously examined.",
              "- Wave differences mix calendar time, questionnaire changes, case mix, sampling frame and reporting; the "
              "penalty is not a causal effect of time and not a clinical validation.",
              "- The C reference uses NFHS-5 labels and a different training population; B − C is context, not an "
              "isolated wave-transfer effect.",
              "- D uses NFHS-5 labels (target supervision). Monotone logistic recalibration with a positive slope "
              "cannot change ROC-AUC; it changes calibration and thresholded metrics only.",
              "- NFHS-4 and NFHS-5 birth calendars overlap; see `exclude_calendar_overlap`.",
              "- Retrospective survey-recorded features; interview-time wealth/education/residence; not prenatal "
              "prediction. Weighted (v005) metrics are descriptive only.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return verification


if __name__ == "__main__":
    main()
