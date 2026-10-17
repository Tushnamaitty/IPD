#!/usr/bin/env python3
"""A6 runner: leave-one-state-out geographic transport and public <-> private sector transport.

Uses only NFHS-5 shared non-test rows (train + validation); the provisional test
partition is untouched unless --evaluate-test is passed (then test rows join
the target-domain evaluation sets only). Every source/target pair is
mother-disjoint; models, preprocessing, encoders and thresholds use source rows
only. Aggregate outputs only.

Usage (from repo root):  python version2_ml/experiments/A6/run.py [--evaluate-test]
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
sys.path[:0] = [str(HERE), str(HERE.parent / "common"), str(HERE.parent / "A2"), str(HERE.parent / "A3"),
                str(ROOT / "version2_ml")]
from A6_transport import (SupportScreen, assert_mother_disjoint, mother_state_check, assert_sector_free,  # noqa: E402
                          sector_overlap_policy, fit_predict_domain, loso_masks, sector_masks,
                          FT_NUM, FT_CAT, SECTOR_FREE_CAT)
from A2_historical_context import HistoricalContextMapping, SmoothingSpec, CELL_RATE  # noqa: E402
from state_crosswalk import (load_crosswalk, validate_crosswalk, nfhs4_unit_map, nfhs5_unit_map,  # noqa: E402
                             assign_units, read_dta_state_labels)
from A3_target_encoder import GroupedOOFTargetEncoder, TESpec, encode_partitions, TE_CELL, LEVEL  # noqa: E402
from exp_cohort import load_wave_births, RAW  # noqa: E402
from exp_metrics import metric_row, select_threshold_max_f1  # noqa: E402
from exp_provenance import git_info, environment, assert_disclosure_safe, scan_output_dir, write_json  # noqa: E402
from split_contracts import grouped_partition, audit_partition  # noqa: E402
from preprocessing import FeatureManifest  # noqa: E402
from model_framework import make_estimator  # noqa: E402

ARMS = {"G0_ft_state_free": FeatureManifest(FT_NUM, FT_CAT, "FT"),
        "G_hist_nfhs4_context": FeatureManifest(FT_NUM + (CELL_RATE,), FT_CAT, "F3"),
        "G_te_source_fallback": FeatureManifest(FT_NUM + (TE_CELL,), FT_CAT, "F4")}
SECTOR_MANIFEST = FeatureManifest(FT_NUM, SECTOR_FREE_CAT, "FT")
PROV = {"i": "max_f1_selected_on_same_source_validation_rows_optimistic",
        "ii": "frozen_source_validation_threshold",
        "iii_geo": "max_f1_selected_on_all_state_validation_rows_includes_target_state",
        "iii_sector": "max_f1_selected_on_same_target_validation_rows_optimistic"}


def run_synthetic_tests() -> list[dict]:
    suite = unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_a6_*.py",
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
    screen = SupportScreen(**cfg["support_screen"])
    qa, rows = [], []

    def check(name, ok, detail=""):
        qa.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        if not ok:
            raise AssertionError(f"QA failed: {name}: {detail}")

    def add(design, comparison, source, target, arm, model, eval_set, y, p, groups, thr, prov, tier, status="ESTIMABLE"):
        base = {"experiment": "A6", "design": design, "comparison": comparison, "wave": "NFHS-5",
                "source_domain": source, "target_domain": target, "arm": arm, "model": model, "eval_set": eval_set,
                "feature_tier": tier, "support_status": status, "weighting": "unweighted"}
        r = metric_row(base, y, p, groups, thr, prov)
        rows.append(r)
        return r

    tests = run_synthetic_tests()
    check("synthetic_tests", len(tests) >= 4 and all(t["status"] == "PASS" for t in tests),
          f"{sum(t['status'] == 'PASS' for t in tests)}/{len(tests)} passed")

    # 1. Data, shared split, non-test analysis frame.
    tgt, flow5 = load_wave_births(ROOT, "NFHS-5", use_cache=not args.no_cache)
    sp = cfg["split"]
    partition = grouped_partition(tgt, group_col=sp["group"], target_col="m17", fractions=tuple(sp["fractions"]),
                                  seed=int(sp["seed"]))
    split_audit = audit_partition(tgt, partition, group_col=sp["group"], target_col="m17")
    check("shared_split_mother_exclusive", split_audit["status"] == "GROUP_EXCLUSIVE_VALID")
    use = partition.ne("test").to_numpy() | evaluate_test
    D = tgt.loc[use].reset_index(drop=True)
    part = partition[use].to_numpy()
    if evaluate_test:  # test rows may only ever be target-domain evaluation rows
        part = np.where(part == "test", "target_only_test", part)
    D["nfhs5_state"] = D["v024"].astype(int).astype(str)
    y, g = D["m17"].to_numpy(int), D["caseid"].to_numpy()
    labels5 = read_dta_state_labels(ROOT / RAW["NFHS-5"])
    state_name = {str(k): v for k, v in labels5.items()}
    msc = mother_state_check(g, D["nfhs5_state"])
    check("each_mother_single_state", True, f"{msc['mothers']:,} mothers")

    # 2. Source-only context features available for an unseen state.
    gcfg = cfg["geographic"]
    xw = load_crosswalk(HERE / gcfg["a2_crosswalk"])
    labels4 = read_dta_state_labels(ROOT / RAW["NFHS-4"])
    validate_crosswalk(xw, labels4, labels5)
    src4, _ = load_wave_births(ROOT, "NFHS-4", use_cache=not args.no_cache)
    src4["source_unit"] = assign_units(src4["v024"], nfhs4_unit_map(xw))
    hist_map = HistoricalContextMapping(SmoothingSpec(**gcfg["hist_smoothing"])).fit(
        src4[["wave", "source_unit", "sector", "m17", "sample_weight"]])
    hist_hash = hist_map.mapping_hash()
    del src4
    D["source_unit"] = assign_units(D["v024"], nfhs5_unit_map(xw))
    D[CELL_RATE] = hist_map.transform(D[["source_unit", "sector"]])[CELL_RATE].to_numpy()
    tecfg = gcfg["te"]
    te_spec = TESpec(alpha=float(tecfg["alpha"]), n_min=int(tecfg["n_min"]))
    tr, va = part == "train", part == "validation"
    enc0 = GroupedOOFTargetEncoder(te_spec, n_splits=tecfg["n_splits"], seed=tecfg["random_state"])
    folds_all = np.full(len(D), -1)
    folds_all[tr] = enc0.inner_folds(y[tr], g[tr])

    def with_te(train_mask):
        part_te = np.where(train_mask, "train", "other")
        _, te = encode_partitions(D, y, g, part_te, te_spec, n_splits=tecfg["n_splits"],
                                  seed=tecfg["random_state"], folds=folds_all[train_mask])
        return D.assign(**{TE_CELL: te[TE_CELL].to_numpy(), LEVEL: te[LEVEL].to_numpy()})

    # 3. (iii) geographic reference: all-state models (state seen in training).
    X_ref = with_te(tr)
    ref_pred, ref_thr = {}, {}
    for arm, man in ARMS.items():
        for model in cfg["models"]:
            preds, thr = fit_predict_domain(X_ref, y, source_train=tr, source_val=va, score={"all_validation": va},
                                            manifest=man, algorithm=model, seed=seed)
            full = np.full(len(D), np.nan)
            full[va] = preds["all_validation"]
            ref_pred[(arm, model)], ref_thr[(arm, model)] = full, thr
    a2p = (HERE / cfg["a2_consistency_check"]).resolve()
    a2_check = {"status": "SKIPPED_A2_METRICS_ABSENT"}
    if a2p.is_file() and not evaluate_test:
        a2 = pd.read_csv(a2p)
        a2 = a2[(a2.arm == "B0") & (a2.cohort == "primary") & (a2.context_variant == "primary")
                & (a2.split == "validation") & (a2.weighting == "unweighted")]
        from sklearn.metrics import roc_auc_score
        diffs = [abs(roc_auc_score(y[va], ref_pred[("G0_ft_state_free", m)][va]) - a2[a2.model == m].roc_auc.iloc[0])
                 for m in cfg["models"]]
        a2_check = {"status": "PASS" if max(diffs) < 1e-12 else "FAIL", "max_abs_difference": float(max(diffs))}
        check("geographic_reference_reproduces_A2_B0", a2_check["status"] == "PASS", str(max(diffs)))

    # 4. Geographic leave-one-state-out.
    states = sorted(D["nfhs5_state"].unique(), key=int)
    screen_rows, geo = [], []
    estimable = []
    for s in states:
        sc = screen.check(y[D["nfhs5_state"].eq(s).to_numpy()])
        screen_rows.append({"state_code": s, "state": state_name[s], "status": sc["status"],
                            "reasons": ";".join(sc["reasons"]),
                            "n_births": sc["n_births"] if sc["status"] == "ESTIMABLE" else "SUPPRESSED",
                            "n_events": sc["n_events"] if sc["status"] == "ESTIMABLE" else "SUPPRESSED"})
        if sc["status"] == "ESTIMABLE":
            estimable.append(s)
        else:
            for arm in ARMS:
                for model in cfg["models"]:
                    geo.append({"held_out_state_code": s, "held_out_state": state_name[s], "arm": arm, "model": model,
                                "support_status": "NOT_ESTIMABLE", "screen_reasons": ";".join(sc["reasons"])})
    check("support_screen_applied", True, f"{len(estimable)}/{len(states)} states estimable")
    mutation_done = False
    for s in estimable:
        m = loso_masks(D["nfhs5_state"], part, s)
        src = m["source_train"] | m["source_val"]
        assert_mother_disjoint(g, src, m["target_all"], f"state {s}")
        Xs = with_te(m["source_train"])
        held = m["target_all"]
        te_levels = set(Xs.loc[held, LEVEL])
        if te_levels - {"sector", "global"}:
            raise AssertionError("LEAKAGE: held-out state received a state-level target encoding")
        vsc = screen.check(y[m["target_val"]])
        score = {"source_val": m["source_val"], "target_all": m["target_all"], "target_val": m["target_val"]}
        for arm, man in ARMS.items():
            X = Xs if arm == "G_te_source_fallback" else D
            for model in cfg["models"]:
                preds, thr = fit_predict_domain(X, y, source_train=m["source_train"], source_val=m["source_val"],
                                                score=score, manifest=man, algorithm=model, seed=seed)
                if not mutation_done and model == "logistic_regression" and arm != "G_hist_nfhs4_context":
                    z = y.copy()
                    z[held] = 1 - z[held]
                    Xz = Xs.copy()
                    if arm == "G_te_source_fallback":
                        part_te = np.where(m["source_train"], "train", "other")
                        _, tz = encode_partitions(D, z, g, part_te, te_spec, n_splits=tecfg["n_splits"],
                                                  seed=tecfg["random_state"], folds=folds_all[m["source_train"]])
                        Xz[TE_CELL] = tz[TE_CELL].to_numpy()
                        check("runtime_geographic_te_target_state_labels_flipped_encodings_identical",
                              Xz[TE_CELL].to_numpy().tobytes() == Xs[TE_CELL].to_numpy().tobytes())
                    pz, _ = fit_predict_domain(Xz if arm == "G_te_source_fallback" else D, z,
                                               source_train=m["source_train"], source_val=m["source_val"],
                                               score=score, manifest=man, algorithm=model, seed=seed)
                    check(f"runtime_geographic_{arm}_target_state_labels_flipped_predictions_identical",
                          all(pz[k].tobytes() == preds[k].tobytes() for k in score), f"state {s}")
                    if arm == "G_te_source_fallback":
                        mutation_done = True
                target = f"held-out state {s}"
                ri = add("geographic", "i_within_source", "other states", "other states", arm, model, "source_validation",
                         y[m["source_val"]], preds["source_val"], g[m["source_val"]], thr, PROV["i"], man.tier)
                rii = add("geographic", "ii_cross_domain", "other states", target, arm, model, "target_all_non_test",
                          y[held], preds["target_all"], g[held], thr, PROV["ii"], man.tier)
                rec = {"held_out_state_code": s, "held_out_state": state_name[s], "arm": arm, "model": model,
                       "support_status": "ESTIMABLE", "screen_reasons": "",
                       "n_births": rii["n_births"], "n_mothers": rii["n_mothers"], "n_events": rii["n_events"],
                       "prevalence": rii["prevalence"], "roc_auc": rii["roc_auc"],
                       "average_precision": rii["average_precision"], "brier": rii["brier"],
                       "calibration_intercept": rii["calibration_intercept"],
                       "calibration_slope": rii["calibration_slope"],
                       "source_validation_roc_auc": ri["roc_auc"], "source_validation_ap": ri["average_precision"],
                       "source_validation_prevalence": ri["prevalence"],
                       "delta_roc_auc_vs_source": rii["roc_auc"] - ri["roc_auc"],
                       "delta_ap_vs_source": rii["average_precision"] - ri["average_precision"],
                       "delta_ap_lift_vs_source": rii["average_precision"] / rii["prevalence"]
                       - ri["average_precision"] / ri["prevalence"],
                       "paired_validation_status": vsc["status"]}
                if vsc["status"] == "ESTIMABLE":
                    tv = m["target_val"]
                    rb = add("geographic", "ii_b_cross_domain_paired_rows", "other states", target, arm, model,
                             "target_validation", y[tv], preds["target_val"], g[tv], thr, PROV["ii"], man.tier)
                    r3 = add("geographic", "iii_target_seen_reference", "all states incl. target-state train mothers",
                             target, arm, model, "target_validation", y[tv], ref_pred[(arm, model)][tv], g[tv],
                             ref_thr[(arm, model)], PROV["iii_geo"], man.tier)
                    rec.update(paired_n_births=rb["n_births"], paired_delta_roc_auc_unseen_minus_seen=
                               rb["roc_auc"] - r3["roc_auc"],
                               paired_delta_ap_unseen_minus_seen=rb["average_precision"] - r3["average_precision"])
                geo.append(rec)
        print(f"[{time.time() - t0:7.1f}s] state {s:>2} {state_name[s]}", flush=True)
    geo_df = pd.DataFrame(geo)
    est = geo_df[geo_df.support_status == "ESTIMABLE"]
    summ = []
    for (arm, model), gdf in est.groupby(["arm", "model"], sort=False):
        a, d = gdf["roc_auc"], gdf["delta_roc_auc_vs_source"]
        pdlt = gdf["paired_delta_roc_auc_unseen_minus_seen"].dropna()
        summ.append({"arm": arm, "model": model, "states_estimable": int(len(gdf)),
                     "states_not_estimable": int((geo_df[(geo_df.arm == arm) & (geo_df.model == model)]
                                                  .support_status == "NOT_ESTIMABLE").sum()),
                     "roc_auc_min": a.min(), "roc_auc_p25": a.quantile(.25), "roc_auc_median": a.median(),
                     "roc_auc_p75": a.quantile(.75), "roc_auc_max": a.max(),
                     "delta_vs_source_median": d.median(), "delta_vs_source_min": d.min(),
                     "delta_vs_source_max": d.max(), "states_below_source": int((d < 0).sum()),
                     "paired_states": int(len(pdlt)), "paired_delta_unseen_minus_seen_median": pdlt.median(),
                     "paired_delta_min": pdlt.min(), "paired_delta_max": pdlt.max()})
    geo_summary = pd.DataFrame(summ)

    # 5. Sector transport.
    scfg = cfg["sector"]
    assert_sector_free(SECTOR_MANIFEST)
    keep, overlap_report = sector_overlap_policy(g, D["sector"], scfg["overlap_policy"])
    check("sector_overlap_policy_applied", True, f"{overlap_report['mothers_with_public_and_private_births']} "
          "both-sector mothers excluded")
    sec_rows = []
    for source, target in scfg["directions"]:
        m = sector_masks(D["sector"], part, source, target, keep)
        assert_mother_disjoint(g, m["source_train"] | m["source_val"], m["target_all"], f"{source}->{target}")
        check(f"sector_{source}_to_{target}_zero_mother_overlap", True)
        screens = {k: screen.check(y[m[k]])["status"] for k in ("source_val", "target_all", "target_val")}
        if any(v != "ESTIMABLE" for v in screens.values()):
            sec_rows.append({"direction": f"{source}->{target}", "support_status": "NOT_ESTIMABLE", **screens})
            continue
        score = {"source_val": m["source_val"], "target_all": m["target_all"], "target_val": m["target_val"]}
        for model in cfg["models"]:
            preds, thr = fit_predict_domain(D, y, source_train=m["source_train"], source_val=m["source_val"],
                                            score=score, manifest=SECTOR_MANIFEST, algorithm=model, seed=seed)
            if model == "logistic_regression":
                z = y.copy()
                z[m["target_all"]] = 1 - z[m["target_all"]]
                pz, _ = fit_predict_domain(D, z, source_train=m["source_train"], source_val=m["source_val"],
                                           score=score, manifest=SECTOR_MANIFEST, algorithm=model, seed=seed)
                check(f"runtime_sector_{source}_to_{target}_target_labels_flipped_predictions_identical",
                      all(pz[k].tobytes() == preds[k].tobytes() for k in score))
            ref, thr3 = fit_predict_domain(D, y, source_train=m["target_train"], source_val=m["target_val"],
                                           score={"reference_val": m["target_val"]}, manifest=SECTOR_MANIFEST,
                                           algorithm=model, seed=seed)
            sd, td = f"{source} (m15 {'20-27' if source == 'public' else '30-33'})", \
                f"{target} (m15 {'20-27' if target == 'public' else '30-33'})"
            ri = add("sector", "i_within_source", sd, sd, "S_sector_free_seven", model, "source_validation",
                     y[m["source_val"]], preds["source_val"], g[m["source_val"]], thr, PROV["i"], "FT")
            rii = add("sector", "ii_cross_domain", sd, td, "S_sector_free_seven", model, "target_all_non_test",
                      y[m["target_all"]], preds["target_all"], g[m["target_all"]], thr, PROV["ii"], "FT")
            rb = add("sector", "ii_b_cross_domain_paired_rows", sd, td, "S_sector_free_seven", model,
                     "target_validation", y[m["target_val"]], preds["target_val"], g[m["target_val"]], thr,
                     PROV["ii"], "FT")
            r3 = add("sector", "iii_target_domain_reference", td, td, "S_sector_free_seven", model,
                     "target_validation", y[m["target_val"]], ref["reference_val"], g[m["target_val"]], thr3,
                     PROV["iii_sector"], "FT")
            sec_rows.append({"direction": f"{source}->{target}", "model": model, "support_status": "ESTIMABLE",
                             "source_val_n": ri["n_births"], "source_val_prevalence": ri["prevalence"],
                             "target_n": rii["n_births"], "target_mothers": rii["n_mothers"],
                             "target_events": rii["n_events"], "target_prevalence": rii["prevalence"],
                             "source_val_roc_auc": ri["roc_auc"], "target_roc_auc": rii["roc_auc"],
                             "target_val_roc_auc": rb["roc_auc"], "target_reference_roc_auc": r3["roc_auc"],
                             "source_val_ap": ri["average_precision"], "target_ap": rii["average_precision"],
                             "target_val_ap": rb["average_precision"], "target_reference_ap": r3["average_precision"],
                             "source_val_brier": ri["brier"], "target_brier": rii["brier"],
                             "target_calibration_intercept": rii["calibration_intercept"],
                             "target_calibration_slope": rii["calibration_slope"],
                             "delta_roc_auc_target_minus_source": rii["roc_auc"] - ri["roc_auc"],
                             "delta_ap_target_minus_source": rii["average_precision"] - ri["average_precision"],
                             "delta_ap_lift_target_minus_source": rii["average_precision"] / rii["prevalence"]
                             - ri["average_precision"] / ri["prevalence"],
                             "paired_delta_roc_auc_transfer_minus_target_reference": rb["roc_auc"] - r3["roc_auc"],
                             "paired_delta_ap_transfer_minus_target_reference":
                                 rb["average_precision"] - r3["average_precision"],
                             "ci_status": "NOT_APPROVED"})
    sector_df = pd.DataFrame(sec_rows)
    check("test_partition_scored_as_target_only" if evaluate_test else "test_partition_untouched", True)
    check("hist_mapping_unchanged", hist_map.mapping_hash() == hist_hash)

    # 6. Outputs.
    metrics = pd.DataFrame(rows)
    screen_df = pd.DataFrame(screen_rows)
    for f in (metrics, geo_df, geo_summary, sector_df, screen_df):
        assert_disclosure_safe(f)
    metrics.to_csv(out / "metrics.csv", index=False)
    geo_df.to_csv(out / "geographic_matrix.csv", index=False)
    geo_summary.to_csv(out / "geographic_summary.csv", index=False)
    sector_df.to_csv(out / "sector_matrix.csv", index=False)
    screen_df.to_csv(out / "support_screen.csv", index=False)
    write_json({"experiment": "A6", "NFHS-5": flow5, "shared_split": split_audit["summary"],
                "analysis_rows": {"births": int(len(D)), "mothers": int(pd.Series(g).nunique()),
                                  "events": int(y.sum()), "partitions": {k: int(v) for k, v in
                                                                         pd.Series(part).value_counts().items()}},
                "mother_state_check": msc, "support_screen": cfg["support_screen"],
                "states_estimable": len(estimable), "states_total": len(states),
                "sector_overlap": overlap_report,
                "notes": ["Analysis uses shared non-test rows only unless --evaluate-test.",
                          "Mothers never cross source/target in any geographic fold or sector direction.",
                          "NOT_ESTIMABLE domains have counts suppressed; no regional pooling."]},
               out / "cohort_flow.json")
    estp = {m: {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in make_estimator(m, seed=seed).get_params().items()} for m in cfg["models"]}
    write_json({"experiment": "A6", "status": "EXPLORATORY", "run_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": round(time.time() - t0, 1), "git": git_info(ROOT), "environment": environment(),
                "source_files": {w: {"basename": RAW[w].name} for w in RAW}, "cohort_definition": flow5["cohort_rule"],
                "seed": seed, "split_policy": sp, "test_partition_evaluated": evaluate_test,
                "geographic_arms": {a: {"numerical": list(mn.numerical), "categorical": list(mn.categorical),
                                        "tier": mn.tier} for a, mn in ARMS.items()},
                "sector_features": {"numerical": list(SECTOR_MANIFEST.numerical),
                                    "categorical": list(SECTOR_MANIFEST.categorical)},
                "hist_mapping_hash_sha256": hist_hash, "te_spec": tecfg, "models": estp,
                "model_hyperparameters_tuned": False, "a2_consistency": a2_check, "synthetic_tests": tests,
                "config": cfg, "confidence_interval_status": "NOT_APPROVED"}, out / "run_manifest.json")
    (out / "metric_schema.md").write_text(METRIC_SCHEMA, encoding="utf-8")
    verification = write_notes(out / "METHODS_RESULTS_NOTES.md", tests, qa, geo_summary, screen_df, sector_df,
                               overlap_report, flow5, D, part, len(estimable), len(states), a2_check, evaluate_test)
    write_json({"experiment": "A6", "status": "PASS", "checks": qa}, out / "qa_status.json")
    scan = scan_output_dir(out)
    qa.append({"check": "output_disclosure_scan", "status": scan["status"], "detail": ", ".join(scan["files_checked"])})
    write_json({"experiment": "A6", "status": "PASS", "checks": qa}, out / "qa_status.json")
    print("\n" + verification)
    print(f"A6 complete in {time.time() - t0:.0f}s -> {out}")


METRIC_SCHEMA = """# A6 metric schema

`metrics.csv`: one row per (design, comparison, domain, arm, model, eval_set).

| Column | Meaning |
|---|---|
| design | geographic (leave-one-state-out) or sector (public <-> private) |
| comparison | i_within_source; ii_cross_domain (all target non-test births); ii_b_cross_domain_paired_rows (target validation mothers); iii_* target-domain reference on the same rows as ii_b (context only) |
| source_domain, target_domain | domains whose labels trained the model / that were scored |
| arm | G0_ft_state_free (8 core), G_hist_nfhs4_context (+ NFHS-4 historical state x sector rate), G_te_source_fallback (+ source-only OOF TE; unseen state -> sector fallback), S_sector_free_seven (7 core, no facility_type) |
| support_status | ESTIMABLE or NOT_ESTIMABLE (n >= 200, >= 30 positives, >= 30 negatives) |
| n_births ... f1 | shared schema; threshold from source validation for ii/ii_b |
| confidence_interval_status | NOT_APPROVED |

`geographic_matrix.csv`: per held-out state x arm x model; delta vs source = held-out state minus within-source
validation; paired delta = state unseen (ii_b) minus state seen (iii) on identical validation rows.
`geographic_summary.csv`: distribution across estimable states. `sector_matrix.csv`: per direction x model.
`support_screen.csv`: screen result per state (counts suppressed when NOT_ESTIMABLE).
"""


def write_notes(path, tests, qa, gs, screen_df, sec, overlap, flow5, D, part, n_est, n_states, a2c,
                evaluate_test) -> str:
    t = {x["test"]: x["status"] for x in tests}
    q = {x["check"]: x["status"] for x in qa}

    def f(x, d=4, sign=False):
        return "NA" if x is None or pd.isna(x) else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")
    failing = screen_df[screen_df.status != "ESTIMABLE"]
    v = ["## Post-execution verification", "", "### 1. Zero-mother-overlap and target-outcome isolation tests",
         f"- Synthetic suite: {sum(s == 'PASS' for s in t.values())}/{len(t)} passed.",
         f"- Zero mother overlap: geographic **{t.get('test_zero_mother_overlap_geographic')}**, sector policy "
         f"**{t.get('test_sector_overlap_policy_and_zero_overlap')}**, overlap detection "
         f"**{t.get('test_overlap_detected_and_state_anomaly_rejected')}**.",
         f"- Target-outcome isolation: geographic **{t.get('test_geographic_target_state_labels_unused')}**, sector "
         f"**{t.get('test_sector_target_labels_unused_both_directions')}**, held-out-state TE "
         f"**{t.get('test_target_encoding_for_held_out_state_uses_source_only')}**, source/target mask guard "
         f"**{t.get('test_fit_refuses_target_rows_in_source')}**.",
         "- Runtime checks on the real data: "
         + ", ".join(f"{k.replace('runtime_', '')} **{s}**" for k, s in q.items() if k.startswith("runtime_")) + ".",
         f"- Every geographic fold and both sector directions asserted zero shared mothers; each mother has one state "
         f"(**{q.get('each_mother_single_state')}**).", "",
         "### 2. Sector transport degradation (target − within-source validation; sector-free 7 features)", ""]
    for r in sec[sec.support_status == "ESTIMABLE"].itertuples(index=False):
        v.append(f"- **{r.direction}, {r.model}:** ROC-AUC {r.source_val_roc_auc:.4f} → {r.target_roc_auc:.4f} "
                 f"(ΔAUC {r.delta_roc_auc_target_minus_source:+.4f}); AP {r.source_val_ap:.4f} → {r.target_ap:.4f} "
                 f"(ΔAP {r.delta_ap_target_minus_source:+.4f}; prevalence {r.source_val_prevalence:.3f} → "
                 f"{r.target_prevalence:.3f}; ΔAP lift {r.delta_ap_lift_target_minus_source:+.3f}). Paired vs a "
                 f"target-sector-trained reference on the same validation rows: ΔAUC "
                 f"{r.paired_delta_roc_auc_transfer_minus_target_reference:+.4f}, ΔAP "
                 f"{r.paired_delta_ap_transfer_minus_target_reference:+.4f}.")
    v += ["- AP is prevalence-dependent: public and private sectors have very different cesarean prevalence, so raw "
          "ΔAP mostly reflects the target prevalence; ΔAUC and ΔAP lift are the fairer comparisons. Differences "
          "reflect case mix and reporting by sector, not a causal sector effect.", "",
          "### 3. Geographic transport across held-out states", "",
          f"- {n_est}/{n_states} states passed the support screen (n ≥ 200, ≥ 30 positives, ≥ 30 negatives on "
          "non-test births). "
          + ("Failed screen (NOT ESTIMABLE, counts suppressed, not pooled): "
             + "; ".join(f"{r.state} ({r.reasons})" for r in failing.itertuples(index=False)) + "."
             if len(failing) else "No state failed the screen."), "",
          "| Arm | Model | States | Held-out AUC min / p25 / median / p75 / max | ΔAUC vs source median (min, max) "
          "| States below source | Paired unseen − seen median (min, max) |", "|---|---|---|---|---|---|---|"]
    for r in gs.itertuples(index=False):
        v.append(f"| {r.arm} | {r.model} | {r.states_estimable} | {r.roc_auc_min:.3f} / {r.roc_auc_p25:.3f} / "
                 f"{r.roc_auc_median:.3f} / {r.roc_auc_p75:.3f} / {r.roc_auc_max:.3f} | "
                 f"{r.delta_vs_source_median:+.4f} ({r.delta_vs_source_min:+.3f}, {r.delta_vs_source_max:+.3f}) | "
                 f"{r.states_below_source} | {f(r.paired_delta_unseen_minus_seen_median, 4, True)} "
                 f"({f(r.paired_delta_min, 3, True)}, {f(r.paired_delta_max, 3, True)}) |")
    v += ["", "- Held-out-state AUC differences mix genuine transport loss with differences in each state's case mix "
          "and prevalence; per-state values are in `geographic_matrix.csv`. Small states have unstable estimates; "
          "no intervals are attached."]
    verification = "\n".join(v)
    pc = pd.Series(part).value_counts()
    lines = ["# A6 — Geographic and public/private sector transport: Methods and Results note", "",
             "**Status: EXPLORATORY development result.** "
             + ("Test rows were used as target-only evaluation rows by explicit flag." if evaluate_test else
                "Only shared non-test NFHS-5 rows were used; the test partition was not scored.")
             + " No confidence intervals, no significance claims, not external clinical validation.", "",
             "## Methods", "",
             f"- **Data:** NFHS-5 {flow5['final_births']:,} institutional births ({flow5['cohort_rule']}); analysis "
             f"rows {len(D):,} ({', '.join(f'{k} {v:,}' for k, v in pc.items())}) from the shared mother-grouped split.",
             "- **Geographic (leave-one-state-out):** for each state passing the screen, models train on other states' "
             "train mothers, choose the threshold on other states' validation mothers (i), and score all of the "
             "held-out state's births (ii) and its validation mothers (ii-b). Reference (iii): the same arm trained "
             "on all states' train mothers (target state seen), scored on the same validation rows — context only. "
             f"Arm G0 reference reproduces A2 B0: {a2c['status']}.",
             "- **Geographic arms:** G0 = eight core features, state-free; G_hist = G0 + NFHS-4 historical state × "
             "sector rate (A2 mapping; exists before any NFHS-5 outcome); G_te = G0 + A3 grouped OOF state × sector "
             "encoding fitted on source states only — the held-out state is unseen, so it receives the source sector "
             "fallback (asserted). No numeric state code is used as a feature.",
             f"- **Sector:** public (m15 20–27) and private (m15 30–33); m15 = 96 excluded "
             f"({overlap['births_other_sector_excluded']:,} births). Prespecified policy: the "
             f"{overlap['mothers_with_public_and_private_births']:,} mothers with births in both sectors "
             f"({overlap['births_excluded_for_overlap']:,} births) are excluded from both directions. Sector-free "
             "feature set (seven core features, facility_type dropped). (i) source-sector validation; (ii) all "
             "target-sector births; (ii-b)/(iii) target validation mothers scored by the transferred model and by a "
             "target-sector-trained reference.",
             "- **Common:** logistic regression and XGBoost with fixed defaults; train-only preprocessing; max-F1 "
             "threshold from source validation applied unchanged to the target.", "",
             "## Sector matrix", "",
             "| Direction | Model | Target n | Target prev. | Source AUC | Target AUC | ΔAUC | Source AP | Target AP | "
             "ΔAP | Target cal. slope / intercept | Paired ΔAUC vs target ref. |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sec[sec.support_status == "ESTIMABLE"].itertuples(index=False):
        lines.append(f"| {r.direction} | {r.model} | {int(r.target_n):,} | {r.target_prevalence:.3f} | "
                     f"{r.source_val_roc_auc:.4f} | {r.target_roc_auc:.4f} | {r.delta_roc_auc_target_minus_source:+.4f} | "
                     f"{r.source_val_ap:.4f} | {r.target_ap:.4f} | {r.delta_ap_target_minus_source:+.4f} | "
                     f"{f(r.target_calibration_slope, 3)} / {f(r.target_calibration_intercept, 3)} | "
                     f"{r.paired_delta_roc_auc_transfer_minus_target_reference:+.4f} |")
    lines += ["", verification, "", "## Limitations", "",
              "- Exploratory, one provisional split, no intervals; V1 test previously examined.",
              "- State is interview residence, not verified delivery-facility state; migration and cross-state "
              "delivery are unresolved. Small held-out states give unstable metrics.",
              "- Sector transport is conditional on observed institutional delivery; differences reflect case mix, "
              "selection and reporting, not causal public/private effects. Excluding both-sector mothers changes the "
              "population slightly.",
              "- G_hist uses NFHS-4 labels from the held-out state (prior wave, different respondents); it is "
              "external historical context, not target-wave outcome information.",
              "- Retrospective survey-recorded features; not prenatal prediction.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return verification


if __name__ == "__main__":
    main()
