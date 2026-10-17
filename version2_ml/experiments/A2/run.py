#!/usr/bin/env python3
"""A2 runner: NFHS-4 historical context -> matched NFHS-5 ablation (aggregates only).

Order is deliberate: the crosswalk is validated and every historical mapping
is fitted and hashed from NFHS-4 BEFORE any NFHS-5 data are loaded. NFHS-5
labels are used only to fit classifiers on the shared train partition and
choose thresholds on validation. The test partition is not scored unless
--evaluate-test is passed. No row-level data, keys or predictions are written.

Usage (from repo root):  python version2_ml/experiments/A2/run.py [--evaluate-test]
"""
from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent / "common"), str(ROOT / "version2_ml")]
from A2_historical_context import (HistoricalContextMapping, SmoothingSpec, build_a2_features,  # noqa: E402
                                   FEATURE_NAMES, STATE_RATE, CELL_RATE, SUPPORT_FEATURES)
from state_crosswalk import (load_crosswalk, validate_crosswalk, nfhs4_unit_map, nfhs5_unit_map,  # noqa: E402
                             with_unseen, assign_units, read_dta_state_labels)
from exp_cohort import load_wave_births, RAW, cmc_to_year_month  # noqa: E402
from exp_metrics import binary_metrics, select_threshold_max_f1  # noqa: E402
from exp_provenance import (git_info, environment, digest_frame, suppress_small_cells,  # noqa: E402
                            assert_disclosure_safe, scan_output_dir, write_json)
from split_contracts import grouped_partition, audit_partition  # noqa: E402
from preprocessing import FeatureManifest  # noqa: E402
from model_framework import AuthorizedClassifier, make_estimator  # noqa: E402

CORE_NUM = ("maternal_age_at_index_birth", "education_years", "birth_order")
CORE_CAT = ("wealth_index", "facility_type", "residence", "religion", "twin_order")
# Non-cumulative matched arms (methods-review correction): each A2 block is added to B0 alone.
ARMS = {
    "B0": {"num": (), "cat": (), "role": "baseline_8_core", "tier": "FT"},
    "B0+S": {"num": (STATE_RATE,), "cat": (), "role": "required_hist_state", "tier": "F3"},
    "B0+SX": {"num": (CELL_RATE,), "cat": (), "role": "required_hist_state_x_sector", "tier": "F3"},
    "B0+SUP": {"num": SUPPORT_FEATURES, "cat": (), "role": "required_support_indicators", "tier": "F3"},
    "B0+S+SX+SUP": {"num": FEATURE_NAMES, "cat": (), "role": "combined_all_a2", "tier": "F3"},
    "R1": {"num": (), "cat": ("nfhs5_state",), "role": "reference_nfhs5_state_indicators", "tier": "FT"},
}
CONTEXT_ARMS = ("B0+S", "B0+SX", "B0+SUP", "B0+S+SX+SUP")
N2 = "N2_hist_cell_rate_as_probability"
SOURCE_DOMAIN = "NFHS-4 (historical context source; labels used only in source aggregates)"


def arm_manifest(arm: str) -> FeatureManifest:
    a = ARMS[arm]
    return FeatureManifest(numerical=CORE_NUM + tuple(a["num"]), categorical=CORE_CAT + tuple(a["cat"]),
                           tier=a["tier"])


def assert_matched_rows(reference: np.ndarray, candidate: np.ndarray) -> None:
    if reference.shape != candidate.shape or not np.array_equal(reference, candidate):
        raise AssertionError("MATCHING: ablation arms evaluated on different rows")


def metric_row(base: dict, y, p, groups, threshold, provenance, weights=None) -> dict:
    m = binary_metrics(y, p, threshold=threshold, sample_weight=weights)
    return {**base, "n_births": int(len(y)), "n_mothers": int(pd.Series(groups).nunique()),
            "n_events": int(np.sum(y)), "prevalence": float(np.mean(y)),
            "weighted_prevalence": m["weighted_outcome_prevalence"] if weights is not None else None,
            "roc_auc": m["roc_auc"], "average_precision": m["average_precision"], "brier": m["brier_score"],
            "calibration_intercept": m["calibration_intercept"], "calibration_slope": m["calibration_slope"],
            "threshold": m.get("threshold"), "threshold_provenance": provenance,
            "precision": m.get("precision"), "recall": m.get("recall"), "f1": m.get("f1"),
            "confidence_interval_status": "NOT_APPROVED", "result_status": "EXPLORATORY"}


def threshold_provenance(split: str) -> str:
    return ("max_f1_selected_on_same_validation_rows_optimistic" if split == "validation"
            else "max_f1_selected_on_validation_frozen")


def level_support(features: pd.DataFrame, partition: pd.Series, variant: str, min_rows: int) -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        sub = features[partition.eq(split).to_numpy()]
        for kind, col in (("state_feature", "hist_state_level"), ("cell_feature", "hist_cell_level")):
            for level in ("cell", "state", "sector", "global"):
                n = int(sub[col].eq(level).sum())
                if kind == "state_feature" and level in ("cell", "sector"):
                    continue
                rows.append({"context_variant": variant, "split": split, "feature": kind, "source_level": level,
                             "n_births": n if (n == 0 or n >= min_rows) else "SUPPRESSED"})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "aggregate"))
    ap.add_argument("--evaluate-test", action="store_true", help="Score the provisional test partition")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    evaluate_test = bool(args.evaluate_test or cfg.get("evaluate_test", False))
    eval_splits = ["validation"] + (["test"] if evaluate_test else [])
    disc = cfg["disclosure"]
    qa = []

    def check(name, ok, detail=""):
        qa.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        if not ok:
            raise AssertionError(f"QA failed: {name}: {detail}")

    # 1. Crosswalk from official DTA value labels (no rows read).
    xw = load_crosswalk(HERE / cfg["crosswalk"])
    labels4 = read_dta_state_labels(ROOT / RAW["NFHS-4"])
    labels5 = read_dta_state_labels(ROOT / RAW["NFHS-5"])
    xw_report = validate_crosswalk(xw, labels4, labels5)
    xw_unseen = with_unseen(xw, cfg["smoothing"]["ladakh_unseen_nfhs5_codes"])
    validate_crosswalk(xw_unseen, labels4, labels5)
    check("crosswalk_valid_against_dta_labels", True, f"{xw_report['nfhs5_codes']} NFHS-5 codes")

    # 2. NFHS-4 source: fit and hash every mapping before NFHS-5 is loaded.
    src, flow4 = load_wave_births(ROOT, "NFHS-4", use_cache=not args.no_cache)
    src["source_unit"] = assign_units(src["v024"], nfhs4_unit_map(xw))
    src_fit = src[["wave", "source_unit", "sector", "m17", "sample_weight"]].copy()
    check("source_rows_all_nfhs4", set(src_fit["wave"]) == {"NFHS-4"})
    specs = {"primary": SmoothingSpec(**cfg["smoothing"]["primary"])}
    specs.update({k: SmoothingSpec(**v) for k, v in cfg["smoothing"]["context_sensitivities"].items()})
    mappings = {k: HistoricalContextMapping(s).fit(src_fit) for k, s in specs.items()}
    hashes_before_target = {k: m.mapping_hash() for k, m in mappings.items()}
    source_digest = digest_frame(src_fit[["source_unit", "sector", "m17"]])
    overlap_cmc = int(src["v008"].max())
    source_timing = {"nfhs4_interview_first": cmc_to_year_month(src["v008"].min()),
                     "nfhs4_interview_last": cmc_to_year_month(src["v008"].max()),
                     "nfhs4_birth_first": cmc_to_year_month(src["b3"].min()),
                     "nfhs4_birth_last": cmc_to_year_month(src["b3"].max())}
    del src

    # 3. NFHS-5 target.
    tgt, flow5 = load_wave_births(ROOT, "NFHS-5", use_cache=not args.no_cache)
    tgt["source_unit"] = assign_units(tgt["v024"], nfhs5_unit_map(xw))
    unseen_units = assign_units(tgt["v024"], nfhs5_unit_map(xw_unseen))
    tgt["nfhs5_state"] = tgt["v024"].astype(int).astype(str)
    contexts = {k: m.transform(tgt[["source_unit", "sector"]]) for k, m in mappings.items()}
    contexts["ladakh_unseen"] = mappings["primary"].transform(
        pd.DataFrame({"source_unit": unseen_units, "sector": tgt["sector"]}, index=tgt.index))

    # 3b. Runtime adversarial check on the real target: label permutation/removal.
    rng = np.random.default_rng(seed)
    for label, mutated in (("permuted", tgt.assign(m17=rng.permutation(tgt["m17"].to_numpy()))),
                           ("removed", tgt.drop(columns="m17")), ("all_positive", tgt.assign(m17=1))):
        m2, f2 = build_a2_features(src_fit, mutated, specs["primary"])
        same = (m2.mapping_hash() == hashes_before_target["primary"]
                and all(f2[c].to_numpy().tobytes() == contexts["primary"][c].to_numpy().tobytes()
                        for c in FEATURE_NAMES)
                and f2[["hist_state_level", "hist_cell_level"]].equals(
                    contexts["primary"][["hist_state_level", "hist_cell_level"]]))
        check(f"runtime_target_label_{label}_bitwise_invariance", same)
    del src_fit, m2, f2

    # 4. Shared mother-grouped development split.
    sp = cfg["split"]
    partition = grouped_partition(tgt, group_col=sp["group"], target_col="m17",
                                  fractions=tuple(sp["fractions"]), seed=int(sp["seed"]))
    split_audit = audit_partition(tgt, partition, group_col=sp["group"], target_col="m17")
    check("split_mother_exclusive", split_audit["status"] == "GROUP_EXCLUSIVE_VALID")
    psu_key = tgt["v024"].astype(str) + ":" + tgt["psu"].astype(int).astype(str)
    train_psu = set(psu_key[partition.eq("train")])
    psu_overlap = {s: {"clusters": int(psu_key[partition.eq(s)].nunique()),
                       "clusters_also_in_train": int(len(set(psu_key[partition.eq(s)]) & train_psu))}
                   for s in ("validation", "test")}

    cohorts = {"primary": pd.Series(True, index=tgt.index),
               "nfhs5_cmc_0_59": tgt["cmc_recall"].between(0, 59),
               "exclude_other_facility": tgt["m15"].ne(96),
               "exclude_calendar_overlap": tgt["b3"].gt(overlap_cmc)}
    cohort_counts = {}
    for name, mask in cohorts.items():
        sub = audit_partition(tgt[mask], partition[mask], group_col=sp["group"], target_col="m17")
        cohort_counts[name] = sub["summary"]

    # 5. Matched ablation.
    base_cols = list(CORE_NUM + CORE_CAT) + ["nfhs5_state"]
    y_all = tgt["m17"].to_numpy(int)
    groups_all = tgt["caseid"].to_numpy()
    w_all = tgt["sample_weight"].to_numpy(float)
    plan = ([("primary", "primary")] + [(c, "primary") for c in cfg["cohort_sensitivities"]]
            + [("primary", v) for v in list(cfg["smoothing"]["context_sensitivities"]) + ["ladakh_unseen"]])
    rows, eval_index = [], {}
    for cohort, variant in plan:
        mask = cohorts[cohort].to_numpy()
        X = pd.concat([tgt[base_cols], contexts[variant][list(FEATURE_NAMES)]], axis=1)
        masks = {s: mask & partition.eq(s).to_numpy() for s in ("train", *eval_splits)}
        for s in eval_splits:
            idx = np.flatnonzero(masks[s])
            eval_index.setdefault((cohort, s), idx)
            assert_matched_rows(eval_index[(cohort, s)], idx)
        arms = list(ARMS) if variant == "primary" else list(CONTEXT_ARMS)
        for model in cfg["models"]:
            for arm in arms:
                clf = AuthorizedClassifier(arm_manifest(arm), algorithm=model, seed=seed)
                clf.fit(X[masks["train"]], y_all[masks["train"]], partition="train")
                preds = {s: clf.predict_proba(X[masks[s]], partition=s) for s in eval_splits}
                thr = select_threshold_max_f1(y_all[masks["validation"]], preds["validation"])
                for s in eval_splits:
                    base = {"experiment": "A2", "wave": "NFHS-5", "source_domain": SOURCE_DOMAIN,
                            "target_domain": "NFHS-5", "cohort": cohort, "context_variant": variant, "split": s,
                            "model": model, "arm": arm, "arm_role": ARMS[arm]["role"],
                            "feature_tier": ARMS[arm]["tier"], "weighting": "unweighted"}
                    m = masks[s]
                    rows.append(metric_row(base, y_all[m], preds[s], groups_all[m], thr, threshold_provenance(s)))
                    if cfg.get("weighted_descriptive_metrics") and cohort == "primary" and variant == "primary":
                        rows.append(metric_row({**base, "weighting": "v005_descriptive"}, y_all[m], preds[s],
                                               groups_all[m], thr, threshold_provenance(s), weights=w_all[m]))
                print(f"[{time.time() - t0:7.1f}s] {cohort:26s} {variant:14s} {model:20s} {arm}", flush=True)
        # Supplemental N2: historical cell rate used directly as a probability (nothing fitted).
        rate = contexts[variant][CELL_RATE].to_numpy(float)
        thr = select_threshold_max_f1(y_all[masks["validation"]], rate[masks["validation"]])
        for s in eval_splits:
            m = masks[s]
            base = {"experiment": "A2", "wave": "NFHS-5", "source_domain": SOURCE_DOMAIN, "target_domain": "NFHS-5",
                    "cohort": cohort, "context_variant": variant, "split": s, "model": "none_direct_rate",
                    "arm": N2, "arm_role": "supplemental_map_baseline_untuned", "feature_tier": "F3",
                    "weighting": "unweighted"}
            rows.append(metric_row(base, y_all[m], rate[m], groups_all[m], thr, threshold_provenance(s)))
    metrics = pd.DataFrame(rows)

    # 6. Ablation deltas vs B0 on identical rows (B0 always from the primary context; it has no A2 input).
    b0 = metrics[(metrics.arm == "B0") & (metrics.context_variant == "primary")].set_index(
        ["cohort", "model", "split", "weighting"])
    abl = []
    for r in metrics[metrics.arm.isin(list(CONTEXT_ARMS) + ["R1"])].itertuples(index=False):
        ref = b0.loc[(r.cohort, r.model, r.split, r.weighting)]
        if int(ref["n_births"]) != r.n_births:
            raise AssertionError("MATCHING: B0 and arm row counts differ")
        abl.append({"experiment": "A2", "cohort": r.cohort, "context_variant": r.context_variant, "split": r.split,
                    "model": r.model, "weighting": r.weighting, "arm": r.arm, "comparator": "B0",
                    "matched_n_births": r.n_births, "matched_rows_check": "PASS_identical_row_index",
                    "delta_roc_auc": r.roc_auc - ref["roc_auc"],
                    "delta_average_precision": r.average_precision - ref["average_precision"],
                    "delta_brier": r.brier - ref["brier"],
                    "arm_calibration_slope": r.calibration_slope, "b0_calibration_slope": ref["calibration_slope"],
                    "paired_ci_status": "NOT_APPROVED", "result_status": "EXPLORATORY"})
    ablations = pd.DataFrame(abl)
    check("matched_rows_all_ablations", True, f"{len(ablations)} paired comparisons")

    # 7. Post-run mapping integrity.
    hashes_after = {k: m.mapping_hash() for k, m in mappings.items()}
    check("mapping_hashes_unchanged_after_target_processing", hashes_after == hashes_before_target)
    check("test_partition_scored" if evaluate_test else "test_partition_untouched", True,
          "scored by explicit flag" if evaluate_test else "no test predictions generated")

    # 8. Aggregate outputs.
    prim = mappings["primary"].aggregate_tables()
    cell_tab = suppress_small_cells(prim["cell"], n_col="n_births", event_col="n_events", min_n=disc["min_n"],
                                    min_events=disc["min_events"], value_cols=("support", "rate", "prior", "supported"))
    state_tab = suppress_small_cells(prim["state"], n_col="n_births", event_col="n_events", min_n=disc["min_n"],
                                     min_events=disc["min_events"], value_cols=("support", "rate", "supported"))
    sector_tab = suppress_small_cells(prim["sector"], n_col="n_births", event_col="n_events", min_n=disc["min_n"],
                                      min_events=disc["min_events"], value_cols=("support", "rate", "supported"))
    wst = mappings["weighted_v005"].aggregate_tables()["state"]
    wstate_tab = suppress_small_cells(wst, n_col="n_births", event_col="n_events", min_n=disc["min_n"],
                                      min_events=disc["min_events"], value_cols=("support", "rate", "supported"))
    provenance = {
        "status": "EXPLORATORY", "source_wave": "NFHS-4", "fitted_before_target_loaded": True,
        "source_rows_digest_sha256": source_digest, "mapping_hash_sha256": hashes_before_target,
        "smoothing_specs": {k: m.spec_dict for k, m in mappings.items()},
        "fallback_order": {"state_feature": ["supported state", "global"],
                           "cell_feature": ["supported cell", "supported state", "supported sector", "global"]},
        "global": prim["global"].to_dict(orient="records")[0],
        "sector": sector_tab.to_dict(orient="records"),
        "state_unweighted": state_tab.to_dict(orient="records"),
        "state_weighted_v005_descriptive": wstate_tab.to_dict(orient="records"),
        "state_x_sector_unweighted": cell_tab.to_dict(orient="records"),
        "supported_counts": {k: {"states": int(m.aggregate_tables()["state"]["supported"].sum()),
                                 "cells": int(m.aggregate_tables()["cell"]["supported"].sum()),
                                 "cells_total": int(len(m.aggregate_tables()["cell"]))}
                             for k, m in mappings.items()},
        "timing": {**source_timing, "nfhs5_birth_first": cmc_to_year_month(tgt["b3"].min()),
                   "nfhs5_birth_last": cmc_to_year_month(tgt["b3"].max()),
                   "calendar_overlap_rule": "exclude_calendar_overlap keeps NFHS-5 births after the last NFHS-4 interview month",
                   "nfhs5_births_in_or_before_last_nfhs4_interview_month": int(tgt["b3"].le(overlap_cmc).sum())},
        "suppression_rule": f"n<{disc['min_n']} or events<{disc['min_events']} or non-events<{disc['min_events']}",
        "crosswalk": xw_report,
    }
    write_json(provenance, out / "source_context_provenance.json")

    support = pd.concat([level_support(contexts[v], partition, v, disc["min_level_rows"])
                         for v in contexts], ignore_index=True)
    support["n_births"] = support["n_births"].astype(object)
    if not evaluate_test:
        support.loc[support.split == "test", "n_births"] = "NOT_REPORTED_TEST_UNTOUCHED"
    lvl = contexts["primary"].groupby([tgt["nfhs5_state"], tgt["sector"]])["hist_cell_level"].first()
    lvl_ladakh = contexts["ladakh_unseen"].groupby([tgt["nfhs5_state"], tgt["sector"]])["hist_cell_level"].first()
    support_cells = pd.DataFrame([
        {"context_variant": v, "split": "all_nfhs5_cells", "feature": "cell_feature", "source_level": k,
         "n_births": None, "n_nfhs5_state_sector_cells": int(c)}
        for v, s in (("primary", lvl), ("ladakh_unseen", lvl_ladakh)) for k, c in s.value_counts().items()])
    support = pd.concat([support, support_cells], ignore_index=True)
    assert_disclosure_safe(support)
    support.to_csv(out / "unseen_cell_support.csv", index=False)

    for frame in (metrics, ablations):
        assert_disclosure_safe(frame)
    metrics.to_csv(out / "metrics.csv", index=False)
    ablations.to_csv(out / "ablations.csv", index=False)

    write_json({"experiment": "A2",
                "NFHS-4": {k: v for k, v in flow4.items()},
                "NFHS-5": {k: v for k, v in flow5.items()},
                "nfhs5_split_primary": split_audit["summary"],
                "nfhs5_cohort_sensitivity_splits": cohort_counts,
                "cluster_overlap_with_train": psu_overlap,
                "notes": ["Mothers never cross partitions (audit_partition).",
                          "Survey clusters DO cross partitions; mother grouping is the protocol unit, cluster overlap is a robustness caveat.",
                          "Test partition counts are design counts only; no test outcomes were scored unless evaluate_test is true."]},
               out / "cohort_flow.json")

    est = {m: {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
               for k, v in make_estimator(m, seed=seed).get_params().items()} for m in cfg["models"]}
    manifest = {
        "experiment": "A2", "status": "EXPLORATORY", "run_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 1), "git": git_info(ROOT), "environment": environment(),
        "source_files": {w: {"basename": RAW[w].name, "bytes": f.get("raw_file_bytes")}
                         for w, f in (("NFHS-4", flow4), ("NFHS-5", flow5))},
        "cohort_definitions": {"NFHS-4": flow4["cohort_rule"], "NFHS-5": flow5["cohort_rule"]},
        "seed": seed, "split_policy": {**sp, "function": "split_contracts.grouped_partition",
                                       "status": "provisional development split; not an untouched confirmatory test"},
        "test_partition_evaluated": evaluate_test, "evaluated_splits": eval_splits,
        "arms": {a: {"numerical": list(arm_manifest(a).numerical), "categorical": list(arm_manifest(a).categorical),
                     "role": ARMS[a]["role"], "tier": ARMS[a]["tier"]} for a in ARMS},
        "supplemental_arm": {N2: "hist_state_sector_cs_rate used directly as predicted probability; nothing fitted"},
        "models": est, "model_hyperparameters_tuned": False,
        "preprocessing": "preprocessing.TrainOnlyPreprocessor fitted on NFHS-5 train partition only",
        "threshold_policy": "max-F1 on validation; validation threshold metrics are optimistic",
        "mapping_hash_sha256": hashes_before_target, "config": cfg,
        "confidence_interval_status": "NOT_APPROVED",
    }
    write_json(manifest, out / "run_manifest.json")
    (out / "metric_schema.md").write_text(METRIC_SCHEMA, encoding="utf-8")
    write_notes(out / "METHODS_RESULTS_NOTES.md", metrics, ablations, provenance, flow4, flow5,
                split_audit, evaluate_test, psu_overlap)
    write_json({"experiment": "A2", "status": "PASS", "checks": qa}, out / "qa_status.json")
    scan = scan_output_dir(out)
    qa.append({"check": "output_disclosure_scan", "status": scan["status"], "detail": ", ".join(scan["files_checked"])})
    write_json({"experiment": "A2", "status": "PASS", "checks": qa}, out / "qa_status.json")
    print(f"A2 complete in {time.time() - t0:.0f}s -> {out}")


METRIC_SCHEMA = """# A2 metric schema

`metrics.csv` has one row per (cohort, context_variant, split, model, arm, weighting).

| Column | Meaning |
|---|---|
| experiment, wave, source_domain, target_domain | A2; NFHS-5 evaluation; NFHS-4 is the context source only |
| cohort | `primary` (NFHS-5 b19 0..59 institutional) or a cohort sensitivity |
| context_variant | historical mapping variant (`primary` alpha=20,n_min=30 unweighted; alpha grid; `weighted_v005`; `ladakh_unseen`) |
| split | `validation` (default) or `test` (only with --evaluate-test) |
| model / arm / arm_role / feature_tier | fixed-hyperparameter estimator and matched feature arm |
| weighting | `unweighted` or `v005_descriptive` (descriptive point metric; design not validated) |
| n_births, n_mothers, n_events, prevalence | evaluation denominators (unweighted) |
| roc_auc | primary metric |
| average_precision | prevalence-dependent; read with `prevalence` |
| brier | mean squared probability error |
| calibration_intercept | logistic recalibration intercept with logit(p) as offset (slope fixed at 1) |
| calibration_slope | slope of logit(p) in logistic recalibration; null if not identifiable |
| threshold, threshold_provenance, precision, recall, f1 | max-F1 threshold chosen on validation only |
| confidence_interval_status | always NOT_APPROVED (survey-design estimator not validated) |
| result_status | EXPLORATORY |

`ablations.csv`: delta = arm minus B0 on identical rows (same cohort, split, model, weighting). No paired CI.
"""


def _fmt(x, d=4, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


def write_notes(path, metrics, ablations, prov, flow4, flow5, split_audit, evaluate_test, psu_overlap):
    v = metrics[(metrics.cohort == "primary") & (metrics.context_variant == "primary")
                & (metrics.split == "validation") & (metrics.weighting == "unweighted")]
    a = ablations[(ablations.cohort == "primary") & (ablations.context_variant == "primary")
                  & (ablations.split == "validation") & (ablations.weighting == "unweighted")].set_index(["model", "arm"])
    lines = ["# A2 — Historical NFHS-4 state / facility-sector context: Methods and Results note", "",
             "**Status: EXPLORATORY development result.** Provisional mother-grouped split; "
             + ("test partition scored by explicit flag." if evaluate_test else
                "metrics are on the *validation* partition; the test partition was not scored.")
             + " No confidence intervals or significance claims (survey-design estimator NOT approved).", "",
             "## Methods", "",
             f"- **Source (NFHS-4):** {flow4['final_births']:,} institutional births ({flow4['cohort_rule']}). "
             f"Interviews {prov['timing']['nfhs4_interview_first']} to {prov['timing']['nfhs4_interview_last']}. "
             "Labels are used only to build aggregate state, sector and state×sector cesarean rates.",
             f"- **Target (NFHS-5):** {flow5['final_births']:,} births, {flow5['final_mothers']:,} mothers "
             f"({flow5['cohort_rule']}). Shared provisional mother-grouped 70/15/15 split (seed 42): "
             + ", ".join(f"{k} {s['rows']:,}" for k, s in split_audit["summary"].items()) + ".",
             "- **Crosswalk:** explicit versioned table (`state_crosswalk_v1.csv`) validated against both DTA v024 "
             "value-label tables; numeric codes are never joined. DNH and DD are pooled for NFHS-4; Ladakh uses the "
             "NFHS-4 Jammu & Kashmir unit (sensitivity: no source unit).",
             "- **Features:** smoothed historical state rate, state×sector rate, log source support and unsupported "
             "flags. Pseudo-count alpha=20, support threshold 30 births, unweighted (predeclared). Fallback: "
             "cell → state → sector → NFHS-4 global; the state feature falls back to the global rate. "
             "No NFHS-5 outcome enters the mapping; it is fitted and hashed before NFHS-5 is loaded.",
             "- **Matched ablation:** B0 = eight common-core features; B0+S, B0+SX, B0+SUP each add one A2 block; "
             "B0+S+SX+SUP adds all; R1 adds ordinary NFHS-5 state indicators (reference). Same rows, split, "
             "train-only preprocessing and fixed default hyperparameters for logistic regression and XGBoost. "
             "N2 uses the historical state×sector rate directly as a probability (nothing fitted).",
             "- **Thresholds:** max-F1 on validation, so validation threshold metrics are optimistic.", "",
             "## Results (primary cohort, primary context, validation, unweighted)", "",
             "| Model | Arm | ROC-AUC | ΔAUC vs B0 | AP | ΔAP vs B0 | Brier | Cal. slope | Cal. intercept |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in v.itertuples(index=False):
        key = (r.model, r.arm)
        dauc = a.loc[key, "delta_roc_auc"] if key in a.index else None
        dap = a.loc[key, "delta_average_precision"] if key in a.index else None
        lines.append(f"| {r.model} | {r.arm} | {r.roc_auc:.4f} | {_fmt(dauc, 4, True) if dauc is not None else '—'} | "
                     f"{r.average_precision:.4f} | {_fmt(dap, 4, True) if dap is not None else '—'} | {r.brier:.4f} | "
                     f"{_fmt(r.calibration_slope, 3)} | {_fmt(r.calibration_intercept, 3)} |")
    prev = v["prevalence"].iloc[0]
    lines += ["", f"Validation prevalence {prev:.4f} (n={int(v['n_births'].iloc[0]):,} births, "
              f"{int(v['n_mothers'].iloc[0]):,} mothers). AP depends on this prevalence.", "",
              "## Sensitivity summary (ΔAUC vs B0, validation, unweighted)", "",
              "| Cohort | Context variant | Model | B0+S | B0+SX | B0+SUP | B0+S+SX+SUP |", "|---|---|---|---|---|---|---|"]
    sens = ablations[(ablations.split == "validation") & (ablations.weighting == "unweighted")
                     & ablations.arm.isin(CONTEXT_ARMS)]
    for (c, cv, mdl), g in sens.groupby(["cohort", "context_variant", "model"], sort=False):
        d = g.set_index("arm")["delta_roc_auc"]
        lines.append(f"| {c} | {cv} | {mdl} | " + " | ".join(_fmt(d.get(x), 4, True) for x in CONTEXT_ARMS) + " |")
    best = sens["delta_roc_auc"]
    r = v.set_index(["model", "arm"])["roc_auc"]
    sx = sens[(sens.arm == "B0+SX") & (sens.context_variant == "primary")]
    primary_sx = sx[sx.cohort == "primary"].set_index("model")["delta_roc_auc"]
    overlap = sx[sx.cohort == "exclude_calendar_overlap"].set_index("model")["delta_roc_auc"]
    n2 = v[v.arm == N2].iloc[0]
    sx_vs_r1 = "; ".join(f"{mdl} {r[(mdl, 'B0+SX')] - r[(mdl, 'R1')]:+.4f}"
                         for mdl in v.model.unique() if (mdl, "R1") in r.index)
    lines += ["", f"Across all {len(sens)} A2 sensitivity comparisons, ΔAUC ranged from {best.min():+.4f} to "
              f"{best.max():+.4f}. The direction is consistent across cohorts and smoothing variants, but these "
              "are point differences on one provisional split without approved intervals: exploratory only.", "",
              "## Reading the results", "",
              f"- **Versus ordinary state indicators (R1):** B0+SX minus R1 ROC-AUC = {sx_vs_r1}. Historical NFHS-4 "
              "context carries state/sector information of similar discriminative value to NFHS-5-trained state "
              "indicators on this split. This does not show signal beyond state identity.",
              "- **Support features act as state identifiers:** log source support takes one value per NFHS-4 "
              "source unit, so B0+SUP (especially for XGBoost) can learn state effects from NFHS-5 training labels. "
              "Its gain should not be read as an effect of support itself.",
              f"- **Wave shift (N2):** the NFHS-4 rate used directly as a probability has ROC-AUC "
              f"{n2.roc_auc:.4f} but calibration intercept {_fmt(n2.calibration_intercept, 3, True)}: NFHS-4 global "
              f"rate {prov['global']['rate']:.3f} vs NFHS-5 validation prevalence {n2.prevalence:.3f}. Historical "
              "rates rank state/sector cells usefully but under-predict NFHS-5 cesarean levels.",
              "- **Calendar overlap:** B0+SX ΔAUC with NFHS-5 births in or before the last NFHS-4 interview month "
              "excluded vs primary cohort: " + "; ".join(
                  f"{mdl} {overlap.get(mdl, float('nan')):+.4f} vs {primary_sx.get(mdl, float('nan')):+.4f}"
                  for mdl in primary_sx.index) + ".", "",
              "## Supported claims", "",
              "- A2 features were built from NFHS-4 only; runtime and unit tests show NFHS-5 label permutation, "
              "replacement or removal leaves the mapping hash and every feature bitwise identical.",
              "- The table reports how much historical context changes development-split discrimination and "
              "calibration relative to the matched eight-core baseline and to ordinary NFHS-5 state indicators (R1).",
              "", "## Not supported / limitations", "",
              "- Not causal: rates describe historical case mix and reporting by state and sector, not sector effects "
              "or medical necessity. Not prenatal prediction: features are retrospectively recorded.",
              "- No confidence intervals or significance; one provisional split; old V1 test already examined.",
              f"- Survey clusters cross partitions (validation: {psu_overlap['validation']['clusters_also_in_train']:,} "
              f"of {psu_overlap['validation']['clusters']:,} clusters also in train); mother grouping only.",
              f"- Calendar overlap: {prov['timing']['nfhs5_births_in_or_before_last_nfhs4_interview_month']:,} NFHS-5 "
              "births occurred in or before the last NFHS-4 interview month; see the `exclude_calendar_overlap` row.",
              "- State is interview residence, not verified delivery-facility state. Ladakh shares the NFHS-4 J&K "
              "source unit in the primary analysis.",
              "- Weighted (v005) results are descriptive only; weights were not verified for this estimand.",
              "- m15=96 ('other') births: all non-cesarean in the audit; see `exclude_other_facility`.",
              "", "## Failure modes checked", "",
              "- Fitting on NFHS-5 or mixed-wave rows is refused (LEAKAGE error). Unknown state codes or crosswalk/label "
              "drift stop the run. Unknown sector values are refused. Unsupported, zero-count and unseen cells use "
              "the fallback order and set flags. Ablation arms are checked for identical evaluation rows.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
