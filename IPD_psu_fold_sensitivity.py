"""PSU-grouped cross-fitting sensitivity for the IPD uncertainty review.

Run from the IPD repository root after IPD_uncertainty_review.py has completed.
Each completed fold is checkpointed under IPD_uncertainty_private/. Original
results are read-only; new aggregate JSON files go to outputs_final/psu_grouped/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import IPD_uncertainty_review as original

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "IPD_uncertainty_private" / "final_review_psu_grouped"
PUBLIC = ROOT / "extensions_work" / "09_uncertainty_review" / "outputs_final" / "psu_grouped"
PRIVATE.mkdir(parents=True, exist_ok=True)
PUBLIC.mkdir(parents=True, exist_ok=True)


def psu_groups(df):
    if df[["sample_stratum_v022", "cluster_number"]].isna().any().any():
        raise ValueError("Stratum/PSU IDs cannot be missing")
    # Combine stratum with PSU because numeric PSU IDs may be reused by strata.
    groups, _ = pd.factorize(pd.MultiIndex.from_frame(
        df[["sample_stratum_v022", "cluster_number"]]), sort=False)
    return groups


def save_fold(path, frame):
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def saved_or_none(path, expected_indices):
    if not path.exists():
        return None
    saved = pd.read_parquet(path)
    if not np.array_equal(saved.row_index.to_numpy(), expected_indices):
        raise ValueError(f"Saved checkpoint {path} does not match current fold; stop")
    print(f"Loaded checkpoint: {path}", flush=True)
    return saved


def complete(label, df, folds):
    target = PRIVATE / label
    pieces = [pd.read_parquet(target / f"fold_{k}.parquet") for k in range(5)]
    pred = pd.concat(pieces, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(pred.row_index.to_numpy(), np.arange(len(df))):
        raise ValueError("Missing or duplicated held-out predictions")
    group = psu_groups(df)
    for k, (_, val) in enumerate(folds):
        if len(set(group[val]) & set(group[np.where(pred.fold.to_numpy() != k)[0]])):
            # All non-heldout rows may contain other folds; this tests each
            # PSU belongs to precisely one fold across the full row file.
            raise AssertionError(f"A PSU is assigned to several folds (fold {k})")
    rows = pd.DataFrame({
        "exposure": (df.facility_type == "private").to_numpy(dtype=int),
        "csection": df.csection.to_numpy(dtype=int),
        "sample_weight_normalized": df.sample_weight_normalized.to_numpy(dtype=float),
        "cluster_number": df.cluster_number.to_numpy(),
        "sample_stratum_v022": df.sample_stratum_v022.to_numpy(),
        "respondent_id": df.respondent_id.to_numpy(),
        "fold": pred.fold.to_numpy(dtype=int),
        "e_hat": np.clip(pred.e_hat.to_numpy(dtype=float),
                         .01 if label == "professor_methods" else 1e-6,
                         .99 if label == "professor_methods" else 1-1e-6),
        "mu1_hat": pred.mu1_hat.to_numpy(dtype=float),
        "mu0_hat": pred.mu0_hat.to_numpy(dtype=float),
    })
    row_file = target / "rows.parquet"
    save_fold(row_file, rows)
    # Review function uses a module-level output directory. Redirect it to a
    # NEW subdirectory only; original aggregate JSONs stay untouched.
    previous_dir = original.PUBLIC
    try:
        original.PUBLIC = PUBLIC
        result = original.review(label, row_file)
    finally:
        original.PUBLIC = previous_dir
    if result["fold_review"]["psus_across_multiple_folds"]:
        raise AssertionError("PSU-grouped fold sensitivity still splits PSUs")
    print(f"Saved private rows: {row_file}", flush=True)


def fit_ext3(round_name):
    df = original.load_round(round_name)
    cfg, util = original.config, original.utils
    cols = list(cfg.CONFOUNDER_COLS)
    W = df[cols].copy()
    for col in cfg.CATEGORICAL_CONFOUNDERS:
        W[col] = W[col].astype("string").fillna("Missing").astype(str)
    a, y = df.exposure.to_numpy(dtype=int), df.csection.to_numpy(dtype=int)
    w = df.sample_weight_normalized.to_numpy(dtype=float)
    groups = psu_groups(df)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(W, a, groups))
    target = PRIVATE / round_name
    target.mkdir(exist_ok=True)
    for k, (tr, val) in enumerate(folds):
        dest = target / f"fold_{k}.parquet"
        if saved_or_none(dest, val) is not None:
            continue
        if set(groups[tr]) & set(groups[val]):
            raise AssertionError("PSU leaked across train/test")
        print(f"{round_name} PSU fold {k+1}/5: fitting", flush=True)
        prop = util.make_propensity_pipeline(cfg.NUMERIC_CONFOUNDERS, cfg.CATEGORICAL_CONFOUNDERS, 42)
        prop.fit(W.iloc[tr], a[tr], clf__sample_weight=w[tr])
        e = prop.predict_proba(W.iloc[val])[:, 1]
        out = util.make_outcome_pipeline(cfg.NUMERIC_CONFOUNDERS, cfg.CATEGORICAL_CONFOUNDERS, 42)
        train = W.iloc[tr].copy()
        train["exposure"] = a[tr]
        out.fit(train, y[tr], clf__sample_weight=w[tr])
        held = W.iloc[val].copy()
        held["exposure"] = 1
        q1 = out.predict_proba(held)[:, 1]
        held["exposure"] = 0
        q0 = out.predict_proba(held)[:, 1]
        save_fold(dest, pd.DataFrame({"row_index":val,"fold":k,"e_hat":e,"mu1_hat":q1,"mu0_hat":q0}))
        print(f"Saved {round_name} PSU fold {k+1}/5", flush=True)
    complete(round_name, df, folds)


def fit_prof():
    prof = original.professor
    df = pd.read_parquet(original.config.V2_PROCESSED_DATA_PATH)
    df = df.loc[df.facility_type.isin(["public", "private"])].reset_index(drop=True)
    if len(df) != 200794:
        raise ValueError("Unexpected professor-methods analytic cohort")
    x = prof.make_features(df)
    a = (df.facility_type == "private").to_numpy(dtype=int)
    y = df.csection.to_numpy(dtype=int)
    w = df.sample_weight_normalized.to_numpy(dtype=float)
    groups = psu_groups(df)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(x, a, groups))
    target = PRIVATE / "professor_methods"
    target.mkdir(exist_ok=True)
    for k, (tr, val) in enumerate(folds):
        dest = target / f"fold_{k}.parquet"
        if saved_or_none(dest, val) is not None:
            continue
        if set(groups[tr]) & set(groups[val]):
            raise AssertionError("PSU leaked across train/test")
        print(f"Professor methods PSU fold {k+1}/5: fitting", flush=True)
        e, _ = prof.stack_predictions(x.iloc[tr].reset_index(drop=True), a[tr], w[tr],
                                      groups[tr], x.iloc[val], 3, 42+k, prof.NUMERIC)
        train = x.iloc[tr].reset_index(drop=True).copy()
        train["exposure"] = a[tr]
        held = x.iloc[val].copy()
        held["exposure"] = 0
        pred = pd.concat([held, held.assign(exposure=1)], ignore_index=True)
        both, _ = prof.stack_predictions(train, y[tr], w[tr], groups[tr],
                                          pred, 3, 142+k, prof.NUMERIC+["exposure"])
        save_fold(dest, pd.DataFrame({"row_index":val,"fold":k,"e_hat":e,
                                      "mu0_hat":both[:len(val)],"mu1_hat":both[len(val):]}))
        print(f"Saved professor methods PSU fold {k+1}/5", flush=True)
    complete("professor_methods", df, folds)


def compare():
    labels = ("nfhs4", "nfhs5", "professor_methods")
    old_dir = original.PUBLIC
    previous = {label:json.loads((old_dir/f"{label}_uncertainty_review.json").read_text()) for label in labels}
    grouped = {label:json.loads((PUBLIC/f"{label}_uncertainty_review.json").read_text()) for label in labels}
    changed = {}
    for label in labels:
        old, new = previous[label], grouped[label]
        changed[label] = {"old_rd_pp":old["aipw_rd_pp"],"psu_grouped_rd_pp":new["aipw_rd_pp"],
                          "change_pp":new["aipw_rd_pp"]-old["aipw_rd_pp"],
                          "psus_split_old":old["fold_review"]["psus_across_multiple_folds"],
                          "psus_split_new":new["fold_review"]["psus_across_multiple_folds"],
                          "psu_grouped_diagnostic_ci_pp":new["aipw_linearized_diagnostic"]["rd_ci_pp"]}
    delta = grouped["nfhs5"]["aipw_rd_pp"]-grouped["nfhs4"]["aipw_rd_pp"]
    se = np.hypot(grouped["nfhs5"]["aipw_linearized_diagnostic"]["rd_se_pp"],
                  grouped["nfhs4"]["aipw_linearized_diagnostic"]["rd_se_pp"])
    from scipy.stats import norm
    z = norm.ppf(.975)
    summary = {"analyses":changed,"psu_grouped_ext3_change_rd_pp":float(delta),
               "psu_grouped_ext3_change_diagnostic_ci_pp":[float(delta-z*se),float(delta+z*se)],
               "psu_grouped_tmle_rd_pp":grouped["professor_methods"]["tmle_rd_pp"],
               "psu_grouped_tmle_diagnostic_ci_pp":grouped["professor_methods"]["tmle_linearized_diagnostic"]["rd_ci_pp"],
               "note":"Sensitivity to keeping PSUs together in BOTH outer and (for Super Learner) inner nuisance folds. Conditional linearized intervals; not a replacement for survey-design review."}
    original.dump(summary, PUBLIC / "psu_grouped_comparison.json")
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("nfhs4","nfhs5","prof","compare"))
    args=p.parse_args()
    if args.stage in ("nfhs4","nfhs5"): fit_ext3(args.stage)
    elif args.stage == "prof": fit_prof()
    else: compare()


if __name__ == "__main__": main()
