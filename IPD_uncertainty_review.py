"""IPD uncertainty review. Run from the IPD repository root on authorized local data.

Private row-level files go only to IPD_uncertainty_private/ inside this repo.
Aggregate diagnostics go to extensions_work/09_uncertainty_review/outputs_final/.
This is a diagnostic audit, not an automatic replacement for manuscript CIs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "extensions_work" / "shared"),
                str(ROOT / "extensions_work" / "03_nfhs4_nfhs5_temporal"),
                str(ROOT / "extensions_work" / "06_professor_methods")]
import config
import utils
import state_crosswalk
import analysis as professor

PRIVATE = ROOT / "IPD_uncertainty_private" / "final_review"
PUBLIC = ROOT / "extensions_work" / "09_uncertainty_review" / "outputs_final"
for directory in (PRIVATE, PUBLIC):
    directory.mkdir(parents=True, exist_ok=True)


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    tmp.replace(path)


def load_round(round_name):
    path = config.NFHS4_PROCESSED_DATA_PATH if round_name == "nfhs4" else config.V2_PROCESSED_DATA_PATH
    df, total, other = utils.load_public_private_analytic(path)
    expected = (195366, 195997, 631) if round_name == "nfhs4" else (200794, 201311, 517)
    if (len(df), total, other) != expected:
        raise ValueError(f"Wrong {round_name} cohort: {(len(df), total, other)}, expected {expected}")
    return state_crosswalk.harmonize_state(df, "NFHS-4" if round_name == "nfhs4" else "NFHS-5")


def fit_ext3(round_name):
    """Same 5-fold estimator as Extension 3, with one atomic checkpoint per fold."""
    df = load_round(round_name)
    cols = list(config.CONFOUNDER_COLS)
    W = df[cols].copy()
    for col in config.CATEGORICAL_CONFOUNDERS:
        W[col] = W[col].astype("string").fillna("Missing").astype(str)
    a = df.exposure.to_numpy(dtype=int)
    y = df.csection.to_numpy(dtype=int)
    w = df.sample_weight_normalized.to_numpy(dtype=float)
    groups = df.respondent_id.to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(W, a, groups))
    target = PRIVATE / round_name
    target.mkdir(exist_ok=True)
    for k, (tr, val) in enumerate(folds):
        dest = target / f"fold_{k}.parquet"
        if dest.exists():
            saved = pd.read_parquet(dest)
            if np.array_equal(saved.row_index.to_numpy(), val) and len(saved) == len(val):
                print(f"{round_name} fold {k + 1}/5: loaded checkpoint", flush=True)
                continue
            raise ValueError(f"Existing checkpoint {dest} has different row indices; stop and investigate")
        if len(set(groups[tr]) & set(groups[val])):
            raise AssertionError("Respondent leaked across folds")
        prop = utils.make_propensity_pipeline(config.NUMERIC_CONFOUNDERS, config.CATEGORICAL_CONFOUNDERS, 42)
        prop.fit(W.iloc[tr], a[tr], clf__sample_weight=w[tr])
        e = prop.predict_proba(W.iloc[val])[:, 1]
        outcome = utils.make_outcome_pipeline(config.NUMERIC_CONFOUNDERS, config.CATEGORICAL_CONFOUNDERS, 42)
        trW = W.iloc[tr].copy()
        trW["exposure"] = a[tr]
        outcome.fit(trW, y[tr], clf__sample_weight=w[tr])
        valW = W.iloc[val].copy()
        valW["exposure"] = 1
        mu1 = outcome.predict_proba(valW)[:, 1]
        valW["exposure"] = 0
        mu0 = outcome.predict_proba(valW)[:, 1]
        checkpoint = pd.DataFrame({"row_index": val, "fold": k, "e_hat": e, "mu1_hat": mu1, "mu0_hat": mu0})
        tmp = dest.with_suffix(".tmp.parquet")
        checkpoint.to_parquet(tmp, index=False)
        tmp.replace(dest)
        print(f"{round_name} fold {k + 1}/5: saved {dest}", flush=True)
    pred = pd.concat([pd.read_parquet(target / f"fold_{k}.parquet") for k in range(5)], ignore_index=True)
    pred = pred.sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(pred.row_index.to_numpy(), np.arange(len(df))):
        raise ValueError("Fold checkpoints do not cover each analytic row exactly once")
    final = pd.DataFrame({"exposure": a, "csection": y, "sample_weight_normalized": w,
        "cluster_number": df.cluster_number.to_numpy(), "sample_stratum_v022": df.sample_stratum_v022.to_numpy(),
        "respondent_id": groups, "fold": pred.fold, "e_hat": np.clip(pred.e_hat, 1e-6, 1 - 1e-6),
        "mu1_hat": pred.mu1_hat, "mu0_hat": pred.mu0_hat})
    dest = target / "rows.parquet"
    final.to_parquet(dest, index=False)
    print(f"Saved private row file: {dest}", flush=True)
    review(round_name, dest)


def fit_prof():
    """Exact professor-methods crossfit architecture, saving completed outer folds."""
    data = config.V2_PROCESSED_DATA_PATH
    df = pd.read_parquet(data)
    df = df.loc[df.facility_type.isin(["public", "private"])].reset_index(drop=True)
    if len(df) != 200794:
        raise ValueError(f"Expected 200794 rows, got {len(df)}")
    x = professor.make_features(df)
    a = (df.facility_type == "private").to_numpy(dtype=int)
    y = df.csection.to_numpy(dtype=int)
    w = df.sample_weight_normalized.to_numpy(dtype=float)
    groups = df.respondent_id.to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(x, a, groups))
    target = PRIVATE / "professor_methods"
    target.mkdir(exist_ok=True)
    for k, (tr, val) in enumerate(folds):
        dest = target / f"fold_{k}.parquet"
        if dest.exists():
            saved = pd.read_parquet(dest)
            if np.array_equal(saved.row_index.to_numpy(), val) and len(saved) == len(val):
                print(f"Professor methods fold {k + 1}/5: loaded checkpoint", flush=True)
                continue
            raise ValueError(f"Checkpoint {dest} differs from current split")
        if len(set(groups[tr]) & set(groups[val])):
            raise AssertionError("Respondent leaked across folds")
        print(f"Professor methods fold {k + 1}/5: fitting", flush=True)
        e, _ = professor.stack_predictions(x.iloc[tr].reset_index(drop=True), a[tr], w[tr], groups[tr],
                                            x.iloc[val], 3, 42 + k, professor.NUMERIC)
        train = x.iloc[tr].copy().reset_index(drop=True)
        train["exposure"] = a[tr]
        val_x = x.iloc[val].copy()
        val_x["exposure"] = 0
        both, _ = professor.stack_predictions(train, y[tr], w[tr], groups[tr],
                         pd.concat([val_x, val_x.assign(exposure=1)], ignore_index=True),
                         3, 142 + k, professor.NUMERIC + ["exposure"])
        checkpoint = pd.DataFrame({"row_index": val, "fold": k, "e_hat": e,
                                  "mu0_hat": both[:len(val)], "mu1_hat": both[len(val):]})
        tmp = dest.with_suffix(".tmp.parquet")
        checkpoint.to_parquet(tmp, index=False)
        tmp.replace(dest)
        print(f"Professor methods fold {k + 1}/5: saved {dest}", flush=True)
    pred = pd.concat([pd.read_parquet(target / f"fold_{k}.parquet") for k in range(5)], ignore_index=True)
    pred = pred.sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(pred.row_index.to_numpy(), np.arange(len(df))):
        raise ValueError("Incomplete fold coverage")
    final = pd.DataFrame({"exposure": a, "csection": y, "sample_weight_normalized": w,
        "cluster_number": df.cluster_number.to_numpy(), "sample_stratum_v022": df.sample_stratum_v022.to_numpy(),
        "respondent_id": groups, "fold": pred.fold, "e_hat": pred.e_hat,
        "mu1_hat": pred.mu1_hat, "mu0_hat": pred.mu0_hat})
    dest = target / "rows.parquet"
    final.to_parquet(dest, index=False)
    print(f"Saved private row file: {dest}", flush=True)
    review("professor_methods", dest)


def ci_from_scores(df, s_rd, s_logrr, rd, rr):
    """With-replacement stratified PSU linearization; singleton contribution omitted and flagged."""
    aggregate = pd.DataFrame({"stratum": df.sample_stratum_v022.to_numpy(),
                              "psu": df.cluster_number.to_numpy(), "rd": s_rd, "logrr": s_logrr})
    pairs = aggregate.groupby(["stratum", "psu"], sort=False, dropna=False)[["rd", "logrr"]].sum().reset_index()
    counts = pairs.groupby("stratum", dropna=False).size()
    variances = {}
    for col in ("rd", "logrr"):
        v = 0.0
        for _, block in pairs.groupby("stratum", dropna=False):
            n = len(block)
            if n > 1:
                v += n / (n - 1) * np.square(block[col] - block[col].mean()).sum()
        variances[col] = float(v)
    z = norm.ppf(.975)
    se_rd, se_log = np.sqrt(variances["rd"]), np.sqrt(variances["logrr"])
    return {"rd_ci_pp": [100 * (rd-z*se_rd), 100 * (rd+z*se_rd)],
            "rr_ci": [float(rr*np.exp(-z*se_log)), float(rr*np.exp(z*se_log))],
            "rd_se_pp": 100*se_rd, "log_rr_se": se_log,
            "psus": len(pairs), "strata": len(counts), "singleton_strata": int((counts==1).sum()),
            "singleton_note": "Singleton strata contribute zero to this diagnostic; confirm with separate design sensitivity."}


def leakage(df):
    pairs = df[["sample_stratum_v022", "cluster_number", "fold"]].drop_duplicates()
    n = pairs.groupby(["sample_stratum_v022", "cluster_number"], dropna=False).fold.nunique()
    return {"psu_count": len(n), "psus_across_multiple_folds": int((n>1).sum()),
            "interpretation": "Respondent grouping does not guarantee PSU-level fold independence."}


def review(label, path):
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    cols = ["exposure", "csection", "sample_weight_normalized", "cluster_number", "sample_stratum_v022",
            "e_hat", "mu1_hat", "mu0_hat"]
    if "e_hat" not in df and "e_hat_clipped" in df:
        df["e_hat"] = df.e_hat_clipped
    if df[cols].isna().any().any():
        raise ValueError(f"Missing values in required review columns of {path}")
    a, y = df.exposure.to_numpy(dtype=int), df.csection.to_numpy(dtype=int)
    w = df.sample_weight_normalized.to_numpy(dtype=float)
    if (w<=0).any() or not np.isfinite(w).all():
        raise ValueError("Survey weights must be positive and finite")
    e = np.clip(df.e_hat.to_numpy(dtype=float), .01 if label == "professor_methods" else 1e-6,
                .99 if label == "professor_methods" else 1-1e-6)
    q1, q0 = df.mu1_hat.to_numpy(dtype=float), df.mu0_hat.to_numpy(dtype=float)
    psi1, psi0 = utils.aipw_psi(y, a, e, q1, q0)
    r1, r0 = np.average(psi1, weights=w), np.average(psi0, weights=w)
    rd, rr = r1-r0, r1/r0
    scores_rd = w/w.sum() * ((psi1-r1)-(psi0-r0))
    scores_log = w/w.sum() * ((psi1-r1)/r1-(psi0-r0)/r0)
    diagnostic = ci_from_scores(df, scores_rd, scores_log, rd, rr)
    out = {"label": label, "n": len(df), "private_n": int(a.sum()), "public_n": int((1-a).sum()),
           "aipw_rd_pp": 100*rd, "aipw_rr": rr, "aipw_linearized_diagnostic": diagnostic,
           "n_propensity_outside_0_05_0_95": int(((e < .05)|(e > .95)).sum()),
           "method_note": "Conditional on held-out nuisance predictions; excludes nuisance-training uncertainty and finite-population corrections."}
    if "fold" in df:
        out["fold_review"] = leakage(df)
    if label == "professor_methods":
        est = professor.estimate(a, y, w, e, q0, q1)
        # Cross-fitted TMLE with separate arm updates; influence curve at targeted fits.
        def target(arm, q, g, epsilon):
            from scipy.special import expit, logit
            q = np.clip(q, professor.TINY, 1-professor.TINY)
            return expit(logit(q) + epsilon/g)
        tq1 = target(1, q1, e, est["tmle"]["fluctuation_epsilon_private"])
        tq0 = target(0, q0, 1-e, est["tmle"]["fluctuation_epsilon_public"])
        t1, t0 = est["tmle"]["private_risk"], est["tmle"]["public_risk"]
        ic1 = tq1 + a/e * (y-tq1)
        ic0 = tq0 + (1-a)/(1-e) * (y-tq0)
        tmrd, tmrr = t1-t0, t1/t0
        tm_diag = ci_from_scores(df, w/w.sum()*((ic1-t1)-(ic0-t0)),
                      w/w.sum()*((ic1-t1)/t1-(ic0-t0)/t0), tmrd, tmrr)
        out["tmle_rd_pp"] = 100*tmrd
        out["tmle_rr"] = tmrr
        out["tmle_linearized_diagnostic"] = tm_diag
        out["tmle_target_score_abs_max"] = max(abs(est["tmle"]["target_score_private"]),
                                                 abs(est["tmle"]["target_score_public"]))
    dest = PUBLIC / f"{label}_uncertainty_review.json"
    dump(out, dest)
    print(json.dumps({"label":label, "RD_pp":round(out["aipw_rd_pp"],4),
          "RD_diagnostic_CI_pp": [round(v,4) for v in diagnostic["rd_ci_pp"]],
          "singleton_strata": diagnostic["singleton_strata"], "saved": str(dest)}, indent=2), flush=True)
    return out


def compare():
    required = [PUBLIC / f"{label}_uncertainty_review.json" for label in ("nfhs4", "nfhs5", "v2", "professor_methods")]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run all fit/review stages first. Missing: {missing}")
    a4, a5, v2, prof = [json.loads(p.read_text()) for p in required]
    # Independently sampled rounds: variances add on the RD scale.
    change = a5["aipw_rd_pp"] - a4["aipw_rd_pp"]
    se = np.hypot(a5["aipw_linearized_diagnostic"]["rd_se_pp"],
                  a4["aipw_linearized_diagnostic"]["rd_se_pp"])
    z = norm.ppf(.975)
    out = {"ext3_change_rd_pp": change, "ext3_change_linearized_diagnostic_ci_pp": [change-z*se, change+z*se],
           "v2_aipw_rd_pp": v2["aipw_rd_pp"], "professor_aipw_rd_pp": prof["aipw_rd_pp"],
           "professor_tmle_rd_pp": prof["tmle_rd_pp"],
           "interpretation": "Separate waves; association only. All new intervals are conditional, linearized diagnostics, pending statistical design review.",
           "psu_split_warning": {p["label"]: p.get("fold_review", {}).get("psus_across_multiple_folds")
                                 for p in (a4,a5,prof)}}
    previous = ROOT/"extensions_work/03_nfhs4_nfhs5_temporal/outputs/temporal_aipw_results.csv"
    if previous.exists():
        table = pd.read_csv(previous)
        checks = {}
        for name, current in (("NFHS-4",a4), ("NFHS-5",a5)):
            match = table.loc[(table.survey_round == name) & (table.confounder_set == "primary")]
            if len(match) != 1: raise ValueError(f"Could not identify one primary {name} published row")
            checks[name] = float(current["aipw_rd_pp"] - match.iloc[0].risk_difference_pct)
        out["ext3_previous_rd_difference_pp"] = checks
        out["ext3_matches_previous_within_0_02_pp"] = all(abs(v)<.02 for v in checks.values())
    prof_json = ROOT/"outputs/metadata/professor_methods_sl_tmle.json"
    if prof_json.exists():
        previous_prof = json.loads(prof_json.read_text())
        out["professor_previous_rd_difference_pp"] = {
            "aipw": prof["aipw_rd_pp"]-100*previous_prof["aipw"]["risk_difference"],
            "tmle": prof["tmle_rd_pp"]-100*previous_prof["tmle"]["risk_difference"]}
    main = ROOT/"outputs/tables/b4_aipw_overall_summary.csv"
    if main.exists():
        row = pd.read_csv(main)
        if len(row) != 1: raise ValueError("V2 summary must have one row")
        out["v2_previous_rd_difference_pp"] = float(v2["aipw_rd_pp"]-row.iloc[0].risk_difference_pct)
    dump(out, PUBLIC / "overall_review.json")
    print(json.dumps(out, indent=2))


def crossround():
    rounds = {name: load_round(name) for name in ("nfhs4", "nfhs5")}
    result = {}
    for name, df in rounds.items():
        rows = {}
        for col in config.CONFOUNDER_COLS:
            rows[col] = {"missing_n": int(df[col].isna().sum()),
                         "missing_pct": float(100*df[col].isna().mean()),
                         "distinct_nonmissing": int(df[col].nunique(dropna=True))}
            if col in config.CATEGORICAL_CONFOUNDERS or col == "wealth_index":
                rows[col]["value_counts"] = {str(k): int(v) for k,v in df[col].value_counts(dropna=False).items()}
        result[name] = {"n":len(df), "variables":rows,
            "states_after_harmonization": sorted(df.state.dropna().unique().tolist()),
            "weight_quantiles": np.quantile(df.sample_weight_normalized, [0,.01,.5,.99,1]).tolist(),
            "facility_counts": df.facility_type.value_counts().to_dict()}
    result["same_harmonized_states"] = result["nfhs4"]["states_after_harmonization"] == result["nfhs5"]["states_after_harmonization"]
    # Audit source value-label metadata without loading millions of survey rows.
    try:
        import pyreadstat
        raw = {"nfhs4":ROOT/"data/raw/nfhs4/IABR74FL.DTA", "nfhs5":ROOT/"data/raw/IABR7EFL.DTA"}
        for name,path in raw.items():
            if not path.exists():
                result[name]["raw_codebook"] = f"Not available: {path}"
                continue
            _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
            labels = {}
            for col in ("m15","m17","s116","v024","v025","v130","v133","v190","bord","b0","v001","v021","v022","v023","v005"):
                if col not in meta.column_names:
                    raise ValueError(f"{name}: {col} absent in raw codebook")
                vset = meta.variable_to_label.get(col)
                labels[col] = {"label": meta.column_names_to_labels.get(col),
                               "values": {str(k):str(v) for k,v in meta.value_labels.get(vset,{}).items()}}
            result[name]["raw_codebook"] = labels
    except ImportError:
        result["raw_codebook_note"] = "pyreadstat unavailable; processed-data audit completed, raw metadata audit skipped"
    dump(result, PUBLIC / "crossround_audit.json")
    print(f"Saved: {PUBLIC/'crossround_audit.json'}; same states: {result['same_harmonized_states']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("crossround","fit-ext3","review-v2","fit-prof","review-existing","compare"))
    p.add_argument("--round", choices=("nfhs4","nfhs5"))
    p.add_argument("--row-file", type=Path, help="Private row-level CSV/parquet file for review-existing")
    p.add_argument("--label", choices=("nfhs4","nfhs5","v2","professor_methods"))
    args = p.parse_args()
    if args.stage == "crossround": crossround()
    elif args.stage == "fit-ext3":
        if not args.round: p.error("fit-ext3 needs --round nfhs4 or --round nfhs5")
        fit_ext3(args.round)
    elif args.stage == "review-v2":
        path = ROOT/"outputs/tables/b4_aipw_row_level_nuisance.csv"
        if not path.exists():
            raise FileNotFoundError(f"V2 notebook row file missing: {path}; do not rerun yet")
        review("v2", path)
    elif args.stage == "fit-prof": fit_prof()
    elif args.stage == "review-existing":
        if args.row_file is None or args.label is None: p.error("Pass --row-file and --label")
        review(args.label, args.row_file)
    else: compare()


if __name__ == "__main__": main()
