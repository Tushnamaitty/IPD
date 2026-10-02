"""Full-design domain and lonely-PSU variance review using saved IPD predictions.

Run from the IPD repository root after IPD_uncertainty_review.py and
IPD_psu_fold_sensitivity.py. No nuisance model is refitted. Private design
checkpoints remain inside IPD_uncertainty_private; only aggregates are output.
These are candidate design-linearized intervals conditional on first-stage
nuisance fits and their regularity assumptions, not automatic publication CIs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.special import expit, logit
from scipy.stats import t

import IPD_uncertainty_review as base

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "IPD_uncertainty_private" / "full_design_ci_review"
OUTPUT = ROOT / "extensions_work" / "09_uncertainty_review" / "outputs_final" / "full_design_ci_review.json"
PRIVATE.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "nfhs4": (ROOT / "data/raw/nfhs4/IABR74FL.DTA", 1315617),
    "nfhs5": (ROOT / "data/raw/IABR7EFL.DTA", 1274250),
}
EXPECTED_ANALYTIC = {"nfhs4": 195366, "nfhs5": 200794}


def full_design(round_name):
    path, expected_rows = SOURCES[round_name]
    if not path.is_file():
        raise FileNotFoundError(f"Raw Birth Recode needed only for PSU/stratum IDs: {path}")
    target = PRIVATE / f"{round_name}_birth_recode_psu_pairs.parquet"
    settings = target.with_suffix(".source.json")
    stat = path.stat()
    provenance = {"source": str(path.resolve()), "size": stat.st_size,
                  "mtime_ns": stat.st_mtime_ns, "raw_rows": expected_rows}
    if target.exists() and settings.exists() and json.loads(settings.read_text()) == provenance:
        pairs = pd.read_parquet(target)
        print(f"Loaded private full-design PSU cache for {round_name}", flush=True)
    else:
        print(f"Reading PSU/stratum columns from {round_name} raw Birth Recode", flush=True)
        raw, _ = pyreadstat.read_dta(str(path), usecols=["v001", "v022"])
        if len(raw) != expected_rows:
            raise ValueError(f"Wrong {round_name} raw file: {len(raw)} rows; expected {expected_rows}")
        if raw[["v001", "v022"]].isna().any().any():
            raise ValueError(f"Missing full-survey PSU/stratum ID in {round_name}")
        pairs = (raw[["v022", "v001"]].drop_duplicates()
                 .rename(columns={"v022":"stratum", "v001":"psu"})
                 .reset_index(drop=True))
        tmp = target.with_suffix(".tmp.parquet")
        pairs.to_parquet(tmp, index=False)
        tmp.replace(target)
        settings.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"Saved private full-design PSU cache for {round_name}", flush=True)
    if pairs.duplicated(["stratum", "psu"]).any():
        raise AssertionError("Duplicate PSU pair in full-design cache")
    count = pairs.groupby("stratum").size()
    return pairs, {"psu_pairs": len(pairs), "strata": len(count),
                   "full_design_singleton_strata": int((count == 1).sum()),
                   "source_birth_recode_rows": expected_rows}


def influence_components(df, method):
    a = df.exposure.to_numpy(dtype=int)
    y = df.csection.to_numpy(dtype=int)
    w = df.sample_weight_normalized.to_numpy(dtype=float)
    if (w <= 0).any() or not np.isfinite(w).all():
        raise ValueError("Invalid survey weights")
    e = np.clip(df.e_hat.to_numpy(dtype=float),
                .01 if "professor" in method else 1e-6,
                .99 if "professor" in method else 1-1e-6)
    q1 = df.mu1_hat.to_numpy(dtype=float)
    q0 = df.mu0_hat.to_numpy(dtype=float)
    if method == "professor_tmle":
        fitted = base.professor.estimate(a, y, w, e, q0, q1)["tmle"]
        q1 = np.clip(q1, base.professor.TINY, 1-base.professor.TINY)
        q0 = np.clip(q0, base.professor.TINY, 1-base.professor.TINY)
        q1 = expit(logit(q1) + fitted["fluctuation_epsilon_private"] / e)
        q0 = expit(logit(q0) + fitted["fluctuation_epsilon_public"] / (1-e))
        r1, r0 = fitted["private_risk"], fitted["public_risk"]
    else:
        r1 = r0 = None
    psi1 = q1 + a/e * (y-q1)
    psi0 = q0 + (1-a)/(1-e) * (y-q0)
    if r1 is None:
        r1, r0 = np.average(psi1, weights=w), np.average(psi0, weights=w)
    rd = float(r1-r0)
    rr = float(r1/r0)
    score1 = psi1-r1
    score0 = psi0-r0
    return {
        "rd": rd, "rr": rr,
        "rd_score": w/w.sum() * (score1-score0),
        "logrr_score": w/w.sum() * (score1/r1-score0/r0),
    }


def variance_with_full_design(full_pairs, analytic, row_score):
    """Domain zeros and stratum PSU centering; no FPC is assumed available."""
    contrib = pd.DataFrame({"stratum":analytic.sample_stratum_v022.to_numpy(),
                            "psu":analytic.cluster_number.to_numpy(),
                            "score":row_score})
    totals = contrib.groupby(["stratum", "psu"], as_index=False)["score"].sum()
    merged = full_pairs.merge(totals, on=["stratum", "psu"], how="left", validate="one_to_one", indicator=True)
    if len(totals) != int(merged.score.notna().sum()):
        raise ValueError("Analytic PSU not present in its raw Birth Recode design")
    merged["score"] = merged.score.fillna(0.0)
    g = merged.groupby("stratum").score.agg(["size", "mean", "sum", "count"])
    sumsquares = merged.assign(square=merged.score.pow(2)).groupby("stratum").square.sum()
    n = g["size"]
    nonlonely = n > 1
    centered_sumsq = (sumsquares - n*g["mean"].pow(2)).clip(lower=0)
    per_stratum = (n[nonlonely] / (n[nonlonely]-1) * centered_sumsq[nonlonely])
    basevar = float(per_stratum.sum())
    lonely = g.loc[~nonlonely]
    grand_mean = float(merged.score.mean())
    adjustvar = basevar + float(((lonely["sum"]-grand_mean)**2).sum())
    averagevar = basevar + len(lonely) * float(per_stratum.mean())
    analytic_counts = totals.groupby("stratum").size()
    analytic_singletons = analytic_counts[analytic_counts == 1]
    full_counts = n.reindex(analytic_singletons.index)
    singleton_pairs = merged.loc[merged.stratum.isin(n[n==1].index), ["stratum", "psu"]]
    singleton_rows = analytic.merge(singleton_pairs,
        left_on=["sample_stratum_v022", "cluster_number"],
        right_on=["stratum", "psu"], how="inner")
    w = analytic.sample_weight_normalized.to_numpy(dtype=float)
    df_design = len(full_pairs)-len(n)
    if df_design < 1:
        raise ValueError("Insufficient survey design degrees of freedom")
    return {
        "variances": {"remove":basevar, "adjust":adjustvar, "average":averagevar},
        "analytic_psu_pairs": len(totals),
        "zero_domain_psu_pairs": int(len(full_pairs)-len(totals)),
        "analytic_singleton_strata": len(analytic_singletons),
        "analytic_singletons_with_other_full_design_psus": int((full_counts>1).sum()),
        "full_design_singleton_analytic_births": len(singleton_rows),
        "full_design_singleton_weight_share_pct": float(100*singleton_rows.sample_weight_normalized.sum()/w.sum()),
        "design_df": int(df_design),
    }


def assess(label, round_name, path, full_pairs):
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if len(frame) != EXPECTED_ANALYTIC[round_name]:
        raise ValueError(f"Wrong cohort in {path}")
    if "e_hat" not in frame and "e_hat_clipped" in frame:
        frame["e_hat"] = frame.e_hat_clipped
    required = ["sample_stratum_v022","cluster_number","sample_weight_normalized",
                "exposure","csection","e_hat","mu0_hat","mu1_hat"]
    if frame[required].isna().any().any():
        raise ValueError(f"Missing required fields: {path}")
    method = "professor_tmle" if "professor_tmle" in label else ("professor_aipw" if "professor" in label else "aipw")
    comp = influence_components(frame, method)
    out = {"label":label,"n":len(frame), "rd_pp":100*comp["rd"], "rr":comp["rr"],
           "raw_source_round":round_name, "ci_method":"stratified full-Birth-Recode-PSU design linearization of cross-fitted influence scores; t critical, df=PSUs-strata; no FPC"}
    rd = variance_with_full_design(full_pairs, frame, comp["rd_score"])
    logrr = variance_with_full_design(full_pairs, frame, comp["logrr_score"])
    if (rd["analytic_psu_pairs"],rd["analytic_singleton_strata"]) != (
        logrr["analytic_psu_pairs"],logrr["analytic_singleton_strata"]):
        raise AssertionError("Inconsistent RD/RR design dimensions")
    crit = t.ppf(.975, rd["design_df"])
    out.update({k:v for k,v in rd.items() if k!="variances"})
    out["intervals_by_singleton_rule"] = {}
    for rule in ("remove","adjust","average"):
        se_rd = np.sqrt(rd["variances"][rule]); se_log = np.sqrt(logrr["variances"][rule])
        out["intervals_by_singleton_rule"][rule] = {
            "rd_ci_pp":[100*(comp["rd"]-crit*se_rd),100*(comp["rd"]+crit*se_rd)],
            "rr_ci":[comp["rr"]*np.exp(-crit*se_log),comp["rr"]*np.exp(crit*se_log)],
            "rd_se_pp":float(100*se_rd), "log_rr_se":float(se_log)}
    print(f"{label}: RD={out['rd_pp']:.4f} pp; adjusted lonely-PSU CI={out['intervals_by_singleton_rule']['adjust']['rd_ci_pp']}",flush=True)
    return out


def main():
    skeletons={}; designs={}
    for round_name in ("nfhs4","nfhs5"):
        skeletons[round_name],designs[round_name]=full_design(round_name)
    files={
        "ext3_nfhs4_original":("nfhs4",ROOT/"IPD_uncertainty_private/final_review/nfhs4/rows.parquet"),
        "ext3_nfhs5_original":("nfhs5",ROOT/"IPD_uncertainty_private/final_review/nfhs5/rows.parquet"),
        "ext3_nfhs4_psu_grouped":("nfhs4",ROOT/"IPD_uncertainty_private/final_review_psu_grouped/nfhs4/rows.parquet"),
        "ext3_nfhs5_psu_grouped":("nfhs5",ROOT/"IPD_uncertainty_private/final_review_psu_grouped/nfhs5/rows.parquet"),
        "v2_original":("nfhs5",ROOT/"outputs/tables/b4_aipw_row_level_nuisance.csv"),
        "professor_aipw_original":("nfhs5",ROOT/"IPD_uncertainty_private/final_review/professor_methods/rows.parquet"),
        "professor_tmle_original":("nfhs5",ROOT/"IPD_uncertainty_private/final_review/professor_methods/rows.parquet"),
        "professor_aipw_psu_grouped":("nfhs5",ROOT/"IPD_uncertainty_private/final_review_psu_grouped/professor_methods/rows.parquet"),
        "professor_tmle_psu_grouped":("nfhs5",ROOT/"IPD_uncertainty_private/final_review_psu_grouped/professor_methods/rows.parquet"),
    }
    missing=[str(p) for _,p in files.values() if not p.is_file()]
    if missing: raise FileNotFoundError(f"Private row prediction files missing: {sorted(set(missing))}")
    output={"full_design":designs,"estimates":{},
            "limitations":["Birth Recode PSU frame may omit PSUs with no recent births; compare to the full NFHS sampling frame if available.",
             "Finite population corrections and any additional sampling stages not supplied.",
             "Influence interval relies on adequate nuisance convergence and valid PSU-level sampling approximations; this script does not prove those conditions.",
             "Singleton-stratum rules are sensitivity choices unless certainty PSU status is verified."]}
    for label,(round_name,path) in files.items():
        output["estimates"][label]=assess(label,round_name,path,skeletons[round_name])
    output["ext3_change"]={}
    for version in ("original","psu_grouped"):
        a=output["estimates"][f"ext3_nfhs4_{version}"]
        b=output["estimates"][f"ext3_nfhs5_{version}"]
        delta=b["rd_pp"]-a["rd_pp"]
        by_rule={}
        for rule in ("remove","adjust","average"):
            se=np.hypot(a["intervals_by_singleton_rule"][rule]["rd_se_pp"],
                        b["intervals_by_singleton_rule"][rule]["rd_se_pp"])
            df=min(a["design_df"],b["design_df"])
            crit=t.ppf(.975,df)
            by_rule[rule]={"change_rd_pp":delta,"change_rd_ci_pp":[delta-crit*se,delta+crit*se]}
        output["ext3_change"][version]=by_rule
    base.dump(output,OUTPUT)
    print(json.dumps({"full_design":designs,
        "ext3_change_psu_grouped_adjust":output["ext3_change"]["psu_grouped"]["adjust"],
        "v2_original_adjust":output["estimates"]["v2_original"]["intervals_by_singleton_rule"]["adjust"]["rd_ci_pp"],
        "tmle_grouped_adjust":output["estimates"]["professor_tmle_psu_grouped"]["intervals_by_singleton_rule"]["adjust"]["rd_ci_pp"],
        "saved":str(OUTPUT)},indent=2))


if __name__=="__main__":main()
