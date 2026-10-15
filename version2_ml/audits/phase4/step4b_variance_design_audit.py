#!/usr/bin/env python3
"""Step 4b: design-support diagnostic on original F0 held-out V1 fold.

No bootstrap/CI computation. Summarizes paired-PSU, singleton, zero-event
cluster, and weight support without exporting IDs, microdata, or predictions.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/"data/processed/df_model_v1.parquet"
OUT=ROOT/"version2_ml/audits/phase4/outputs/step4b_variance_design_audit.json"

def main():
    if not SRC.is_file():raise FileNotFoundError(SRC)
    df=pd.read_parquet(SRC)
    needed=["respondent_id","csection","primary_sampling_unit",
            "sample_stratum_v022","sample_weight"]
    if any(c not in df for c in needed):
        raise ValueError("Missing required columns: "+str([c for c in needed if c not in df]))
    y=pd.to_numeric(df["csection"],errors="raise").astype(int)
    groups=df["respondent_id"]
    # Historical notebook used X (36 predictors), y, groups and shuffled stratified fold.
    tr,te=next(StratifiedGroupKFold(n_splits=5,shuffle=True,
                   random_state=42).split(np.zeros((len(df),1)),y,groups))
    sample=df.iloc[te].copy()
    stratum=sample["sample_stratum_v022"]
    psu=sample["primary_sampling_unit"]
    w=pd.to_numeric(sample["sample_weight"],errors="coerce")
    valid=(stratum.notna() & psu.notna() & w.notna() & np.isfinite(w) & (w>0))
    if not valid.all():raise ValueError("Invalid stratum/PSU/weight in held-out fold")
    weight_values=w.to_numpy(dtype=np.float64)
    kish=float(weight_values.sum()**2/np.dot(weight_values,weight_values))
    if not np.isfinite(kish) or not (0 < kish <= len(weight_values)):
        raise ValueError("Invalid Kish effective sample size")
    units=pd.DataFrame({"s":stratum.to_numpy(),"p":psu.to_numpy(),
                        "y":y.iloc[te].to_numpy(),"w":w.to_numpy()})
    cluster=units.groupby(["s","p"],dropna=False).agg(
        births=("y","size"),positives=("y","sum"),weight_sum=("w","sum")).reset_index()
    scount=cluster.groupby("s").size()
    single=int(scount.eq(1).sum())
    pairplus=int(scount.ge(2).sum())
    total=int(scount.size)
    report={
      "status":"VARIANCE_DESIGN_SUPPORT_DIAGNOSTIC_ONLY",
      "historical_holdout_births":int(len(sample)),
      "historical_holdout_mothers":int(groups.iloc[te].nunique()),
      "strata":total,"strata_at_least_two_psus":pairplus,
      "singleton_psu_strata":single,
      "singleton_strata_fraction":float(single/total),
      "distinct_psu_stratum_pairs":int(len(cluster)),
      "psu_pairs_no_positive_births":int(cluster["positives"].eq(0).sum()),
      "psu_pairs_all_positive_births":int(cluster["positives"].eq(cluster["births"]).sum()),
      "weight_kish_effective_births":kish,
      "max_births_in_psu":int(cluster["births"].max()),
      "max_psu_weight_share":float(cluster["weight_sum"].max()/cluster["weight_sum"].sum()),
      "variance_estimator_approved":False,
      "confidence_intervals_computed":False,
      "analysis_limitations":[
        "First grouped V1 fold reused, not an independent population probability subsample.",
        "Survey sample has multistage design; v022+PSU identification may not fully encode design.",
        "Sampling with replacement within observed test fold ignores original draw and selection.",
        "Singleton PSU strata require documented variance rule; no automatic zero-variance fix.",
        "Kish n ignores clustering and stratification.",
        "A bootstrap of model predictions with fixed model captures evaluation variation only; refitting changes estimand.",
        "No row-level identifiers, PSU mapping, or small state subgroup counts are saved."
      ]
    }
    previous=ROOT/"version2_ml/audits/phase4/outputs/step4_weighted_metrics_pilot.json"
    if previous.is_file():
        earlier=json.loads(previous.read_text(encoding="utf-8"))
        reference=earlier.get("test_weight_effective_n_kish")
        if reference is not None:
            report["comparison_step4_kish_absolute_difference"]=float(abs(kish-float(reference)))
            report["comparison_step4_kish_matches"]=bool(np.isclose(kish,float(reference),rtol=1e-10,atol=1e-7))
            if not report["comparison_step4_kish_matches"]:
                report["status"]="KISH_MISMATCH_INVESTIGATE"
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+chr(10),encoding="utf-8")
    print("Saved:",OUT)
    print("Strata:",total,"singleton:",single,"PSU pairs:",len(cluster))
    print("Variance estimator approved: NO")
if __name__=="__main__":main()
