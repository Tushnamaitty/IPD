#!/usr/bin/env python3
"""Phase 4 Step 1: local real NFHS-5 mother-grouped split pilot; aggregates only.

Reads only minimal columns in chunks; no rows, mother IDs, or assignments saved.
Proportions 70/15/15 are development PILOT, not final split freeze.
"""
from pathlib import Path
import sys,json
import pandas as pd
import pyreadstat

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"version2_ml"))
from split_contracts import grouped_partition,audit_partition
SOURCE=ROOT/"data/raw/IABR7EFL.DTA"
OUT=ROOT/"version2_ml/audits/phase4/outputs/step1_grouped_split_pilot.json"

def main():
    if not SOURCE.is_file():raise FileNotFoundError(SOURCE)
    chunks=[]
    use=["caseid","bidx","m15","m17","b19"]
    for c,_ in pyreadstat.read_file_in_chunks(pyreadstat.read_dta,str(SOURCE),
                    chunksize=75000,usecols=use,apply_value_formats=False):
        fac=pd.to_numeric(c["m15"],errors="coerce")
        age=pd.to_numeric(c["b19"],errors="coerce")
        mask=(fac.between(20,27)|fac.between(30,33)|fac.eq(96))&age.between(0,59)
        rows=c.loc[mask,["caseid","bidx","m17"]].copy()
        chunks.append(rows)
    data=pd.concat(chunks,ignore_index=True)
    keys=data[["caseid","bidx"]]
    if keys.duplicated().any():raise ValueError("Duplicate birth keys; aborting")
    groups=data["caseid"].astype("string")
    if groups.isna().any() or groups.str.strip().eq("").any():
        raise ValueError("Missing mother key")
    partition=grouped_partition(data,group_col="caseid",target_col="m17",seed=42)
    check=audit_partition(data,partition,group_col="caseid",target_col="m17")
    report={
       "status":"REAL_SPLIT_PILOT_VALID" if check["status"]=="GROUP_EXCLUSIVE_VALID" else "INVALID",
       "cohort_n":int(len(data)), "mother_n":int(groups.nunique()),
       "split":check["summary"],
       "identity_overlap_guard":"passed",
       "split_rule":"provisional 70/15/15, grouped by caseid, hash seed 42, stratified by mother outcome mean",
       "final_test_independence_certified":False,
       "survey_bootstrap_validated":False,
       "notes":["Original V1 test performance was already examined; don't call new partition independent confirmation.",
                "Pilot checks sample sizes and mother exclusivity, not state/PSU/stratum balance.",
                "Never commit per-mother or per-birth assignments in public repo.",
                "Splitting groups by outcome mean is only a development heuristic; final design must be frozen before confirmatory claims."]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    print("Saved:",OUT)

if __name__=="__main__":main()
