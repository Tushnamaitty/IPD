#!/usr/bin/env python3
"""Local NFHS-5 aggregate split diagnostics; never export row-level records."""
from pathlib import Path
import sys, json
import pandas as pd
import pyreadstat

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"version2_ml"))
from split_contracts import grouped_partition, audit_partition

SOURCE=ROOT/"data/raw/IABR7EFL.DTA"
OUTPUT=ROOT/"version2_ml/audits/phase4/outputs/step2_split_design_diagnostics.json"
COLS=["caseid","bidx","m15","m17","b19","v024","v001","v022","v005"]

def main():
    if not SOURCE.is_file():raise FileNotFoundError(SOURCE)
    batches=[]
    for df,_ in pyreadstat.read_file_in_chunks(pyreadstat.read_dta,str(SOURCE),
              chunksize=75000,usecols=COLS,apply_value_formats=False):
        n=lambda c:pd.to_numeric(df[c],errors="coerce")
        mask=(n("m15").between(20,27)|n("m15").between(30,33)|n("m15").eq(96)) & n("b19").between(0,59)
        batches.append(df.loc[mask,COLS].copy())
    df=pd.concat(batches,ignore_index=True)
    part=grouped_partition(df,seed=42)
    verified=audit_partition(df,part)
    df["split"]=part.to_numpy()
    f=pd.to_numeric(df["m15"],errors="coerce")
    df["sector"]=f.map(lambda x: "public" if 20<=x<=27 else ("private" if 30<=x<=33 else "other"))
    df["state_code"]=df["v024"].astype("string")
    df["outcome"]=pd.to_numeric(df["m17"],errors="coerce")
    coverage={}
    for split in ("train","validation","test"):
        subset=df.loc[df["split"].eq(split)]
        coverage[split]={
            "n":int(len(subset)),
            "states":int(subset["state_code"].nunique()),
            "psu_stratum_pairs":int(len(set(zip(subset["v022"],subset["v001"])))),
            "strata":int(subset["v022"].nunique()),
            "missing_psu":int(subset["v001"].isna().sum()),
            "missing_stratum":int(subset["v022"].isna().sum()),
        }
    groups={}
    warnings=[]
    for kind,col in (("sector","sector"),("state","state_code")):
        groups[kind]=[]
        for category,subset in df.groupby(col,dropna=False):
            counts={split:{"n":int((subset["split"]==split).sum()),
                           "positive":int(subset.loc[subset["split"]==split,"outcome"].sum())}
                    for split in ("train","validation","test")}
            if min(d["n"] for d in counts.values())<30:
                warnings.append(f"{kind}: category with <30 observations in a partition; suppressed")
                continue
            groups[kind].append({"category":str(category),"by_partition":counts})
    report={
        "status":"AGGREGATE_SPLIT_DIAGNOSTICS_REVIEW_REQUIRED",
        "n":int(len(df)),
        "group_exclusivity":verified["status"],
        "partition_summary":verified["summary"],
        "coverage":coverage,
        "categories":groups,
        "warnings":warnings,
        "limitations":[
            "Provisional mother-grouped random split, not an unseen-state validation split.",
            "PSU/stratum coverage is descriptive only; no design-based inference validated.",
            "Small groups suppressed, but publication disclosure review still required.",
            "No claim of independent confirmatory testing: historical V1 test already inspected.",
            "No row-level records, PSU IDs or mother IDs written."
        ]
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"Processed {len(df)} births; report saved: {OUTPUT}")
    print("Warnings:",len(warnings))

if __name__=="__main__":
    main()
