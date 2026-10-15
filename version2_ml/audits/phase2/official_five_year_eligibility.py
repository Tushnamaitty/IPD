#!/usr/bin/env python3
"""Audit DHS five-year eligibility via b19 and CMC, locally on NFHS BR data.

This generates aggregate JSON only. Use b19<60 if available per DHS-7 Guide;
compare against v008-b3<60, do not silently relabel a historical V1 cohort.
"""
from pathlib import Path
import json
import pandas as pd
import pyreadstat

ROOT=Path(__file__).resolve().parents[3]
SOURCES={"NFHS-4":ROOT/"data/raw/nfhs4/IABR74FL.DTA",
         "NFHS-5":ROOT/"data/raw/IABR7EFL.DTA"}
OUT=ROOT/"version2_ml/audits/phase2/outputs/official_five_year_eligibility.json"

def check(path):
    _,meta=pyreadstat.read_dta(str(path),metadataonly=True)
    available=set(meta.column_names)
    needed={"m15","v008","b3"}
    if not needed.issubset(available):
        raise ValueError(f"Missing required columns in {path.name}: {needed-available}")
    b19="b19" in available
    usecols=list(needed|({"b19"} if b19 else set()))
    counts={k:0 for k in (
        "institutional_rows","cmc_0_59","cmc_exact_60","cmc_other",
        "b19_valid_0_59","b19_exact_60","b19_61_plus","b19_missing_or_invalid",
        "cmc_0_59_and_b19_not","b19_0_59_and_cmc_not",
        "cmc_exact_60_and_b19_0_59","cmc_exact_60_and_b19_not",
    )}
    for chunk,_ in pyreadstat.read_file_in_chunks(
            pyreadstat.read_dta,str(path),chunksize=75000,
            usecols=usecols,apply_value_formats=False):
        numeric=lambda name:pd.to_numeric(chunk[name],errors="coerce")
        m15=numeric("m15")
        subset=chunk.loc[m15.between(20,27)|m15.between(30,33)|m15.eq(96)]
        convert=lambda name:pd.to_numeric(subset[name],errors="coerce")
        cmc=convert("v008")-convert("b3")
        cmc_good=cmc.between(0,59)
        counts["institutional_rows"]+=len(subset)
        counts["cmc_0_59"]+=int(cmc_good.sum())
        counts["cmc_exact_60"]+=int(cmc.eq(60).sum())
        counts["cmc_other"]+=int((~cmc.between(0,60)).sum())
        if b19:
            age=convert("b19")
            valid=age.notna() & age.between(0,1200)
            good=valid & age.lt(60)
            counts["b19_valid_0_59"]+=int(good.sum())
            counts["b19_exact_60"]+=int((valid & age.eq(60)).sum())
            counts["b19_61_plus"]+=int((valid & age.gt(60)).sum())
            counts["b19_missing_or_invalid"]+=int((~valid).sum())
            counts["cmc_0_59_and_b19_not"]+=int((cmc_good & ~good).sum())
            counts["b19_0_59_and_cmc_not"]+=int((good & ~cmc_good).sum())
            counts["cmc_exact_60_and_b19_0_59"]+=int((cmc.eq(60)&good).sum())
            counts["cmc_exact_60_and_b19_not"]+=int((cmc.eq(60)&~good).sum())
    return {"b19_available":b19,"counts":counts}

def main():
    result={"status":"FIVE_YEAR_B19_VS_CMC_AUDIT",
            "official_rule":"DHS-7 guidance: b19<60 when available; otherwise v008-b3<60",
            "cohort_policy_signed_off":False,"waves":{},
            "limitations":["This checks values, not the official NFHS sampling implementation.",
                           "Do not compare model performance across unmatched birth cohorts.",
                           "No survey bootstrap or model fitting."]}
    for wave,source in SOURCES.items():
        if not source.exists():raise FileNotFoundError(source)
        result["waves"][wave]=check(source)
        print(wave,result["waves"][wave])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("Saved:",OUT)

if __name__=="__main__":main()
