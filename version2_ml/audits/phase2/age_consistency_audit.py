#!/usr/bin/env python3
"""Investigate interview age versus age-at-birth rounding in NFHS BR files.

Produces aggregate JSON only. Does not modify raw files, train models, or
certify feature timing. Run locally from any working directory.
"""
from pathlib import Path
import json
import pandas as pd
import pyreadstat

ROOT=Path(__file__).resolve().parents[3]
SOURCES={"NFHS-4": ROOT/"data/raw/nfhs4/IABR74FL.DTA",
         "NFHS-5": ROOT/"data/raw/IABR7EFL.DTA"}
OUTPUT=ROOT/"version2_ml/audits/phase2/outputs/age_consistency_audit.json"
USECOLS=["m15","v011","v012","v008","b3"]


def audit(path):
    keys=["institutional","fractional_discrepancy","completed_year_discrepancy",
          "interview_younger_than_age_at_birth_fractional",
          "interview_younger_than_age_at_birth_completed",
          "age_at_interview_cmc_vs_v012_mismatch",
          "age_at_birth_cmc_missing_or_negative",
          "interview_before_birth_cmc","recall_60"]
    result=dict.fromkeys(keys,0)
    for chunk,_ in pyreadstat.read_file_in_chunks(
        pyreadstat.read_dta,str(path),chunksize=75000,
        usecols=USECOLS,apply_value_formats=False):
        num=lambda c: pd.to_numeric(chunk[c],errors="coerce")
        facility=num("m15")
        f=chunk.loc[facility.between(20,27)|facility.between(30,33)|facility.eq(96)]
        n=lambda c: pd.to_numeric(f[c],errors="coerce")
        birth_age_months=n("b3")-n("v011")
        interview_age_months=n("v008")-n("v011")
        completed_birth_age=(birth_age_months//12)
        completed_interview_age=(interview_age_months//12)
        stated_age=n("v012")
        result["institutional"]+=len(f)
        result["fractional_discrepancy"]+=int((stated_age.lt(birth_age_months/12)).sum())
        result["completed_year_discrepancy"]+=int((stated_age.lt(completed_birth_age)).sum())
        result["interview_younger_than_age_at_birth_fractional"]+=int(
            (completed_interview_age.lt(birth_age_months/12)).sum())
        result["interview_younger_than_age_at_birth_completed"]+=int(
            (completed_interview_age.lt(completed_birth_age)).sum())
        result["age_at_interview_cmc_vs_v012_mismatch"]+=int(
            ((completed_interview_age.ne(stated_age)) &
             completed_interview_age.notna() & stated_age.notna()).sum())
        result["age_at_birth_cmc_missing_or_negative"]+=int(
            (birth_age_months.isna()|birth_age_months.lt(0)).sum())
        result["interview_before_birth_cmc"]+=int((n("v008").lt(n("b3"))).sum())
        result["recall_60"]+=int((n("v008")-n("b3")).eq(60).sum())
    return result


def main():
    results={"status":"AGGREGATE_ROUNDING_DIAGNOSTIC_ONLY",
             "waves":{},"age_feature_fully_approved":False,
             "notes":[
                 "v012 is completed interview years; comparison with fractional birth years is not like-for-like.",
                 "CMC-based completed ages provide diagnostic checks, not proof of accurate dates or interview timing.",
                 "Interview v012 vs derived CMC age may differ from CMC/imputation conventions.",
                 "No survey inferential analysis or questionnaire review performed."]}
    for wave,path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Local NFHS file not found: {path}")
        results["waves"][wave]=audit(path)
        print(wave,results["waves"][wave]["institutional"],"institutional records")
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print("Saved:",OUTPUT)


if __name__=="__main__":
    main()
