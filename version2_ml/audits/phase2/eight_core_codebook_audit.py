#!/usr/bin/env python3
"""Local Phase 2 eight-core metadata and categorical-code audit.

Requires pyreadstat. Writes only aggregate JSON, never row-level NFHS data.
Value-label dictionaries from DHS files are summarized; semantic equivalence
still requires manual review of questionnaires, especially interview timing.
"""
from pathlib import Path
import json
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {"NFHS-4": ROOT / "data/raw/nfhs4/IABR74FL.DTA",
           "NFHS-5": ROOT / "data/raw/IABR7EFL.DTA"}
OUT = ROOT / "version2_ml/audits/phase2/outputs/eight_core_codebook_audit.json"
FEATURES = {"education_years":"v133","wealth_index":"v190","birth_order":"bord",
            "facility_type":"m15","residence":"v025","religion":"v130",
            "twin_order":"b0"}
COLUMNS = ["m15","b3","v011",*FEATURES.values()]


def audit(path):
    _, metadata = pyreadstat.read_dta(str(path), metadataonly=True)
    available = set(metadata.column_names)
    missing = sorted(set(COLUMNS)-available)
    if missing:
        raise ValueError(f"Missing fields in {path.name}: {missing}")
    labels = dict(zip(metadata.column_names, metadata.column_labels))
    output = {"institutional_n":0,"age_at_birth_missing_or_negative":0,
              "fields":{}}
    for feature, raw in FEATURES.items():
        output["fields"][feature] = {
            "raw_variable":raw, "label":labels.get(raw),
            "defined_value_labels":{str(k):str(v) for k,v in
                metadata.variable_value_labels.get(raw,{}).items()},
            "missing":0,"observed_values":{}
        }
    for chunk,_ in pyreadstat.read_file_in_chunks(
        pyreadstat.read_dta,str(path),chunksize=75000,
        usecols=COLUMNS,apply_value_formats=False):
        n=lambda col:pd.to_numeric(chunk[col],errors="coerce")
        m15=n("m15")
        subset=chunk.loc[m15.between(20,27)|m15.between(30,33)|m15.eq(96)]
        num=lambda col:pd.to_numeric(subset[col],errors="coerce")
        output["institutional_n"]+=len(subset)
        age=num("b3")-num("v011")
        output["age_at_birth_missing_or_negative"]+=int((age.isna()|age.lt(0)).sum())
        for name, raw in FEATURES.items():
            s=subset[raw]
            result=output["fields"][name]
            result["missing"]+=int(s.isna().sum())
            for val,count in s.value_counts(dropna=True).items():
                key=str(int(val)) if pd.notna(val) and float(val).is_integer() else str(val)
                result["observed_values"][key]=result["observed_values"].get(key,0)+int(count)
    return output


def main():
    report={"status":"EIGHT_CORE_METADATA_AND_CODES_REVIEW_REQUIRED",
            "feature_freeze_approved":False,"waves":{},
            "caveats":["Raw labels/codes are descriptive evidence, not questionnaire equivalence.",
                       "Detailed m15 facility codes are filtered to broad institutions only.",
                       "Index-age derived from b3-v011 is a candidate, not source timing certification.",
                       "Small category counts may appear: review disclosure before sharing publicly.",
                       "Survey-weighted inference and final model feature building not performed."]}
    for wave,path in SOURCES.items():
        if not path.exists():raise FileNotFoundError(path)
        report["waves"][wave]=audit(path)
        print(wave,report["waves"][wave]["institutional_n"],"institutional births")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Saved:",OUT)


if __name__=="__main__":
    main()
