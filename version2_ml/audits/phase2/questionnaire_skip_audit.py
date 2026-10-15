#!/usr/bin/env python3
"""Phase 2: test whether corrected pregnancy items have most-recent-birth skip patterns.

Aggregate-only local NFHS BR audit. No respondent-level exports.
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import pyreadstat

ROOT=Path(__file__).resolve().parents[3]
FILES={"NFHS-4":ROOT/"data/raw/nfhs4/IABR74FL.DTA",
       "NFHS-5":ROOT/"data/raw/IABR7EFL.DTA"}
MAPPINGS={
    "counselled_vaginal_bleeding":("s419a","s420a"),
    "counselled_convulsions":("s419b","s420b"),
    "counselled_prolonged_labour":("s419c","s420c"),
    "counselled_severe_abdominal_pain":("s419d","s420d"),
    "counselled_high_blood_pressure":("s419e","s420e"),
    "pregnancy_convulsions":("s432","s434"),
    "pregnancy_swelling":("s433","s435"),
}
CORES=["m15","m17","bidx","bord","v008","b3"]
GROUPS=("bidx_1","bidx_gt1","bidx_other")

def run_wave(path, wave):
    idx=0 if wave=="NFHS-4" else 1
    cols=CORES+[c[idx] for c in MAPPINGS.values()]
    out={g:{"births":0,"features":{k:{"nonmissing":0,"missing":0,"no":0,"yes":0,"unknown_8":0,"unexpected":0}
                                  for k in MAPPINGS}} for g in GROUPS}
    for chunk,_ in pyreadstat.read_file_in_chunks(
        pyreadstat.read_dta,str(path),chunksize=75000,
        usecols=cols,apply_value_formats=False):
        num=lambda key:pd.to_numeric(chunk[key],errors="coerce")
        facility=num("m15")
        f=chunk.loc[facility.between(20,27)|facility.between(30,33)|facility.eq(96)]
        bidx=pd.to_numeric(f["bidx"],errors="coerce")
        sets={"bidx_1":bidx.eq(1),"bidx_gt1":bidx.gt(1),
              "bidx_other":bidx.isna()|bidx.le(0)}
        for group,mask in sets.items():
            s=f.loc[mask]
            out[group]["births"]+=len(s)
            for name,raws in MAPPINGS.items():
                val=pd.to_numeric(s[raws[idx]],errors="coerce")
                d=out[group]["features"][name]
                d["missing"]+=int(val.isna().sum())
                d["nonmissing"]+=int(val.notna().sum())
                d["no"]+=int(val.eq(0).sum())
                d["yes"]+=int(val.eq(1).sum())
                d["unknown_8"]+=int(val.eq(8).sum())
                d["unexpected"]+=int((val.notna()&~val.isin([0,1,8])).sum())
    return out

def main():
    report={"status":"SKIP_PATTERN_DESCRIPTIVE_DIAGNOSTIC",
            "conclusions_not_automatic":True,
            "notes":[
                "bidx=1 is most recent birth in BR; compare item missingness by bidx.",
                "Aggregate observations cannot prove formal questionnaire universe/route equivalence.",
                "A source-specific item may be asked only for a most recent birth and be inappropriate for older-birth prediction.",
                "No model fitting or survey inference. Assess small-cell disclosure before sharing publicly.",
            ],"waves":{}}
    for wave,path in FILES.items():
        if not path.is_file():raise FileNotFoundError(f"Local dataset not found: {path}")
        report["waves"][wave]=run_wave(path,wave)
        print(wave,{k:v["births"] for k,v in report["waves"][wave].items()})
    dest=ROOT/"version2_ml/audits/phase2/outputs/questionnaire_skip_audit.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Saved:",dest)

if __name__=="__main__":main()
