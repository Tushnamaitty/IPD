#!/usr/bin/env python3
"""Phase 2 remaining 29 predictors: local, aggregate cross-wave source audit.

No source data are uploaded. Reports only coverage/code distributions, not
semantic equivalence or actual V1 preprocessing reproduction.
"""
from pathlib import Path
import json
import pandas as pd
import pyreadstat

ROOT=Path(__file__).resolve().parents[3]
WAVES={"NFHS-4":ROOT/"data/raw/nfhs4/IABR74FL.DTA",
       "NFHS-5":ROOT/"data/raw/IABR7EFL.DTA"}
OUT=ROOT/"version2_ml/audits/phase2/outputs/remaining29_feature_audit.json"
# Historical 36 names minus the 7 corrected counselling/complication variables.
FEATURES={
 "maternal_age":["v012"],"education_years":["v133"],"wealth_index":["v190"],
 "birth_order":["bord"],"total_children_ever_born":["v201"],
 "preceding_birth_interval":["b11"],"age_at_first_birth":["v212"],
 "bmi":["v445"],"first_anc_timing":["m13"],"anc_visits":["m14"],
 "facility_type":["m15"],"residence":["v025"],"religion":["v130"],
 "social_group":["s116"],"twin_order":["b0"],
 "anc_doctor":["m2a"],"anc_nurse_midwife":["m2b"],
 "anc_traditional_attendant":["m2g"],"anc_community_worker":["m2h"],
 "anc_anganwadi_worker":["m2i"],"anc_asha_worker":["m2j"],
 "anc_other_provider":["m2k"],"no_anc_provider":["m2n"],
 "health_insurance":["v481"],
 "weight_measured_during_pregnancy":["m42a"],
 "bp_measured_during_pregnancy":["m42c"],
 "urine_sample_during_pregnancy":["m42d"],
 "blood_sample_during_pregnancy":["m42e"],
 "told_about_pregnancy_complications":["m43"]
}

def wave_audit(path):
    _,meta=pyreadstat.read_dta(str(path),metadataonly=True)
    available=set(meta.column_names)
    selected=sorted({"m15","bidx"}|{sources[0] for sources in FEATURES.values() if sources[0] in available})
    # Use actual code values (no formatted labels); optional features absent in wave remain flagged.
    result={"feature_count":len(FEATURES),"available_raw_features":{},
            "missing_raw_features":[], "cohort_total":0,
            "birth_index_1":0,"birth_index_gt1":0}
    for name,sources in FEATURES.items():
        col=sources[0]
        if col in available:
            result["available_raw_features"][name]={
                "source":col,"total_nonmissing":0,"total_missing":0,
                "bidx1_nonmissing":0,"bidx1_missing":0,
                "bidx_gt1_nonmissing":0,"bidx_gt1_missing":0,
                "observed_numeric_min":None,"observed_numeric_max":None,
                "non_numeric_nonmissing":0}
        else:
            result["missing_raw_features"].append({"feature":name,"source":col})
    for chunk,_ in pyreadstat.read_file_in_chunks(
            pyreadstat.read_dta,str(path),chunksize=75000,usecols=selected,
            apply_value_formats=False):
        f=pd.to_numeric(chunk["m15"],errors="coerce")
        use=chunk.loc[f.between(20,27)|f.between(30,33)|f.eq(96)]
        bidx=pd.to_numeric(use["bidx"],errors="coerce")
        recent=bidx.eq(1)
        older=bidx.gt(1)
        result["cohort_total"]+=len(use)
        result["birth_index_1"]+=int(recent.sum())
        result["birth_index_gt1"]+=int(older.sum())
        for name,info in result["available_raw_features"].items():
            series=use[info["source"]]
            numeric=pd.to_numeric(series,errors="coerce")
            info["total_nonmissing"]+=int(series.notna().sum())
            info["total_missing"]+=int(series.isna().sum())
            info["bidx1_nonmissing"]+=int((recent&series.notna()).sum())
            info["bidx1_missing"]+=int((recent&series.isna()).sum())
            info["bidx_gt1_nonmissing"]+=int((older&series.notna()).sum())
            info["bidx_gt1_missing"]+=int((older&series.isna()).sum())
            info["non_numeric_nonmissing"]+=int((series.notna()&numeric.isna()).sum())
            if numeric.notna().any():
                mn=float(numeric.min())
                mx=float(numeric.max())
                info["observed_numeric_min"]=mn if info["observed_numeric_min"] is None else min(mn,info["observed_numeric_min"])
                info["observed_numeric_max"]=mx if info["observed_numeric_max"] is None else max(mx,info["observed_numeric_max"])
    return result

def main():
    assert len(FEATURES)==29, "Expected 29 non-seven V1 features"
    report={"status":"REMAINING29_RAW_AVAILABILITY_AUDIT_NOT_SEMANTIC_SIGNOFF",
            "waves":{}, "certified_feature_count":0,
            "limitations":["No raw-to-derived transformations validated.",
                           "Variable labels and questionnaire routing must be reviewed separately.",
                           "Missingness percentages differ by wave and birth-index eligibility.",
                           "Numeric min/max may include survey missing or special codes; do not interpret without metadata.",
                           "Interview-time measurement and birth-time availability unresolved for several features.",
                           "No design-based inference, respondent outputs, or model fitting."]}
    for wave,path in WAVES.items():
        if not path.is_file(): raise FileNotFoundError(f"Missing local dataset: {path}")
        report["waves"][wave]=wave_audit(path)
        print(wave,"institutional births:",report["waves"][wave]["cohort_total"],
              "missing raw feature sources:",len(report["waves"][wave]["missing_raw_features"]))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Saved:",OUT)

if __name__=="__main__":main()
