#!/usr/bin/env python3
"""A4/A5/A6 readiness audit. LOCAL NFHS BR files; outputs aggregates only.

A4: prior observed institutional CS among *survey-window* births only.
A5: common eight-core feature construction in both waves.
A6: feasibility of holding out complete states and facility sectors.
No prediction, training, model fitting, or row-level output. No publication signoff.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import pyreadstat

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"version2_ml"))
from feature_builder import make_core_features,feature_build_summary
SOURCES={"NFHS-4":ROOT/"data/raw/nfhs4/IABR74FL.DTA",
         "NFHS-5":ROOT/"data/raw/IABR7EFL.DTA"}
OUT=ROOT/"version2_ml/audits/phase5/outputs/conditional_experiment_gate_audit.json"
COLS=["caseid","bidx","b3","b19","v008","v011","v133","v190",
      "bord","m15","m17","v025","v130","b0","v024"]

def wave_rows(path,wave):
    _,meta=pyreadstat.read_dta(str(path),metadataonly=True)
    cols=[x for x in COLS if x in meta.column_names]
    if any(x not in cols for x in COLS if x!="b19"):
        raise ValueError("Required source variable missing")
    batches=[]
    for x,_ in pyreadstat.read_file_in_chunks(pyreadstat.read_dta,
           str(path),chunksize=70000,usecols=cols,apply_value_formats=False):
        # Prior history must use all available births, including home deliveries.
        if wave=="NFHS-5":
            valid=pd.to_numeric(x["b19"],errors="coerce").between(0,59)
        else:
            valid=(pd.to_numeric(x["v008"],errors="coerce")-
                   pd.to_numeric(x["b3"],errors="coerce")).between(0,59)
        batches.append(x.loc[valid].copy())
    return pd.concat(batches,ignore_index=True)

def audit(wave,data):
    n=lambda k:pd.to_numeric(data[k],errors="coerce")
    fac=n("m15")
    institutional=fac.between(20,27)|fac.between(30,33)|fac.eq(96)
    x=data.loc[institutional].copy()
    if x[["caseid","bidx"]].duplicated().any():
        raise ValueError("Duplicate index births detected")
    if x["caseid"].isna().any():raise ValueError("Missing mother identifier")
    features=make_core_features(x,wave=wave)
    built=feature_build_summary(features)
    # A4 prior observed history: for each indexed birth, determine if any
    # earlier-in-calendar-month birth has m17=1 and institutional m15.
    # Same-month multiplets cannot be ordered as preceding deliveries.
    prior=data.loc[n("m15").between(20,27)|n("m15").between(30,33)|n("m15").eq(96)]
    prior=prior.loc[pd.to_numeric(prior["m17"],errors="coerce").eq(1),
                    ["caseid","b3"]].copy()
    prior["b3"]=pd.to_numeric(prior["b3"],errors="coerce")
    earliest=prior.groupby("caseid")["b3"].min()
    previous=earliest.reindex(x["caseid"]).to_numpy()<pd.to_numeric(x["b3"],errors="coerce").to_numpy()
    order=pd.to_numeric(x["bord"],errors="coerce")
    # Missing prior observation is distinct from established lifetime negative.
    history={
        "index_births":int(len(x)),
        "first_live_birth":int(order.eq(1).sum()),
        "higher_birth_order":int(order.gt(1).sum()),
        "observed_earlier_institutional_cs":int(np.sum(previous)),
        "higher_order_no_observed_earlier_institutional_cs":int((order.gt(1)&~previous).sum()),
        "observed_history_means_lifetime_history":False,
        "same_month_prior_birth_is_not_counted":True
    }
    state=x["v024"].astype("string")
    outcome=pd.to_numeric(x["m17"],errors="coerce")
    sector=fac.loc[x.index].map(
        lambda v:"public" if 20<=v<=27 else("private" if 30<=v<=33 else "other"))
    domain={}
    for key,values in (("state",state),("sector",sector)):
        temp=pd.DataFrame({"domain":values.to_numpy(),"y":outcome.to_numpy()})
        grp=temp.groupby("domain",dropna=False)["y"].agg(["size","sum"])
        eligible=(grp["size"]>=200)&(grp["sum"]>=30)&((grp["size"]-grp["sum"])>=30)
        domain[key]={
            "groups_total":int(len(grp)),
            "groups_meeting_preliminary_200_rows_30_each_class":int(eligible.sum()),
            "groups_failing_screen":int((~eligible).sum()),
            "domain_holdout_split_built":False,
            "screen_is_only_feasibility_not_final_eligibility":True
        }
    return {"institutional_rows":len(x),"core":built,
            "A4":history,"A5":{"common_core_built":True,"requires_source_only_fitting":True,
             "cross_wave_questionnaire_and_timing_review_pending":True},
            "A6":domain}

def main():
    result={"status":"CONDITIONAL_EXPERIMENT_READINESS_AUDIT",
        "approved_for_final_publication":False,"wave_results":{},
        "limitations":[
          "A4 survey-window recorded prior institutional CS is NOT lifetime previous CS.",
          "This A4 indicator uses all sampled births in 60-month window, not women outside recall.",
          "A5 harmonized eight core features are a model-development baseline; full semantic review remains.",
          "A6 domain sample counts do not prove accuracy/generalization and no held-out model is fitted.",
          "Small-group counts intentionally not exported; domain and weight-based inference unvalidated."
        ]}
    for wave,p in SOURCES.items():
        if not p.is_file():raise FileNotFoundError(p)
        print("Checking",wave)
        result["wave_results"][wave]=audit(wave,wave_rows(p,wave))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+chr(10),encoding="utf-8")
    print("Saved:",OUT)
if __name__=="__main__":main()
