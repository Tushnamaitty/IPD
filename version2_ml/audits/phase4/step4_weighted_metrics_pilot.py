#!/usr/bin/env python3
"""Phase 4 Step 4: survey-weighted F0 point-metric pilot only.

Reruns the historical F0/XGBoost split and preprocessing on LOCAL df_model_v1.
Outputs aggregate weighted/unweighted metrics and design coverage diagnostics.
NO confidence intervals, bootstrap, row-level data or fitted models are saved.
Weighted performance on a historical test subset is NOT a design-unbiased
population performance estimator. Its validity must be reviewed separately.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"version2_ml"))
from feature_builder import F0_NUM,F0_CAT,F0_BIN,validate_f0_historical
SRC=ROOT/"data/processed/df_model_v1.parquet"
OUT=ROOT/"version2_ml/audits/phase4/outputs/step4_weighted_metrics_pilot.json"
EXCLUDE=("respondent_id","birth_index","cluster_number","household_number",
         "respondent_line_number","sample_weight","primary_sampling_unit",
         "sample_stratum_v022","sample_stratum_v023","state","district",
         "delivery_place_code","survey_weight","csection")

def metric(y,p,w):
    return {"roc_auc":float(roc_auc_score(y,p,sample_weight=w)),
            "average_precision":float(average_precision_score(y,p,sample_weight=w)),
            "brier":float(brier_score_loss(y,p,sample_weight=w))}

def main():
    if not SRC.is_file():raise FileNotFoundError(SRC)
    df=pd.read_parquet(SRC)
    validate_f0_historical(df)
    required={"respondent_id","csection","sample_weight",
              "primary_sampling_unit","sample_stratum_v022"}
    missing=sorted(required-set(df.columns))
    if missing:raise ValueError("Required design or outcome fields absent: "+str(missing))
    X=df.drop(columns=[name for name in EXCLUDE if name in df]).copy()
    if set(X)!=set(F0_NUM+F0_CAT+F0_BIN):
        raise ValueError("Historical predictor manifest mismatch")
    y=pd.to_numeric(df["csection"],errors="raise").astype(int)
    if not set(y.unique()).issubset({0,1}):raise ValueError("Nonbinary outcomes")
    groups=df["respondent_id"]
    if groups.isna().any():raise ValueError("Missing respondent grouping")
    w=pd.to_numeric(df["sample_weight"],errors="coerce")
    if w.isna().any() or (~np.isfinite(w)).any() or (w<=0).any():
        raise ValueError("Sample weight missing, nonfinite, or nonpositive")
    for c in F0_NUM:
        X[c]=np.asarray(pd.to_numeric(X[c],errors="coerce"),dtype=np.float64)
    for c in F0_CAT+F0_BIN:
        X[c]=X[c].astype("string").fillna("Missing").astype(str).to_numpy(dtype=object)
    tr,te=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(X,y,groups))
    if set(groups.iloc[tr]).intersection(groups.iloc[te]):
        raise RuntimeError("Mother overlap across train/test")
    prep=ColumnTransformer([
       ("numeric",Pipeline([("imputer",SimpleImputer(strategy="median",
           add_indicator=True,missing_values=np.nan))]),list(F0_NUM)),
       ("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=True),list(F0_CAT)),
       ("binary",OneHotEncoder(handle_unknown="ignore",sparse_output=True),list(F0_BIN))
    ],remainder="drop",sparse_threshold=0.3,verbose_feature_names_out=True)
    Xtr=prep.fit_transform(X.iloc[tr])
    Xte=prep.transform(X.iloc[te])
    yt=y.iloc[te].to_numpy()
    pos=int(y.iloc[tr].sum())
    if pos<=0:raise ValueError("No training positives")
    ratio=int((y.iloc[tr]==0).sum())/pos
    model=XGBClassifier(n_estimators=200,learning_rate=0.05,max_depth=6,
        subsample=0.8,colsample_bytree=0.8,scale_pos_weight=ratio,
        eval_metric="logloss",random_state=42,n_jobs=-1)
    model.fit(Xtr,y.iloc[tr])
    scores=model.predict_proba(Xte)[:,1]
    wt=w.iloc[te].to_numpy(dtype=float)
    strata=df["sample_stratum_v022"].iloc[te]
    psu=df["primary_sampling_unit"].iloc[te]
    valid=(~strata.isna())&(~psu.isna())
    if not bool(valid.all()):
        raise ValueError("Missing stratum or PSU in historical test")
    design_pairs=pd.DataFrame({"stratum":strata.to_numpy(),
                               "psu":psu.to_numpy()}).drop_duplicates()
    by_stratum=design_pairs.groupby("stratum").size()
    report={
        "status":"WEIGHTED_POINT_METRICS_PILOT_REVIEW_REQUIRED",
        "historical_split":"V1 first StratifiedGroupKFold(5, shuffle=True, seed=42)",
        "test_births":int(len(te)),"test_positive":int(yt.sum()),
        "test_unique_mothers":int(groups.iloc[te].nunique()),
        "test_strata":int(by_stratum.size),
        "test_unique_stratum_psu_pairs":int(len(design_pairs)),
        "test_single_psu_strata_count":int((by_stratum==1).sum()),
        "test_weight_min":float(np.min(wt)),
        "test_weight_max":float(np.max(wt)),
        "test_weight_effective_n_kish":float(wt.sum()**2/(wt@wt)),
        "unweighted":metric(yt,scores,None),
        "survey_weighted_descriptive":metric(yt,scores,wt),
        "survey_bootstrap_approved":False,
        "confidence_intervals_reported":False,
        "caveats":[
          "Survey weights are used only to compute descriptive test-point metrics.",
          "A mother-grouped subset is not itself a full sample-design survey; weights alone do not establish design-unbiased population model performance.",
          "PSU/stratum pairs and singleton counts are diagnostics, not variance estimates.",
          "Preprocessing and training are unweighted to reproduce historical F0 exactly.",
          "Test split was previously examined; performance is not virgin confirmatory evidence.",
          "No bootstraps or confidence intervals computed. The original cohort is not modified."
        ]}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Saved:",OUT)
    print("Unweighted:",report["unweighted"])
    print("Weighted descriptive:",report["survey_weighted_descriptive"])

if __name__=="__main__":main()
