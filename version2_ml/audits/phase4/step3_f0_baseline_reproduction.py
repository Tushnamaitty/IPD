#!/usr/bin/env python3
"""Phase 4 Step 3: independently rerun V1 XGBoost from historical 36-feature parquet.

Uses notebook 03/04's StratifiedGroupKFold(5, shuffle=True, random_state=42),
train-only tree preprocessing, XGB hyperparameters, and same test fold. Requires
local data/processed/df_model_v1.parquet. Saves aggregate result only. Does not
use the new V2 70/15/15 test split. No individuals or predictions are persisted.
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
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"version2_ml"))
from feature_builder import F0_NUM,F0_CAT,F0_BIN,validate_f0_historical

SOURCE=ROOT/"data/processed/df_model_v1.parquet"
OUT=ROOT/"version2_ml/audits/phase4/outputs/step3_f0_baseline_reproduction.json"
EXCLUDE=("respondent_id","birth_index","cluster_number","household_number",
         "respondent_line_number","sample_weight","primary_sampling_unit",
         "sample_stratum_v022","sample_stratum_v023","state","district",
         "delivery_place_code","survey_weight","csection")
def main():
    if not SOURCE.is_file():
        raise FileNotFoundError("Original V1 parquet absent: "+str(SOURCE))
    df=pd.read_parquet(SOURCE)
    manifest=validate_f0_historical(df)
    for name in ("csection","respondent_id"):
        if name not in df:raise ValueError("Missing "+name)
    X=df.drop(columns=[name for name in EXCLUDE if name in df.columns]).copy()
    expected=set(F0_NUM+F0_CAT+F0_BIN)
    if set(X.columns)!=expected:
        raise ValueError("Historical X set mismatch. Missing: "
            +str(sorted(expected-set(X.columns)))+" extras: "
            +str(sorted(set(X.columns)-expected)))
    y=pd.to_numeric(df["csection"],errors="raise").astype(int)
    if not set(y.unique()).issubset({0,1}):raise ValueError("Nonbinary outcomes")
    groups=df["respondent_id"]
    if groups.isna().any():raise ValueError("Missing mother identifiers")
    for col in F0_NUM:
        X[col]=np.asarray(pd.to_numeric(X[col],errors="coerce"),dtype=np.float64)
    for col in F0_CAT+F0_BIN:
        X[col]=X[col].astype("string").fillna("Missing").astype(str).to_numpy(dtype=object)
    train_idx,test_idx=next(StratifiedGroupKFold(
        n_splits=5,shuffle=True,random_state=42).split(X,y,groups))
    if set(groups.iloc[train_idx]).intersection(groups.iloc[test_idx]):
        raise RuntimeError("Mother overlap across train/test")
    prep=ColumnTransformer(transformers=[
       ("numeric",Pipeline([("imputer",SimpleImputer(strategy="median",
                                  add_indicator=True,missing_values=np.nan))]),list(F0_NUM)),
       ("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=True),list(F0_CAT)),
       ("binary",OneHotEncoder(handle_unknown="ignore",sparse_output=True),list(F0_BIN))],
       remainder="drop",sparse_threshold=0.3,verbose_feature_names_out=True)
    Xt=prep.fit_transform(X.iloc[train_idx])
    Xe=prep.transform(X.iloc[test_idx])
    train_y=y.iloc[train_idx]; test_y=y.iloc[test_idx]
    ratio=float((train_y.eq(0).sum())/(train_y.eq(1).sum()))
    model=XGBClassifier(n_estimators=200,learning_rate=0.05,max_depth=6,
             subsample=0.8,colsample_bytree=0.8,scale_pos_weight=ratio,
             eval_metric="logloss",random_state=42,n_jobs=-1)
    model.fit(Xt,train_y)
    prob=model.predict_proba(Xe)[:,1]
    pred=model.predict(Xe)
    result={"status":"F0_HISTORICAL_XGB_REPRODUCTION_RUN",
      "n_total":int(len(df)),"mothers_total":int(groups.nunique()),
      "n_train":len(train_idx),"n_test":len(test_idx),
      "mothers_train":int(groups.iloc[train_idx].nunique()),
      "mothers_test":int(groups.iloc[test_idx].nunique()),
      "test_positive":int(test_y.sum()),"features_raw":36,
      "features_encoded":int(Xt.shape[1]),
      "model":"XGBoost original notebook 04 baseline",
      "roc_auc":float(roc_auc_score(test_y,prob)),
      "average_precision":float(average_precision_score(test_y,prob)),
      "f1_default_predict":float(f1_score(test_y,pred,zero_division=0)),
      "historical_reference_roc_auc":0.8033,
      "historical_reference_ap":0.5330,
      "historical_reference_f1":0.5469,
      "matching_tolerance_not_prespecified":True,
      "reproduced_original_result_certified":False,
      "limitations":["Historical values were recorded from project context; validate notebook source result/provenance.",
        "Uses original 80/20 grouped fold, NOT V2 70/15/15 partition.",
        "No baseline comparison across unmatched test rows.",
        "No design-valid bootstrap; original test previously inspected.",
        "No individual records, IDs, predictions or pretrained model written."]}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("Saved",OUT)
    print("AUC",result["roc_auc"],"AP",result["average_precision"],"F1",result["f1_default_predict"])

if __name__=="__main__":main()
