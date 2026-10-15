#!/usr/bin/env python3
"""Phase 4 Step 3: inspect original V1 notebooks before attempting F0 reproduction.

Outputs only a notebook-code provenance manifest and expected locked metrics.
Does not claim that F0 has been reproduced; no NFHS data read or exported.
"""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
FILES=["notebooks/03_machine_learning_preprocessing.ipynb",
       "notebooks/04_baseline_machine_learning_models.ipynb",
       "notebooks/05_hyperparameter_tuning.ipynb",
       "notebooks/07_calibration_threshold_optimization.ipynb"]
OUT=ROOT/"version2_ml/audits/phase4/outputs/step3_f0_provenance.json"
KEYWORDS=("train_test_split","GroupShuffleSplit","GroupKFold",
"XGBClassifier","LogisticRegression","CatBoost","roc_auc_score",
"average_precision_score","f1_score","SimpleImputer","OneHotEncoder",
"StandardScaler","Pipeline","ColumnTransformer","GridSearchCV",
"RandomizedSearchCV","model.fit","fit_transform","dropna","fillna",
"read_parquet","read_csv","to_parquet","joblib","pickle","v445","v012",
"m15","m17","caseid","bidx")
def main():
    record={"status":"F0_ORIGINAL_NOTEBOOK_AUDIT_PENDING_REPRODUCTION",
      "original_f0_locked_reference":{"nfhs5_institutional_births":201311,
        "mothers":158429,"feature_count":36,
        "xgboost_roc_auc":0.8033,"xgboost_average_precision":0.5330,
        "xgboost_f1":0.5469},
      "notebooks":[],"blocking_gates":[
        "Confirm original train/validation/test row memberships or faithfully recreate splits.",
        "Recreate 36 original transformations, special code handling, feature order and imputation.",
        "Recover exact package versions, random seeds, thresholds and XGBoost hyperparameters.",
        "Re-run on identical evaluation births and compare metrics to locked reference.",
        "Do not compare F0 and F1 on unmatched cohorts or pretend inspected historical test is virgin."
      ]}
    for file in FILES:
        p=ROOT/file
        if not p.is_file():
            record["notebooks"].append({"path":file,"status":"MISSING"})
            continue
        blob=p.read_bytes()
        nb=json.loads(blob)
        cells=[("".join(c.get("source",[]))) for c in nb.get("cells",[]) if c.get("cell_type")=="code"]
        matches={k:[i for i,cell in enumerate(cells) if k.lower() in cell.lower()]
                 for k in KEYWORDS}
        invalid=[]
        for i,c in enumerate(cells):
            try:ast.parse(c)
            except SyntaxError: invalid.append(i)
        record["notebooks"].append({
           "path":file,"status":"PRESENT","sha256":hashlib.sha256(blob).hexdigest(),
           "code_cells":len(cells),"keyword_cell_indices":{k:v for k,v in matches.items() if v},
           "non_standard_python_cell_indices":invalid,
           "code_only_sha256":hashlib.sha256(("\n#CELL\n".join(cells)).encode()).hexdigest()
        })
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print("Saved:",OUT)
    print("Missing notebooks:",sum(x["status"]=="MISSING" for x in record["notebooks"]))
    print("F0 reproduced: NO (code provenance only)")
if __name__=="__main__":main()
