"""Reproducible, aggregate-only NFHS-5 state-context analysis for Extension 4.

Input is the frozen V2 Parquet, not reconstructed raw NFHS data. The notebook
calls run_extension4; no precomputed results or row-level caches are read.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
NUMERIC = ["birth_order", "wealth_index", "education_years"]
CATEGORICAL = ["residence", "religion", "social_group", "twin_order", "state"]
CONFOUNDERS = NUMERIC + CATEGORICAL
REQUIRED = CONFOUNDERS + ["facility_type", "csection", "respondent_id", "cluster_number",
                          "sample_stratum_v022", "sample_weight_normalized"]
STATE_NAMES = {
    1:"Jammu & Kashmir",2:"Himachal Pradesh",3:"Punjab",4:"Chandigarh",5:"Uttarakhand",
    6:"Haryana",7:"Delhi",8:"Rajasthan",9:"Uttar Pradesh",10:"Bihar",11:"Sikkim",
    12:"Arunachal Pradesh",13:"Nagaland",14:"Manipur",15:"Mizoram",16:"Tripura",
    17:"Meghalaya",18:"Assam",19:"West Bengal",20:"Jharkhand",21:"Odisha",
    22:"Chhattisgarh",23:"Madhya Pradesh",24:"Gujarat",
    25:"Dadra & Nagar Haveli and Daman & Diu",27:"Maharashtra",28:"Andhra Pradesh",
    29:"Karnataka",30:"Goa",31:"Lakshadweep",32:"Kerala",33:"Tamil Nadu",
    34:"Puducherry",35:"Andaman & Nicobar Islands",36:"Telangana",37:"Ladakh",
}
OUTPUTS = ["context_sources.csv", "context_merge_audit.csv", "context_descriptives.csv",
           "context_overlap_diagnostics.csv", "context_balance_diagnostics.csv",
           "context_modifier_results.csv", "context_bootstrap_results.csv",
           "context_sensitivity_results.csv", "context_leave_one_state_out.csv",
           "context_effect_plot.png", "context_metadata.json"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _weights_ess(w):
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum()) if len(w) and np.square(w).sum() else np.nan


def _weighted_mean(x, w):
    return float(np.dot(np.asarray(x, float), np.asarray(w, float)) / np.sum(w))


def _estimate(d):
    w = d["sample_weight_normalized"].to_numpy(float)
    r1, r0 = _weighted_mean(d["psi1"], w), _weighted_mean(d["psi0"], w)
    return {"adjusted_private_risk_pct": 100*r1, "adjusted_public_risk_pct": 100*r0,
            "risk_difference_pct": 100*(r1-r0), "risk_ratio": r1/r0 if r0 else np.nan}


def _check_source(source, expected):
    source = source.copy()
    assert source.state_code_v024.is_unique and len(source) == 36, "Expected 36 unique source state rows."
    assert set(source.state_code_v024) == set(STATE_NAMES), "State crosswalk mismatch."
    for r in source.itertuples():
        name = "Delhi" if r.state_name_raw == "NCT of Delhi" else r.state_name_raw
        assert name == STATE_NAMES[int(r.state_code_v024)], (r.state_code_v024, name)
    den = pd.to_numeric(source.required_total_specialist_posts, errors="coerce")
    pos = pd.to_numeric(source.in_position_total_specialists, errors="coerce")
    assert (den.dropna() >= 0).all() and (pos.dropna() >= 0).all()
    source["availability_pct"] = np.where(den.gt(0) & pos.notna(), 100*pos/den, np.nan)
    if expected:
        assert source.availability_pct.notna().sum() == 34
        assert source.loc[source.availability_pct.isna(), "state_code_v024"].sort_values().tolist() == [4, 7]
    return source


def _propensity():
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                           ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=42))])


def _outcome(model, jobs):
    if model == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("Install xgboost before running the real Extension 4 analysis.") from e
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median", add_indicator=True), NUMERIC + ["exposure"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
        clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42,
                            n_jobs=jobs, tree_method="hist", n_estimators=200, max_depth=4,
                            learning_rate=0.05)
    elif model == "logistic_test_only":
        pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), NUMERIC+["exposure"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(model)
    return Pipeline([("pre", pre), ("clf", clf)])


def _crossfit(d, model, jobs):
    W = d[CONFOUNDERS].copy().reset_index(drop=True)
    for col in CATEGORICAL:
        W[col] = W[col].astype("string").fillna("Missing").astype(str)
    a = d.exposure.to_numpy(int)
    y = d.csection.to_numpy(int)
    sw = d.sample_weight_normalized.to_numpy(float)
    group = d.respondent_id.to_numpy()
    e, mu1, mu0 = (np.full(len(d), np.nan) for _ in range(3))
    folds = np.full(len(d), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(splitter.split(W, a, group)):
        assert not set(group[tr]).intersection(group[te]), "Respondent leakage in cross-fit."
        assert np.all(folds[te] == -1), "Duplicate held-out prediction."
        folds[te] = fold
        pp = _propensity()
        pp.fit(W.iloc[tr], a[tr], clf__sample_weight=sw[tr])
        e[te] = pp.predict_proba(W.iloc[te])[:, 1]
        we = W.copy() if fold == 0 else we
        if fold == 0:
            we["exposure"] = a
        op = _outcome(model, jobs)
        op.fit(we.iloc[tr], y[tr], clf__sample_weight=sw[tr])
        hold = we.iloc[te].copy()
        hold["exposure"] = 1
        mu1[te] = op.predict_proba(hold)[:, 1]
        hold["exposure"] = 0
        mu0[te] = op.predict_proba(hold)[:, 1]
        print(f"  fold {fold+1}/5: {len(te):,} held-out births", flush=True)
        del pp, op, hold
    assert np.all(folds >= 0)
    assert all(np.isfinite(x).all() for x in (e,mu1,mu0))
    assert all(((x>=0)&(x<=1)).all() for x in (e,mu1,mu0))
    return np.clip(e, 1e-6, 1-1e-6), mu1, mu0, folds


def _balance_and_overlap(d):
    overlap, balance = [], []
    for g, sub in d.groupby("context_group", sort=False):
        a = sub.exposure.to_numpy(int); sw = sub.sample_weight_normalized.to_numpy(float)
        e = sub.e_hat.to_numpy(float)
        ipw = sw*np.where(a==1,1/e,1/(1-e))
        for fac, keep in [("overall", np.ones(len(sub),bool)),("private",a==1),("public",a==0)]:
            q = e[keep]
            overlap.append({"context_group":g,"facility_type":fac,"n":int(keep.sum()),
                "propensity_min":float(q.min()),"propensity_max":float(q.max()),
                "propensity_mean_survey_weighted":_weighted_mean(q,sw[keep]),
                "n_extreme_lt0.05_or_gt0.95":int(((q<.05)|(q>.95)).sum()),
                "pct_extreme":float(100*((q<.05)|(q>.95)).mean()),
                "ess_survey_weight":_weights_ess(sw[keep]),"ess_ipw_diagnostic":_weights_ess(ipw[keep])})
        X = pd.DataFrame(index=sub.index)
        for col in NUMERIC:
            x = pd.to_numeric(sub[col], errors="coerce")
            X[col] = x.fillna(x.median()).astype(float)
            if x.isna().any(): X[col+"_missing"] = x.isna().astype(float)
        for col in CATEGORICAL:
            cat = sub[col].astype("string").fillna("Missing")
            for val in sorted(cat.unique()):
                X[f"{col}={val}"] = cat.eq(val).to_numpy(float)
        for name in X.columns:
            x = X[name].to_numpy(float)
            def smd(weight):
                w1,w0=weight[a==1],weight[a==0]
                x1,x0=x[a==1],x[a==0]
                m1,m0=_weighted_mean(x1,w1),_weighted_mean(x0,w0)
                v1=_weighted_mean((x1-m1)**2,w1);v0=_weighted_mean((x0-m0)**2,w0)
                den=np.sqrt((v1+v0)/2)
                return float((m1-m0)/den) if den>1e-12 else (0.0 if abs(m1-m0)<1e-12 else np.nan)
            before, after=smd(sw),smd(ipw)
            balance.append({"context_group":g,"covariate":name,"smd_before":before,
                            "smd_after_ipw":after,"flag_after_abs_smd_gt_0_1":bool(np.isnan(after) or abs(after)>.1)})
    return pd.DataFrame(overlap), pd.DataFrame(balance)


def _bootstrap(d, n_bootstrap, seed):
    # One resample across the whole cohort per replicate: both context groups
    # share a PSU draw, including when a stratum contains states in both groups.
    q=d[["sample_stratum_v022","cluster_number","context_group","sample_weight_normalized","psi1","psi0"]].copy()
    w=q.sample_weight_normalized.to_numpy(float)
    low=(q.context_group=="low").to_numpy(float); high=1-low
    for name,mask in [("low",low),("high",high)]:
        q[name+"_w"]=w*mask
        q[name+"_p1"]=w*q.psi1.to_numpy(float)*mask
        q[name+"_p0"]=w*q.psi0.to_numpy(float)*mask
    cols=[f"{g}_{v}" for g in ("low","high") for v in ("w","p1","p0")]
    cluster=q.groupby(["sample_stratum_v022","cluster_number"],dropna=False,sort=True)[cols].sum().reset_index()
    by_stratum=[v[cols].to_numpy(float) for _,v in cluster.groupby("sample_stratum_v022",sort=True,dropna=False)]
    rng=np.random.RandomState(seed)
    rows=[]
    for b in range(n_bootstrap):
        total=np.zeros(6)
        for arr in by_stratum:
            idx=rng.choice(len(arr),size=len(arr),replace=True)
            total+=arr[idx].sum(axis=0)
        if total[0]<=0 or total[3]<=0: raise RuntimeError(f"Bootstrap replicate {b} omitted a context group.")
        r1lo,r0lo=total[1]/total[0],total[2]/total[0]
        r1hi,r0hi=total[4]/total[3],total[5]/total[3]
        rdlo,rdhi=100*(r1lo-r0lo),100*(r1hi-r0hi)
        rows.append({"replicate":b,"rd_low_pct":rdlo,"rd_high_pct":rdhi,
                     "delta_high_minus_low_pct":rdhi-rdlo,
                     "rr_low":r1lo/r0lo if r0lo else np.nan,"rr_high":r1hi/r0hi if r0hi else np.nan})
    return pd.DataFrame(rows),len(cluster),len(by_stratum)


def run_analysis(df, source, out: Path, *, input_path=None, source_path=None,
                 n_bootstrap=500, seed=42, model="xgboost", jobs=4, expected_counts=True):
    out=Path(out)
    assert not any((out/f).exists() for f in OUTPUTS), "Move old output files to a backup directory before this clean run."
    absent=set(REQUIRED)-set(df.columns)
    assert not absent, f"Missing V2 columns: {sorted(absent)}"
    if expected_counts:
        assert len(df)==201311
        assert df.facility_type.value_counts().to_dict()=={"public":150299,"private":50495,"other":517}
    assert df.csection.dropna().isin([0,1]).all() and df.csection.notna().all()
    assert df.respondent_id.notna().all() and df.cluster_number.notna().all() and df.sample_stratum_v022.notna().all()
    assert df.sample_weight_normalized.notna().all() and (df.sample_weight_normalized>0).all()
    source=_check_source(source,expected_counts)
    analytic=df[df.facility_type.isin(["public","private"])].copy()
    if expected_counts: assert len(analytic)==200794
    assert set(analytic.state.dropna().astype(int)) <= set(source.state_code_v024)
    rural=analytic[analytic.residence==2].copy()
    if expected_counts: assert len(rural)==156717
    rural=rural.merge(source[["state_code_v024","state_name_raw","availability_pct"]],
                      left_on="state",right_on="state_code_v024",how="left",validate="many_to_one",indicator=True)
    assert (rural._merge=="both").all()
    missing=int(rural.availability_pct.isna().sum())
    if expected_counts: assert missing==134
    valid=rural.loc[rural.availability_pct.notna()].copy().reset_index(drop=True)
    if expected_counts: assert len(valid)==156583
    threshold=float(source.availability_pct.dropna().median())
    valid["context_group"]=np.where(valid.availability_pct>=threshold,"high","low")
    valid["exposure"]=(valid.facility_type=="private").astype(int)
    if expected_counts:
        assert np.isclose(threshold,18.31488545410447)
        assert valid.groupby("context_group").size().to_dict()=={"low":66625,"high":89958}
        assert valid.groupby("context_group").state.nunique().to_dict()=={"low":17,"high":17}
    print(f"Rural retained: {len(valid):,}; Z median: {threshold:.6f}; missing Z: {missing}",flush=True)
    # Simple diagnostics use raw data and are recreated on every run.
    desc=[]
    for (g,fac),q in valid.groupby(["context_group","facility_type"],sort=True):
        group=valid[valid.context_group==g]
        desc.append({"context_group":g,"Z_definition":f"Z {'>=' if g=='high' else '<'} {threshold:.6f}%",
                     "n_states":int(group.state.nunique()),"facility_type":fac,"n":len(q),
                     "weighted_share_pct":100*q.sample_weight_normalized.sum()/group.sample_weight_normalized.sum(),
                     "weighted_csection_rate_pct":100*_weighted_mean(q.csection,q.sample_weight_normalized),
                     "unweighted_csection_rate_pct":100*float(q.csection.mean()),
                     "effective_sample_size_survey_weight":_weights_ess(q.sample_weight_normalized)})
    desc=pd.DataFrame(desc)
    assert len(desc)==4
    assert np.allclose(desc.groupby("context_group").weighted_share_pct.sum(),100)
    merge_audit=[]
    for r in source.itertuples():
        q=rural[rural.state==r.state_code_v024]
        merge_audit.append({"geo_key":int(r.state_code_v024),"state_name":r.state_name_raw,
            "source_period":"31 March 2021","Z":r.availability_pct,"matched":len(q)>0,
            "rural_births_pubpriv":len(q),"unique_respondents":int(q.respondent_id.nunique()),
            "weighted_share_of_rural_retained":float(q.sample_weight_normalized.sum()/valid.sample_weight_normalized.sum()),
            "n_public":int((q.facility_type=="public").sum()),"n_private":int((q.facility_type=="private").sum()),
            "survey_weighted_csection_rate_pct":100*_weighted_mean(q.csection,q.sample_weight_normalized) if len(q) else np.nan,
            "missing_reason":"RHS denominator zero/not applicable" if pd.isna(r.availability_pct) else ""})
    merge_audit=pd.DataFrame(merge_audit)
    frames=[]
    for g in ("low","high"):
        sub=valid.loc[valid.context_group==g].copy().reset_index(drop=True)
        print(f"Cross-fitting {g}: {len(sub):,} births",flush=True)
        e,mu1,mu0,folds=_crossfit(sub,model,jobs)
        a=sub.exposure.to_numpy(int);y=sub.csection.to_numpy(int)
        sub["e_hat"]=e
        sub["psi1"]=mu1+a*(y-mu1)/e
        sub["psi0"]=mu0+(1-a)*(y-mu0)/(1-e)
        assert np.isfinite(sub[["psi1","psi0"]].to_numpy()).all()
        frames.append(sub)
    scored=pd.concat(frames,ignore_index=True)
    estimates={g:_estimate(scored[scored.context_group==g]) for g in ("low","high")}
    delta=estimates["high"]["risk_difference_pct"]-estimates["low"]["risk_difference_pct"]
    boot,n_psus,n_strata=_bootstrap(scored,n_bootstrap,seed)
    cilo,cihi={g:np.percentile(boot[f"rd_{g}_pct"],[2.5,97.5]) for g in ("low","high")}.values()
    delta_ci=np.percentile(boot.delta_high_minus_low_pct,[2.5,97.5])
    result=[]
    for g,ci in [("low",(cilo,cihi)[0]),("high",(cilo,cihi)[1])]:
        q=scored[scored.context_group==g]
        d={"context_group":g,"threshold_pct":threshold,"n_states":int(q.state.nunique()),
           "n":len(q),"n_public":int((q.exposure==0).sum()),"n_private":int(q.exposure.sum()),
           **estimates[g],"risk_difference_ci_low_pct":ci[0],"risk_difference_ci_high_pct":ci[1]}
        result.append(d)
    result.append({"context_group":"high_minus_low (Delta)","threshold_pct":threshold,
                   "risk_difference_pct":delta,"risk_difference_ci_low_pct":delta_ci[0],
                   "risk_difference_ci_high_pct":delta_ci[1]})
    modifier=pd.DataFrame(result)
    overlap,balance=_balance_and_overlap(scored)
    sensitivity=[]
    for g in ("low","high"):
        q=scored[scored.context_group==g]
        keep=q.e_hat.between(.05,.95,inclusive="both")
        if not keep.any(): raise ValueError(f"No overlap after trimming for {g}.")
        sensitivity.append({"context_group":g,"sensitivity":"trim propensity outside [0.05,0.95]",
            "retained_n":int(keep.sum()),"excluded_n":int((~keep).sum()),**_estimate(q[keep])})
    sensitivity.append({"context_group":"high_minus_low","sensitivity":"trim propensity outside [0.05,0.95]",
            "risk_difference_pct":sensitivity[1]["risk_difference_pct"]-sensitivity[0]["risk_difference_pct"]})
    influence=[]
    for state,name in sorted(STATE_NAMES.items()):
        if state not in set(scored.state): continue
        remain=scored[scored.state!=state]
        low=_estimate(remain[remain.context_group=="low"])["risk_difference_pct"]
        high=_estimate(remain[remain.context_group=="high"])["risk_difference_pct"]
        influence.append({"omitted_state_code":state,"omitted_state_name":name,"retained_n":len(remain),
                          "rd_low_pct":low,"rd_high_pct":high,"delta_high_minus_low_pct":high-low})
    influence=pd.DataFrame(influence)
    if expected_counts: assert len(influence)==34
    # Plot derived directly from new estimates and their paired-bootstrap CI.
    fig,ax=plt.subplots(figsize=(7,4.1))
    vals=[estimates[g]["risk_difference_pct"] for g in ("low","high")]
    bounds=[(cilo,cihi)[i] for i in (0,1)]
    ax.errorbar(vals,[0,1],xerr=[[v-b[0] for v,b in zip(vals,bounds)],
                                [b[1]-v for v,b in zip(vals,bounds)]],fmt="o",capsize=4,color="#214c72")
    ax.set_yticks([0,1],["Lower CHC specialist availability","Higher CHC specialist availability"])
    ax.invert_yaxis();ax.set_xlabel("Adjusted private - public risk difference (pp)")
    ax.set_title(f"Rural CHC specialist availability\nHigh - low: {delta:+.2f} pp [{delta_ci[0]:+.2f}, {delta_ci[1]:+.2f}]",fontsize=11)
    ax.grid(axis="x",alpha=.25);fig.tight_layout()
    out.mkdir(parents=True,exist_ok=True)
    desc.to_csv(out/"context_descriptives.csv",index=False)
    merge_audit.to_csv(out/"context_merge_audit.csv",index=False)
    pd.DataFrame([{"indicator_id":"rhs20_21_chc_specialist_availability",
                   "source_title":"Rural Health Statistics 2020-21, Table 23, total specialists at rural CHCs",
                   "publisher":"Ministry of Health and Family Welfare, Government of India",
                   "URL":"https://www.mohfw.gov.in/sites/default/files/rhs20-21_2.pdf",
                   "reference_date":"31 March 2021","source_file_sha256":_sha256(Path(source_path)) if source_path else "",
                   "formula":"100 * total specialists in position / required total specialist posts",
                   "scope":"State-level rural public CHCs; residential geography, not delivery-facility location",
                   "timing_limitation":"Near-contemporaneous; not uniformly pre-delivery"}]).to_csv(out/"context_sources.csv",index=False)
    overlap.to_csv(out/"context_overlap_diagnostics.csv",index=False)
    balance.to_csv(out/"context_balance_diagnostics.csv",index=False)
    modifier.to_csv(out/"context_modifier_results.csv",index=False)
    boot.to_csv(out/"context_bootstrap_results.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(out/"context_sensitivity_results.csv",index=False)
    influence.to_csv(out/"context_leave_one_state_out.csv",index=False)
    fig.savefig(out/"context_effect_plot.png",dpi=170);plt.close(fig)
    meta={"input_path":str(input_path) if input_path else "in-memory validation input",
          "input_sha256":_sha256(Path(input_path)) if input_path else None,
          "source_path":str(source_path) if source_path else "in-memory validation source",
          "source_sha256":_sha256(Path(source_path)) if source_path else None,
          "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                      "scikit_learn":sklearn.__version__,"xgboost":__import__("xgboost").__version__ if model=="xgboost" else None},
          "cohort":{"v2_total":len(df),"public_private":len(analytic),"rural_public_private":len(rural),
                    "missing_z":missing,"retained":len(valid),"states":int(valid.state.nunique())},
          "indicator":{"name":"rural CHC specialist availability", "threshold_pct":threshold,
                       "split":"unweighted median across valid states; high >= median"},
          "models":{"propensity":"survey-weighted logistic", "outcome":model,
                    "crossfit":"5-fold respondent-grouped within each context group", "seed":42,
                    "clipping_epsilon":1e-6},
          "bootstrap":{"replicates":n_bootstrap,"seed":seed,"psu_key":["sample_stratum_v022","cluster_number"],
                       "n_psu_pairs":n_psus,"n_strata":n_strata,"one_joint_draw":True,
                       "nuisance_models_refit":False,"interval":"percentile 2.5/97.5; conditional on fitted nuisances and observed states"},
          "results":{"low_rd_pct":estimates["low"]["risk_difference_pct"],
                     "high_rd_pct":estimates["high"]["risk_difference_pct"],
                     "delta_high_minus_low_pct":delta,"delta_ci_pct":[float(delta_ci[0]),float(delta_ci[1])]},
          "state_influence":{"n_omitted_states":len(influence),
                             "min_delta_pct":float(influence.delta_high_minus_low_pct.min()),
                             "max_delta_pct":float(influence.delta_high_minus_low_pct.max())},
          "interpretation":"Contextual association in rural residential institutional births; no staffing or facility causal effect.",
          "output_files":OUTPUTS}
    (out/"context_metadata.json").write_text(json.dumps(meta,indent=2,allow_nan=False),encoding="utf-8")
    assert len(desc)==4 and len(boot)==n_bootstrap and len(influence)==valid.state.nunique()
    assert np.allclose(boot.rd_high_pct-boot.rd_low_pct,boot.delta_high_minus_low_pct)
    print(f"Delta high - low: {delta:+.4f} pp [{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}]",flush=True)
    print("Generated:",", ".join(OUTPUTS),flush=True)
    return modifier,meta


def run_extension4():
    repo=ROOT.parents[1]
    v2=Path(os.environ.get("IPD_EXT04_V2_PATH",repo/"data"/"processed"/"df_model_v2.parquet"))
    source=Path(os.environ.get("IPD_EXT04_SOURCE_PATH",ROOT/"local_source"/"rhs20-21_table23_transcribed.csv"))
    out=Path(os.environ.get("IPD_EXT04_OUTPUT_DIR",ROOT/"outputs"))
    if not v2.is_file(): raise FileNotFoundError(f"Frozen V2 Parquet missing: {v2}")
    if not source.is_file(): raise FileNotFoundError(f"RHS source transcription missing: {source}")
    if v2.suffix.lower() != ".parquet": raise ValueError("Real-data run must read the frozen V2 Parquet.")
    print("Reading frozen V2:",v2,flush=True)
    return run_analysis(pd.read_parquet(v2),pd.read_csv(source),out,input_path=v2,source_path=source,
                        n_bootstrap=500,seed=42,model="xgboost",
                        jobs=int(os.environ.get("IPD_XGB_N_JOBS","4")),expected_counts=True)
