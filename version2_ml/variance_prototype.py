"""Exploratory stratified-PSU resampling engine; NOT a certified DHS variance estimator.

Uses fixed predictions and replicates PSUs within each observed stratum.
Singleton policies:
- error (default): abort; no quietly invented certainty
- certainty: retain single PSU; explicitly flagged sensitivity ONLY
No publication-valid CI can be inferred without multistage design review.
"""
import numpy as np
import pandas as pd

def kish_effective_n(weights):
    w=np.asarray(weights,dtype=np.float64)
    if w.ndim!=1 or len(w)==0 or not np.isfinite(w).all() or (w<=0).any():
        raise ValueError("Invalid weights")
    return float(w.sum()**2/np.dot(w,w))

def validate_design(strata,psu,weights):
    s=pd.Series(strata).reset_index(drop=True)
    p=pd.Series(psu).reset_index(drop=True)
    w=np.asarray(weights,dtype=np.float64)
    if len(s)!=len(p) or len(s)!=len(w):
        raise ValueError("Array lengths differ")
    if s.isna().any() or p.isna().any():
        raise ValueError("Missing PSU or stratum")
    effective=kish_effective_n(w)
    units=pd.DataFrame({"stratum":s,"psu":p}).drop_duplicates()
    npsu=units.groupby("stratum").size()
    return {"n_rows":len(w),"n_strata":len(npsu),
            "n_psu_pairs":len(units),
            "singleton_strata":int((npsu==1).sum()),
            "kish_n":effective}

def resample_psu_indices(strata,psu,*,repetitions=200,seed=42,
                         singleton_policy="error"):
    """Yield replicate row indices, PSU sampling with replacement within stratum."""
    if singleton_policy not in ("error","certainty"):
        raise ValueError("Unknown singleton policy")
    if repetitions<1:raise ValueError("No replicates requested")
    s=pd.Series(strata).reset_index(drop=True)
    p=pd.Series(psu).reset_index(drop=True)
    if len(s)!=len(p) or s.isna().any() or p.isna().any():
        raise ValueError("Invalid stratum/PSU")
    by_stratum={}
    for (stratum,cluster),grp in pd.DataFrame({"s":s,"p":p}).groupby(["s","p"],sort=False):
        by_stratum.setdefault(stratum,[]).append(grp.index.to_numpy(dtype=np.int64))
    if singleton_policy=="error" and any(len(v)==1 for v in by_stratum.values()):
        raise ValueError("Singleton PSU stratum; choose explicit sensitivity policy")
    rng=np.random.default_rng(seed)
    for _ in range(repetitions):
        selected=[]
        for clusters in by_stratum.values():
            draws=rng.integers(0,len(clusters),size=len(clusters))
            selected.extend(clusters[i] for i in draws)
        yield np.concatenate(selected)
