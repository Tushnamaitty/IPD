"""Reusable helper functions shared by the Extension 2 and Extension 3 notebooks.

These reimplement -- without editing -- the exact conventions used in
notebooks/v2/06_propensity_overlap_diagnostics.ipynb and
notebooks/v2/07_aipw_primary_analysis.ipynb: survey-weighted nuisance
models, respondent-grouped cross-fitting, the AIPW pseudo-outcome formula,
and PSU-cluster bootstrap resampling within strata. Centralizing them here
means both extensions use identical machinery instead of two divergent
copy-pasted implementations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_public_private_analytic(v2_path, expected_total_rows=None, expected_analytic_rows=None):
    """Load df_model_v2 (read-only) and restrict to public/private facility records.

    Mirrors notebooks/v2/06 and 07 section 2 exactly: 'other'-sector records
    are excluded, exposure is coded private=1/public=0, and row counts are
    asserted rather than assumed.
    """
    df = pd.read_parquet(v2_path)
    n_loaded = len(df)
    if expected_total_rows is not None:
        assert n_loaded == expected_total_rows, (
            f"df_model_v2 row count is {n_loaded}, expected {expected_total_rows}. Stop and investigate."
        )

    n_other = (df["facility_type"] == "other").sum()
    analytic = df[df["facility_type"].isin(["public", "private"])].copy().reset_index(drop=True)
    assert set(analytic["facility_type"].unique()) <= {"public", "private"}, \
        "Non public/private facility type leaked into the analytic sample."

    analytic["exposure"] = (analytic["facility_type"] == "private").astype(int)
    assert analytic["exposure"].isna().sum() == 0, "Missing exposure values found."
    assert analytic["csection"].isna().sum() == 0, "Missing outcome values found."

    if expected_analytic_rows is not None:
        assert len(analytic) == expected_analytic_rows, (
            f"Analytic sample is {len(analytic)} rows, expected {expected_analytic_rows}. "
            f"(n_other excluded = {n_other})"
        )

    return analytic, n_loaded, int(n_other)


# ---------------------------------------------------------------------------
# Nuisance-model pipelines (identical specification to notebook 07)
# ---------------------------------------------------------------------------

def make_propensity_pipeline(numeric_confounders, categorical_confounders, random_state=42):
    """Survey-weighted Logistic Regression propensity model -- confounders only."""
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric_confounders),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_confounders),
    ])
    return Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=random_state))])


def make_outcome_pipeline(numeric_confounders, categorical_confounders, random_state=42, model_type="xgboost"):
    """Survey-weighted S-learner outcome model: confounders + exposure as a feature.

    model_type="xgboost" matches notebook 07's default (fixed, modest
    hyperparameters, no tuning). model_type="logistic_regression" is the
    documented alternate.
    """
    outcome_numeric = list(numeric_confounders) + ["exposure"]

    if model_type == "xgboost":
        from xgboost import XGBClassifier

        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="median", add_indicator=True), outcome_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_confounders),
        ])
        clf = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            random_state=random_state, n_jobs=-1, tree_method="hist",
            n_estimators=200, max_depth=4, learning_rate=0.05,
        )
    elif model_type == "logistic_regression":
        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), outcome_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_confounders),
        ])
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("pre", preprocessor), ("clf", clf)])


# ---------------------------------------------------------------------------
# Cross-fitting
# ---------------------------------------------------------------------------

def cross_fit_nuisances(
    W, exposure, outcome, groups, sample_weight,
    numeric_confounders, categorical_confounders,
    n_folds=5, random_state=42, outcome_model_type="xgboost",
):
    """Respondent-grouped StratifiedGroupKFold cross-fitting for e_hat, mu1_hat, mu0_hat.

    Identical discipline to notebooks/v2/07 section 4/6: no respondent's
    records appear in both the training and held-out portions of any fold,
    and no model ever predicts on data it was trained on.

    Returns (e_hat, mu1_hat, mu0_hat, fold_assignments).
    """
    n = len(W)
    W = W.reset_index(drop=True)
    exposure = np.asarray(exposure)
    outcome = np.asarray(outcome)
    groups = np.asarray(groups)
    sample_weight = np.asarray(sample_weight)

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_indices = list(splitter.split(W, exposure, groups))
    fold_assignments = np.full(n, -1, dtype=int)

    for fold_id, (train_idx, test_idx) in enumerate(fold_indices):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        leak = train_groups & test_groups
        assert not leak, f"Fold {fold_id}: {len(leak)} groups appear in both train and held-out portions."
        fold_assignments[test_idx] = fold_id

    covered = np.concatenate([test_idx for _, test_idx in fold_indices])
    assert len(covered) == n and len(set(covered.tolist())) == n, \
        "Row coverage across folds is incomplete or overlapping."

    e_hat = np.full(n, np.nan)
    mu1_hat = np.full(n, np.nan)
    mu0_hat = np.full(n, np.nan)

    W_with_exposure = W.copy()
    W_with_exposure["exposure"] = exposure

    for fold_id, (train_idx, test_idx) in enumerate(fold_indices):
        prop_pipeline = make_propensity_pipeline(numeric_confounders, categorical_confounders, random_state)
        prop_pipeline.fit(W.iloc[train_idx], exposure[train_idx], clf__sample_weight=sample_weight[train_idx])
        e_hat[test_idx] = prop_pipeline.predict_proba(W.iloc[test_idx])[:, 1]

        outcome_pipeline = make_outcome_pipeline(
            numeric_confounders, categorical_confounders, random_state, outcome_model_type
        )
        outcome_pipeline.fit(
            W_with_exposure.iloc[train_idx], outcome[train_idx], clf__sample_weight=sample_weight[train_idx]
        )

        held_out_W = W_with_exposure.iloc[test_idx].copy()
        as_treated = held_out_W.copy()
        as_treated["exposure"] = 1
        mu1_hat[test_idx] = outcome_pipeline.predict_proba(as_treated)[:, 1]

        as_control = held_out_W.copy()
        as_control["exposure"] = 0
        mu0_hat[test_idx] = outcome_pipeline.predict_proba(as_control)[:, 1]

    assert not np.isnan(e_hat).any(), "NaN found in e_hat."
    assert not np.isnan(mu1_hat).any(), "NaN found in mu1_hat."
    assert not np.isnan(mu0_hat).any(), "NaN found in mu0_hat."

    return e_hat, mu1_hat, mu0_hat, fold_assignments


# ---------------------------------------------------------------------------
# AIPW pseudo-outcomes and point estimates (identical formula to notebook 07)
# ---------------------------------------------------------------------------

def aipw_psi(y, a, e, mu1, mu0):
    """Per-record AIPW pseudo-outcomes under treatment (psi1) and control (psi0)."""
    psi1 = mu1 + (a / e) * (y - mu1)
    psi0 = mu0 + ((1 - a) / (1 - e)) * (y - mu0)
    return psi1, psi0


def aipw_point_estimate(y, a, e, mu1, mu0, weights):
    """Survey-weighted AIPW risk under treatment/control, risk difference, risk ratio."""
    psi1, psi0 = aipw_psi(y, a, e, mu1, mu0)
    r1 = np.average(psi1, weights=weights)
    r0 = np.average(psi0, weights=weights)
    return r1, r0, r1 - r0, r1 / r0


# ---------------------------------------------------------------------------
# Full cross-fitted AIPW + PSU-bootstrap pipeline (round-agnostic convenience
# wrapper -- used by Extension 3 to run the identical Notebook-07 pipeline
# independently within each NFHS round)
# ---------------------------------------------------------------------------

def run_cross_fitted_aipw(
    df, confounder_cols, numeric_confounders, categorical_confounders,
    outcome_col="csection", exposure_col="exposure", weight_col="sample_weight_normalized",
    group_col="respondent_id", psu_col="cluster_number", stratum_col="sample_stratum_v022",
    n_folds=5, random_state=42, n_bootstrap=500, bootstrap_seed=42, outcome_model_type="xgboost",
):
    """End-to-end cross-fitted AIPW pipeline, mirroring notebooks/v2/07 exactly:
    respondent-grouped cross-fitting, survey-weighted nuisance models, the AIPW
    point estimate, and a PSU-cluster bootstrap within strata.

    Returns a dict of point estimates, bootstrap replicate arrays, and the
    row-level nuisance predictions.
    """
    W = df[confounder_cols].copy()
    for col in categorical_confounders:
        W[col] = W[col].astype("string").fillna("Missing").astype(str)

    exposure = df[exposure_col].to_numpy()
    outcome = df[outcome_col].astype(int).to_numpy()
    weight = df[weight_col].to_numpy()
    groups = df[group_col].to_numpy()
    psu = df[psu_col].to_numpy()
    stratum = df[stratum_col].to_numpy()

    e_hat_raw, mu1_hat, mu0_hat, fold_assignments = cross_fit_nuisances(
        W, exposure, outcome, groups, weight, numeric_confounders, categorical_confounders,
        n_folds=n_folds, random_state=random_state, outcome_model_type=outcome_model_type,
    )
    e_hat_clipped = np.clip(e_hat_raw, 1e-6, 1 - 1e-6)

    r1, r0, rd, rr = aipw_point_estimate(outcome, exposure, e_hat_clipped, mu1_hat, mu0_hat, weight)

    rng = np.random.RandomState(bootstrap_seed)
    rd_reps = np.empty(n_bootstrap)
    rr_reps = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = resample_indices_within_strata(psu, stratum, rng)
        _, _, rd_b, rr_b = aipw_point_estimate(
            outcome[idx], exposure[idx], e_hat_clipped[idx], mu1_hat[idx], mu0_hat[idx], weight[idx]
        )
        rd_reps[b] = rd_b
        rr_reps[b] = rr_b

    return {
        "r1": r1, "r0": r0, "risk_difference": rd, "risk_ratio": rr,
        "rd_reps": rd_reps, "rr_reps": rr_reps,
        "rd_ci": (float(np.percentile(rd_reps, 2.5)), float(np.percentile(rd_reps, 97.5))),
        "rr_ci": (float(np.percentile(rr_reps, 2.5)), float(np.percentile(rr_reps, 97.5))),
        "e_hat_clipped": e_hat_clipped, "mu1_hat": mu1_hat, "mu0_hat": mu0_hat,
        "fold_assignments": fold_assignments, "n": len(df),
    }


# ---------------------------------------------------------------------------
# PSU-cluster bootstrap (identical to notebook 07 section 10)
# ---------------------------------------------------------------------------

def resample_indices_within_strata(psu_ids, stratum_ids, rng):
    """Resample PSUs with replacement, within each sampling stratum."""
    psu_ids = np.asarray(psu_ids)
    stratum_ids = np.asarray(stratum_ids)
    row_positions = np.arange(len(psu_ids))
    out = []
    for s in np.unique(stratum_ids):
        stratum_mask = stratum_ids == s
        psus_in_stratum = np.unique(psu_ids[stratum_mask])
        n_psu = len(psus_in_stratum)
        drawn_psus = rng.choice(psus_in_stratum, size=n_psu, replace=True)
        for drawn_psu in drawn_psus:
            out.append(row_positions[stratum_mask & (psu_ids == drawn_psu)])
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Balance / overlap diagnostics (identical to notebook 06)
# ---------------------------------------------------------------------------

def effective_sample_size(weights):
    weights = np.asarray(weights)
    return (weights.sum() ** 2) / (weights ** 2).sum()


def smd(x1, w1, x0, w0):
    """Weighted standardized mean difference between two groups."""
    def weighted_mean_var(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        mean = np.average(x, weights=w)
        var = np.average((x - mean) ** 2, weights=w)
        return mean, var

    m1, v1 = weighted_mean_var(x1, w1)
    m0, v0 = weighted_mean_var(x0, w0)
    pooled_sd = np.sqrt((v1 + v0) / 2)
    return (m1 - m0) / pooled_sd if pooled_sd > 0 else np.nan
