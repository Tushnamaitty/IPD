"""Cross-fitted Super Learner AIPW and pooled logistic-fluctuation TMLE.

Run from the repository root. This is a methodological sensitivity analysis,
not an estimate of the effect of an intervention on facility policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC = ["birth_order", "wealth_index", "education_years"]
CATEGORICAL = ["residence", "religion", "social_group", "twin_order", "state"]
CLIP = 0.01  # prespecified practical stability threshold; report its impact
TINY = 1e-6


def make_features(df):
    x = df[NUMERIC + CATEGORICAL].copy()
    for c in CATEGORICAL:
        x[c] = x[c].astype("string").fillna("Missing").astype(str)
    return x


def learner(kind, seed, numeric):
    if kind == "logistic":
        pre = ColumnTransformer([
            ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                               ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                          unknown_value=-1))]), CATEGORICAL),
        ], sparse_threshold=0)
        clf = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                             learning_rate=0.06, l2_regularization=1,
                                             random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_predict(model, x_train, y_train, w_train, x_predict):
    model.fit(x_train, y_train, clf__sample_weight=w_train)
    return np.clip(model.predict_proba(x_predict)[:, 1], TINY, 1 - TINY)


def stack_predictions(x, y, weights, groups, x_predict, inner_folds, seed, numeric):
    """Learn convex ensemble weights using training-only, grouped OOF predictions."""
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes must occur in each outer training sample")
    cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    oof = np.full((len(y), 2), np.nan)
    for tr, val in cv.split(x, y, groups):
        if len(set(groups[tr]) & set(groups[val])):
            raise AssertionError("Respondent leakage in inner folds")
        for j, kind in enumerate(("logistic", "tree")):
            oof[val, j] = fit_predict(learner(kind, seed, numeric), x.iloc[tr], y[tr],
                                      weights[tr], x.iloc[val])
    if not np.isfinite(oof).all():
        raise AssertionError("Incomplete inner OOF prediction coverage")
    norm_w = weights / np.sum(weights)

    def loss(alpha):
        p = np.clip(alpha * oof[:, 0] + (1 - alpha) * oof[:, 1], TINY, 1 - TINY)
        return -np.sum(norm_w * (y * np.log(p) + (1 - y) * np.log1p(-p)))

    opt = minimize_scalar(loss, bounds=(0, 1), method="bounded")
    candidates = [(0., loss(0.)), (1., loss(1.)), (float(opt.x), loss(opt.x))]
    alpha = min(candidates, key=lambda v: v[1])[0]
    pred = []
    for kind in ("logistic", "tree"):
        pred.append(fit_predict(learner(kind, seed, numeric), x, y, weights, x_predict))
    return alpha * pred[0] + (1 - alpha) * pred[1], {"logistic": alpha, "tree": 1-alpha, "inner_logloss": loss(alpha)}


def crossfit(df, outer_folds=5, inner_folds=3, seed=42):
    x = make_features(df)
    a = (df.facility_type == "private").to_numpy(dtype=int)
    y = df.csection.to_numpy(dtype=int)
    w = df.sample_weight_normalized.to_numpy(dtype=float)
    groups = df.respondent_id.to_numpy()
    fold = np.full(len(df), -1)
    e, q0, q1 = [np.full(len(df), np.nan) for _ in range(3)]
    model_weights = []
    cv = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    for fold_id, (tr, val) in enumerate(cv.split(x, a, groups)):
        if len(set(groups[tr]) & set(groups[val])):
            raise AssertionError("Respondent leakage in outer folds")
        print(f"Outer fold {fold_id+1}/{outer_folds}: {len(tr)} training, {len(val)} held out", flush=True)
        e[val], ew = stack_predictions(x.iloc[tr].reset_index(drop=True), a[tr], w[tr],
                                       groups[tr], x.iloc[val], inner_folds, seed+fold_id, NUMERIC)
        train_x = x.iloc[tr].copy().reset_index(drop=True)
        train_x["exposure"] = a[tr]
        held_x = x.iloc[val].copy()
        held_x["exposure"] = 0
        predict_x = pd.concat([held_x, held_x.assign(exposure=1)], ignore_index=True)
        both, qw = stack_predictions(train_x, y[tr], w[tr], groups[tr],
                                      predict_x, inner_folds, seed+fold_id+100,
                                      NUMERIC+["exposure"])
        q0[val], q1[val] = both[:len(val)], both[len(val):]
        fold[val] = fold_id
        model_weights.append({"fold": fold_id, "propensity": ew, "outcome": qw})
    if (fold < 0).any() or not all(np.isfinite(v).all() for v in (e, q0, q1)):
        raise AssertionError("Missing held-out prediction")
    return a, y, w, e, q0, q1, fold, model_weights


def estimate(a, y, w, e, q0, q1):
    """AIPW and bounded TMLE using the same cross-fitted nuisance predictions."""
    e = np.clip(e, CLIP, 1 - CLIP)
    q0, q1 = np.clip(q0, TINY, 1-TINY), np.clip(q1, TINY, 1-TINY)
    def arm_target(arm, q, g):
        observed = a == arm
        clever = 1/g[observed]
        def score(epsilon):
            return np.sum(w[observed]*clever*(y[observed] -
                expit(logit(q[observed])+epsilon*clever))) / np.sum(w)
        epsilon = brentq(score, -50., 50.)
        return expit(logit(q)+epsilon/g), epsilon, score(epsilon)

    updated1, epsilon1, score1 = arm_target(1, q1, e)
    updated0, epsilon0, score0 = arm_target(0, q0, 1-e)
    tmle1, tmle0 = np.average(updated1, weights=w), np.average(updated0, weights=w)
    aipw1 = np.average(q1 + a/e*(y-q1), weights=w)
    aipw0 = np.average(q0 + (1-a)/(1-e)*(y-q0), weights=w)
    return {
        "aipw": {"private_risk": aipw1, "public_risk": aipw0,
                 "risk_difference": aipw1-aipw0, "risk_ratio": aipw1/aipw0},
        "tmle": {"private_risk": tmle1, "public_risk": tmle0,
                 "risk_difference": tmle1-tmle0, "risk_ratio": tmle1/tmle0,
                 "fluctuation_epsilon_private": epsilon1,
                 "fluctuation_epsilon_public": epsilon0,
                 "target_score_private": score1, "target_score_public": score0},
    }


def bootstrap(a, y, w, e, q0, q1, psu, strata, count, seed):
    """Conditional intervals: resample PSUs within strata, keep nuisance fits fixed."""
    rng = np.random.default_rng(seed)
    blocks = []
    frame = pd.DataFrame({"stratum": strata, "psu": psu, "row": np.arange(len(a))})
    for _, st in frame.groupby("stratum", sort=False):
        blocks.append([v.row.to_numpy() for _, v in st.groupby("psu", sort=False)])
    reps = {k: {m: [] for m in ("risk_difference", "risk_ratio")} for k in ("aipw", "tmle")}
    for b in range(count):
        idx = np.concatenate([np.concatenate([block[j] for j in rng.integers(len(block), size=len(block))])
                              for block in blocks])
        res = estimate(a[idx], y[idx], w[idx], e[idx], q0[idx], q1[idx])
        for name in reps:
            for measure in reps[name]:
                reps[name][measure].append(res[name][measure])
        if (b+1) % 50 == 0:
            print(f"Bootstrap {b+1}/{count}", flush=True)
    return {name: {measure+"_ci95": np.percentile(v, [2.5, 97.5]).tolist()
                   for measure, v in values.items()} for name, values in reps.items()}


def run(args):
    df = pd.read_parquet(args.data)
    required = NUMERIC + CATEGORICAL + ["facility_type", "csection", "respondent_id",
                                      "sample_weight_normalized", "cluster_number",
                                      "sample_stratum_v022", "delivery_place_code"]
    missing = sorted(set(required)-set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not args.allow_test_cohort and len(df) != 201311:
        raise ValueError(f"Expected 201311 V2 input rows; got {len(df)}")
    df = df.loc[df.facility_type.isin(["public", "private"])].reset_index(drop=True)
    if not args.allow_test_cohort and len(df) != 200794:
        raise ValueError(f"Expected 200794 public/private deliveries; got {len(df)}")
    if df[required].loc[:, ["facility_type", "csection", "respondent_id", "cluster_number",
                           "sample_stratum_v022"]].isna().any().any():
        raise ValueError("Missing exposure, outcome or survey/group identifiers")
    if not set(df.csection.unique()) <= {0, 1}:
        raise ValueError("Outcome must be binary")
    w = df.sample_weight_normalized.to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Survey weights must be finite and positive")
    a, y, w, e, q0, q1, fold, blends = crossfit(df, args.outer_folds, args.inner_folds, args.seed)
    result = estimate(a, y, w, e, q0, q1)
    result["diagnostics"] = {
        "input_rows": 201311 if not args.allow_test_cohort else None,
        "analytic_rows": len(df), "public_rows": int((a == 0).sum()),
        "private_rows": int((a == 1).sum()),
        "propensity_quantiles": np.quantile(e, [0, .01, .05, .5, .95, .99, 1]).tolist(),
        "propensity_clipped_at_0.01": int(((e < CLIP) | (e > 1-CLIP)).sum()),
        "propensity_outside_0.05_0.95": int(((e < .05) | (e > .95)).sum()),
        "fold_weights": blends,
    }
    result["delivery_place_breakdown"] = [
        {"sector": str(sector), "m15": str(code), "n": int(len(sub)),
         "weighted_csection_rate": float(np.average(sub.csection, weights=sub.sample_weight_normalized))}
        for (sector, code), sub in df.groupby(["facility_type", "delivery_place_code"], dropna=False)
    ]
    if args.bootstrap:
        result["conditional_bootstrap"] = bootstrap(a, y, w, e, q0, q1,
            df.cluster_number.to_numpy(), df.sample_stratum_v022.to_numpy(),
            args.bootstrap, args.seed)
    result["interpretation"] = "Survey-weighted standardized public/private association; not a policy intervention effect."
    result["ci_note"] = "Conditional percentile PSU bootstrap with fixed nuisance fits; does not include nuisance training uncertainty."
    result["cohort_type"] = "TEST DATA ONLY" if args.allow_test_cohort else "NFHS-5 V2"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"aipw": result["aipw"], "tmle": result["tmle"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/df_model_v2.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metadata/professor_methods_sl_tmle.json"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-test-cohort", action="store_true", help="Synthetic validation only; labels output as TEST DATA ONLY")
    run(parser.parse_args())
