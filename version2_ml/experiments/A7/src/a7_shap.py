"""A7 SHAP analysis with family aggregation and rank stability.

TreeExplainer on the fitted XGBoost model; contributions are recorded in the
margin (log-odds) space on a seeded validation sample. Encoded one-hot columns
are aggregated back to their source constructs using the fitted
ColumnTransformer's own category lists, so no column-name parsing is needed.
SHAP values are predictive attributions, not causal effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def family_map(classifier) -> dict[str, str]:
    pipeline = classifier.preprocessor.pipeline
    manifest = classifier.preprocessor.manifest
    mapping = {}
    for column in manifest.numerical:
        mapping[f"numeric__{column}"] = column
    encoder = pipeline.named_transformers_["categorical"].named_steps["onehot"]
    categories = encoder.categories_
    for column, values in zip(manifest.categorical, categories):
        for value in values:
            mapping[f"categorical__{column}_{value}"] = column
    return mapping


def explained_feature_names(classifier) -> list[str]:
    return list(classifier.preprocessor.pipeline.get_feature_names_out())


def shap_family_table(
    classifier,
    frame: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> dict:
    import shap

    names = explained_feature_names(classifier)
    mapping = family_map(classifier)
    missing = sorted(set(names) - set(mapping))
    if missing:
        raise RuntimeError(f"Encoded features without a family mapping: {missing}")
    if len(frame) > sample_size:
        sample = frame.sample(n=sample_size, random_state=seed)
    else:
        sample = frame
    transformed = classifier.preprocessor.transform(sample)
    explainer = shap.TreeExplainer(classifier.estimator)
    values = explainer.shap_values(transformed)
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise RuntimeError("Unexpected SHAP value shape")
    mean_abs = np.abs(values).mean(axis=0)
    by_family = {}
    for index, name in enumerate(names):
        family = mapping[name]
        by_family.setdefault(family, 0.0)
        by_family[family] += float(mean_abs[index])
    total = float(sum(by_family.values()))
    ordered = sorted(by_family.items(), key=lambda item: item[1], reverse=True)
    table = []
    for rank, (family, value) in enumerate(ordered, start=1):
        table.append(
            {
                "family": family,
                "mean_abs_shap_margin": round(value, 6),
                "share_of_total": round(value / total, 6) if total else 0.0,
                "rank": rank,
            }
        )
    return {
        "n_sample_rows": int(len(sample)),
        "n_encoded_features": int(len(names)),
        "n_families": len(table),
        "explainer": "shap.TreeExplainer, tree_path_dependent, margin (log-odds) contributions",
        "families": table,
    }


def _family_ranks(table: dict) -> dict[str, int]:
    return {row["family"]: row["rank"] for row in table["families"]}


def rank_stability(tables: list[dict]) -> dict:
    from scipy.stats import spearmanr

    families = [row["family"] for row in tables[0]["families"]]
    matrix = []
    for table in tables:
        ranks = _family_ranks(table)
        if set(ranks) != set(families):
            raise RuntimeError("SHAP family sets differ across models")
        matrix.append([ranks[family] for family in families])
    correlations = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            rho = spearmanr(matrix[i], matrix[j]).statistic
            correlations.append(float(rho))
    return {
        "families": families,
        "rank_matrix": matrix,
        "pairwise_spearman": [round(value, 6) for value in correlations],
        "min_pairwise_spearman": round(min(correlations), 6) if correlations else None,
        "n_models": len(matrix),
    }
