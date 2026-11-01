"""Aggregate output schema (handoff Section 8) shared by A1/A4/A7 outputs.

Rows describe experiment runs, never births. Absent fields are written as an
explicit ``not_applicable`` string, never silently zero-filled. Small positive
counts can be disclosure-suppressed via :func:`suppress_small_count`.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

NOT_APPLICABLE = "not_applicable"
UNCERTAINTY_STATUS = "not_validated_point_estimate_only"
DEFAULT_SUPPRESSION_THRESHOLD = 10

COLUMNS = (
    "experiment_id",
    "wave",
    "cohort_rule",
    "evaluation_design",
    "split_id",
    "fold",
    "source_domain",
    "target_domain",
    "model",
    "feature_tier",
    "baseline_id",
    "n_births",
    "n_mothers",
    "n_events",
    "prevalence",
    "roc_auc",
    "average_precision",
    "brier_score",
    "calibration_intercept",
    "calibration_slope",
    "threshold",
    "precision",
    "recall",
    "f1",
    "weighted_descriptive_flag",
    "uncertainty_status",
    "repo_commit",
    "seed",
    "software_versions",
    "feature_manifest_hash",
    "split_manifest_hash",
)


def build_row(**fields) -> dict:
    """Return a schema-ordered row; absent fields become explicit not_applicable."""
    unknown = sorted(set(fields) - set(COLUMNS))
    if unknown:
        raise ValueError(f"Unknown schema fields: {unknown}")
    row = {}
    for column in COLUMNS:
        if column not in fields:
            row[column] = NOT_APPLICABLE
            continue
        value = fields[column]
        if value is None:
            raise ValueError(
                f"Field {column}: use explicit '{NOT_APPLICABLE}' instead of None"
            )
        if isinstance(value, float) and math.isnan(value):
            raise ValueError(
                f"Field {column}: use explicit '{NOT_APPLICABLE}' instead of NaN"
            )
        row[column] = value
    return row


def suppress_small_count(value, *, threshold: int = DEFAULT_SUPPRESSION_THRESHOLD):
    """Mask positive counts below the disclosure threshold; zero stays zero."""
    count = int(value)
    if threshold < 2:
        raise ValueError("Suppression threshold must be at least 2")
    if 0 < count < threshold:
        return f"<{threshold} (suppressed)"
    return count


def write_rows(rows, destination) -> Path:
    """Write schema rows to CSV in locked column order."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        unknown = sorted(set(row) - set(COLUMNS))
        if unknown:
            raise ValueError(f"Unknown schema fields in row: {unknown}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, NOT_APPLICABLE) for column in COLUMNS})
    return path
