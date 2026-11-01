"""A1 state/area context features for matched NFHS-5 ablations.

State/UT labels are verbatim from the authorized local NFHS-5 DTA value
metadata (read 2026-10-17): 36 units, code 26 absent because Dadra & Nagar
Haveli and Daman & Diu are merged at code 25. State codes are never treated
as numeric ordinals. No external area descriptor is implemented: no time-valid
source with full provenance is available, so the handoff fallback applies
(categorical state plus residence; residence is already inside the eight-core
F1). See docs/A1_A4_A7_REWORK_PLAN.md and docs/A1_A4_A7_DISCREPANCY_LOG.md.
"""
from __future__ import annotations

import pandas as pd

try:
    from feature_builder import CORE_NAMES
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Add version2_ml to sys.path before importing a1_state_context"
    ) from exc

STATE_LABELS = {
    1: "jammu & kashmir",
    2: "himachal pradesh",
    3: "punjab",
    4: "chandigarh",
    5: "uttarakhand",
    6: "haryana",
    7: "nct of delhi",
    8: "rajasthan",
    9: "uttar pradesh",
    10: "bihar",
    11: "sikkim",
    12: "arunachal pradesh",
    13: "nagaland",
    14: "manipur",
    15: "mizoram",
    16: "tripura",
    17: "meghalaya",
    18: "assam",
    19: "west bengal",
    20: "jharkhand",
    21: "odisha",
    22: "chhattisgarh",
    23: "madhya pradesh",
    24: "gujarat",
    25: "dadra & nagar haveli and daman & diu",
    27: "maharashtra",
    28: "andhra pradesh",
    29: "karnataka",
    30: "goa",
    31: "lakshadweep",
    32: "kerala",
    33: "tamil nadu",
    34: "puducherry",
    35: "andaman & nicobar islands",
    36: "telangana",
    37: "ladakh",
}
UNKNOWN_STATE_TOKEN = "unknown_state"
STATE_FEATURE = "state_ut"
INTERACTION_FEATURE = "state_ut_x_residence"
INTERACTION_SEPARATOR = " | "

F1_NUMERICAL = (
    "maternal_age_at_index_birth",
    "education_years",
    "wealth_index",
    "birth_order",
)
F1_CATEGORICAL = ("facility_type", "residence", "religion", "twin_order")
F1_FEATURES = CORE_NAMES
A1_FEATURE_NAMES = CORE_NAMES + (STATE_FEATURE,)
A1_INTERACTION_FEATURE_NAMES = A1_FEATURE_NAMES + (INTERACTION_FEATURE,)

EXTERNAL_CONTEXT_SPEC_FIELDS = (
    "variable",
    "geo_level",
    "source",
    "reference_year",
    "coverage",
    "join_key",
    "post_outcome_timing_risk",
    "missingness_plan",
    "fallback",
)


def map_states(values) -> pd.Series:
    """Map raw NFHS-5 v024 codes to verified labels; missing becomes explicit token."""
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    observed = numeric.dropna()
    if ((observed % 1) != 0).any():
        raise ValueError("State codes must be integer-valued")
    unknown = sorted({int(v) for v in observed} - set(STATE_LABELS))
    if unknown:
        raise ValueError(
            f"Unmapped NFHS-5 state codes (consistent mapping required): {unknown}"
        )
    labels = numeric.map(
        lambda value: STATE_LABELS[int(value)]
        if pd.notna(value)
        else UNKNOWN_STATE_TOKEN
    )
    labels = labels.astype("string")
    labels.name = STATE_FEATURE
    return labels


def build_state_frame(
    core: pd.DataFrame, state_labels, *, include_interaction: bool = False
) -> pd.DataFrame:
    """Append state (and optional residence interaction) in locked column order."""
    if tuple(core.columns) != CORE_NAMES:
        raise ValueError(
            "Core frame must contain exactly the eight CORE_NAMES columns in order"
        )
    labels = pd.Series(state_labels).astype("string")
    if len(labels) != len(core):
        raise ValueError("State label length mismatch")
    if not labels.index.equals(core.index):
        raise ValueError("State labels must be aligned to the core frame index")
    if labels.isna().any():
        raise ValueError("Mapped state labels must not be missing")
    allowed = set(STATE_LABELS.values()) | {UNKNOWN_STATE_TOKEN}
    unexpected = sorted(set(labels.unique()) - allowed)
    if unexpected:
        raise ValueError(f"Unrecognized mapped state labels: {unexpected}")
    frame = core.copy()
    frame[STATE_FEATURE] = labels.to_numpy()
    if include_interaction:
        residence = frame["residence"].astype("string")
        frame[INTERACTION_FEATURE] = (
            frame[STATE_FEATURE] + INTERACTION_SEPARATOR + residence
        ).astype("string")
        expected = A1_INTERACTION_FEATURE_NAMES
    else:
        expected = A1_FEATURE_NAMES
    if tuple(frame.columns) != expected:
        raise RuntimeError("A1 feature order violates the locked contract")
    return frame


def assert_unique_birth_keys(frame: pd.DataFrame, *, key_columns=("caseid", "bidx")) -> None:
    missing_columns = [column for column in key_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing birth key columns: {missing_columns}")
    keys = frame[list(key_columns)]
    if keys.isna().any().any():
        raise ValueError("Missing birth key components")
    for column in key_columns:
        blank = frame[column].astype("string").str.strip().eq("")
        if blank.any():
            raise ValueError(f"Blank birth key component in {column}")
    duplicates = int(keys.duplicated().sum())
    if duplicates:
        raise ValueError(f"Repeated birth keys present: {duplicates}")


def support_screen(cells, target, *, min_rows: int, min_class: int) -> dict:
    """Training-row support screen; counts stay in memory and are masked on write."""
    cell_series = pd.Series(cells).astype("string")
    labels = pd.to_numeric(pd.Series(target), errors="coerce")
    if len(cell_series) != len(labels):
        raise ValueError("Support screen length mismatch")
    if cell_series.isna().any() or labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Support screen requires nonmissing cells and binary target")
    table = pd.DataFrame(
        {"cell": cell_series.to_numpy(), "y": labels.astype(int).to_numpy()}
    )
    grouped = table.groupby("cell", sort=True)["y"].agg(["size", "sum"])
    report = []
    for cell, row in grouped.iterrows():
        n_rows = int(row["size"])
        n_positive = int(row["sum"])
        n_negative = n_rows - n_positive
        passes = n_rows >= min_rows and n_positive >= min_class and n_negative >= min_class
        report.append(
            {
                "cell": str(cell),
                "n_rows": n_rows,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "passes": bool(passes),
            }
        )
    failing = [cell for cell in report if not cell["passes"]]
    if not report:
        gate_passed, reason = False, "no_cells"
    elif failing:
        names = "; ".join(cell["cell"] for cell in failing[:5])
        if len(failing) > 5:
            names += "; ..."
        gate_passed, reason = False, f"insufficient_support: {names}"
    else:
        gate_passed, reason = True, "all_cells_pass"
    return {
        "min_rows": int(min_rows),
        "min_class": int(min_class),
        "cells": report,
        "n_cells": len(report),
        "gate_passed": bool(gate_passed),
        "reason": reason,
    }


def assert_external_context_specs(specs) -> dict:
    """Date-provenance gate for proposed external area descriptors (none used)."""
    if specs is None:
        specs = []
    for index, spec in enumerate(specs):
        missing = sorted(set(EXTERNAL_CONTEXT_SPEC_FIELDS) - set(spec or {}))
        if missing:
            raise ValueError(
                f"External context spec {index} missing provenance fields: {missing}"
            )
    if specs:
        return {"n_specs": len(specs), "status": "PROVENANCE_COMPLETE"}
    return {"n_specs": 0, "status": "NO_EXTERNAL_CONTEXT_DECLARED"}
