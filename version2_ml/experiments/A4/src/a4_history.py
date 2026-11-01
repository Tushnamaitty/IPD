"""A4 observed-history builder for the four-category prior-delivery variable.

Frozen semantics (handoff section 6; decisions recorded in
docs/A1_A4_A7_DISCREPANCY_LOG.md):
- FIRST_RECORDED_LIVE_BIRTH: the index birth is the first recorded live birth
  (bord == 1); live-birth order, not first pregnancy.
- OBSERVED_PREVIOUS_INSTITUTIONAL_CS: bord > 1 and at least one strictly
  earlier-calendar-month eligible survey-window birth from the same mother was
  an institutional delivery (m15 broad institutional rule) with m17 == 1.
- HIGHER_ORDER_NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS: bord > 1 and at least one
  strictly earlier institutional survey-window birth was observed, none with
  m17 == 1. This is an observed negative within survey recall, not a lifetime
  negatation.
- HIGHER_ORDER_EARLIER_INSTITUTIONAL_HISTORY_UNOBSERVED: bord > 1 and no
  strictly earlier institutional survey-window record was found; includes
  earlier noninstitutional births and older births outside survey recall.

Missing or ambiguous records never convert to a negative: they yield the
explicit UNKNOWN_HISTORY level plus a history_quality_flag token. Same-month
multiple births never count as previous deliveries (strict b3 ordering only).
The timeline is built in memory and never exported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from a4_facility_detail import (
    INSTITUTIONAL_CODES,
    KNOWN_CODES,
    OTHER_UNKNOWN_SECTOR_CODE,
)

FIRST_RECORDED_LIVE_BIRTH = "first_recorded_live_birth"
OBSERVED_PREVIOUS_INSTITUTIONAL_CS = "observed_previous_institutional_cs"
NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS = (
    "higher_order_no_observed_previous_institutional_cs"
)
UNOBSERVED_HISTORY = "higher_order_earlier_institutional_history_unobserved"
UNKNOWN_HISTORY = "unknown_history"

HISTORY_LEVELS = (
    FIRST_RECORDED_LIVE_BIRTH,
    OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
    NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
    UNOBSERVED_HISTORY,
    UNKNOWN_HISTORY,
)

HISTORY_FEATURE = "history_four_level"
HAS_ANY_FEATURE = "has_any_observed_earlier_institutional_birth"
QUALITY_FEATURE = "history_quality_flag"

QUALITY_OK = "ok"
QUALITY_INCONSISTENT = "inconsistent_bord1_with_earlier_records"
QUALITY_AMBIGUOUS = "ambiguous_prior_institutional_outcome"
QUALITY_MISSING_DATES = "timeline_incomplete_missing_dates"

HAS_ANY_TRUE = "true"
HAS_ANY_FALSE = "false"
HAS_ANY_UNKNOWN = "unknown"

M17_VALID = (0, 1)
M17_CODED_MISSING = (8, 9)

REQUIRED_ROSTER_COLUMNS = ("caseid", "bidx", "b3", "b19", "m15", "m17")
REQUIRED_INDEX_COLUMNS = ("caseid", "bidx", "b3", "bord", "m15")


def _strict_numeric(values, label: str) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    original = pd.Series(values)
    if (original.notna() & numeric.isna()).any():
        raise ValueError(f"{label}: non-numeric values present")
    observed = numeric.dropna()
    if ((observed % 1) != 0).any():
        raise ValueError(f"{label}: non-integer codes present")
    return numeric


def _validate_roster(roster: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_ROSTER_COLUMNS if c not in roster.columns]
    if missing:
        raise ValueError(f"History roster is missing columns: {missing}")
    frame = roster[list(REQUIRED_ROSTER_COLUMNS)].copy()
    keys = frame[["caseid", "bidx"]]
    if keys.isna().any().any():
        raise ValueError("History roster has missing birth key components")
    if int(keys.duplicated().sum()):
        raise ValueError(
            "History roster has duplicate (caseid, bidx) records; "
            "ordering would be ambiguous"
        )
    frame["m15"] = _strict_numeric(frame["m15"], "m15")
    frame["m17"] = _strict_numeric(frame["m17"], "m17")
    frame["b3"] = _strict_numeric(frame["b3"], "b3")
    frame["b19"] = _strict_numeric(frame["b19"], "b19")
    bad_facility = sorted(set(frame["m15"].dropna().unique()) - set(KNOWN_CODES))
    if bad_facility:
        raise ValueError(f"Unexpected NFHS-5 m15 facility codes: {bad_facility}")
    bad_outcome = sorted(
        set(frame["m17"].dropna().unique()) - set(M17_VALID + M17_CODED_MISSING)
    )
    if bad_outcome:
        raise ValueError(f"Invalid m17 outcome codes in history roster: {bad_outcome}")
    return frame


def _validate_index(index: pd.DataFrame, include_other_unknown_sector: bool) -> pd.DataFrame:
    missing = [c for c in REQUIRED_INDEX_COLUMNS if c not in index.columns]
    if missing:
        raise ValueError(f"Index births are missing columns: {missing}")
    frame = index[list(REQUIRED_INDEX_COLUMNS)].copy()
    keys = frame[["caseid", "bidx"]]
    if keys.isna().any().any():
        raise ValueError("Index births have missing birth key components")
    if int(keys.duplicated().sum()):
        raise ValueError("Index births contain duplicate (caseid, bidx) records")
    frame["b3"] = _strict_numeric(frame["b3"], "b3")
    frame["m15"] = _strict_numeric(frame["m15"], "m15")
    frame["bord"] = _strict_numeric(frame["bord"], "bord")
    if frame["bord"].isna().any() or (frame["bord"] < 1).any():
        raise ValueError("Index births require valid positive birth order values")
    if frame["m15"].isna().any():
        raise ValueError("Index births require a recorded m15 facility code")
    allowed = set(INSTITUTIONAL_CODES)
    if not include_other_unknown_sector:
        allowed = allowed - {OTHER_UNKNOWN_SECTOR_CODE}
    bad = sorted(set(frame["m15"].dropna().unique()) - allowed)
    if bad:
        raise ValueError(
            "Index births contain non-institutional m15 codes for this rule: "
            f"{bad}"
        )
    return frame


def build_history_features(
    roster: pd.DataFrame,
    index: pd.DataFrame,
    *,
    include_other_unknown_sector: bool = True,
    wave: str = "NFHS-5",
) -> tuple[pd.DataFrame, dict]:
    if wave != "NFHS-5":
        raise NotImplementedError(
            "Observed-history builder is implemented only for verified NFHS-5 codes."
        )
    frame = _validate_roster(roster)
    idx = _validate_index(index, include_other_unknown_sector)

    eligible = frame[frame["b19"].between(0, 59)]
    n_outside_frame = int(len(frame) - len(eligible))
    mother_missing_date = (
        eligible.assign(_miss=eligible["b3"].isna())
        .groupby("caseid")["_miss"]
        .any()
    )

    timeline = eligible[eligible["b3"].notna()].copy()
    if not include_other_unknown_sector:
        timeline = timeline[timeline["m15"] != OTHER_UNKNOWN_SECTOR_CODE].copy()
    inst_set = set(INSTITUTIONAL_CODES)
    if not include_other_unknown_sector:
        inst_set = inst_set - {OTHER_UNKNOWN_SECTOR_CODE}

    is_inst = timeline["m15"].isin(inst_set)
    missing_m15 = timeline["m15"].isna()
    coded_missing_outcome = timeline["m17"].isna() | timeline["m17"].isin(M17_CODED_MISSING)
    timeline["_inst"] = is_inst.to_numpy(dtype="int64")
    timeline["_cs"] = (is_inst & timeline["m17"].eq(1)).to_numpy(dtype="int64")
    timeline["_ambig"] = (
        missing_m15 | (is_inst & coded_missing_outcome)
    ).to_numpy(dtype="int64")

    blocks = (
        timeline.groupby(["caseid", "b3"], as_index=False)
        .agg(
            block_inst=("_inst", "sum"),
            block_cs=("_cs", "sum"),
            block_ambig=("_ambig", "sum"),
        )
        .sort_values(["caseid", "b3"], kind="mergesort")
    )
    blocks["earlier_inst"] = (
        blocks.groupby("caseid")["block_inst"].cumsum() - blocks["block_inst"]
    )
    blocks["earlier_cs"] = (
        blocks.groupby("caseid")["block_cs"].cumsum() - blocks["block_cs"]
    )
    blocks["earlier_ambig"] = (
        blocks.groupby("caseid")["block_ambig"].cumsum() - blocks["block_ambig"]
    )
    blocks_map = blocks.set_index(["caseid", "b3"])[
        ["earlier_inst", "earlier_cs", "earlier_ambig"]
    ]

    merged = idx.join(blocks_map, on=["caseid", "b3"])
    no_match = merged["earlier_inst"].isna() & idx["b3"].notna()
    if no_match.any():
        raise RuntimeError(
            "Index births are missing from the roster history timeline; "
            "cannot derive observed history safely"
        )

    earlier_inst = merged["earlier_inst"].fillna(0).astype("int64").to_numpy()
    earlier_cs = merged["earlier_cs"].fillna(0).astype("int64").to_numpy()
    earlier_ambig = merged["earlier_ambig"].fillna(0).astype("int64").to_numpy()
    bord = idx["bord"].to_numpy()
    unknown_dates = (
        idx["caseid"].map(mother_missing_date).fillna(False).to_numpy(dtype=bool)
        | idx["b3"].isna().to_numpy()
    )

    first = bord == 1
    observed_cs = (~first) & (earlier_cs > 0)
    ambiguous = (~first) & ~observed_cs & ((earlier_ambig > 0) | unknown_dates)
    no_observed = (~first) & ~observed_cs & ~ambiguous & (earlier_inst > 0)
    level = np.select(
        [first, observed_cs, ambiguous, no_observed],
        [
            FIRST_RECORDED_LIVE_BIRTH,
            OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
            UNKNOWN_HISTORY,
            NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
        ],
        default=UNOBSERVED_HISTORY,
    )

    has_any = np.where(
        earlier_inst > 0,
        HAS_ANY_TRUE,
        np.where((earlier_ambig > 0) | unknown_dates, HAS_ANY_UNKNOWN, HAS_ANY_FALSE),
    )

    inconsistent = first & ((earlier_inst > 0) | (earlier_ambig > 0))
    ambiguous_flag = earlier_ambig > 0
    flags = []
    for inc, amb, mdt in zip(inconsistent, ambiguous_flag, unknown_dates):
        parts = []
        if inc:
            parts.append(QUALITY_INCONSISTENT)
        if amb:
            parts.append(QUALITY_AMBIGUOUS)
        if mdt:
            parts.append(QUALITY_MISSING_DATES)
        flags.append(";".join(parts) if parts else QUALITY_OK)

    features = pd.DataFrame(
        {
            HISTORY_FEATURE: pd.Series(level, index=idx.index, dtype="string"),
            HAS_ANY_FEATURE: pd.Series(has_any, index=idx.index, dtype="string"),
            QUALITY_FEATURE: pd.Series(flags, index=idx.index, dtype="string"),
        }
    )

    diagnostics = {
        "n_roster_rows": int(len(frame)),
        "n_eligible_rows": int(len(eligible)),
        "n_timeline_rows": int(len(timeline)),
        "n_outside_recall_frame": n_outside_frame,
        "n_mothers_roster": int(frame["caseid"].nunique()),
        "n_index_rows": int(len(idx)),
        "include_other_unknown_sector": bool(include_other_unknown_sector),
        "n_mothers_with_missing_dates": int(mother_missing_date.sum()),
        "n_index_b3_missing": int(idx["b3"].isna().sum()),
        "level_counts": {
            str(key): int(value)
            for key, value in features[HISTORY_FEATURE].value_counts().items()
        },
        "has_any_counts": {
            str(key): int(value)
            for key, value in features[HAS_ANY_FEATURE].value_counts().items()
        },
        "quality_counts": {
            str(key): int(value)
            for key, value in features[QUALITY_FEATURE].value_counts().items()
        },
    }
    return features, diagnostics


def history_support_counts(features: pd.DataFrame, target) -> list[dict]:
    values = pd.to_numeric(pd.Series(target), errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError("History support requires a non-missing binary target")
    if len(values) != len(features):
        raise ValueError("History support length mismatch")
    rows = []
    for group, column in (
        (HISTORY_FEATURE, HISTORY_FEATURE),
        (HAS_ANY_FEATURE, HAS_ANY_FEATURE),
        (QUALITY_FEATURE, QUALITY_FEATURE),
    ):
        frame = pd.DataFrame(
            {"value": features[column].astype("string").to_numpy(), "y": values.to_numpy()}
        )
        grouped = frame.groupby("value", sort=True)["y"].agg(["size", "sum"])
        for value, row in grouped.iterrows():
            n_rows = int(row["size"])
            n_positive = int(row["sum"])
            rows.append(
                {
                    "group": group,
                    "value": str(value),
                    "n_births": n_rows,
                    "n_positive": n_positive,
                    "n_negative": n_rows - n_positive,
                }
            )
    return rows
