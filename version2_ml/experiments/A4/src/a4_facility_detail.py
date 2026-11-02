"""A4 verified m15 facility-detail mapping for NFHS-5.

Labels are verbatim from the authorized local NFHS-5 birth recode (IABR7EFL)
value metadata read on 2026-10-17. NFHS-4 harmonization requires the same
local label verification of IABR74FL.DTA, which is not available in this
authorized environment, so it is explicitly not implemented (the crosswalk
records this as pending rather than guessing labels).
"""
from __future__ import annotations

import pandas as pd

LABEL_SOURCE = "NFHS-5 IABR7EFL.DTA m15 value metadata (local read 2026-10-17)"

INSTITUTIONAL_CODES = frozenset(range(20, 28)) | frozenset(range(30, 34)) | {96}
NONINSTITUTIONAL_HOME_CODES = frozenset({10, 11, 12, 13})
OTHER_UNKNOWN_SECTOR_CODE = 96
KNOWN_CODES = INSTITUTIONAL_CODES | NONINSTITUTIONAL_HOME_CODES

FACILITY_DETAIL_LABELS = {
    10: "home",
    11: "respondent's home",
    12: "other home",
    13: "parents' home",
    20: "public sector",
    21: "public: govt./munic. hospital",
    22: "public: govt. dispensary",
    23: "public: uhc/uhp/ufwc",
    24: "public: chc/rural hospital/block phc",
    25: "public: phc/additional phc",
    26: "public: sub-centre",
    27: "other public sector health facility",
    30: "private sector",
    31: "private: hospital/maternity home/clinic",
    32: "other private sector health facility",
    33: "ngo or trust hospital/clinic",
    96: "other",
}

BROAD_CLASS = {
    **{code: "home" for code in NONINSTITUTIONAL_HOME_CODES},
    **{code: "public" for code in range(20, 28)},
    **{code: "private" for code in range(30, 34)},
    OTHER_UNKNOWN_SECTOR_CODE: "other_unknown_sector",
}

DETAIL_NOTES = {
    20: "Sector stated without any facility type; kept as its own level.",
    30: "Sector stated without any facility type; kept as its own level.",
    96: "Includes unknown-sector statements; D2 includes these rows in the index cohort and the observed-history timeline.",
    10: "Generic home statement without further detail.",
}

DETAIL_FEATURE = "facility_detail"


def _strict_codes(values, label: str) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    original = pd.Series(values)
    if (original.notna() & numeric.isna()).any():
        raise ValueError(f"{label}: non-numeric values present")
    observed = numeric.dropna()
    if ((observed % 1) != 0).any():
        raise ValueError(f"{label}: non-integer codes present")
    return numeric


def _assert_wave(wave: str) -> None:
    if wave != "NFHS-5":
        raise NotImplementedError(
            "Facility-detail harmonization is implemented only for the verified "
            "NFHS-5 labels; NFHS-4 requires local IABR74FL.DTA label verification."
        )


def facility_detail(values, *, wave: str = "NFHS-5") -> pd.Series:
    _assert_wave(wave)
    numeric = _strict_codes(values, "m15")
    unmapped = sorted(set(numeric.dropna().unique()) - set(FACILITY_DETAIL_LABELS))
    if unmapped:
        raise ValueError(f"Unexpected NFHS-5 m15 facility codes: {unmapped}")
    if numeric.isna().any():
        raise ValueError(
            "m15 is missing on rows passed to facility_detail; coding audit required"
        )
    labels = numeric.astype(int).map(FACILITY_DETAIL_LABELS).astype("string")
    labels.name = DETAIL_FEATURE
    return labels


def facility_broad_class(values, *, wave: str = "NFHS-5") -> pd.Series:
    _assert_wave(wave)
    numeric = _strict_codes(values, "m15")
    unmapped = sorted(set(numeric.dropna().unique()) - set(BROAD_CLASS))
    if unmapped:
        raise ValueError(f"Unexpected NFHS-5 m15 facility codes: {unmapped}")
    if numeric.isna().any():
        raise ValueError("m15 is missing on rows passed to facility_broad_class")
    classes = numeric.astype(int).map(BROAD_CLASS).astype("string")
    classes.name = "facility_broad_class"
    return classes


def crosswalk_rows(*, include_pending_nfhs4: bool = True) -> list[dict]:
    rows = []
    for code in sorted(FACILITY_DETAIL_LABELS):
        rows.append(
            {
                "wave": "NFHS-5",
                "m15_code": code,
                "broad_class": BROAD_CLASS[code],
                "detail_label": FACILITY_DETAIL_LABELS[code],
                "notes": DETAIL_NOTES.get(code, ""),
                "verification_source": LABEL_SOURCE,
            }
        )
    if include_pending_nfhs4:
        rows.append(
            {
                "wave": "NFHS-4",
                "m15_code": "not_verified",
                "broad_class": "not_verified",
                "detail_label": "not_implemented_pending_local_label_verification",
                "notes": (
                    "IABR74FL.DTA is not available in the authorized environment; "
                    "harmonization is blocked until labels are verified locally."
                ),
                "verification_source": "pending",
            }
        )
    return rows
