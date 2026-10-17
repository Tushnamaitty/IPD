"""A2 cross-wave state/UT crosswalk: explicit versioned table, fail-closed validation.

State numeric codes are NEVER joined across waves: NFHS-4 v024 is alphabetical,
NFHS-5 v024 follows a different scheme. Each NFHS-5 code maps to one NFHS-4
source unit through the table, and every label is checked against the DTA
value labels. Boundary changes (Ladakh split, DNH/DD merger) are explicit.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

STATUSES = {"unchanged", "merged", "boundary_changed", "split_child", "no_source_unit"}
_ALIASES = {"nct of delhi": "delhi", "orissa": "odisha", "uttaranchal": "uttarakhand",
            "chattisgarh": "chhattisgarh", "a and n islands": "andaman and nicobar islands"}
COLUMNS = ["nfhs5_code", "nfhs5_label", "source_unit", "nfhs4_codes", "nfhs4_labels",
           "boundary_status", "note"]


def canonical(label) -> str:
    s = re.sub(r"[#*]", "", str(label)).strip().lower().replace("&", " and ")
    s = " ".join(s.split())
    return _ALIASES.get(s, s)


def load_crosswalk(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(table.columns) != COLUMNS:
        raise ValueError("Crosswalk columns differ from versioned schema")
    return table


def _codes(cell: str) -> list[int]:
    return [int(x) for x in cell.split(";") if x.strip()]


def _labels(cell: str) -> list[str]:
    return [x for x in cell.split(";") if x.strip()]


def validate_crosswalk(table: pd.DataFrame, nfhs4_labels: dict, nfhs5_labels: dict) -> dict:
    """Check the table against DTA value labels. Raises on any inconsistency."""
    lab4 = {int(k): canonical(v) for k, v in nfhs4_labels.items()}
    lab5 = {int(k): canonical(v) for k, v in nfhs5_labels.items()}
    codes5 = [int(c) for c in table["nfhs5_code"]]
    if len(codes5) != len(set(codes5)):
        raise ValueError("CROSSWALK: duplicate NFHS-5 code")
    if set(codes5) != set(lab5):
        raise ValueError(f"CROSSWALK: NFHS-5 code coverage mismatch: "
                         f"missing={sorted(set(lab5) - set(codes5))} extra={sorted(set(codes5) - set(lab5))}")
    unit_of_nfhs4: dict[int, str] = {}
    for row in table.itertuples(index=False):
        code5 = int(row.nfhs5_code)
        if row.boundary_status not in STATUSES:
            raise ValueError(f"CROSSWALK: unknown boundary status for NFHS-5 code {code5}")
        if canonical(row.nfhs5_label) != lab5[code5]:
            raise ValueError(f"CROSSWALK: NFHS-5 label mismatch for code {code5}")
        c4, l4 = _codes(row.nfhs4_codes), _labels(row.nfhs4_labels)
        if row.boundary_status == "no_source_unit":
            if c4 or row.source_unit:
                raise ValueError(f"CROSSWALK: no_source_unit row {code5} must not name a source")
            continue
        if not row.source_unit or not c4 or len(c4) != len(l4):
            raise ValueError(f"CROSSWALK: incomplete source mapping for NFHS-5 code {code5}")
        if (row.boundary_status == "merged") != (len(c4) > 1):
            raise ValueError(f"CROSSWALK: only 'merged' rows may pool several NFHS-4 codes ({code5})")
        for code4, text in zip(c4, l4):
            if code4 not in lab4 or canonical(text) != lab4[code4]:
                raise ValueError(f"CROSSWALK: NFHS-4 label mismatch for code {code4}")
            if unit_of_nfhs4.setdefault(code4, row.source_unit) != row.source_unit:
                raise ValueError(f"CROSSWALK: NFHS-4 code {code4} assigned to two source units")
        if row.boundary_status == "unchanged" and canonical(row.nfhs5_label) != canonical(l4[0]):
            raise ValueError(f"CROSSWALK: 'unchanged' row {code5} has differing names")
    if set(unit_of_nfhs4) != set(lab4):
        raise ValueError(f"CROSSWALK: NFHS-4 codes not covered: {sorted(set(lab4) - set(unit_of_nfhs4))}")
    status = table["boundary_status"].value_counts().sort_index()
    return {"status": "CROSSWALK_VALID", "nfhs5_codes": len(codes5), "nfhs4_codes": len(lab4),
            "source_units": len(set(unit_of_nfhs4.values())),
            "boundary_status_counts": {k: int(v) for k, v in status.items()},
            "numeric_codes_joined_directly": False}


def nfhs4_unit_map(table: pd.DataFrame) -> dict[int, str]:
    out = {}
    for row in table.itertuples(index=False):
        for code4 in _codes(row.nfhs4_codes):
            out[code4] = row.source_unit
    return out


def nfhs5_unit_map(table: pd.DataFrame) -> dict[int, str | None]:
    return {int(r.nfhs5_code): (r.source_unit or None) for r in table.itertuples(index=False)}


def with_unseen(table: pd.DataFrame, nfhs5_codes) -> pd.DataFrame:
    """Sensitivity variant: declare listed NFHS-5 units as having no NFHS-4 source."""
    out = table.copy()
    mask = out["nfhs5_code"].astype(int).isin([int(c) for c in nfhs5_codes])
    if mask.sum() != len(set(nfhs5_codes)):
        raise ValueError("Unknown NFHS-5 code in unseen-unit variant")
    out.loc[mask, ["source_unit", "nfhs4_codes", "nfhs4_labels"]] = ""
    out.loc[mask, "boundary_status"] = "no_source_unit"
    return out


def assign_units(codes: pd.Series, mapping: dict) -> pd.Series:
    """Map v024 codes to source units; unknown codes fail closed, declared gaps -> <NA>."""
    numeric = pd.to_numeric(codes, errors="coerce")
    unknown = sorted(set(numeric.dropna().astype(int)) - set(mapping))
    if numeric.isna().any() or unknown:
        raise ValueError(f"CROSSWALK: state code(s) absent from versioned table: {unknown}")
    return numeric.astype(int).map(mapping).astype("string")


def read_dta_state_labels(path: Path) -> dict[int, str]:
    import pyreadstat
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    labels = meta.value_labels.get(meta.variable_to_label.get("v024"), {})
    if not labels:
        raise ValueError("No v024 value-label table; state names are not guessed")
    return {int(k): str(v) for k, v in labels.items()}
