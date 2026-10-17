"""Shared local NFHS birth-table loader for V2 experiments (A2/A3/A5/A6).

Reads the raw BR files in chunks and keeps the audited broad institutional
cohort. The returned frame is RESTRICTED (mother keys, outcomes): keep it in
memory or in the gitignored data/processed/ cache only. Never export rows.
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "version2_ml") not in sys.path:
    sys.path.insert(0, str(ROOT / "version2_ml"))
from feature_builder import make_core_features, CORE_NAMES  # noqa: E402

RAW = {"NFHS-4": Path("data/raw/nfhs4/IABR74FL.DTA"),
       "NFHS-5": Path("data/raw/IABR7EFL.DTA")}
EXPECTED_INSTITUTIONAL = {"NFHS-4": 195997, "NFHS-5": 201311}
COHORT_RULE = {"NFHS-4": "institutional m15 in 20-27/30-33/96; v008-b3 in 0..59 (b19 unavailable)",
               "NFHS-5": "institutional m15 in 20-27/30-33/96; b19 in 0..59"}
_READ = ["caseid", "bidx", "b3", "b19", "v008", "v011", "v133", "v190", "bord", "m15",
         "m17", "v025", "v130", "b0", "v024", "v005", "v001", "v022"]
CACHE_DIR = Path("data/processed/v2_experiments")


def _institutional(m15: pd.Series) -> pd.Series:
    return m15.between(20, 27) | m15.between(30, 33) | m15.eq(96)


def _read_raw(path: Path, wave: str):
    import pyreadstat
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    cols = [c for c in _READ if c in meta.column_names]
    missing = [c for c in _READ if c not in cols and not (c == "b19" and wave == "NFHS-4")]
    if missing:
        raise ValueError(f"{wave}: required raw columns absent: {missing}")
    flow = {"raw_rows": 0, "recall_eligible": 0, "institutional_and_recall": 0}
    keep = []
    for chunk, _ in pyreadstat.read_file_in_chunks(pyreadstat.read_dta, str(path), chunksize=75000,
                                                   usecols=cols, apply_value_formats=False):
        num = lambda c: pd.to_numeric(chunk[c], errors="coerce")
        if wave == "NFHS-5":
            recall = num("b19").between(0, 59)
        else:
            recall = (num("v008") - num("b3")).between(0, 59)
        inst = _institutional(num("m15"))
        flow["raw_rows"] += int(len(chunk))
        flow["recall_eligible"] += int(recall.sum())
        flow["institutional_and_recall"] += int((recall & inst).sum())
        keep.append(chunk.loc[recall & inst].copy())
    return pd.concat(keep, ignore_index=True), flow


def build_birth_table(raw: pd.DataFrame, wave: str, flow: dict) -> tuple[pd.DataFrame, dict]:
    """Validate and derive the canonical restricted birth table from eligible raw rows."""
    y = pd.to_numeric(raw["m17"], errors="coerce")
    valid = y.isin([0, 1])
    flow = dict(flow, invalid_or_missing_outcome_excluded=int((~valid).sum()))
    raw = raw.loc[valid].reset_index(drop=True)
    caseid = raw["caseid"].astype("string")
    if caseid.isna().any() or caseid.str.strip().eq("").any():
        raise ValueError(f"{wave}: missing mother identifier")
    if raw[["caseid", "bidx"]].duplicated().any():
        raise ValueError(f"{wave}: duplicate (caseid, bidx) birth keys")
    core = make_core_features(raw, wave=wave)
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")
    w = num("v005") / 1e6
    if not (np.isfinite(w) & (w > 0)).all():
        raise ValueError(f"{wave}: invalid survey weights")
    out = pd.DataFrame({
        "caseid": caseid.to_numpy(), "bidx": num("bidx").astype(int).to_numpy(),
        "wave": wave, "v024": num("v024").astype(int).to_numpy(),
        "m15": num("m15").astype(int).to_numpy(), "m17": y[valid].astype(int).to_numpy(),
        "b3": num("b3").to_numpy(), "v008": num("v008").to_numpy(),
        "cmc_recall": (num("v008") - num("b3")).to_numpy(),
        "b19": num("b19").to_numpy() if "b19" in raw else np.nan,
        "sample_weight": w.to_numpy(), "psu": num("v001").to_numpy(),
    })
    for name in CORE_NAMES:
        out[name] = core[name].to_numpy()
    out["sector"] = out["facility_type"]
    flow["final_births"] = int(len(out))
    flow["final_mothers"] = int(out["caseid"].nunique())
    flow["final_events"] = int(out["m17"].sum())
    flow["expected_institutional_births"] = EXPECTED_INSTITUTIONAL[wave]
    flow["matches_expected"] = flow["final_births"] == EXPECTED_INSTITUTIONAL[wave]
    flow["cohort_rule"] = COHORT_RULE[wave]
    flow["sector_births"] = {k: int(v) for k, v in out["sector"].value_counts().sort_index().items()}
    return out, flow


def load_wave_births(root: Path, wave: str, *, use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    if wave not in RAW:
        raise ValueError("Unrecognized NFHS wave")
    path = Path(root) / RAW[wave]
    if not path.is_file():
        raise FileNotFoundError(f"Authorized local input missing: {RAW[wave]}")
    stat = path.stat()
    cache = Path(root) / CACHE_DIR / f"{wave.replace('-', '').lower()}_births.pkl"
    key = (path.name, stat.st_size, stat.st_mtime_ns, "v1")
    if use_cache and cache.is_file():
        with cache.open("rb") as fh:
            stored = pickle.load(fh)
        if stored.get("key") == key:
            return stored["frame"], stored["flow"]
    raw, flow = _read_raw(path, wave)
    frame, flow = build_birth_table(raw, wave, flow)
    flow["raw_file_basename"] = path.name
    flow["raw_file_bytes"] = int(stat.st_size)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as fh:
            pickle.dump({"key": key, "frame": frame, "flow": flow}, fh)
    return frame, flow


def cmc_to_year_month(cmc: float) -> str:
    c = int(cmc)
    return f"{1900 + (c - 1) // 12}-{(c - 1) % 12 + 1:02d}"
