"""V1_spec input: key-for-key reconciliation of the V1 36-feature parquet.

Mirrors run_a7._f0_stage: identical row count, every cohort key matched, no
cohort-only / parquet-only / duplicate keys, zero outcome (csection vs m17)
and facility (delivery_place_code vs m15) mismatches. Any failure returns
status not_run; nothing is fitted on an unreconciled table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from feature_builder import F0_BIN, F0_CAT, F0_NAMES, F0_NUM
from preprocessing import FeatureManifest

F0_MANIFEST = FeatureManifest(F0_NUM + F0_BIN, F0_CAT, "F0")
_EXTRA = ["respondent_id", "birth_index", "csection", "delivery_place_code"]


def reconcile_f0(parquet_path, cohort: pd.DataFrame, keys: np.ndarray) -> tuple[dict, pd.DataFrame | None]:
    path = Path(parquet_path)
    if not path.exists():
        return {"status": "not_run_parquet_missing", "reconciliation": {}}, None
    pq = pd.read_parquet(path, columns=list(F0_NAMES) + _EXTRA)
    pq_keys = (pq["respondent_id"].astype("string") + "|"
               + pq["birth_index"].map(lambda value: str(int(value))).astype("string"))
    key_set, pq_set = set(keys), set(pq_keys)
    rec = {
        "parquet_rows": int(len(pq)),
        "cohort_rows": int(len(cohort)),
        "matched_keys": int(pq_keys.isin(key_set).sum()),
        "cohort_only_keys": int(len(key_set - pq_set)),
        "parquet_only_keys": int(len(pq_set - key_set)),
        "duplicate_parquet_keys": int(pq_keys.duplicated().sum()),
    }
    exact = (rec["parquet_rows"] == len(cohort) and rec["matched_keys"] == len(cohort)
             and rec["cohort_only_keys"] == 0 and rec["parquet_only_keys"] == 0 and rec["duplicate_parquet_keys"] == 0)
    if not exact:
        return {"status": "not_run_reconciliation_failed", "reconciliation": rec}, None
    aligned = pq.set_index(pq_keys).loc[keys]
    rec["outcome_mismatch"] = int((pd.to_numeric(aligned["csection"], errors="coerce").to_numpy()
                                   != cohort["m17"].to_numpy()).sum())
    rec["facility_mismatch"] = int((pd.to_numeric(aligned["delivery_place_code"], errors="coerce").to_numpy()
                                    != pd.to_numeric(cohort["m15"]).to_numpy()).sum())
    if rec["outcome_mismatch"] or rec["facility_mismatch"]:
        return {"status": "not_run_reconciliation_failed", "reconciliation": rec}, None
    return {"status": "reconciled", "reconciliation": rec}, aligned[list(F0_NAMES)].reset_index(drop=True)
