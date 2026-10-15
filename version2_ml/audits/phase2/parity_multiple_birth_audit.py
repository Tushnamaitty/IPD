#!/usr/bin/env python3
"""IPD V2 Phase 2: parity/birth-order and multiple-birth aggregate audit.

Read only local NFHS BR microdata, emit aggregate JSON. No row-level exports.
This cannot verify full lifetime pregnancy history or prior cesarean status.
"""
from pathlib import Path
import json
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[3]
DATA = {
    "NFHS-4": ROOT / "data/raw/nfhs4/IABR74FL.DTA",
    "NFHS-5": ROOT / "data/raw/IABR7EFL.DTA",
}
OUTPUT = ROOT / "version2_ml/audits/phase2/outputs/parity_multiple_birth_audit.json"
COLS = ["caseid", "bidx", "bord", "b0", "b3", "v201", "m15", "m17"]


def wave_audit(path):
    total = {k: 0 for k in (
        "institutional_n", "first_birth_bord1_n", "bord_gt1_n",
        "bord_missing_or_invalid_n", "b0_missing_n", "b0_zero_n",
        "b0_nonzero_n", "b0_unexpected_code_n",
        "ceb_less_bord_n", "ceb_equal_bord_n", "ceb_greater_bord_n",
        "duplicate_caseid_bidx_rows_chunk", "same_mother_same_birth_month_n_chunk",
    )}
    codes = {}
    # Within-chunk diagnostics only: do NOT interpret zero cross-chunk duplicate
    # counts as full-data uniqueness. Mother/birth membership spans chunks.
    for chunk, _ in pyreadstat.read_file_in_chunks(
        pyreadstat.read_dta, str(path), chunksize=75000,
        usecols=COLS, apply_value_formats=False
    ):
        num = lambda name: pd.to_numeric(chunk[name], errors="coerce")
        fac = num("m15")
        eligible = fac.between(20,27) | fac.between(30,33) | fac.eq(96)
        x = chunk.loc[eligible].copy()
        n = lambda name: pd.to_numeric(x[name], errors="coerce")
        b, multiple, ceb = n("bord"), n("b0"), n("v201")
        total["institutional_n"] += len(x)
        total["first_birth_bord1_n"] += int(b.eq(1).sum())
        total["bord_gt1_n"] += int(b.gt(1).sum())
        total["bord_missing_or_invalid_n"] += int((b.isna() | b.lt(1)).sum())
        total["b0_missing_n"] += int(multiple.isna().sum())
        total["b0_zero_n"] += int(multiple.eq(0).sum())
        total["b0_nonzero_n"] += int((multiple.notna() & multiple.ne(0)).sum())
        total["b0_unexpected_code_n"] += int((multiple.notna() & ~multiple.isin([0,1,2,3,4,5])).sum())
        total["ceb_less_bord_n"] += int(ceb.lt(b).sum())
        total["ceb_equal_bord_n"] += int(ceb.eq(b).sum())
        total["ceb_greater_bord_n"] += int(ceb.gt(b).sum())
        total["duplicate_caseid_bidx_rows_chunk"] += int(x[["caseid","bidx"]].duplicated(keep=False).sum())
        total["same_mother_same_birth_month_n_chunk"] += int(x[["caseid","b3"]].duplicated(keep=False).sum())
        observed = multiple.value_counts(dropna=False)
        for key, val in observed.items():
            name = "MISSING" if pd.isna(key) else str(int(key))
            codes[name] = codes.get(name, 0) + int(val)
    return {"counts": total, "b0_numeric_code_counts": codes}


def main():
    output = {
        "status": "PARITY_AND_MULTIPLE_BIRTH_AGGREGATE_DIAGNOSTIC_ONLY",
        "waves": {},
        "feature_freeze_approved": False,
        "caveats": [
            "DHS b0 codes need confirmation against NFHS BR questionnaire/recode documentation; nonzero is not automatically number of fetuses.",
            "bord is birth order of a live birth, not number of previous pregnancies or previous cesareans.",
            "Same-mother same-month checks are INCOMPLETE across chunks and among noninstitutional births.",
            "Comparisons among only institutional births do not reconstruct complete earlier birth histories.",
            "Counts are unweighted; no inferential or bootstrap validation.",
            "Review small aggregates for disclosure risk before publishing.",
        ],
    }
    for wave, path in DATA.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing local NFHS file: {path}")
        output["waves"][wave] = wave_audit(path)
        print(wave, output["waves"][wave]["counts"]["institutional_n"], "institutional births")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
