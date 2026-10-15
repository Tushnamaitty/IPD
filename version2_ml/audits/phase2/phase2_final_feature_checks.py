#!/usr/bin/env python3
"""Aggregate Phase 2 age/parity/recollection checks, executed ONLY on local raw NFHS data.

pip install pyreadstat
python version2_ml/audits/phase2/phase2_final_feature_checks.py
Outputs aggregate JSON; never saves respondent-level rows or sensitive data.
"""
from __future__ import annotations
import json
from pathlib import Path
import pyreadstat
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INPUTS = {"NFHS-4": ROOT / "data/raw/nfhs4/IABR74FL.DTA",
          "NFHS-5": ROOT / "data/raw/IABR7EFL.DTA"}
COLUMNS = ["m15", "m17", "v008", "b3", "v011", "v012", "v201", "bord", "b0"]
OUT = ROOT / "version2_ml/audits/phase2/outputs/phase2_final_feature_checks.json"


def aggregate(source: Path):
    count = {k: 0 for k in (
        "institutional_n", "valid_y_n", "age_cmc_missing_n", "age_cmc_negative_n",
        "age_birth_under_10_n", "age_birth_over_55_n", "age_interview_lower_than_birth_n",
        "parity_cmp_nonmissing_n", "interview_ceb_greater_bord_n",
        "interview_ceb_less_bord_n", "invalid_bord_n",
        "plurality_non_singleton_code_n", "recall_0_59_n", "recall_60_n",
        "recall_other_n")}
    for chunk, _ in pyreadstat.read_file_in_chunks(
        pyreadstat.read_dta, str(source), chunksize=75000,
        usecols=COLUMNS, apply_value_formats=False
    ):
        n = lambda s: pd.to_numeric(chunk[s], errors="coerce")
        f = n("m15")
        eligible = f.between(20, 27) | f.between(30, 33) | f.eq(96)
        if not eligible.any():
            continue
        s = chunk.loc[eligible]
        z = lambda col: pd.to_numeric(s[col], errors="coerce")
        count["institutional_n"] += len(s)
        count["valid_y_n"] += int(z("m17").isin((0, 1)).sum())
        months = z("b3") - z("v011")
        age = months / 12
        count["age_cmc_missing_n"] += int(months.isna().sum())
        count["age_cmc_negative_n"] += int(months.lt(0).sum())
        count["age_birth_under_10_n"] += int(age.lt(10).sum())
        count["age_birth_over_55_n"] += int(age.gt(55).sum())
        count["age_interview_lower_than_birth_n"] += int(z("v012").lt(age).sum())
        bord, ceb = z("bord"), z("v201")
        valid = bord.notna() & ceb.notna()
        count["parity_cmp_nonmissing_n"] += int(valid.sum())
        count["interview_ceb_greater_bord_n"] += int((valid & ceb.gt(bord)).sum())
        count["interview_ceb_less_bord_n"] += int((valid & ceb.lt(bord)).sum())
        count["invalid_bord_n"] += int((bord.isna() | bord.le(0)).sum())
        count["plurality_non_singleton_code_n"] += int(z("b0").fillna(0).ne(0).sum())
        delta = z("v008") - z("b3")
        count["recall_0_59_n"] += int(delta.between(0, 59).sum())
        count["recall_60_n"] += int(delta.eq(60).sum())
        count["recall_other_n"] += int((~delta.between(0, 60)).sum())
    return count


def main():
    report = {
        "status": "AGGREGATE_FEATURE_DIAGNOSTICS_ONLY",
        "source_data_published": False,
        "feature_freeze_approved": False,
        "notes": ["Age at delivery = (b3-v011)/12 is only a candidate derived variable.",
                  "Birth order is not equivalent to previous-CS history.",
                  "All questionnaire and 60-month eligibility decisions still require review.",
                  "Counts are unweighted; no design-valid inference is performed."],
        "waves": {}
    }
    for wave, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Local raw file not found: {path}")
        report["waves"][wave] = aggregate(path)
        print(f"{wave} institutional rows: {report['waves'][wave]['institutional_n']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
