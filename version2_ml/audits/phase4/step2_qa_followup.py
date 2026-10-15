#!/usr/bin/env python3
"""Local aggregate QA of NFHS-5 m15=96 outcome coding and small-state support."""
from pathlib import Path
import json
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/raw/IABR7EFL.DTA"
OUTPUT = ROOT / "version2_ml/audits/phase4/outputs/step2_qa_followup.json"

def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    counts = {"total": 0, "other": 0, "other_yes": 0,
              "other_no": 0, "other_unknown": 0}
    state_counts = {}
    for chunk, _ in pyreadstat.read_file_in_chunks(
        pyreadstat.read_dta, str(SOURCE), chunksize=75000,
        usecols=["m15", "m17", "b19", "v024"], apply_value_formats=False
    ):
        facility = pd.to_numeric(chunk["m15"], errors="coerce")
        age = pd.to_numeric(chunk["b19"], errors="coerce")
        institutional = facility.between(20,27) | facility.between(30,33) | facility.eq(96)
        selected = chunk.loc[institutional & age.between(0,59)]
        counts["total"] += len(selected)
        for key, value in selected["v024"].value_counts(dropna=False).items():
            category = str(key)
            state_counts[category] = state_counts.get(category, 0) + int(value)
        other = selected.loc[pd.to_numeric(selected["m15"],errors="coerce").eq(96)]
        outcome = pd.to_numeric(other["m17"], errors="coerce")
        counts["other"] += len(other)
        counts["other_yes"] += int(outcome.eq(1).sum())
        counts["other_no"] += int(outcome.eq(0).sum())
        counts["other_unknown"] += int((~outcome.isin([0,1])).sum())
    report = {
        "status": "DESCRIPTIVE_QA_REVIEW_REQUIRED",
        "eligible_births": counts["total"],
        "other_facility_births": counts["other"],
        "other_facility_cesarean": counts["other_yes"],
        "other_facility_non_cesarean": counts["other_no"],
        "other_facility_outcome_unknown": counts["other_unknown"],
        "number_states": len(state_counts),
        "number_states_under_200_births_total": sum(x<200 for x in state_counts.values()),
        "interpretation": "Zero events need checking against DHS questionnaire; not evidence that all other facilities are truly vaginal births.",
        "privacy": "No state-level small counts, microdata, mother IDs, PSU IDs, or predictions saved."
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
