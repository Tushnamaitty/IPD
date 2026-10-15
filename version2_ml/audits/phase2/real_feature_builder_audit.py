#!/usr/bin/env python3
"""Local NFHS-4/5 real-data smoke audit for candidate F1/FT eight-core builder.

No individual-level records or model predictions are written. The reports
contain only high-level counts and errors. This is NOT F0 reproduction or an
approval of timing/semantics or survey inference.
"""
from pathlib import Path
import sys
import json
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "version2_ml"))
from feature_builder import make_core_features, feature_build_summary

FILES = {
    "NFHS-4": ROOT / "data/raw/nfhs4/IABR74FL.DTA",
    "NFHS-5": ROOT / "data/raw/IABR7EFL.DTA",
}
COLS = ["m15", "b3", "v011", "v133", "v190", "bord", "v025", "v130", "b0", "b19", "v008"]
DEST = ROOT / "version2_ml/audits/phase2/outputs/real_feature_builder_audit.json"


def audit(path, wave):
    _, metadata = pyreadstat.read_dta(str(path), metadataonly=True)
    available = set(metadata.column_names)
    required = [c for c in COLS if c != "b19"]
    if not set(required).issubset(available):
        raise ValueError("Missing columns: " + str(sorted(set(required) - available)))
    usecols = required + (["b19"] if "b19" in available else [])
    out = {
        "raw_institutional": 0, "official_5y_eligible": 0,
        "cmc_5y_eligible": 0, "output_rows": 0,
        "feature_names": None, "missing_by_feature": {},
        "errors": [], "b19_available": "b19" in available
    }
    for chunk_num, (chunk, _) in enumerate(pyreadstat.read_file_in_chunks(
            pyreadstat.read_dta, str(path), chunksize=60000,
            usecols=usecols, apply_value_formats=False), start=1):
        n = lambda column: pd.to_numeric(chunk[column], errors="coerce")
        facility = n("m15")
        eligible = facility.between(20, 27) | facility.between(30, 33) | facility.eq(96)
        selected = chunk.loc[eligible]
        out["raw_institutional"] += len(selected)
        cmc = pd.to_numeric(selected["v008"], errors="coerce") - pd.to_numeric(selected["b3"], errors="coerce")
        cmc_good = cmc.between(0, 59)
        out["cmc_5y_eligible"] += int(cmc_good.sum())
        if "b19" in available:
            child_age = pd.to_numeric(selected["b19"], errors="coerce")
            official = child_age.between(0, 59)
        else:
            official = cmc_good
        out["official_5y_eligible"] += int(official.sum())
        frame = selected.loc[official]
        if frame.empty:
            continue
        try:
            features = make_core_features(frame, wave=wave)
            summ = feature_build_summary(features)
            out["output_rows"] += summ["rows"]
            if out["feature_names"] is None:
                out["feature_names"] = summ["columns"]
            elif out["feature_names"] != summ["columns"]:
                raise RuntimeError("Feature schema changed between chunks")
            for name, count in summ["missing_by_feature"].items():
                out["missing_by_feature"][name] = out["missing_by_feature"].get(name, 0) + count
        except Exception as exc:
            # Exposes only class/error text, not record data.
            out["errors"].append(f"Chunk {chunk_num}: {type(exc).__name__}: {exc}")
    out["cohort_preserved"] = out["official_5y_eligible"] == out["output_rows"]
    out["status"] = ("PASS_AGGREGATE_FEATURE_BUILD" if out["cohort_preserved"]
                     and len(out["errors"]) == 0 else "REVIEW_REQUIRED")
    return out


def main():
    report = {
        "status": "REAL_DATA_FEATURE_BUILDER_AUDIT",
        "feature_freeze_approved": False,
        "baseline_f0_reproduced": False,
        "survey_inference_validated": False,
        "notes": [
            "Eight conservative candidate features only, not original F0 reconstruction.",
            "NFHS-5 b19<60 eligibility and NFHS-4 CMC 0..59 are provisional harmonized rules.",
            "No raw microdata, IDs, or predictions are saved.",
            "Input data and special-code handling still require contextual verification."
        ],
        "waves": {}
    }
    for wave, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing local input file: {path}")
        report["waves"][wave] = audit(path, wave)
        print(wave, report["waves"][wave]["status"],
              report["waves"][wave]["output_rows"], "constructed rows")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved:", DEST)


if __name__ == "__main__":
    main()
