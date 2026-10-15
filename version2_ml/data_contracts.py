"""IPD V2 Phase 3 Step 2: aggregate-only data contract checks (no raw exports)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

MAPPINGS = {
    "counsel_bleeding": ("s419a", "s420a"),
    "counsel_convulsions": ("s419b", "s420b"),
    "counsel_labour": ("s419c", "s420c"),
    "counsel_abdominal_pain": ("s419d", "s420d"),
    "counsel_blood_pressure": ("s419e", "s420e"),
    "pregnancy_convulsions": ("s432", "s434"),
    "pregnancy_swelling": ("s433", "s435"),
}
REQUIRED = {"caseid", "bidx", "m15", "m17", "v008", "b3", "v005", "v001", "v022"}
INSTITUTIONAL = set(range(20, 28)) | set(range(30, 34)) | {96}


def audit_births(df: pd.DataFrame, wave: str) -> dict:
    """Validate a local BR dataframe, returning counts and diagnostics only."""
    if wave not in ("NFHS-4", "NFHS-5"):
        raise ValueError("wave must be NFHS-4 or NFHS-5")
    idx = 0 if wave == "NFHS-4" else 1
    needed = REQUIRED | {pair[idx] for pair in MAPPINGS.values()}
    missing = sorted(needed - set(df.columns))
    report = {"wave": wave, "raw_rows": int(len(df)), "errors": [], "warnings": []}
    if missing:
        report["errors"].append("Missing columns: " + ", ".join(missing))
        report["status"] = "INVALID"
        return report

    numeric = lambda c: pd.to_numeric(df[c], errors="coerce")
    facility, outcome = numeric("m15"), numeric("m17")
    institution = facility.isin(INSTITUTIONAL)
    recall = (numeric("v008") - numeric("b3"))[institution]
    y = outcome[institution]
    birth_id = df[["caseid", "bidx"]].copy()
    ids = birth_id["caseid"].astype("string")
    missing_ids = int((ids.isna() | ids.str.strip().eq("")).sum())
    duplicate_keys = int(birth_id.duplicated(keep=False).sum())
    report.update(
        institutional_rows=int(institution.sum()),
        public_rows=int((institution & facility.between(20, 27)).sum()),
        private_rows=int((institution & facility.between(30, 33)).sum()),
        other_rows=int((institution & facility.eq(96)).sum()),
        invalid_institutional_outcomes=int((~y.isin([0, 1])).sum()),
        recall_0_59=int(recall.between(0, 59).sum()),
        recall_exactly_60=int(recall.eq(60).sum()),
        recall_outside_0_60_or_missing=int((~recall.between(0, 60)).sum()),
        missing_mother_ids=missing_ids,
        missing_birth_indices=int(numeric("bidx").isna().sum()),
        duplicate_birth_keys=duplicate_keys,
        invalid_weights=int((numeric("v005").isna() | (numeric("v005") <= 0)).sum()),
        missing_psu=int(df["v001"].isna().sum()),
        missing_strata=int(df["v022"].isna().sum()),
    )
    for field in ("invalid_institutional_outcomes", "missing_mother_ids",
                  "missing_birth_indices", "duplicate_birth_keys", "invalid_weights",
                  "missing_psu", "missing_strata"):
        if report[field]:
            report["errors"].append(f"{field}: {report[field]}")
    if report["recall_exactly_60"]:
        report["warnings"].append("Exactly 60-month births present; eligibility not automatically changed.")
    if report["recall_outside_0_60_or_missing"]:
        report["warnings"].append("Recall dates outside 0..60 months or missing.")
    mapping_check = {}
    for label, pair in MAPPINGS.items():
        s = pd.to_numeric(df.loc[institution, pair[idx]], errors="coerce")
        allowed = {0, 1, 8} if label.startswith("pregnancy_") else {0, 1}
        unexpected = int((s.notna() & ~s.isin(allowed)).sum())
        mapping_check[label] = {
            "raw_column": pair[idx],
            "system_missing": int(s.isna().sum()),
            "unknown_8": int(s.eq(8).sum()),
            "unexpected_codes": unexpected,
        }
        if unexpected:
            report["errors"].append(f"{label}: {unexpected} unexpected codes")
    report["seven_mapping_checks"] = mapping_check
    expected = 195997 if wave == "NFHS-4" else 201311
    if report["institutional_rows"] != expected:
        report["warnings"].append("Broad institutional count differs from historical expected count.")
    report["status"] = "INVALID" if report["errors"] else "DATA_CONTRACT_VALID"
    report["limitations"] = [
        "Valid codes do not establish questionnaire/skip-universe equivalence.",
        "This test cannot prove leakage-safe splits, feature timing, or valid survey inference.",
        "No row-level outputs or model training.",
    ]
    return report


def write_aggregate(report: dict, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
