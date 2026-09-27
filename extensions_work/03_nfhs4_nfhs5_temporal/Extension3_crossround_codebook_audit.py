"""Audit every raw NFHS-4/5 variable used by Extension 3, without model fitting.

Place in IPD/extensions_work/03_nfhs4_nfhs5_temporal/ and run from the IPD root.
Only aggregate metadata and category counts are written to outputs/codebook_audit/.
The audit checks file labels and observed analytic-domain values, but cannot by
itself certify identical survey questionnaire wording or wealth construction.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from state_crosswalk import NFHS4, NFHS5

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "codebook_audit"
SOURCES = {
    "NFHS-4": ROOT / "data" / "raw" / "nfhs4" / "IABR74FL.DTA",
    "NFHS-5": ROOT / "data" / "raw" / "IABR7EFL.DTA",
}
VARS = ["m15", "m17", "v024", "v025", "v130", "v133", "v190", "s116",
        "bord", "b0", "v001", "v022", "v005", "caseid"]
ROLES = {
    "m15": "exposure", "m17": "outcome", "v024": "state", "v025": "residence",
    "v130": "religion", "v133": "education_years", "v190": "wealth_index",
    "s116": "social_group", "bord": "birth_order", "b0": "twin_order",
    "v001": "cluster_number", "v022": "sample_stratum_v022",
    "v005": "sample_weight_normalized", "caseid": "respondent_id",
}
CATEGORICAL = ["m15", "m17", "v024", "v025", "v130", "v190", "s116", "b0"]
EXPECTED = {
    "NFHS-4": {"institutional": 195997, "public": 141028, "private": 54338,
               "other": 631, "analytic": 195366},
    "NFHS-5": {"institutional": 201311, "public": 150299, "private": 50495,
               "other": 517, "analytic": 200794},
}


def simple(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def printable_code(value):
    if pd.isna(value):
        return "<missing>"
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def clean_labels(labels):
    return {printable_code(k): str(v).strip() for k, v in (labels or {}).items()}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def metadata(path):
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True, apply_value_formats=False)
    if not set(VARS).issubset(meta.column_names):
        raise ValueError(f"{path} missing {sorted(set(VARS) - set(meta.column_names))}")
    variables = {}
    for var in VARS:
        variables[var] = {
            "role": ROLES[var],
            "variable_label": (meta.column_names_to_labels or {}).get(var, "") or "",
            "value_labels": clean_labels((meta.variable_value_labels or {}).get(var, {})),
        }
    return variables


def domain(round_name, path, variables):
    """Read only audited columns and retain aggregate analytic-domain counts."""
    counts = Counter()
    missing = Counter()
    observed = {v: Counter() for v in CATEGORICAL + ["v133", "bord"]}
    low_high = {v: [float("inf"), float("-inf")] for v in ["v005", "v133", "bord"]}
    weight_sum = 0.0
    state_codes = set()
    psu_strata = set()
    id_count = 0
    for chunk_index, (frame, _) in enumerate(pyreadstat.read_file_in_chunks(
            pyreadstat.read_dta, str(path), chunksize=100000, usecols=VARS,
            apply_value_formats=False, user_missing=False), start=1):
        counts["raw_birth_rows"] += len(frame)
        place = pd.to_numeric(frame["m15"], errors="coerce")
        institutional = place.notna() & ~place.isin([10, 11, 12, 13])
        group = np.select([place.between(20, 27), place.between(30, 33)],
                          ["public", "private"], default="other")
        counts["institutional"] += int(institutional.sum())
        counts["public"] += int((institutional & (group == "public")).sum())
        counts["private"] += int((institutional & (group == "private")).sum())
        counts["other"] += int((institutional & (group == "other")).sum())
        selected = frame.loc[institutional & (group != "other")]
        counts["analytic"] += len(selected)
        delivery = pd.to_numeric(frame["m17"], errors="coerce")
        counts["other_recorded_m15_96"] += int((place == 96).sum())
        counts["missing_m17_recorded"] += int(delivery.loc[institutional].isna().sum())
        counts["missing_m17_analytic"] += int(delivery.loc[selected.index].isna().sum())
        for var in VARS:
            missing[var] += int(selected[var].isna().sum())
        for var, counter in observed.items():
            vc = selected[var].value_counts(dropna=False)
            counter.update({printable_code(k): int(v) for k, v in vc.items()})
        for var in low_high:
            series = pd.to_numeric(selected[var], errors="coerce")
            if series.notna().any():
                low_high[var][0] = min(low_high[var][0], float(series.min()))
                low_high[var][1] = max(low_high[var][1], float(series.max()))
        weight_sum += float(pd.to_numeric(selected["v005"], errors="coerce").sum() / 1e6)
        state_codes.update(printable_code(x) for x in selected["v024"].dropna().unique())
        psu_strata.update(zip(selected["v022"].astype(str), selected["v001"].astype(str)))
        id_count += int(selected["caseid"].notna().sum())
        print(f"{round_name}: read chunk {chunk_index}, analytic births so far {counts['analytic']}", flush=True)
    report = {
        "round": round_name, "source_name": path.name, "source_bytes": path.stat().st_size,
        "source_mtime_ns": path.stat().st_mtime_ns, "counts": dict(counts),
        "missing_by_variable": dict(missing),
        "observed_code_counts_analytic": {k: dict(sorted(v.items())) for k, v in observed.items()},
        "numeric_min_max_analytic": {k: v for k, v in low_high.items()},
        "sum_normalized_v005_analytic": weight_sum,
        "observed_state_codes": sorted(state_codes, key=lambda x: int(x)),
        "unique_analytic_psu_stratum_pairs": len(psu_strata),
        "nonmissing_analytic_caseid": id_count,
        "variables": variables,
    }
    save_json(OUT / f"{round_name.lower().replace('-', '')}_raw_audit.json", report)
    return report


def compare(a, b):
    rows = []
    for var in VARS:
        av, bv = a["variables"][var], b["variables"][var]
        al, bl = av["value_labels"], bv["value_labels"]
        codes4, codes5 = set(a["observed_code_counts_analytic"].get(var, {})), set(b["observed_code_counts_analytic"].get(var, {}))
        label_codes = set(al) | set(bl)
        conflicts = [k for k in sorted(label_codes) if k in al and k in bl and simple(al[k]) != simple(bl[k])]
        if var == "v024":
            status = "ROUND_SPECIFIC_STATE_MAP_REVIEW"
        elif var in ("v190", "v005", "v001", "v022", "caseid"):
            status = "CONSTRUCT_OR_DESIGN_REVIEW"
        elif simple(av["variable_label"]) != simple(bv["variable_label"]) or conflicts:
            status = "LABEL_DIFFERENCE_REVIEW"
        else:
            status = "METADATA_LABELS_MATCH"
        rows.append({
            "variable": var, "model_role": ROLES[var], "audit_status": status,
            "nfhs4_variable_label": av["variable_label"], "nfhs5_variable_label": bv["variable_label"],
            "nfhs4_value_labels_json": json.dumps(al, ensure_ascii=False),
            "nfhs5_value_labels_json": json.dumps(bl, ensure_ascii=False),
            "differing_label_codes": ";".join(conflicts),
            "observed_only_nfhs4": ";".join(sorted(codes4 - codes5)),
            "observed_only_nfhs5": ";".join(sorted(codes5 - codes4)),
            "nfhs4_missing_analytic": a["missing_by_variable"][var],
            "nfhs5_missing_analytic": b["missing_by_variable"][var],
            "nfhs4_special_cleaned_code_count": a["observed_code_counts_analytic"].get(var, {}).get("97" if var == "v133" else "8", 0) if var in ("v133", "s116") else "",
            "nfhs5_special_cleaned_code_count": b["observed_code_counts_analytic"].get(var, {}).get("97" if var == "v133" else "8", 0) if var in ("v133", "s116") else "",
            "review_note": ("Within-round wealth quintiles can be compared as ordinal categories; construction and real purchasing power need independent documentation." if var == "v190" else
                            "Identifiers, weights, PSU and strata are round-specific; labels alone do not verify comparable sampling design." if var in ("v005", "v001", "v022", "caseid") else
                            "Verify harmonized geography and territorial changes against the raw state labels." if var == "v024" else
                            "Check full questionnaire wording and any questionnaire-specific category changes before declaring substantive equivalence."),
        })
    state_check = {}
    for label, report, mapping in [("NFHS-4", a, NFHS4), ("NFHS-5", b, NFHS5)]:
        raw_labels = report["variables"]["v024"]["value_labels"]
        unmapped = sorted(set(report["observed_state_codes"]) - {str(x) for x in mapping})
        # Territorial merges and Delhi naming are intentional documented aliases.
        exempt = {"NFHS-4": {"8", "9", "14", "25"},
                  "NFHS-5": {"1", "7", "25", "37"}}[label]
        mismatches = {code: {"raw": raw_labels.get(code, ""), "map": mapping[int(code)]}
                      for code in report["observed_state_codes"] if code in raw_labels
                      and code not in exempt and simple(raw_labels[code]) != simple(mapping[int(code)])}
        state_check[label] = {"unmapped_analytic_codes": unmapped,
                              "unexpected_label_disagreements": mismatches,
                              "harmonized_geographies": sorted(set(mapping.values()))}
    same_geographies = (set(NFHS4.values()) == set(NFHS5.values()) and len(set(NFHS4.values())) == 35)
    result = {
        "scope": "All Extension 3 model variables plus exposure, outcome and design columns; raw metadata and analytic-domain counts",
        "source_files": {"NFHS-4": a["source_name"], "NFHS-5": b["source_name"]},
        "analytic_counts_match_expected": {
            name: all(report["counts"].get(k) == EXPECTED[name][k] for k in EXPECTED[name])
            for name, report in [("NFHS-4", a), ("NFHS-5", b)]},
        "state_crosswalk": state_check, "same_35_harmonized_geographies": same_geographies,
        "unresolved_external_review": [
            "Read the survey-specific questionnaire/codebook text for every covariate; raw DTA variable/value labels alone cannot prove full semantic equivalence.",
            "Verify wave-specific v190 wealth-index construction before making between-wave level comparisons.",
            "Verify survey sampling, weight construction and domain variance interpretation with a survey-methods reviewer.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "crossround_variable_audit.csv", index=False)
    save_json(OUT / "crossround_audit_summary.json", result)
    print("Saved aggregate audit in:", OUT, flush=True)
    print("Cohort count match:", result["analytic_counts_match_expected"], flush=True)
    print("35 shared mapped geographies:", same_geographies, flush=True)
    print("Unexpected state-label disagreements:", {k: v["unexpected_label_disagreements"] for k, v in state_check.items()}, flush=True)
    print("Review rows flagged:", [r["variable"] for r in rows if r["audit_status"] != "METADATA_LABELS_MATCH"], flush=True)


def main():
    for path in SOURCES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    reports = []
    for name, path in SOURCES.items():
        variables = metadata(path)
        cache = OUT / f"{name.lower().replace('-', '')}_raw_audit.json"
        if cache.exists():
            previous = json.loads(cache.read_text(encoding="utf-8"))
            if previous.get("source_bytes") == path.stat().st_size and previous.get("source_mtime_ns") == path.stat().st_mtime_ns:
                print(f"{name}: using saved aggregate checkpoint", flush=True)
                reports.append(previous)
                continue
            raise RuntimeError(f"Source changed since {cache} was saved; move the previous audit before rerunning")
        reports.append(domain(name, path, variables))
    compare(*reports)


if __name__ == "__main__":
    main()
