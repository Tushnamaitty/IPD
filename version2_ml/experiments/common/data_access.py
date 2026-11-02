"""Shared data access and frozen-contract helpers for authorized local runs.

Used by A4 and A7 so the frozen C1 cohort definition, input hashing,
environment capture and split provenance live in one audited place instead of
being copied per runner. Reads only the approved local NFHS-5 subset; never
writes row-level or identifying data.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

READ_COLUMNS = (
    "caseid",
    "bidx",
    "b3",
    "b19",
    "v008",
    "v011",
    "m15",
    "m17",
    "v024",
    "v025",
    "v130",
    "v133",
    "v190",
    "bord",
    "b0",
)

INSTITUTIONAL_CODES = frozenset(range(20, 28)) | frozenset(range(30, 34)) | {96}
C1_COHORT_RULE = "c1_m15_institutional_b19_0_59_m17_0_1"
EXPECTED_C1 = {
    "n_births": 201311,
    "n_mothers": 158429,
    "n_events": 44742,
    "m15_96_rows": 517,
    "states_observed": 36,
}
FROZEN_SPLIT_HASH = (
    "6ff45fcb76af00a6de5a33a225c482b1c9b3153f35544d887d8e3b2c3467904b"
)
FROZEN_TRAIN_DIGEST = (
    "b96807778dbdfddc17d1446db1a9b918ba49a8eac9b09a9f710799d0cfeec37f"
)
FROZEN_VALID_DIGEST = (
    "800e7130876165e2f2e72ba40555b742295baa411d03f04197ee4920234e8499"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_versions(extra: dict | None = None) -> dict:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for module_name, key in (
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("matplotlib", "matplotlib"),
    ):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        versions[key] = module.__version__
    if extra:
        versions.update(extra)
    return versions


def repo_state(repo_root: Path | None = None) -> dict:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"repo_commit": commit, "branch": branch}
    except Exception:
        return {"repo_commit": "unavailable", "branch": "unavailable"}


def read_dta_subset(path: Path, *, extra_columns: tuple = ()) -> pd.DataFrame:
    import pyreadstat

    columns = list(READ_COLUMNS)
    for column in extra_columns:
        if column not in columns:
            columns.append(column)
    frame, _ = pyreadstat.read_dta(
        str(path), usecols=columns, apply_value_formats=False
    )
    return frame


def build_c1_cohort(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    facility = pd.to_numeric(raw["m15"], errors="coerce")
    outcome = pd.to_numeric(raw["m17"], errors="coerce")
    months = pd.to_numeric(raw["b19"], errors="coerce")
    mask = (
        facility.isin(sorted(INSTITUTIONAL_CODES))
        & months.between(0, 59)
        & outcome.isin([0, 1])
    )
    cohort = raw.loc[mask].reset_index(drop=True)
    cohort["m15"] = pd.to_numeric(cohort["m15"], errors="coerce")
    cohort["m17"] = pd.to_numeric(cohort["m17"], errors="coerce").astype(int)
    state_codes = pd.to_numeric(cohort["v024"], errors="coerce")
    summary = {
        "n_births": int(len(cohort)),
        "n_mothers": int(cohort["caseid"].astype("string").nunique()),
        "n_events": int(cohort["m17"].sum()),
        "n_controls": int((cohort["m17"] == 0).sum()),
        "m15_96_rows": int((cohort["m15"] == 96).sum()),
        "states_observed": int(state_codes.nunique()),
        "missing_state": int(state_codes.isna().sum()),
        "duplicate_birth_keys": int(cohort[["caseid", "bidx"]].duplicated().sum()),
        "missing_caseid": int(cohort["caseid"].isna().sum()),
        "missing_bidx": int(cohort["bidx"].isna().sum()),
    }
    return cohort, summary


def assert_frozen_c1(summary: dict) -> list[str]:
    if (
        summary["n_births"] != EXPECTED_C1["n_births"]
        or summary["n_mothers"] != EXPECTED_C1["n_mothers"]
    ):
        raise RuntimeError(
            "Frozen C1 cohort mismatch: observed "
            f"{summary['n_births']} births / {summary['n_mothers']} mothers vs "
            f"expected {EXPECTED_C1['n_births']} / {EXPECTED_C1['n_mothers']}; "
            "re-audit the input."
        )
    if summary["n_events"] != EXPECTED_C1["n_events"]:
        raise RuntimeError(
            "Frozen C1 event count mismatch: observed "
            f"{summary['n_events']} vs expected {EXPECTED_C1['n_events']}."
        )
    if summary["duplicate_birth_keys"] or summary["missing_caseid"] or summary["missing_bidx"]:
        raise RuntimeError("C1 cohort integrity failure; re-audit the input.")
    if summary["missing_state"]:
        raise RuntimeError(
            "Missing v024 values are incompatible with the consistent state mapping."
        )
    warnings = []
    if summary["m15_96_rows"] != EXPECTED_C1["m15_96_rows"]:
        warnings.append("m15=96 count differs from the Phase-0 verified 517.")
    if summary["states_observed"] != EXPECTED_C1["states_observed"]:
        warnings.append("Observed state count differs from the verified 36.")
    return warnings
