"""Shared provenance, small-cell suppression and disclosure checks."""
from __future__ import annotations
import hashlib
import importlib
import json
import platform
import subprocess
from pathlib import Path
import pandas as pd

FORBIDDEN_COLUMNS = {"caseid", "bidx", "respondent_id", "birth_key", "m17", "csection",
                     "v001", "psu", "primary_sampling_unit", "v022", "v023", "v005",
                     "sample_weight", "partition", "prediction", "probability", "score"}
PACKAGES = ("numpy", "pandas", "sklearn", "scipy", "xgboost", "pyreadstat", "yaml")
SUPPRESSED = "SUPPRESSED"


def _git(root, *args):
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info(root: Path) -> dict:
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {"commit": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "base_version2_ml_development": _git(root, "rev-parse", "origin/version2-ml-development"),
            "tracked_files_dirty": bool(status) if status is not None else None}


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001 - record absence, never fail provenance
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def digest_frame(frame: pd.DataFrame) -> str:
    """Order-sensitive SHA-256 over values; a digest, not an exportable identifier."""
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def suppress_small_cells(table: pd.DataFrame, *, n_col: str, event_col: str, min_n: int = 50,
                         min_events: int = 10, value_cols=()) -> pd.DataFrame:
    """Replace counts/rates of cells with n<min_n, events<min_events or non-events<min_events."""
    out = table.copy()
    n = pd.to_numeric(out[n_col])
    e = pd.to_numeric(out[event_col])
    small = (n < min_n) | (e < min_events) | ((n - e) < min_events)
    cols = [n_col, event_col, *value_cols]
    out[cols] = out[cols].astype(object)
    out.loc[small, cols] = SUPPRESSED
    out["suppressed"] = small.to_numpy()
    return out


def assert_disclosure_safe(frame: pd.DataFrame, *, max_rows: int = 5000) -> None:
    bad = sorted(set(map(str, frame.columns)) & FORBIDDEN_COLUMNS)
    if bad:
        raise ValueError(f"DISCLOSURE: restricted columns in aggregate output: {bad}")
    if len(frame) > max_rows:
        raise ValueError("DISCLOSURE: aggregate output suspiciously row-level")


def scan_output_dir(folder: Path) -> dict:
    """Fail-closed scan of written aggregate CSV/JSON outputs for restricted fields."""
    checked = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix == ".csv":
            assert_disclosure_safe(pd.read_csv(path))
        elif path.suffix == ".json":
            def walk(node):
                if isinstance(node, dict):
                    bad = set(map(str, node)) & (FORBIDDEN_COLUMNS - {"partition"})
                    if bad:
                        raise ValueError(f"DISCLOSURE: restricted key(s) {sorted(bad)} in {path.name}")
                    for v in node.values():
                        walk(v)
                elif isinstance(node, list):
                    for v in node:
                        walk(v)
            walk(json.loads(path.read_text(encoding="utf-8")))
        else:
            continue
        checked.append(path.name)
    return {"status": "PASS", "files_checked": checked}


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
