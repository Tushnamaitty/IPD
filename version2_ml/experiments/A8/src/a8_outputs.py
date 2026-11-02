"""Aggregate output writers and MANIFEST.json for A8 (reusable by later experiments).

Text outputs are written with LF line endings so the sha256 recorded in the
manifest equals the hash of the committed git blob on every platform.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from exp_provenance import environment, git_info, scan_output_dir
from manifests import stable_hash

MANIFEST_NAME = "MANIFEST.json"


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def write_text(text: str, path: Path) -> Path:
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    return path


def write_json(payload, path: Path) -> Path:
    return write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", path)


def config_hash(config: dict) -> str:
    return stable_hash(config)


def write_manifest(out_dir: Path, *, experiment_id: str, config: dict, config_path: Path, repo_root: Path,
                   seeds, split_manifest_hashes: dict, extra: dict | None = None) -> dict:
    """Scan outputs for restricted fields, then record sha256 of every output file."""
    out_dir = Path(out_dir)
    scan = scan_output_dir(out_dir)
    files = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != MANIFEST_NAME:
            files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_bytes(path)})
    manifest = {
        "experiment_id": experiment_id,
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(repo_root),
        "config_file": str(Path(config_path).name),
        "config_sha256": sha256_bytes(config_path),
        "config_hash": config_hash(config),
        "software": environment(),
        "seeds": [int(s) for s in seeds],
        "split_manifest_hashes": {str(k): v for k, v in split_manifest_hashes.items()},
        "disclosure_scan": scan,
        "hash_rule": "SHA-256 over the file bytes as written; text files use LF line endings. "
                     "MANIFEST.json itself is not listed.",
        "files": files,
        **(extra or {}),
    }
    write_json(manifest, out_dir / MANIFEST_NAME)
    return manifest
