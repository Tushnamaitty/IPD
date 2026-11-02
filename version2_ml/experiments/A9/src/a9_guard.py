"""Score-once guard for each seed's test partition (PROTOCOL.md section 4).

A guard file is created with status "started" before the first test
prediction and completed with the seed's aggregate results. A completed guard
is reused without re-scoring; a started guard blocks the seed until a human
decides. Guards live in a git-ignored private/ folder and hold aggregates only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STARTED, COMPLETED = "started", "completed"


class GuardBlocked(RuntimeError):
    pass


def guard_path(directory: Path, seed: int) -> Path:
    return Path(directory) / f"test_scored_{int(seed)}.json"


def read_guard(directory: Path, seed: int) -> dict | None:
    path = guard_path(directory, seed)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") == STARTED:
        raise GuardBlocked(
            f"{path.name} is 'started' but never completed: seed {seed}'s test partition may already have been "
            "scored. Re-scoring is blocked; a human must decide how to proceed."
        )
    if payload.get("status") != COMPLETED:
        raise GuardBlocked(f"{path.name} has unknown status {payload.get('status')!r}")
    return payload


def start_guard(directory: Path, seed: int, protocol_commit: str) -> Path:
    path = guard_path(directory, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": int(seed), "status": STARTED, "protocol_commit": protocol_commit,
               "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with path.open("x", encoding="utf-8", newline="\n") as handle:  # "x": fails if it already exists
        json.dump(payload, handle, indent=2)
    return path


def complete_guard(directory: Path, seed: int, results: dict) -> Path:
    path = guard_path(directory, seed)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != STARTED:
        raise GuardBlocked(f"{path.name} is not in 'started' state")
    payload.update({"status": COMPLETED, "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "results": results})
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8", newline="\n")
    return path
