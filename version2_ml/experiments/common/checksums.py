"""Privacy-safe row-membership digests for matched comparisons.

Digests are one-way SHA-256 values computed in memory; sorted birth keys are
never written or printed. Equal digests prove two analyses used identical row
sets without revealing the rows.
"""
from __future__ import annotations

import hashlib


def membership_digest(keys, *, seed: int = 42) -> str:
    """SHA-256 over sorted stringified keys, salted with seed and length."""
    tokens = sorted(str(key) for key in keys)
    digest = hashlib.sha256()
    digest.update(f"membership|seed={int(seed)}|n={len(tokens)}|".encode("utf-8"))
    for token in tokens:
        digest.update(token.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def assert_matched_membership(*digests) -> None:
    """Raise if any supplied membership digest differs from the others."""
    if len(digests) < 2:
        raise ValueError("Need at least two digests to prove a matched comparison")
    if any(not d for d in digests):
        raise ValueError("Empty membership digest")
    if len(set(digests)) != 1:
        raise ValueError("MATCHED-ROW VIOLATION: membership digests differ")
