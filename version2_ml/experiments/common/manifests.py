"""SHA-256 manifest hashing for feature sets and mother-grouped splits.

Hashes are computed from canonical, row-order-independent serializations.
Split manifests contain mother identifiers only transiently in memory; the
published artifact is the one-way digest.
"""
from __future__ import annotations

import hashlib
import json


def stable_hash(payload) -> str:
    """SHA-256 over a canonical JSON serialization of any hashable payload."""
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def feature_manifest_hash(*, tier, numerical, categorical, builder_id, extra=None) -> str:
    return stable_hash(
        {
            "kind": "feature_manifest",
            "tier": tier,
            "numerical": list(numerical),
            "categorical": list(categorical),
            "builder_id": builder_id,
            "extra": extra,
        }
    )


def split_manifest_hash(*, split_id, seed, fractions, assignments) -> str:
    """Hash sorted (group, partition-label) pairs; independent of row order."""
    pairs = sorted((str(group), str(label)) for group, label in assignments)
    return stable_hash(
        {
            "kind": "split_manifest",
            "split_id": split_id,
            "seed": int(seed),
            "fractions": [float(x) for x in fractions],
            "assignments": pairs,
        }
    )
