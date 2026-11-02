"""A9 PSU-grouped, outcome-rate-stratified 70/15/15 splits (PROTOCOL.md section 3).

Same deterministic hashing scheme as split_contracts.grouped_partition
(sha256(f"{seed}|{group}")), grouped by survey cluster v021 and stratified on
five fixed PSU outcome-rate strata instead of the binary mother bucket.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from split_contracts import audit_partition

LABELS = ("train", "validation", "test")


def assert_psu_nesting(frame: pd.DataFrame, *, psu_col: str = "v021", mother_col: str = "caseid") -> dict:
    """Every mother lies in exactly one PSU, so PSU grouping is mother-exclusive."""
    psu = pd.to_numeric(frame[psu_col], errors="coerce")
    if psu.isna().any():
        raise ValueError("Missing PSU code")
    per_mother = pd.DataFrame({"m": frame[mother_col].astype("string").to_numpy(), "p": psu.to_numpy()}) \
        .groupby("m")["p"].nunique()
    spanning = int((per_mother > 1).sum())
    if spanning:
        raise ValueError(f"PSU NESTING VIOLATION: {spanning} mother(s) appear in more than one PSU")
    return {"n_psus": int(psu.nunique()), "n_mothers": int(len(per_mother)), "mothers_in_multiple_psus": 0}


def _stratum(rate: pd.Series, edges) -> pd.Series:
    edges = [float(e) for e in edges]
    stratum = pd.Series(len(edges), index=rate.index, dtype=int)
    stratum[rate == edges[0]] = 0
    for k in range(1, len(edges)):
        stratum[(rate > edges[k - 1]) & (rate <= edges[k])] = k
    return stratum


def psu_stratified_partition(frame: pd.DataFrame, *, seed: int, group_col: str = "v021", target_col: str = "m17",
                             fractions=(0.7, 0.15, 0.15), edges=(0.0, 0.2, 0.4, 0.6),
                             mother_col: str = "caseid") -> pd.Series:
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("Expected three fractions summing to 1")
    assert_psu_nesting(frame, psu_col=group_col, mother_col=mother_col)
    group = pd.to_numeric(frame[group_col]).astype(int).astype(str)
    y = pd.to_numeric(frame[target_col])
    if not y.isin([0, 1]).all():
        raise ValueError("Target must be 0/1")
    rate = pd.DataFrame({"g": group.to_numpy(), "y": y.to_numpy()}).groupby("g", sort=True)["y"].mean()
    strata = _stratum(rate, edges)

    def h(token):
        return int.from_bytes(hashlib.sha256(f"{seed}|{token}".encode("utf-8")).digest()[:8], "big")

    assignments = {}
    for _, members in strata.groupby(strata):
        order = sorted(members.index.tolist(), key=h)
        n = len(order)
        a, b = int(round(fractions[0] * n)), int(round((fractions[0] + fractions[1]) * n))
        for key in order[:a]:
            assignments[key] = LABELS[0]
        for key in order[a:b]:
            assignments[key] = LABELS[1]
        for key in order[b:]:
            assignments[key] = LABELS[2]
    result = pd.Series(group.map(assignments).to_numpy(), index=frame.index, name="partition")
    if result.isna().any():
        raise RuntimeError("Unassigned PSU")
    audit_partition(frame, result, group_col=group_col, target_col=target_col)
    audit_partition(frame, result, group_col=mother_col, target_col=target_col)
    return result


def overlap_counts(frame: pd.DataFrame, partition: pd.Series, column: str) -> dict:
    sets = {label: set(frame.loc[partition.eq(label).to_numpy(), column].astype(str)) for label in LABELS}
    return {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }


def partition_summary(frame: pd.DataFrame, partition: pd.Series, *, psu_col="v021", mother_col="caseid",
                      target_col="m17") -> dict:
    out = {}
    for label in LABELS:
        mask = partition.eq(label).to_numpy()
        y = frame.loc[mask, target_col].to_numpy()
        out[label] = {
            "n_births": int(mask.sum()),
            "n_mothers": int(frame.loc[mask, mother_col].nunique()),
            "n_psus": int(frame.loc[mask, psu_col].nunique()),
            "n_events": int(y.sum()),
            "prevalence": float(np.mean(y)),
        }
    return out
