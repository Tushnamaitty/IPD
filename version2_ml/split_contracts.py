"""Phase 3 Step 3: group-exclusive partitions and transfer guards.

No raw identifiers are written, and this module does not create an independent
confirmatory holdout. The previously inspected V1 test remains disclosed.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd


def _check(frame, group_col, target_col):
    for column in (group_col, target_col):
        if column not in frame:
            raise ValueError(f"Missing required column: {column}")
    if frame.empty:
        raise ValueError("Cannot partition an empty dataset")
    group = frame[group_col].astype("string")
    if group.isna().any() or group.str.strip().eq("").any():
        raise ValueError("Missing or blank group identifier")
    target = pd.to_numeric(frame[target_col], errors="coerce")
    if not target.isin([0, 1]).all():
        raise ValueError("Target must contain only nonmissing 0/1 values")
    return group, target.astype(int)


def grouped_partition(frame: pd.DataFrame, *, group_col="caseid",
                      target_col="m17", fractions=(0.7, 0.15, 0.15),
                      seed=42) -> pd.Series:
    """Deterministic mother-exclusive 3-way split.

    Splits are *approximately* stratified using the per-mother outcome rate.
    Fractions are provisional, not a frozen protocol. Actual proportions vary
    with mother sizes, and class balance must be audited on local data.
    """
    if len(fractions) != 3 or any(not 0 < x < 1 for x in fractions):
        raise ValueError("Expected three positive fractions")
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("Split fractions must sum to 1")
    group, y = _check(frame, group_col, target_col)
    stats = pd.DataFrame({"group": group.to_numpy(), "y": y.to_numpy()})
    aggregated = stats.groupby("group", sort=True)["y"].agg(["mean", "size"])
    if len(aggregated) < 3:
        raise ValueError("At least three distinct mothers required")
    # Stable hash avoids randomized Python hash salts and row-order sensitivity.
    def h(token):
        digest = hashlib.sha256(f"{seed}|{token}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")
    aggregated["bucket"] = (aggregated["mean"] >= 0.5).astype(int)
    assignments = {}
    labels = ("train", "validation", "test")
    for _, subset in aggregated.groupby("bucket"):
        order = sorted(subset.index.tolist(), key=h)
        n = len(order)
        a, b = int(round(fractions[0] * n)), int(round((fractions[0] + fractions[1]) * n))
        for key in order[:a]:
            assignments[str(key)] = labels[0]
        for key in order[a:b]:
            assignments[str(key)] = labels[1]
        for key in order[b:]:
            assignments[str(key)] = labels[2]
    result = group.map(assignments)
    if result.isna().any():
        raise RuntimeError("Unassigned group")
    audit_partition(frame, result, group_col=group_col, target_col=target_col)
    return pd.Series(result.to_numpy(), index=frame.index, name="partition")


def audit_partition(frame: pd.DataFrame, partition: pd.Series, *,
                    group_col="caseid", target_col="m17") -> dict:
    group, y = _check(frame, group_col, target_col)
    if len(partition) != len(frame):
        raise ValueError("Partition length mismatch")
    p = pd.Series(partition.to_numpy(), index=frame.index)
    expected = {"train", "validation", "test"}
    if not p.isin(expected).all():
        raise ValueError("Unexpected partition labels")
    g = pd.DataFrame({"group": group.to_numpy(), "part": p.to_numpy()})
    if g.groupby("group")["part"].nunique().gt(1).any():
        raise ValueError("LEAKAGE: a mother appears in multiple partitions")
    summary = {}
    for label in ("train", "validation", "test"):
        mask = p.eq(label).to_numpy()
        summary[label] = {"rows": int(mask.sum()), "mothers": int(g.loc[mask, "group"].nunique()),
                          "positives": int(y.to_numpy()[mask].sum()),
                          "negatives": int(mask.sum() - y.to_numpy()[mask].sum())}
    for label, x in summary.items():
        if x["rows"] == 0 or x["positives"] == 0 or x["negatives"] == 0:
            raise ValueError(f"{label}: empty partition or outcome class; revise development-only split")
    return {"status": "GROUP_EXCLUSIVE_VALID", "summary": summary,
            "independent_final_test_certified": False,
            "notes": ["V1 test exposure prevents automatic independent-test claims.",
                      "Split ratios are provisional until protocol freeze.",
                      "Never commit row-level assignments or mother IDs."]}


def guard_wave_transfer(train_wave: pd.Series, eval_wave: pd.Series,
                        *, adaptation_ids=None, evaluation_ids=None) -> dict:
    """Guard declared wave direction and disjoint target adaptation/evaluation sets.

    Does not verify how preprocessing was fitted or whether labels were accessed.
    """
    if set(train_wave.dropna().astype(str)) != {"NFHS-4"}:
        raise ValueError("Wave-transfer training must use NFHS-4 only")
    if set(eval_wave.dropna().astype(str)) != {"NFHS-5"}:
        raise ValueError("Wave-transfer evaluation must use NFHS-5 only")
    if train_wave.isna().any() or eval_wave.isna().any():
        raise ValueError("Missing wave labels")
    if (adaptation_ids is None) != (evaluation_ids is None):
        raise ValueError("Provide both adaptation and evaluation ID sets or neither")
    if adaptation_ids is not None and (set(adaptation_ids) & set(evaluation_ids)):
        raise ValueError("LEAKAGE: target adaptation/evaluation overlap")
    return {"status": "WAVE_DIRECTION_VALID",
            "adaptation_disjoint_checked": adaptation_ids is not None,
            "target_label_use_not_verified": True}
