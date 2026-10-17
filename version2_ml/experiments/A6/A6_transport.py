"""A6: geographic (leave-one-state-out) and public <-> private sector transport on NFHS-5.

Guarantees enforced in code:
  * every mother belongs to exactly one state (state is household residence);
  * source-training/validation mothers and target-domain mothers never overlap
    (assert_mother_disjoint raises a LEAKAGE error);
  * models, preprocessing, encoders and thresholds see only source rows; the
    target outcome array is only indexed for evaluation;
  * held-out domains pass a predeclared support screen (n, positives, negatives)
    or are reported NOT ESTIMABLE with counts suppressed;
  * sector transport uses a sector-free feature set (facility_type rejected);
    mothers with births in both sectors are excluded from both directions.
No causal interpretation of state or sector; aggregate outputs only.
"""
from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "version2_ml", ROOT / "version2_ml" / "experiments" / "common"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from preprocessing import FeatureManifest  # noqa: E402
from model_framework import AuthorizedClassifier  # noqa: E402
from exp_metrics import select_threshold_max_f1  # noqa: E402

FT_NUM = ("maternal_age_at_index_birth", "education_years", "birth_order")
FT_CAT = ("wealth_index", "facility_type", "residence", "religion", "twin_order")
SECTOR_FREE_CAT = tuple(c for c in FT_CAT if c != "facility_type")
SECTORS = {"public": range(20, 28), "private": range(30, 34)}
OVERLAP_POLICIES = ("exclude_overlap_mothers",)


@dataclass(frozen=True)
class SupportScreen:
    min_n: int = 200
    min_pos: int = 30
    min_neg: int = 30

    def check(self, y) -> dict:
        y = np.asarray(y)
        n, pos = int(len(y)), int(y.sum())
        neg = n - pos
        reasons = [r for r, bad in (("n<%d" % self.min_n, n < self.min_n), ("positives<%d" % self.min_pos,
                   pos < self.min_pos), ("negatives<%d" % self.min_neg, neg < self.min_neg)) if bad]
        return {"status": "NOT_ESTIMABLE" if reasons else "ESTIMABLE", "reasons": reasons,
                "n_births": n, "n_events": pos}


def assert_mother_disjoint(groups, mask_a, mask_b, label: str = "") -> None:
    g = np.asarray(groups)
    shared = set(g[np.asarray(mask_a)]) & set(g[np.asarray(mask_b)])
    if shared:
        raise ValueError(f"LEAKAGE: {len(shared)} mother(s) in both source and target {label}".strip())


def mother_state_check(groups, states) -> dict:
    per = pd.Series(np.asarray(states)).groupby(pd.Series(np.asarray(groups))).nunique()
    bad = int(per.gt(1).sum())
    if bad:
        raise ValueError(f"Identifier anomaly: {bad} mother(s) recorded in more than one state")
    return {"mothers": int(len(per)), "mothers_in_multiple_states": 0}


def assert_sector_free(manifest: FeatureManifest) -> None:
    names = set(manifest.numerical) | set(manifest.categorical)
    if "facility_type" in names or any(n.startswith(("te_", "hist_")) and "sector" in n for n in names):
        raise ValueError("Sector transport requires a sector-free feature set")


def sector_of(m15) -> np.ndarray:
    m = pd.to_numeric(pd.Series(np.asarray(m15)), errors="coerce")
    return np.where(m.between(20, 27), "public", np.where(m.between(30, 33), "private",
                                                          np.where(m.eq(96), "other", "invalid")))


def sector_overlap_policy(groups, sector, policy: str = "exclude_overlap_mothers") -> tuple[np.ndarray, dict]:
    """Keep-mask for public/private transfer; m15=96 dropped; both-sector mothers handled by policy."""
    if policy not in OVERLAP_POLICIES:
        raise ValueError(f"Unknown overlap policy; prespecified options: {OVERLAP_POLICIES}")
    g, s = pd.Series(np.asarray(groups)), pd.Series(np.asarray(sector))
    if s.eq("invalid").any():
        raise ValueError("Invalid facility code for sector transfer")
    pp = s.isin(["public", "private"])
    both = s[pp].groupby(g[pp]).nunique()
    overlap = set(both.index[both.gt(1)])
    keep = pp & ~g.isin(overlap)
    return keep.to_numpy(), {"policy": policy, "births_other_sector_excluded": int((~pp).sum()),
                             "mothers_with_public_and_private_births": len(overlap),
                             "births_excluded_for_overlap": int((pp & g.isin(overlap)).sum()),
                             "births_kept": int(keep.sum())}


def fit_predict_domain(frame: pd.DataFrame, y, *, source_train, source_val, score: dict,
                       manifest: FeatureManifest, algorithm: str, seed: int = 42) -> tuple[dict, float]:
    """Fit on source_train rows only, threshold on source_val rows only, score the named masks.

    `y` may span all rows; only y[source_train] and y[source_val] are read.
    """
    st, sv = np.asarray(source_train), np.asarray(source_val)
    if (st & sv).any():
        raise ValueError("Source train and validation rows overlap")
    for name, m in score.items():
        if name.startswith("target") and (np.asarray(m) & (st | sv)).any():
            raise ValueError(f"LEAKAGE: target mask '{name}' overlaps source rows")
    y = np.asarray(y)
    clf = AuthorizedClassifier(manifest, algorithm=algorithm, seed=seed)
    clf.fit(frame[st], y[st], partition="train")
    thr = select_threshold_max_f1(y[sv], clf.predict_proba(frame[sv], partition="validation"))
    preds = {k: clf.predict_proba(frame[np.asarray(m)], partition="validation") for k, m in score.items()}
    return preds, thr


def loso_masks(states, partition, held_out) -> dict:
    """Leave-one-state-out masks over shared non-test rows."""
    s, p = np.asarray(states).astype(str), np.asarray(partition)
    out = s != str(held_out)
    return {"source_train": out & (p == "train"), "source_val": out & (p == "validation"),
            "target_all": ~out, "target_val": ~out & (p == "validation")}


def sector_masks(sector, partition, source: str, target: str, keep) -> dict:
    s, p, k = np.asarray(sector), np.asarray(partition), np.asarray(keep)
    if source == target or {source, target} != {"public", "private"}:
        raise ValueError("Sector directions are public->private or private->public")
    return {"source_train": k & (s == source) & (p == "train"), "source_val": k & (s == source) & (p == "validation"),
            "target_all": k & (s == target), "target_val": k & (s == target) & (p == "validation"),
            "target_train": k & (s == target) & (p == "train")}
