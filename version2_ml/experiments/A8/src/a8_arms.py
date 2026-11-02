"""A8 arm registry and leakage-safe feature construction.

Arms extend the A7 tier F1_detail_history (a7_candidates.TIERS) with A1's
verbatim state indicator and/or A3's grouped out-of-fold state x sector target
encoding. The encoder is fitted on fit-pool labels only (OOF within the fit
pool); calibration and validation rows are encoded with the 100% fit-pool
table, exactly as A3's encode_partitions treats non-training partitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from a1_state_context import STATE_FEATURE
from A3_target_encoder import TE_CELL, TESpec, encode_partitions
from a7_candidates import TIERS, estimator_factory
from model_framework import AuthorizedClassifier
from preprocessing import FeatureManifest

BASE_TIER = "F1_detail_history"
ARM_ORDER = ("H0", "H1", "H2", "H3")
ARMS = {
    "H0": {"categorical": (), "numerical": (), "manifest_tier": "F2"},
    "H1": {"categorical": (STATE_FEATURE,), "numerical": (), "manifest_tier": "F3"},
    "H2": {"categorical": (), "numerical": (TE_CELL,), "manifest_tier": "F4"},
    "H3": {"categorical": (STATE_FEATURE,), "numerical": (TE_CELL,), "manifest_tier": "F4"},
}
TE_STATE_KEY = "nfhs5_state"   # A3 key name: v024 code as string
TE_SECTOR_KEY = "sector"       # A3 key name: facility_type public/private/other
FIT_LABEL = "train"            # encode_partitions fits on rows labelled "train" only


def arm_manifest(arm: str) -> FeatureManifest:
    spec, base = ARMS[arm], TIERS[BASE_TIER]
    return FeatureManifest(
        tuple(base["numerical"]) + spec["numerical"],
        tuple(base["categorical"]) + spec["categorical"],
        spec["manifest_tier"],
    )


def added_feature_count(arm: str) -> int:
    return len(ARMS[arm]["categorical"]) + len(ARMS[arm]["numerical"])


def te_key_frame(state_codes, facility_type, index) -> pd.DataFrame:
    """A3 key columns: state as the integer v024 code string, sector = facility_type."""
    codes = pd.to_numeric(pd.Series(np.asarray(state_codes)), errors="coerce")
    if codes.isna().any():
        raise ValueError("Missing v024 state code")
    return pd.DataFrame(
        {
            TE_STATE_KEY: codes.astype(int).astype(str).to_numpy(),
            TE_SECTOR_KEY: pd.Series(np.asarray(facility_type)).astype("string").to_numpy(),
        },
        index=index,
    )


def encode_state_sector(keys: pd.DataFrame, y, groups, fit_mask, *, alpha: float, n_min: int,
                        n_splits: int, seed: int):
    """A3 encode_partitions with the fit pool as its only labelled ("train") partition.

    Returns (encoder, te_state_sector array aligned to ``keys``). Labels of
    non-fit rows are replaced by 0 before the call, so they cannot reach the
    encoder even in principle; A3 additionally slices them away.
    """
    fit_mask = np.asarray(fit_mask, dtype=bool)
    y_fit_only = np.where(fit_mask, np.asarray(y, dtype=int), 0)
    partition = pd.Series(np.where(fit_mask, FIT_LABEL, "held_out"), index=keys.index)
    encoder, te = encode_partitions(
        keys, y_fit_only, np.asarray(groups), partition, TESpec(alpha=float(alpha), n_min=int(n_min)),
        n_splits=int(n_splits), seed=int(seed), state_col=TE_STATE_KEY, sector_col=TE_SECTOR_KEY,
    )
    if encoder.full_table.tables["n"] != int(fit_mask.sum()):
        raise RuntimeError("Target-encoder full table was not fitted on exactly the fit pool")
    return encoder, te[TE_CELL].to_numpy(dtype=float)


def arm_frame(base: pd.DataFrame, state_labels, te_values, arm: str) -> pd.DataFrame:
    """Base tier frame plus the arm's added columns, in manifest order."""
    frame = base.copy()
    if STATE_FEATURE in ARMS[arm]["categorical"]:
        frame[STATE_FEATURE] = pd.Series(np.asarray(state_labels), index=base.index).astype("string")
    if TE_CELL in ARMS[arm]["numerical"]:
        values = np.asarray(te_values, dtype=float)
        if values.shape != (len(base),) or not np.isfinite(values).all():
            raise ValueError("Target-encoding values missing or misaligned")
        frame[TE_CELL] = values
    manifest = arm_manifest(arm)
    return frame[list(manifest.numerical) + list(manifest.categorical)]


def fit_arm(model: str, arm: str, params: dict, frame: pd.DataFrame, y, seed: int) -> AuthorizedClassifier:
    """a7_candidates.fit_candidate with the arm's manifest instead of an A7 tier."""
    classifier = AuthorizedClassifier(arm_manifest(arm), algorithm=model, seed=seed)
    classifier.estimator = estimator_factory(model, params, seed)
    classifier.fit(frame, y, partition="train")
    return classifier
