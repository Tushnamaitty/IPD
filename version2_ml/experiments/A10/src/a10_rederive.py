"""Verified deterministic re-derivation of A9's V2 test predictions.

A9 kept no per-birth predictions or model state. For one seed this rebuilds
A9's PSU split and calibration carve with A9's own functions, refits V2 with
A9's fit_seed, predicts the test rows, and then REQUIRES:
- split manifest hash and all membership digests equal to the A9 guard;
- V2 validation thresholds equal to the guard (exact);
- all 11 V2 test metrics from a9_metrics.test_metrics equal to the guard (exact).
Any mismatch raises; nothing is returned or written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from a7_candidates import carve_calibration_partition
from a9_metrics import METRICS, test_metrics
from a9_models import fit_seed, predict_test
from a9_splits import psu_stratified_partition
from checksums import membership_digest
from manifests import split_manifest_hash


class RederivationMismatch(RuntimeError):
    pass


def rederive_seed(seed: int, cohort: pd.DataFrame, keys, frames_v2: dict, a9_config: dict, params: dict,
                  guard_results: dict) -> dict:
    split = a9_config["split"]
    psu = pd.to_numeric(cohort[split["group_col"]]).astype(int).astype(str)
    partition = psu_stratified_partition(cohort, seed=seed, group_col=split["group_col"], target_col=split["target_col"],
                                         fractions=tuple(split["fractions"]), edges=tuple(split["strata_edges"]),
                                         mother_col=split["mother_col"])
    audit = guard_results["audit"]
    split_hash = split_manifest_hash(split_id=split["split_id_template"].format(seed=seed), seed=seed,
                                     fractions=split["fractions"], assignments=dict(zip(psu.tolist(), partition.tolist())).items())
    if split_hash != audit["split_manifest_hash"]:
        raise RederivationMismatch(f"seed {seed}: split manifest hash differs from A9")
    masks = {label: partition.eq(label).to_numpy() for label in ("train", "validation", "test")}
    for label, mask in masks.items():
        if membership_digest(np.asarray(keys)[mask], seed=seed) != audit["membership_digests"][label]:
            raise RederivationMismatch(f"seed {seed}: {label} membership differs from A9")
    train = np.flatnonzero(masks["train"])
    carve_cfg = a9_config["calibration"]["carve"]
    carve = carve_calibration_partition(cohort.iloc[train].reset_index(drop=True), group_col=carve_cfg["group_col"],
                                        calibration_fraction=float(carve_cfg["subset_fraction"]), seed=int(carve_cfg["seed"]))
    fit_pos, calib_pos = train[carve.eq("fit").to_numpy()], train[carve.eq("calibration").to_numpy()]
    valid_pos, test_pos = np.flatnonzero(masks["validation"]), np.flatnonzero(masks["test"])
    y = cohort[split["target_col"]].to_numpy(dtype=int)
    models = fit_seed(frames_v2, a9_config["models"], params, fit_positions=fit_pos, calib_positions=calib_pos,
                      valid_positions=valid_pos, y=y, seed=seed)
    thresholds = models["V2"]["thresholds"]
    if thresholds != guard_results["thresholds"]["V2"]:
        raise RederivationMismatch(f"seed {seed}: V2 thresholds {thresholds} differ from A9 {guard_results['thresholds']['V2']}")
    p_test = predict_test(models, frames_v2, test_pos)["V2"]
    point = test_metrics(y[test_pos], p_test, thresholds)
    stored = guard_results["point"]["V2"]
    mismatched = [key for key in METRICS if point[key] != stored[key]]
    if mismatched:
        raise RederivationMismatch(f"seed {seed}: V2 test metrics differ from A9: {mismatched}")
    return {"seed": seed, "test_positions": test_pos, "p_test": p_test, "threshold": thresholds["f1_optimal"],
            "psu_test": psu.to_numpy()[test_pos], "verified_metrics": len(METRICS)}
