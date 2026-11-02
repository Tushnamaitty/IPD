"""A10 subgroup metrics, pooled PSU bootstrap, gaps and predeclared flags.

Pooled rows are the concatenated test partitions of the A9 seeds; each row
carries its seed's V2 probability and thresholded label. Only aggregates are
returned: point metrics, CI bounds, replicate-wise gaps and suppressed bins.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from a7_calibration import calibration_intercept_slope
from a9_metrics import calibration_bins, percentile_ci
from aggregate_schema import suppress_small_count
from exp_metrics import binary_metrics

METRICS = ("prevalence", "roc_auc", "average_precision", "brier_score", "calibration_intercept",
           "calibration_slope", "observed_rate", "mean_predicted", "recall_at_threshold", "precision_at_threshold")
NOT_ESTIMABLE = "not estimable"


def group_metrics(y, p, label) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    label = np.asarray(label, dtype=bool)
    if len(np.unique(y)) < 2:
        return {key: np.nan for key in METRICS}
    point = binary_metrics(y, p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        intercept, slope = calibration_intercept_slope(y, p)
    tp = float(np.sum(label & (y == 1)))
    predicted, positives = float(label.sum()), float(y.sum())
    return {"prevalence": float(y.mean()), "roc_auc": float(point["roc_auc"]),
            "average_precision": float(point["average_precision"]), "brier_score": float(point["brier_score"]),
            "calibration_intercept": float(intercept), "calibration_slope": float(slope),
            "observed_rate": float(y.mean()), "mean_predicted": float(p.mean()),
            "recall_at_threshold": tp / positives if positives else np.nan,
            "precision_at_threshold": tp / predicted if predicted else np.nan}


def supported(y, *, min_births: int, min_per_class: int) -> bool:
    y = np.asarray(y, dtype=int)
    events = int(y.sum())
    return len(y) >= min_births and events >= min_per_class and len(y) - events >= min_per_class


class PooledTest:
    """Concatenated test rows of all seeds, with per-seed PSU structure for the bootstrap."""

    def __init__(self, seeds: list, y, p, label, psu, cohort_index, keys):
        self.seeds = np.asarray(seeds)
        self.y = np.asarray(y, dtype=int)
        self.p = np.asarray(p, dtype=float)
        self.label = np.asarray(label, dtype=bool)
        self.psu = np.asarray(psu)
        self.cohort_index = np.asarray(cohort_index)
        self.keys = np.asarray(keys)
        self.rows_of = []
        for seed in pd.unique(self.seeds):
            rows = np.flatnonzero(self.seeds == seed)
            codes, inverse = np.unique(self.psu[rows], return_inverse=True)
            order = np.argsort(inverse, kind="stable")
            starts = np.searchsorted(inverse[order], np.arange(len(codes)))
            ends = np.append(starts[1:], len(order))
            self.rows_of.append([rows[order[s:e]] for s, e in zip(starts, ends)])

    def replicate_indices(self, rng):
        parts = []
        for groups in self.rows_of:
            draw = rng.integers(0, len(groups), len(groups))
            parts.extend(groups[k] for k in draw)
        return np.concatenate(parts)


def analyse(pooled: PooledTest, members: dict, config: dict) -> dict:
    """Point estimates, per-seed ranges, bootstrap CIs and replicate gaps for every subgroup."""
    screen = config["support_screen"]
    boot = config["bootstrap"]
    groups = []
    for dimension, block in members.items():
        for group, cohort_mask in block.items():
            mask = cohort_mask[pooled.cohort_index]
            ok = supported(pooled.y[mask], min_births=int(screen["min_births"]), min_per_class=int(screen["min_per_class"]))
            entry = {"dimension": dimension, "group": group, "mask": mask, "supported": ok,
                     "n": int(mask.sum()), "n_unique_births": int(len(np.unique(pooled.keys[mask]))),
                     "events": int(pooled.y[mask].sum())}
            if ok:
                entry["point"] = group_metrics(pooled.y[mask], pooled.p[mask], pooled.label[mask])
                per_seed = []
                for seed in pd.unique(pooled.seeds):
                    seed_mask = mask & (pooled.seeds == seed)
                    if supported(pooled.y[seed_mask], min_births=int(screen["min_births"]),
                                 min_per_class=int(screen["min_per_class"])):
                        per_seed.append(group_metrics(pooled.y[seed_mask], pooled.p[seed_mask], pooled.label[seed_mask]))
                entry["per_seed"] = per_seed
                entry["replicates"] = {key: np.full(int(boot["replicates"]), np.nan) for key in METRICS}
            groups.append(entry)
    rng = np.random.default_rng(int(boot["seed"]))
    live = [g for g in groups if g["supported"]]
    for b in range(int(boot["replicates"])):
        idx = pooled.replicate_indices(rng)
        for entry in live:
            sel = idx[entry["mask"][idx]]
            values = group_metrics(pooled.y[sel], pooled.p[sel], pooled.label[sel])
            for key in METRICS:
                entry["replicates"][key][b] = values[key]
    gaps = []
    for dimension in members:
        block = [g for g in live if g["dimension"] == dimension]
        if len(block) < 2:
            gaps.append({"dimension": dimension, "n_supported_groups": len(block), "gap": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "max_group": NOT_ESTIMABLE, "min_group": NOT_ESTIMABLE})
            continue
        aucs = {g["group"]: g["point"]["roc_auc"] for g in block}
        reps = np.vstack([g["replicates"]["roc_auc"] for g in block])
        low, high, n_valid = percentile_ci(reps.max(axis=0) - reps.min(axis=0))
        gaps.append({"dimension": dimension, "n_supported_groups": len(block),
                     "max_group": max(aucs, key=aucs.get), "max_roc_auc": max(aucs.values()),
                     "min_group": min(aucs, key=aucs.get), "min_roc_auc": min(aucs.values()),
                     "gap": max(aucs.values()) - min(aucs.values()), "ci_low": low, "ci_high": high,
                     "replicates_valid": n_valid})
    return {"groups": groups, "gaps": gaps}


def flags_for(entry: dict, config: dict) -> list:
    if not entry["supported"]:
        return ["UNDER_SUPPORTED"]
    point = entry["point"]
    out = []
    if point["roc_auc"] < 0.70:
        out.append("LOW_DISCRIMINATION")
    if abs(point["calibration_intercept"]) > 0.10 or not 0.8 <= point["calibration_slope"] <= 1.2:
        out.append("MISCALIBRATED")
    return out


def metrics_rows(result: dict, config: dict, min_cell: int) -> list:
    rows = []
    for entry in result["groups"]:
        row = {"dimension": entry["dimension"], "group": entry["group"],
               "n": suppress_small_count(entry["n"], threshold=min_cell),
               "n_unique_births": suppress_small_count(entry["n_unique_births"], threshold=min_cell),
               "events": suppress_small_count(entry["events"], threshold=min_cell),
               "non_events": suppress_small_count(entry["n"] - entry["events"], threshold=min_cell),
               "supported": entry["supported"], "flags": ";".join(flags_for(entry, config)) or "none"}
        for key in METRICS:
            if not entry["supported"]:
                row.update({key: NOT_ESTIMABLE, f"{key}_ci_low": NOT_ESTIMABLE, f"{key}_ci_high": NOT_ESTIMABLE})
                continue
            low, high, _ = percentile_ci(entry["replicates"][key])
            row.update({key: entry["point"][key], f"{key}_ci_low": low, f"{key}_ci_high": high})
        if entry["supported"]:
            for key in ("roc_auc", "average_precision", "brier_score", "calibration_slope", "calibration_intercept"):
                values = [s[key] for s in entry["per_seed"]]
                row[f"{key}_seed_min"] = float(np.min(values)) if values else NOT_ESTIMABLE
                row[f"{key}_seed_max"] = float(np.max(values)) if values else NOT_ESTIMABLE
            row["n_seeds_supported"] = len(entry["per_seed"])
            row["roc_auc_replicates_valid"] = percentile_ci(entry["replicates"]["roc_auc"])[2]
        rows.append(row)
    return rows


def flag_rows(result: dict, config: dict, min_cell: int) -> list:
    rows = []
    for entry in result["groups"]:
        for flag in flags_for(entry, config):
            if flag == "UNDER_SUPPORTED":
                rows.append({"dimension": entry["dimension"], "group": entry["group"], "flag": flag,
                             "n": suppress_small_count(entry["n"], threshold=min_cell),
                             "events": suppress_small_count(entry["events"], threshold=min_cell),
                             "non_events": suppress_small_count(entry["n"] - entry["events"], threshold=min_cell),
                             "evidence": f"support screen requires >= {config['support_screen']['min_births']} births and "
                                         f">= {config['support_screen']['min_per_class']} per class; not estimable"})
                continue
            point = entry["point"]
            auc_low, auc_high, _ = percentile_ci(entry["replicates"]["roc_auc"])
            if flag == "LOW_DISCRIMINATION":
                evidence = f"ROC-AUC {point['roc_auc']:.4f} (95% CI {auc_low:.4f}-{auc_high:.4f}) < 0.70"
            else:
                evidence = (f"intercept {point['calibration_intercept']:+.3f} (limit |.| <= 0.10); "
                            f"slope {point['calibration_slope']:.3f} (limit 0.8-1.2)")
            rows.append({"dimension": entry["dimension"], "group": entry["group"], "flag": flag,
                         "n": suppress_small_count(entry["n"], threshold=min_cell),
                         "events": suppress_small_count(entry["events"], threshold=min_cell),
                         "non_events": suppress_small_count(entry["n"] - entry["events"], threshold=min_cell),
                         "roc_auc": point["roc_auc"], "calibration_intercept": point["calibration_intercept"],
                         "calibration_slope": point["calibration_slope"], "evidence": evidence})
    return rows


def sector_bins(pooled: PooledTest, members: dict, config: dict, min_cell: int) -> pd.DataFrame:
    frames = []
    for group, cohort_mask in members["facility_sector"].items():
        mask = cohort_mask[pooled.cohort_index]
        if mask.sum() < int(config["calibration_bins"]["n_bins"]):
            continue
        bins = calibration_bins(pooled.y[mask], pooled.p[mask], n_bins=int(config["calibration_bins"]["n_bins"]),
                                min_cell=min_cell)
        frames.append(bins.assign(facility_sector=group))
    return pd.concat(frames, ignore_index=True)
