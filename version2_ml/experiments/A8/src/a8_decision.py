"""A8 predeclared decision rule (config.yaml: decision_rule), applied mechanically."""
from __future__ import annotations

import pandas as pd

from a8_arms import added_feature_count


def decide(deltas: pd.DataFrame, rule: dict, seeds) -> dict:
    """Return the selected tier and the evidence for it.

    An arm passes only if, for ``rule['model']``, delta ROC-AUC >= the minimum
    AND delta AP >= the minimum on every required seed. Among passing arms the
    one with the fewest added features wins; ties go to ``rule['tie_break']``.
    """
    seeds = sorted(int(s) for s in seeds)
    block = deltas[deltas["model"] == rule["model"]]
    per_seed, passed = {}, []
    for arm in sorted(block["arm"].unique()):
        rows = block[block["arm"] == arm].set_index("seed")
        missing = sorted(set(seeds) - set(rows.index.astype(int)))
        if missing:
            raise ValueError(f"{arm}: no deltas for seeds {missing}; the rule needs all seeds")
        checks = {}
        for seed in seeds:
            d_auc = float(rows.loc[seed, "delta_roc_auc"])
            d_ap = float(rows.loc[seed, "delta_average_precision"])
            checks[str(seed)] = {
                "delta_roc_auc": round(d_auc, 6),
                "delta_average_precision": round(d_ap, 6),
                "passes": bool(d_auc >= float(rule["min_delta_roc_auc"]) and d_ap >= float(rule["min_delta_average_precision"])),
            }
        per_seed[arm] = checks
        if all(item["passes"] for item in checks.values()):
            passed.append(arm)
    if passed:
        fewest = min(added_feature_count(arm) for arm in passed)
        candidates = sorted(arm for arm in passed if added_feature_count(arm) == fewest)
        selected_arm = rule["tie_break"] if rule["tie_break"] in candidates else candidates[0]
    else:
        selected_arm = rule["comparator"]
    return {
        "selected_arm": selected_arm,
        "selected_tier": rule["tier_names"][selected_arm],
        "rule": rule["adopt_arm_if"],
        "thresholds": {
            "min_delta_roc_auc": float(rule["min_delta_roc_auc"]),
            "min_delta_average_precision": float(rule["min_delta_average_precision"]),
            "model": rule["model"],
            "seeds": seeds,
        },
        "passed_arms": passed,
        "per_seed_deltas": per_seed,
    }
