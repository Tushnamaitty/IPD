"""A9 aggregation across seeds, predeclared statements, paper table and figures.

Inputs are the per-seed aggregate results stored in the guard files
(point metrics, bootstrap replicate metric values, curves, bins). Pooled
estimate = mean across seeds; pooled CI = percentile interval of the
across-seed mean of replicate b (PROTOCOL.md section 8).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from a9_metrics import METRICS, percentile_ci
from a9_models import FEATURE_LABELS, MODEL_ORDER

COMPARISONS = (("V2", "V1_spec"), ("V2", "F1_core"), ("V2", "V2_LR"))
MODEL_COLORS = {"V2": "#2a78d6", "V2_LR": "#7a5cd6", "F1_core": "#8a8986", "V1_spec": "#e0782b"}
SURFACE, INK, MUTED, GRIDLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"

STATEMENTS = {
    "superiority": ("V2 discrimination is superior to the V1 specification",
                    "V2 discrimination is not shown to be superior to the V1 specification"),
    "calibration": ("V2 is well calibrated", "V2 is not shown to be well calibrated"),
    "equivalence": ("XGBoost and LR are equivalent", "XGBoost and LR are not shown to be equivalent"),
}


def _models(results) -> list:
    present = set(results[0]["point"])
    return [m for m in MODEL_ORDER if m in present]


def _replicates(result, model, metric) -> np.ndarray:
    return np.asarray(result["replicates"][model][metric], dtype=float)


def test_metrics_table(results, protocol_commit) -> pd.DataFrame:
    rows = []
    for result in results:
        for model in _models(results):
            row = {"protocol_commit": protocol_commit, "seed": result["seed"], "model": model,
                   "features": FEATURE_LABELS[model], **result["test_counts"],
                   "threshold_f1_optimal": result["thresholds"][model]["f1_optimal"],
                   "threshold_recall_80": result["thresholds"][model]["recall_80"]}
            for metric in METRICS:
                low, high, n_valid = percentile_ci(_replicates(result, model, metric))
                row.update({metric: result["point"][model][metric], f"{metric}_ci_low": low,
                            f"{metric}_ci_high": high})
            row["bootstrap_replicates_valid"] = n_valid
            rows.append(row)
    return pd.DataFrame(rows)


def _pooled(results, values_fn) -> tuple[float, float, float, list]:
    per_seed = [values_fn(r)[0] for r in results]
    stacked = np.vstack([values_fn(r)[1] for r in results])
    low, high, _ = percentile_ci(stacked.mean(axis=0))
    return float(np.mean(per_seed)), low, high, per_seed


def paired_deltas_table(results, protocol_commit) -> pd.DataFrame:
    models = _models(results)
    rows = []
    for a, b in COMPARISONS:
        if a not in models or b not in models:
            continue
        for metric in METRICS:
            def values(result, metric=metric):
                point = result["point"][a][metric] - result["point"][b][metric]
                return point, _replicates(result, a, metric) - _replicates(result, b, metric)
            for result in results:
                point, reps = values(result)
                low, high, n_valid = percentile_ci(reps)
                rows.append({"protocol_commit": protocol_commit, "comparison": f"{a} - {b}", "seed": str(result["seed"]),
                             "metric": metric, "delta": point, "ci_low": low, "ci_high": high,
                             "replicates_valid": n_valid})
            pooled, low, high, _ = _pooled(results, values)
            rows.append({"protocol_commit": protocol_commit, "comparison": f"{a} - {b}", "seed": "pooled",
                         "metric": metric, "delta": pooled, "ci_low": low, "ci_high": high,
                         "replicates_valid": int(np.isfinite(np.vstack([values(r)[1] for r in results]).mean(axis=0)).sum())})
    return pd.DataFrame(rows)


def pooled_summary_table(results, protocol_commit) -> pd.DataFrame:
    rows = []
    for model in _models(results):
        for metric in METRICS:
            pooled, low, high, per_seed = _pooled(
                results, lambda r, m=model, k=metric: (r["point"][m][k], _replicates(r, m, k)))
            rows.append({"protocol_commit": protocol_commit, "model": model, "features": FEATURE_LABELS[model],
                         "metric": metric, "pooled_mean": pooled, "pooled_ci_low": low, "pooled_ci_high": high,
                         "seed_min": float(np.min(per_seed)), "seed_max": float(np.max(per_seed)),
                         "n_seeds": len(per_seed)})
    return pd.DataFrame(rows)


def statements(deltas: pd.DataFrame, pooled: pd.DataFrame, config: dict, v1_status: str) -> dict:
    cfg = config["statements"]
    out = {}
    sup = deltas[(deltas.comparison == "V2 - V1_spec") & (deltas.metric == "roc_auc") & (deltas.seed != "pooled")]
    if v1_status == "reconciled" and len(sup):
        n_pos = int((sup["ci_low"] > cfg["superiority"]["lower_bound_gt"]).sum())
        holds = n_pos >= int(cfg["superiority"]["min_seeds"])
        evidence = (f"lower 95% bound of delta ROC-AUC (V2 - V1_spec) > 0 on {n_pos} of {len(sup)} seeds "
                    f"(required >= {cfg['superiority']['min_seeds']}); per-seed lower bounds: "
                    + ", ".join(f"seed {r.seed}: {r.ci_low:+.4f}" for r in sup.itertuples()))
    else:
        holds, evidence = False, f"V1_spec {v1_status}; no paired comparison possible"
    out["superiority"] = {"holds": holds, "statement": STATEMENTS["superiority"][0 if holds else 1], "evidence": evidence}

    v2 = pooled[pooled.model == "V2"].set_index("metric")
    slope, intercept = float(v2.loc["calibration_slope", "pooled_mean"]), float(v2.loc["calibration_intercept", "pooled_mean"])
    lo, hi = cfg["calibration"]["slope_range"]
    holds = lo <= slope <= hi and abs(intercept) <= float(cfg["calibration"]["max_abs_intercept"])
    out["calibration"] = {"holds": holds, "statement": STATEMENTS["calibration"][0 if holds else 1],
                          "evidence": f"pooled slope {slope:.3f} (required {lo}-{hi}); pooled intercept "
                                      f"{intercept:+.3f} (required |.| <= {cfg['calibration']['max_abs_intercept']})"}

    eq = deltas[(deltas.comparison == "V2 - V2_LR") & (deltas.metric == "roc_auc")]
    pooled_eq = eq[eq.seed == "pooled"].iloc[0]
    margin = float(cfg["equivalence"]["margin"])
    holds = -margin <= pooled_eq.ci_low and pooled_eq.ci_high <= margin
    per_seed = ", ".join(f"seed {r.seed}: [{r.ci_low:+.4f}, {r.ci_high:+.4f}]" for r in eq[eq.seed != "pooled"].itertuples())
    out["equivalence"] = {"holds": holds, "statement": STATEMENTS["equivalence"][0 if holds else 1],
                          "evidence": f"pooled delta ROC-AUC (V2 - V2_LR) {pooled_eq.delta:+.4f}, 95% CI "
                                      f"[{pooled_eq.ci_low:+.4f}, {pooled_eq.ci_high:+.4f}] (margin ±{margin}); "
                                      f"per seed: {per_seed}"}
    return out


def paper_table(pooled: pd.DataFrame, recorded_v1: dict) -> str:
    def cell(model, metric, digits=4):
        row = pooled[(pooled.model == model) & (pooled.metric == metric)].iloc[0]
        return f"{row.pooled_mean:.{digits}f} ({row.pooled_ci_low:.{digits}f}-{row.pooled_ci_high:.{digits}f})"

    lines = ["| Model | Features | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier (95% CI) | Cal. slope | Cal. intercept |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for model in [m for m in MODEL_ORDER if m in set(pooled.model)]:
        slope = pooled[(pooled.model == model) & (pooled.metric == "calibration_slope")].iloc[0].pooled_mean
        intercept = pooled[(pooled.model == model) & (pooled.metric == "calibration_intercept")].iloc[0].pooled_mean
        lines.append(f"| {model} | {FEATURE_LABELS[model]} | {cell(model, 'roc_auc')} | {cell(model, 'average_precision')} | "
                     f"{cell(model, 'brier_score')} | {slope:.3f} | {intercept:+.3f} |")
    lines.append(f"| V1 (recorded) | {recorded_v1['label']} | {recorded_v1['roc_auc']:.4f} (not available) | "
                 f"{recorded_v1['average_precision']:.4f} (not available) | {recorded_v1['brier_score']:.4f} (not available) | "
                 "not reported | not reported |")
    return "\n".join(lines)


def _style(axis):
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRIDLINE, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=MUTED, labelsize=8)


def fig_roc_pr(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    grid = np.asarray(results[0]["curves"][_models(results)[0]]["grid"])
    for model in _models(results):
        tpr = np.array([r["curves"][model]["tpr_at_fpr"] for r in results])
        prec = np.array([r["curves"][model]["precision_at_recall"] for r in results])
        for row_t, row_p in zip(tpr, prec):
            axes[0].plot(grid, row_t, color=MODEL_COLORS[model], linewidth=0.6, alpha=0.35)
            axes[1].plot(grid, row_p, color=MODEL_COLORS[model], linewidth=0.6, alpha=0.35)
        axes[0].plot(grid, tpr.mean(axis=0), color=MODEL_COLORS[model], linewidth=2, label=model)
        axes[1].plot(grid, prec.mean(axis=0), color=MODEL_COLORS[model], linewidth=2, label=model)
    axes[0].plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--")
    for axis, (xl, yl, title) in zip(axes, (("false positive rate", "true positive rate", "ROC (test)"),
                                            ("recall", "precision", "Precision-recall (test)"))):
        _style(axis)
        axis.set_xlabel(xl, color=MUTED, fontsize=9)
        axis.set_ylabel(yl, color=MUTED, fontsize=9)
        axis.set_title(title, color=INK, fontsize=10, loc="left")
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle("A9: test curves, 5 PSU-grouped seeds (thin) and their mean (thick)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_calibration(bins: pd.DataFrame, path):
    models = [m for m in MODEL_ORDER if m in set(bins.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.4), facecolor=SURFACE, sharey=True)
    for axis, model in zip(np.atleast_1d(axes), models):
        _style(axis)
        block = bins[(bins.model == model) & (~bins.suppressed)]
        for _, seed_block in block.groupby("seed"):
            axis.plot(seed_block.mean_predicted.astype(float), seed_block.observed_rate.astype(float), marker="o",
                      markersize=3, linewidth=1, color=MODEL_COLORS[model], alpha=0.7)
        axis.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--")
        axis.set_title(model, color=INK, fontsize=10, loc="left")
        axis.set_xlabel("mean predicted", color=MUTED, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("observed rate", color=MUTED, fontsize=9)
    fig.suptitle("A9: calibration on test, 10 quantile bins per seed (Platt-calibrated)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_forest(deltas: pd.DataFrame, path):
    comparisons = [c for c in (f"{a} - {b}" for a, b in COMPARISONS) if c in set(deltas.comparison)]
    metrics = (("roc_auc", "Δ ROC-AUC"), ("average_precision", "Δ PR-AUC"))
    fig, axes = plt.subplots(len(comparisons), 2, figsize=(10, 2.3 * len(comparisons) + 0.6), facecolor=SURFACE,
                             squeeze=False)
    for i, comparison in enumerate(comparisons):
        for j, (metric, label) in enumerate(metrics):
            axis = axes[i, j]
            _style(axis)
            block = deltas[(deltas.comparison == comparison) & (deltas.metric == metric)].reset_index(drop=True)
            ypos = np.arange(len(block))[::-1]
            for y, row in zip(ypos, block.itertuples()):
                pooled = row.seed == "pooled"
                color = INK if pooled else MODEL_COLORS["V2"]
                axis.plot([row.ci_low, row.ci_high], [y, y], color=color, linewidth=2.2 if pooled else 1.4)
                axis.plot(row.delta, y, marker="D" if pooled else "o", color=color, markersize=6 if pooled else 4.5)
            axis.axvline(0, color=MUTED, linewidth=0.8)
            if comparison == "V2 - V2_LR" and metric == "roc_auc":
                for edge in (-0.01, 0.01):
                    axis.axvline(edge, color=MUTED, linewidth=0.8, linestyle="--")
            axis.set_yticks(ypos)
            axis.set_yticklabels([f"seed {s}" if s != "pooled" else "pooled" for s in block.seed], fontsize=8)
            axis.set_title(f"{comparison}: {label}", color=INK, fontsize=9, loc="left")
            axis.grid(axis="y", visible=False)
    fig.suptitle("A9: paired test deltas with 95% PSU cluster-bootstrap CIs", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
