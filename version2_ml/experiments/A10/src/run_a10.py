"""A10 CLI runner: V2 subgroup performance and calibration on the A9 test partitions.

1. synthetic A10 tests must pass;
2. A9 guards for all seeds must be completed under the recorded A9 protocol;
3. per seed: verified deterministic re-derivation of V2 test predictions
   (split hash, membership digests, thresholds and all 11 stored test metrics
   must match A9 exactly, otherwise stop with nothing written); the
   re-derivation is logged in the A9 guard;
4. pooled subgroup metrics, PSU bootstrap CIs (B = 1000, seed 2028), gaps,
   predeclared flags; aggregate outputs and MANIFEST.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_HERE = Path(__file__).resolve()
_A10 = _HERE.parents[1]
_BASE = _HERE.parents[3]
_REPO = _BASE.parent
_A9 = _BASE / "experiments" / "A9"
for _entry in (
    _BASE,
    _BASE / "experiments" / "common",
    _BASE / "experiments" / "A1" / "src",
    _BASE / "experiments" / "A3",
    _BASE / "experiments" / "A4" / "src",
    _BASE / "experiments" / "A7" / "src",
    _BASE / "experiments" / "A8" / "src",
    _A9 / "src",
    _HERE.parent,
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import run_a9  # noqa: E402  (A9 data loading and parameter checks, reused unchanged)
from a10_analysis import PooledTest, analyse, flag_rows, metrics_rows, sector_bins  # noqa: E402
from a10_rederive import RederivationMismatch, rederive_seed  # noqa: E402
from a10_subgroups import DIMENSION_LABELS, DIMENSIONS, memberships  # noqa: E402
from a8_outputs import write_csv, write_manifest, write_text  # noqa: E402
from a9_guard import GuardBlocked, guard_path, read_guard  # noqa: E402
from a9_models import model_frames  # noqa: E402
from data_access import software_versions  # noqa: E402

SURFACE, INK, MUTED, GRIDLINE, BLUE, RED = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6", "#d03b3b"


class A10Error(RuntimeError):
    pass


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="A10 subgroup checks")
    parser.add_argument("--nfhs5-path", default=os.environ.get("NFHS5_PATH"))
    parser.add_argument("--config", default=str(_A10 / "config.yaml"))
    parser.add_argument("--out-dir", default=str(_A10 / "outputs"))
    args = parser.parse_args(argv)
    if not args.nfhs5_path:
        parser.error("NFHS5_PATH is not set and --nfhs5-path was not given")
    return args


def _run_synthetic_tests() -> str:
    suite = unittest.defaultTestLoader.discover(str(_A10 / "tests"), pattern="test_*.py", top_level_dir=str(_A10 / "tests"))
    result = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0:
        raise A10Error(f"Synthetic A10 tests failed ({len(result.failures)} failures, {len(result.errors)} errors)")
    return f"{result.testsRun} synthetic tests OK"


def log_rederivation(guard_dir: Path, seed: int, entry: dict) -> None:
    """Append a verified-re-derivation record to the A9 guard (status and results untouched)."""
    path = guard_path(guard_dir, seed)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("rederivations", []).append(entry)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8", newline="\n")


def _fmt(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else str(value)


def fig_forest(metrics: pd.DataFrame, pooled_auc: float, path: Path) -> None:
    rows = metrics[metrics.supported].copy()
    labels, ypos, y, dims = [], [], 0, []
    for dimension in DIMENSIONS:
        block = rows[rows.dimension == dimension]
        if block.empty:
            continue
        dims.append((y, DIMENSION_LABELS[dimension]))
        y += 1
        for row in block.itertuples():
            labels.append((y, row))
            y += 1
        y += 0.6
    fig, axis = plt.subplots(figsize=(8.5, max(6, 0.24 * y)), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    for ypos_, row in labels:
        low_flag = row.roc_auc < 0.70
        color = RED if low_flag else BLUE
        axis.plot([row.roc_auc_ci_low, row.roc_auc_ci_high], [ypos_, ypos_], color=color, linewidth=1.4)
        axis.plot(row.roc_auc, ypos_, "o", color=color, markersize=4)
    axis.axvline(pooled_auc, color=INK, linewidth=0.9, label=f"all test rows {pooled_auc:.3f}")
    axis.axvline(0.70, color=RED, linewidth=0.9, linestyle="--", label="0.70 flag line")
    axis.set_yticks([p for p, _ in labels] + [p for p, _ in dims])
    axis.set_yticklabels([str(r.group) for _, r in labels] + [name for _, name in dims], fontsize=7)
    for tick, (pos, _) in zip(axis.get_yticklabels()[len(labels):], dims):
        tick.set_fontweight("bold")
        tick.set_color(INK)
    axis.invert_yaxis()
    axis.grid(axis="x", color=GRIDLINE, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=MUTED, labelsize=7)
    axis.set_xlabel("ROC-AUC (pooled test rows, 95% PSU bootstrap CI)", color=MUTED, fontsize=8)
    axis.legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("A10: V2 ROC-AUC by subgroup (supported groups only)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_calibration_by_sector(bins: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    sectors = [s for s in ("public", "private", "other") if s in set(bins.facility_sector)]
    fig, axes = plt.subplots(1, len(sectors), figsize=(3.4 * len(sectors), 3.5), facecolor=SURFACE, sharey=True)
    for axis, sector in zip(np.atleast_1d(axes), sectors):
        axis.set_facecolor(SURFACE)
        block = bins[(bins.facility_sector == sector) & (~bins.suppressed)]
        axis.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--")
        if not block.empty:
            axis.plot(block.mean_predicted.astype(float), block.observed_rate.astype(float), "o-", color=BLUE,
                      markersize=4, linewidth=1.2)
        row = metrics[(metrics.dimension == "facility_sector") & (metrics.group == sector)].iloc[0]
        note = (f"slope {_fmt(row.calibration_slope)}, intercept {_fmt(row.calibration_intercept)}"
                if row.supported else "not estimable")
        axis.set_title(f"{sector}\n{note}", color=INK, fontsize=9, loc="left")
        axis.set_xlabel("mean predicted", color=MUTED, fontsize=8)
        axis.grid(color=GRIDLINE, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(colors=MUTED, labelsize=7)
        if block.empty:
            axis.text(0.5, 0.5, "all bins suppressed (< 10)", ha="center", color=MUTED, fontsize=8, transform=axis.transAxes)
    np.atleast_1d(axes)[0].set_ylabel("observed rate", color=MUTED, fontsize=8)
    fig.suptitle("A10: V2 calibration by facility sector (pooled test, 10 quantile bins)", x=0.01, ha="left", color=INK,
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _methods(state) -> str:
    m, flags, gaps = state["metrics"], state["flags"], state["gaps"]
    lines = ["# A10 methods and results: V2 subgroup checks", "",
             f"- source: A9 V2 (protocol `{state['config']['source']['a9_protocol_commit'][:7]}`, A9 commit "
             f"`{state['config']['source']['a9_commit']}`), seeds {state['config']['source']['seeds']}",
             "- predictions: verified deterministic re-derivation. Per seed, the split hash, membership digests, V2 "
             "thresholds and all 11 V2 test metrics stored in the A9 guard were reproduced exactly "
             f"({state['verified']} values); A9 results are unchanged.",
             f"- repo_commit at run: `{state['git']['commit']}`; generated_at_utc: {state['generated']}",
             f"- pooled test rows: {state['n_rows']:,} birth-test observations "
             f"({state['n_unique']:,} unique births; a birth can be in more than one seed's test partition)",
             f"- all pooled test rows: ROC-AUC {state['overall']['roc_auc']:.4f}, AP {state['overall']['average_precision']:.4f}, "
             f"slope {state['overall']['calibration_slope']:.3f}, intercept {state['overall']['calibration_intercept']:+.3f}",
             "- PSU bootstrap within each seed's test partition, B = 1,000, seed 2028; percentile 95% CIs.", "",
             "Within-subgroup ROC-AUC is usually lower than the all-rows ROC-AUC because between-group differences "
             "in risk are removed. **A flag is a reporting requirement, not a model failure.**", "",
             "## Predeclared flags", "",
             "- LOW_DISCRIMINATION: ROC-AUC < 0.70",
             "- MISCALIBRATED: |calibration intercept| > 0.10 or slope outside 0.8-1.2",
             "- UNDER_SUPPORTED: < 200 births or < 30 per class in the pooled test rows (not estimable)", "",
             "## Flag table", ""]
    if flags.empty:
        lines.append("No subgroup is flagged.")
    else:
        lines += ["| dimension | group | flag | n | events | evidence |", "| --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {r.dimension} | {r.group} | {r.flag} | {r.n} | {r.events} | {r.evidence} |" for r in flags.itertuples()]
    lines += ["", "## ROC-AUC gaps within dimensions (largest minus smallest supported group)", "",
              "| dimension | groups | max (group) | min (group) | gap (95% CI) |", "| --- | --- | --- | --- | --- |"]
    for g in gaps.itertuples():
        if g.n_supported_groups < 2:
            lines.append(f"| {g.dimension} | {g.n_supported_groups} | not estimable | not estimable | not estimable |")
        else:
            lines.append(f"| {g.dimension} | {g.n_supported_groups} | {g.max_roc_auc:.4f} ({g.max_group}) | "
                         f"{g.min_roc_auc:.4f} ({g.min_group}) | {g.gap:.4f} ({g.ci_low:.4f}-{g.ci_high:.4f}) |")
    lines += ["", "## Subgroup metrics (pooled; 95% CI; per-seed ROC-AUC range)", "",
              "| dimension | group | n | events | ROC-AUC (95% CI) | seeds | AP | Brier | intercept | slope | obs / pred | recall | precision | flags |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in m.itertuples():
        if not r.supported:
            lines.append(f"| {r.dimension} | {r.group} | {r.n} | {r.events} | not estimable | | | | | | | | | {r.flags} |")
            continue
        lines.append(f"| {r.dimension} | {r.group} | {r.n} | {r.events} | {r.roc_auc:.3f} ({r.roc_auc_ci_low:.3f}-{r.roc_auc_ci_high:.3f}) | "
                     f"{_fmt(r.roc_auc_seed_min)}-{_fmt(r.roc_auc_seed_max)} | {r.average_precision:.3f} | {r.brier_score:.3f} | "
                     f"{r.calibration_intercept:+.3f} | {r.calibration_slope:.3f} | {r.observed_rate:.3f} / {r.mean_predicted:.3f} | "
                     f"{r.recall_at_threshold:.3f} | {r.precision_at_threshold:.3f} | {r.flags} |")
    lines += ["", "## Re-derivation log", ""]
    lines += [f"- seed {v['seed']}: {v['verified_metrics']} metrics, thresholds, split hash and 3 membership digests matched A9"
              for v in state["verifications"]]
    lines += ["", "## Disclosure", "",
              "- Aggregate outputs only; counts below 10 suppressed (births, events, non-events); no keys or per-birth predictions written.",
              "- Unweighted; CIs ignore survey strata and are not design-based."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    a9_config = yaml.safe_load((_A9 / "config.yaml").read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    guard_dir = _A9 / "outputs" / "private"
    min_cell = int(config["disclosure"]["min_cell"])
    qa = []
    try:
        qa.append({"check": "synthetic_a10_tests", "status": "PASS", "detail": _run_synthetic_tests()})
        protocol = config["source"]["a9_protocol_commit"]
        guards = {}
        for seed in config["source"]["seeds"]:
            guard = read_guard(guard_dir, seed)
            if guard is None or guard["results"]["protocol_commit"] != protocol:
                raise A10Error(f"seed {seed}: no completed A9 guard under protocol {protocol[:7]}")
            guards[int(seed)] = guard
        qa.append({"check": "a9_guards_completed_under_protocol", "status": "PASS", "detail": protocol[:7]})
        params = run_a9._params(a9_config)
        print("loading NFHS-5 cohort ...", flush=True)
        data = run_a9._load(a9_config, Path(args.nfhs5_path), qa)
        frames_v2 = {"V2": model_frames(data["base"], data["states"], None)["V2"]}
        parts, verifications = [], []
        for seed in config["source"]["seeds"]:
            seed = int(seed)
            print(f"seed {seed}: re-deriving V2 test predictions and verifying against A9 ...", flush=True)
            rederived = rederive_seed(seed, data["cohort"], data["keys"], frames_v2, a9_config, params,
                                      guards[seed]["results"])
            verifications.append({"seed": seed, "verified_metrics": rederived["verified_metrics"]})
            log_rederivation(guard_dir, seed, {"experiment": "A10", "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                               "purpose": "subgroup checks", "verified_metrics_exact": rederived["verified_metrics"],
                                               "thresholds_exact": True, "split_and_membership_exact": True})
            parts.append(rederived)
        qa.append({"check": "rederivation_matches_a9_exactly", "status": "PASS",
                   "detail": f"{sum(v['verified_metrics'] for v in verifications)} metrics + thresholds + split/membership, 5 seeds"})
    except (A10Error, GuardBlocked, RederivationMismatch, run_a9.ProtocolError) as exc:
        print(f"\nSTOPPED: {exc}\nNo outputs written.")
        return 1

    cohort_index = np.concatenate([p["test_positions"] for p in parts])
    pooled = PooledTest(
        seeds=np.concatenate([np.full(len(p["test_positions"]), p["seed"]) for p in parts]),
        y=data["cohort"]["m17"].to_numpy(dtype=int)[cohort_index],
        p=np.concatenate([p["p_test"] for p in parts]),
        label=np.concatenate([p["p_test"] >= p["threshold"] for p in parts]),
        psu=np.concatenate([p["psu_test"] for p in parts]),
        cohort_index=cohort_index, keys=data["keys"][cohort_index],
    )
    members = memberships(data["base"], data["states"], config)
    print(f"bootstrapping {sum(len(b) for b in members.values())} subgroups (B = {config['bootstrap']['replicates']}) ...", flush=True)
    result = analyse(pooled, members, config)
    from a10_analysis import group_metrics
    overall = group_metrics(pooled.y, pooled.p, pooled.label)
    metrics = pd.DataFrame(metrics_rows(result, config, min_cell))
    flags = pd.DataFrame(flag_rows(result, config, min_cell),
                         columns=["dimension", "group", "flag", "n", "events", "non_events", "roc_auc",
                                  "calibration_intercept", "calibration_slope", "evidence"])
    gaps = pd.DataFrame(result["gaps"])
    bins = sector_bins(pooled, members, config, min_cell)

    from exp_provenance import git_info

    git = git_info(_REPO)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*"):
        if stale.is_file():
            stale.unlink()
    stamp = {"a9_protocol_commit": config["source"]["a9_protocol_commit"]}
    write_csv(metrics.assign(**stamp), out_dir / "a10_subgroup_metrics.csv")
    write_csv(flags.assign(**stamp), out_dir / "a10_flags.csv")
    write_csv(gaps.assign(**stamp), out_dir / "a10_gaps.csv")
    write_csv(bins.assign(**stamp), out_dir / "a10_calibration_bins_by_sector.csv")
    fig_forest(metrics, overall["roc_auc"], out_dir / "fig_a10_subgroup_forest.png")
    fig_calibration_by_sector(bins, metrics, out_dir / "fig_a10_calibration_by_sector.png")
    write_text(_methods({"config": config, "metrics": metrics, "flags": flags, "gaps": gaps, "git": git,
                         "verified": sum(v["verified_metrics"] for v in verifications), "verifications": verifications,
                         "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "n_rows": int(len(pooled.y)), "n_unique": int(len(np.unique(pooled.keys))),
                         "overall": overall}), out_dir / "a10_methods_results.md")
    write_manifest(out_dir, experiment_id="A10", config=config, config_path=config_path, repo_root=_REPO,
                   seeds=config["source"]["seeds"],
                   split_manifest_hashes={s: g["results"]["audit"]["split_manifest_hash"] for s, g in guards.items()},
                   extra={"a9_protocol_commit": config["source"]["a9_protocol_commit"], "bootstrap": config["bootstrap"],
                          "software_versions_runtime": software_versions(), "qa": qa,
                          "prediction_source": "verified deterministic re-derivation of A9 V2 (see qa)"})

    print("\nFlag table:")
    print(flags.drop(columns=["roc_auc", "calibration_intercept", "calibration_slope"]).to_string(index=False)
          if not flags.empty else "  no subgroup flagged")
    print("\nROC-AUC gaps:")
    print(gaps[["dimension", "n_supported_groups", "max_group", "min_group", "gap", "ci_low", "ci_high"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\noutputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
