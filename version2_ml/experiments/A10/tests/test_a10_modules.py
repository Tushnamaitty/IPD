"""A10 synthetic tests. No NFHS records are read or required."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_BASE = Path(__file__).resolve().parents[3]
for _entry in (
    _BASE,
    _BASE / "experiments" / "common",
    _BASE / "experiments" / "A1" / "src",
    _BASE / "experiments" / "A3",
    _BASE / "experiments" / "A4" / "src",
    _BASE / "experiments" / "A7" / "src",
    _BASE / "experiments" / "A8" / "src",
    _BASE / "experiments" / "A9" / "src",
    _BASE / "experiments" / "A9" / "tests",
    _BASE / "experiments" / "A10" / "src",
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import run_a9  # noqa: E402
from a10_analysis import PooledTest, analyse, flag_rows, flags_for, group_metrics, metrics_rows, sector_bins, supported  # noqa: E402
from a10_rederive import RederivationMismatch, rederive_seed  # noqa: E402
from a10_subgroups import memberships  # noqa: E402
from a9_models import fit_seed, model_frames, predict_test  # noqa: E402
from exp_provenance import assert_disclosure_safe  # noqa: E402
from test_a9_modules import CONFIG as A9_CONFIG, FAST, positions, synthetic  # noqa: E402

CONFIG = yaml.safe_load((_BASE / "experiments" / "A10" / "config.yaml").read_text(encoding="utf-8"))


def small_config(replicates=20, min_births=40, min_per_class=8):
    config = copy.deepcopy(CONFIG)
    config["bootstrap"]["replicates"] = replicates
    config["support_screen"] = {"min_births": min_births, "min_per_class": min_per_class}
    return config


class SubgroupTests(unittest.TestCase):
    def test_age_cut_points_and_partitions(self):
        _, base, _, states, _ = synthetic(40)
        base = base.copy()
        base.loc[base.index[:4], "maternal_age_at_index_birth"] = [19.999, 20.0, 34.999, 35.0]
        members = memberships(base, states, CONFIG)["maternal_age"]
        self.assertEqual([k for k in ("<20", "20-34", "35+") if members[k][0]], ["<20"])
        self.assertTrue(members["20-34"][1] and members["20-34"][2] and members["35+"][3])

    def test_birth_order_and_sector_groups(self):
        _, base, _, states, _ = synthetic(40)
        members = memberships(base, states, CONFIG)
        order = base["birth_order"].to_numpy()
        self.assertTrue(np.array_equal(members["birth_order"]["first_birth"], order == 1))
        self.assertTrue(np.array_equal(members["facility_sector"]["private"], base.facility_type.eq("private").to_numpy()))
        self.assertEqual(len(members["previous_cs_history"]), 5)


class MetricAndFlagTests(unittest.TestCase):
    def test_threshold_precision_recall(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        p = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3])
        m = group_metrics(y, p, p >= 0.5)
        self.assertAlmostEqual(m["recall_at_threshold"], 2 / 3)
        self.assertAlmostEqual(m["precision_at_threshold"], 2 / 3)
        self.assertAlmostEqual(m["observed_rate"], 0.5)
        self.assertTrue(np.isnan(group_metrics(np.zeros(5, int), np.full(5, 0.2), np.zeros(5, bool))["roc_auc"]))

    def test_support_screen_edges(self):
        self.assertTrue(supported(np.r_[np.ones(30), np.zeros(170)], min_births=200, min_per_class=30))
        self.assertFalse(supported(np.r_[np.ones(29), np.zeros(171)], min_births=200, min_per_class=30))
        self.assertFalse(supported(np.r_[np.ones(30), np.zeros(169)], min_births=200, min_per_class=30))

    def test_flag_edges(self):
        def entry(auc, intercept, slope):
            return {"supported": True, "point": {"roc_auc": auc, "calibration_intercept": intercept,
                                                 "calibration_slope": slope}}
        self.assertEqual(flags_for(entry(0.70, 0.10, 0.8), CONFIG), [])
        self.assertEqual(flags_for(entry(0.70, -0.10, 1.2), CONFIG), [])
        self.assertEqual(flags_for(entry(0.6999, 0.0, 1.0), CONFIG), ["LOW_DISCRIMINATION"])
        self.assertEqual(flags_for(entry(0.8, 0.1001, 1.0), CONFIG), ["MISCALIBRATED"])
        self.assertEqual(flags_for(entry(0.8, 0.0, 1.2001), CONFIG), ["MISCALIBRATED"])
        self.assertEqual(flags_for({"supported": False}, CONFIG), ["UNDER_SUPPORTED"])


class PooledAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cohort, base, _, states, keys = synthetic(500)
        rng = np.random.default_rng(3)
        parts = []
        for seed in (11, 22):
            pos = positions(cohort, seed=seed)["test"]
            parts.append((seed, pos, np.clip(0.2 + 0.5 * cohort.m17.to_numpy()[pos] + rng.normal(0, 0.2, len(pos)), 0.01, 0.99)))
        index = np.concatenate([p[1] for p in parts])
        cls.pooled = PooledTest(seeds=np.concatenate([np.full(len(p[1]), p[0]) for p in parts]),
                                y=cohort.m17.to_numpy()[index], p=np.concatenate([p[2] for p in parts]),
                                label=np.concatenate([p[2] for p in parts]) >= 0.4,
                                psu=cohort.v021.astype(int).astype(str).to_numpy()[index], cohort_index=index,
                                keys=keys[index])
        cls.members = memberships(base, states, CONFIG)
        cls.config = small_config()
        cls.result = analyse(cls.pooled, cls.members, cls.config)

    def test_deterministic_bootstrap(self):
        again = analyse(self.pooled, self.members, self.config)
        a = [g["replicates"]["roc_auc"] for g in self.result["groups"] if g["supported"]]
        b = [g["replicates"]["roc_auc"] for g in again["groups"] if g["supported"]]
        self.assertTrue(all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b)))

    def test_gap_is_max_minus_min_of_supported_groups(self):
        for gap in self.result["gaps"]:
            block = [g for g in self.result["groups"] if g["dimension"] == gap["dimension"] and g["supported"]]
            if len(block) >= 2:
                aucs = [g["point"]["roc_auc"] for g in block]
                self.assertAlmostEqual(gap["gap"], max(aucs) - min(aucs))
                self.assertLessEqual(gap["ci_low"], gap["ci_high"])

    def test_unsupported_groups_have_no_metrics_and_small_counts_suppressed(self):
        rows = pd.DataFrame(metrics_rows(self.result, self.config, 10))
        unknown = rows[rows.group == "unknown_history"].iloc[0]
        self.assertFalse(unknown["supported"])
        self.assertEqual(unknown["roc_auc"], "not estimable")
        self.assertEqual(unknown["flags"], "UNDER_SUPPORTED")
        for column in ("n", "n_unique_births", "events", "non_events"):
            numeric = pd.to_numeric(rows[column], errors="coerce").dropna()
            self.assertTrue(((numeric == 0) | (numeric >= 10)).all(), column)
        assert_disclosure_safe(rows)

    def test_flag_rows_carry_evidence(self):
        flags = pd.DataFrame(flag_rows(self.result, self.config, 10))
        self.assertTrue(flags["evidence"].str.len().gt(0).all())
        self.assertIn("UNDER_SUPPORTED", set(flags.flag))

    def test_sector_bins_suppressed(self):
        bins = sector_bins(self.pooled, self.members, self.config, 10)
        small = bins[bins.suppressed]
        self.assertTrue((small["n_births"] == "SUPPRESSED").all())


class RederivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort, base, f0, states, cls.keys = synthetic()
        cls.frames = model_frames(base, states, f0)
        cls.frames_v2 = {"V2": cls.frames["V2"]}
        cls.tmp = tempfile.TemporaryDirectory()
        config = copy.deepcopy(A9_CONFIG)
        config["bootstrap"]["replicates"] = 5
        cls.a9_config = config
        cls.guard = run_a9._score_seed(11, {"cohort": cls.cohort, "keys": cls.keys}, cls.frames, config, FAST,
                                       Path(cls.tmp.name), "p" * 40)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rederive(self, guard):
        return rederive_seed(11, self.cohort, self.keys, self.frames_v2, self.a9_config, FAST, guard)

    def test_rederivation_reproduces_a9_exactly(self):
        out = self.rederive(self.guard)
        self.assertEqual(out["verified_metrics"], 11)
        self.assertEqual(len(out["p_test"]), self.guard["test_counts"]["n_births"])

    def test_tampered_metric_stops(self):
        guard = copy.deepcopy(self.guard)
        guard["point"]["V2"]["roc_auc"] += 1e-15
        with self.assertRaises(RederivationMismatch):
            self.rederive(guard)

    def test_tampered_split_or_threshold_stops(self):
        for mutate in (lambda g: g["audit"].__setitem__("split_manifest_hash", "0" * 64),
                       lambda g: g["thresholds"]["V2"].__setitem__("f1_optimal", 0.99)):
            guard = copy.deepcopy(self.guard)
            mutate(guard)
            with self.assertRaises(RederivationMismatch):
                self.rederive(guard)

    def test_test_labels_do_not_reach_predictions(self):
        pos = positions(self.cohort, seed=11)
        y = self.cohort.m17.to_numpy()
        mutated = y.copy()
        mutated[pos["test"]] = 1 - mutated[pos["test"]]
        preds = []
        for labels in (y, mutated):
            models = fit_seed(self.frames_v2, self.a9_config["models"], FAST, fit_positions=pos["fit"],
                              calib_positions=pos["calib"], valid_positions=pos["valid"], y=labels, seed=11)
            preds.append(predict_test(models, self.frames_v2, pos["test"])["V2"])
        self.assertEqual(preds[0].tobytes(), preds[1].tobytes())


class EndToEndOutputTests(unittest.TestCase):
    def test_outputs_and_manifest(self):
        import run_a10
        from a8_outputs import write_manifest

        cohort, base, _, states, keys = synthetic(500)
        pos = positions(cohort, seed=11)["test"]
        p = np.clip(0.2 + 0.5 * cohort.m17.to_numpy()[pos], 0.01, 0.99)
        pooled = PooledTest(seeds=np.full(len(pos), 11), y=cohort.m17.to_numpy()[pos], p=p, label=p >= 0.4,
                            psu=cohort.v021.astype(int).astype(str).to_numpy()[pos], cohort_index=pos, keys=keys[pos])
        config = small_config(replicates=10)
        members = memberships(base, states, config)
        result = analyse(pooled, members, config)
        metrics = pd.DataFrame(metrics_rows(result, config, 10))
        flags = pd.DataFrame(flag_rows(result, config, 10))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            run_a10.write_csv(metrics, out / "a10_subgroup_metrics.csv")
            run_a10.write_csv(flags, out / "a10_flags.csv")
            run_a10.write_csv(pd.DataFrame(result["gaps"]), out / "a10_gaps.csv")
            run_a10.fig_forest(metrics, 0.85, out / "forest.png")
            run_a10.fig_calibration_by_sector(sector_bins(pooled, members, config, 10), metrics, out / "cal.png")
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("x: 1\n", encoding="utf-8")
            manifest = write_manifest(out, experiment_id="A10", config=config, config_path=config_path,
                                      repo_root=_BASE.parent, seeds=[11], split_manifest_hashes={})
            self.assertEqual(len(manifest["files"]), 5)


if __name__ == "__main__":
    unittest.main()
