"""A9 synthetic tests. No NFHS records are read or required."""
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
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from a1_state_context import map_states  # noqa: E402
from a7_candidates import carve_calibration_partition  # noqa: E402
from a9_guard import GuardBlocked, complete_guard, read_guard, start_guard  # noqa: E402
from a9_metrics import METRICS, calibration_bins, percentile_ci, psu_bootstrap, test_metrics  # noqa: E402
from a9_models import fit_seed, model_frames, predict_test  # noqa: E402
from a9_report import paired_deltas_table, paper_table, pooled_summary_table, statements  # noqa: E402
from a9_splits import assert_psu_nesting, overlap_counts, psu_stratified_partition  # noqa: E402
from a9_v1 import reconcile_f0  # noqa: E402
from feature_builder import F0_BIN, F0_CAT, F0_NAMES, F0_NUM  # noqa: E402

CONFIG = yaml.safe_load((_BASE / "experiments" / "A9" / "config.yaml").read_text(encoding="utf-8"))
FAST = {
    "a7_selected_xgboost": {"n_estimators": 15, "max_depth": 3, "learning_rate": 0.2, "subsample": 0.8,
                            "colsample_bytree": 0.8, "eval_metric": "logloss", "tree_method": "hist"},
    "a7_selected_logistic_regression": {"C": 1.0, "solver": "liblinear", "max_iter": 1000, "class_weight": "balanced"},
    "a7_reference_xgboost": {"n_estimators": 15, "max_depth": 4, "learning_rate": 0.2, "subsample": 0.8,
                             "colsample_bytree": 0.8, "eval_metric": "logloss", "tree_method": "hist"},
}
THRESHOLDS = {"f1_optimal": 0.3, "recall_80": 0.2}


def synthetic(n_psus=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    mother = 0
    for psu in range(1, n_psus + 1):
        state = int(rng.choice([1, 9, 10, 27, 33]))
        psu_effect = rng.normal(0, 0.8)
        for _ in range(rng.integers(2, 6)):
            mother += 1
            for bidx in range(1, rng.integers(1, 3) + 1):
                rows.append({"caseid": f"c{mother:05d}", "bidx": bidx, "v021": float(1000 + psu), "v024": state,
                             "psu_effect": psu_effect})
    cohort = pd.DataFrame(rows)
    n = len(cohort)
    facility = rng.choice(["public", "private", "other"], n, p=(0.6, 0.35, 0.05))
    bord = rng.integers(1, 5, n)
    history = np.where(bord == 1, "first_recorded_live_birth",
                       rng.choice(["observed_previous_institutional_cs", "higher_order_no_observed_previous_institutional_cs"], n))
    logit = -1.3 + 1.2 * (facility == "private") + 1.8 * (history == "observed_previous_institutional_cs") + cohort.psu_effect
    cohort["m17"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cohort["m15"] = np.where(facility == "public", 21, np.where(facility == "private", 31, 96))
    cohort["v005"] = rng.integers(500_000, 2_000_000, n)
    base = pd.DataFrame({
        "maternal_age_at_index_birth": rng.uniform(16, 45, n), "education_years": rng.integers(0, 21, n).astype(float),
        "wealth_index": rng.integers(1, 6, n).astype(str), "birth_order": bord.astype(float),
        "facility_type": facility, "residence": rng.choice(["urban", "rural"], n), "religion": rng.choice(["1", "2"], n),
        "twin_order": "0",
        "facility_detail": np.where(facility == "public", "public: govt./munic. hospital",
                                    np.where(facility == "private", "private: hospital/maternity home/clinic", "other")),
        "history_four_level": history,
        "has_any_observed_earlier_institutional_birth": np.where(bord == 1, "false", "true"),
        "history_quality_flag": "ok",
    })
    f0 = pd.DataFrame({name: rng.normal(size=n) for name in F0_NUM})
    for name in F0_BIN:
        f0[name] = rng.integers(0, 2, n)
    for name in F0_CAT:
        f0[name] = rng.choice(["a", "b"], n)
    f0["facility_type"] = facility
    keys = (cohort["caseid"] + "|" + cohort["bidx"].astype(str)).to_numpy()
    return cohort, base, f0[list(F0_NAMES)], map_states(cohort["v024"]).to_numpy(), keys


def positions(cohort, seed=11):
    part = psu_stratified_partition(cohort, seed=seed)
    train = np.flatnonzero(part.eq("train").to_numpy())
    carve = carve_calibration_partition(cohort.iloc[train].reset_index(drop=True), group_col="v021",
                                        calibration_fraction=0.2, seed=4208)
    return {"fit": train[carve.eq("fit").to_numpy()], "calib": train[carve.eq("calibration").to_numpy()],
            "valid": np.flatnonzero(part.eq("validation").to_numpy()), "test": np.flatnonzero(part.eq("test").to_numpy())}


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.cohort = synthetic()[0]

    def test_mother_in_two_psus_rejected(self):
        extra_birth_elsewhere = self.cohort.iloc[[0]].assign(bidx=9, v021=self.cohort.v021.max() + 1)
        bad = pd.concat([self.cohort, extra_birth_elsewhere], ignore_index=True)
        with self.assertRaises(ValueError):
            assert_psu_nesting(bad)

    def test_zero_psu_and_mother_overlap(self):
        part = psu_stratified_partition(self.cohort, seed=22)
        for column in ("v021", "caseid"):
            self.assertEqual(sum(overlap_counts(self.cohort, part, column).values()), 0, column)

    def test_deterministic_and_seed_dependent(self):
        a = psu_stratified_partition(self.cohort, seed=33)
        self.assertTrue(a.equals(psu_stratified_partition(self.cohort, seed=33)))
        self.assertFalse(a.equals(psu_stratified_partition(self.cohort, seed=44)))

    def test_fractions_by_psu(self):
        part = psu_stratified_partition(self.cohort, seed=55)
        share = pd.Series(part.to_numpy(), index=self.cohort.v021).groupby(level=0).first().value_counts(normalize=True)
        self.assertAlmostEqual(share["train"], 0.70, delta=0.03)
        self.assertAlmostEqual(share["validation"], 0.15, delta=0.03)


class V1ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.cohort, _, self.f0, _, self.keys = synthetic(60)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "df_model_v1.parquet"

    def tearDown(self):
        self.tmp.cleanup()

    def parquet(self, frame=None):
        pq = self.f0.copy()
        pq["respondent_id"] = self.cohort["caseid"].to_numpy()
        pq["birth_index"] = self.cohort["bidx"].to_numpy()
        pq["csection"] = self.cohort["m17"].to_numpy()
        pq["delivery_place_code"] = self.cohort["m15"].to_numpy()
        (pq if frame is None else frame(pq)).sample(frac=1, random_state=0).to_parquet(self.path)

    def test_reconciled_and_aligned_to_cohort_order(self):
        self.parquet()
        info, aligned = reconcile_f0(self.path, self.cohort, self.keys)
        self.assertEqual(info["status"], "reconciled")
        self.assertTrue(np.allclose(aligned[F0_NUM[0]].to_numpy(), self.f0[F0_NUM[0]].to_numpy()))

    def test_missing_file_not_run(self):
        self.assertEqual(reconcile_f0(self.path, self.cohort, self.keys)[0]["status"], "not_run_parquet_missing")

    def test_outcome_mismatch_not_run(self):
        self.parquet(lambda pq: pq.assign(csection=1 - pq["csection"]))
        info, aligned = reconcile_f0(self.path, self.cohort, self.keys)
        self.assertEqual(info["status"], "not_run_reconciliation_failed")
        self.assertIsNone(aligned)

    def test_duplicate_key_not_run(self):
        self.parquet(lambda pq: pd.concat([pq, pq.iloc[[0]]]))
        self.assertEqual(reconcile_f0(self.path, self.cohort, self.keys)[0]["status"], "not_run_reconciliation_failed")


class GuardTests(unittest.TestCase):
    def test_score_once_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_guard(tmp, 11))
            start_guard(tmp, 11, "abc")
            with self.assertRaises(FileExistsError):
                start_guard(tmp, 11, "abc")
            with self.assertRaises(GuardBlocked):
                read_guard(tmp, 11)
            complete_guard(tmp, 11, {"x": 1})
            self.assertEqual(read_guard(tmp, 11)["results"], {"x": 1})
            with self.assertRaises(GuardBlocked):
                complete_guard(tmp, 11, {"x": 2})


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.psu = np.repeat(np.arange(150), 5)
        self.y = (rng.random(len(self.psu)) < 0.3).astype(int)
        self.p = np.clip(0.3 + 0.4 * (self.y - 0.3) + rng.normal(0, 0.15, len(self.y)), 0.01, 0.99)

    def test_paired_draws_give_zero_deltas_for_identical_models(self):
        reps = psu_bootstrap(self.y, {"A": self.p, "B": self.p.copy()}, self.psu, {"A": THRESHOLDS, "B": THRESHOLDS},
                             replicates=40, seed=2027)
        for metric in METRICS:
            self.assertTrue(np.array_equal(reps["A"][metric], reps["B"][metric], equal_nan=True), metric)

    def test_deterministic_with_seed_and_ci_brackets_point(self):
        a = psu_bootstrap(self.y, {"A": self.p}, self.psu, {"A": THRESHOLDS}, replicates=60, seed=2027)
        b = psu_bootstrap(self.y, {"A": self.p}, self.psu, {"A": THRESHOLDS}, replicates=60, seed=2027)
        self.assertTrue(np.array_equal(a["A"]["roc_auc"], b["A"]["roc_auc"]))
        low, high, n = percentile_ci(a["A"]["roc_auc"])
        point = test_metrics(self.y, self.p, THRESHOLDS)["roc_auc"]
        self.assertEqual(n, 60)
        self.assertTrue(low <= point <= high)

    def test_undefined_replicates_are_dropped(self):
        low, high, n = percentile_ci([0.7, np.nan, 0.8, 0.75])
        self.assertEqual(n, 3)
        self.assertTrue(np.isnan(test_metrics(np.zeros(10, int), np.full(10, 0.2), THRESHOLDS)["roc_auc"]))

    def test_calibration_bins_suppress_small_cells(self):
        bins = calibration_bins(self.y[:60], self.p[:60], n_bins=10, min_cell=10)
        self.assertTrue(bins["suppressed"].all())
        self.assertTrue((bins["n_births"] == "SUPPRESSED").all())


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort, cls.base, cls.f0, cls.states, _ = synthetic()
        cls.pos = positions(cls.cohort)
        cls.frames = model_frames(cls.base, cls.states, cls.f0)
        cls.reference = cls.fit_and_score(cls.cohort["m17"].to_numpy())

    @classmethod
    def fit_and_score(cls, y):
        models = fit_seed(cls.frames, CONFIG["models"], FAST, fit_positions=cls.pos["fit"],
                          calib_positions=cls.pos["calib"], valid_positions=cls.pos["valid"], y=y, seed=11)
        return models, predict_test(models, cls.frames, cls.pos["test"])

    def mutate(self, where):
        y = self.cohort["m17"].to_numpy().copy()
        y[where] = 1 - np.random.default_rng(5).permutation(y[where])
        return y

    def test_all_four_models_fitted(self):
        self.assertEqual(sorted(self.reference[0]), ["F1_core", "V1_spec", "V2", "V2_LR"])

    def test_test_labels_never_reach_models_thresholds_or_predictions(self):
        models, p_test = self.fit_and_score(self.mutate(self.pos["test"]))
        for name, model in self.reference[0].items():
            self.assertEqual(models[name]["thresholds"], model["thresholds"], name)
            self.assertEqual(models[name]["p_valid"].tobytes(), model["p_valid"].tobytes(), name)
            self.assertEqual(p_test[name].tobytes(), self.reference[1][name].tobytes(), name)

    def test_validation_labels_only_move_thresholds(self):
        models, p_test = self.fit_and_score(self.mutate(self.pos["valid"]))
        for name in self.reference[0]:
            self.assertEqual(p_test[name].tobytes(), self.reference[1][name].tobytes(), name)

    def test_v2_uses_final_tier_with_state(self):
        frame, manifest = self.frames["V2"]
        self.assertIn("state_ut", manifest.categorical)
        self.assertEqual(len(manifest.numerical) + len(manifest.categorical), 13)


def fake_results(v2_minus_v1=(0.03, 0.03, 0.03, 0.03, 0.03), lr_gap=0.002, slope=1.0, intercept=0.0, spread=0.01):
    rng = np.random.default_rng(0)
    out = []
    for seed, gap in zip((11, 22, 33, 44, 55), v2_minus_v1):
        point, reps = {}, {}
        shared = rng.normal(0, spread, 200)  # paired bootstrap: draws are shared, so noise is mostly common
        for model, shift in (("V2", 0.0), ("V2_LR", -lr_gap), ("F1_core", -0.05), ("V1_spec", -gap)):
            noise = shared + rng.normal(0, spread / 10, 200)
            values = {m: 0.5 for m in METRICS}
            values.update({"roc_auc": 0.85 + shift, "average_precision": 0.65 + shift, "brier_score": 0.12 - shift / 10,
                           "calibration_slope": slope, "calibration_intercept": intercept})
            point[model] = values
            reps[model] = {m: (np.full(200, v) + (noise if m in ("roc_auc", "average_precision") else 0)).tolist()
                           for m, v in values.items()}
        out.append({"seed": seed, "point": point, "replicates": reps,
                    "thresholds": {m: THRESHOLDS for m in point}, "test_counts": {"n_births": 100}})
    return out


class ReportTests(unittest.TestCase):
    def analyse(self, results, v1="reconciled"):
        deltas = paired_deltas_table(results, "p")
        pooled = pooled_summary_table(results, "p")
        return deltas, pooled, statements(deltas, pooled, CONFIG, v1)

    def test_superiority_needs_four_of_five_seeds(self):
        self.assertTrue(self.analyse(fake_results())[2]["superiority"]["holds"])
        weak = fake_results(v2_minus_v1=(0.03, 0.03, 0.03, 0.0, 0.0))
        st = self.analyse(weak)[2]["superiority"]
        self.assertFalse(st["holds"])
        self.assertEqual(st["statement"], "V2 discrimination is not shown to be superior to the V1 specification")

    def test_superiority_negative_when_v1_not_run(self):
        self.assertFalse(self.analyse(fake_results(), v1="not_run_parquet_missing")[2]["superiority"]["holds"])

    def test_calibration_statement_edges(self):
        self.assertTrue(self.analyse(fake_results(slope=1.1, intercept=-0.05))[2]["calibration"]["holds"])
        self.assertFalse(self.analyse(fake_results(slope=1.11))[2]["calibration"]["holds"])
        self.assertFalse(self.analyse(fake_results(intercept=0.051))[2]["calibration"]["holds"])

    def test_equivalence_uses_pooled_ci(self):
        self.assertTrue(self.analyse(fake_results(lr_gap=0.002))[2]["equivalence"]["holds"])
        self.assertFalse(self.analyse(fake_results(lr_gap=0.009))[2]["equivalence"]["holds"])

    def test_pooled_delta_is_mean_of_seed_deltas(self):
        deltas = self.analyse(fake_results(v2_minus_v1=(0.01, 0.02, 0.03, 0.04, 0.05)))[0]
        block = deltas[(deltas.comparison == "V2 - V1_spec") & (deltas.metric == "roc_auc")]
        self.assertAlmostEqual(float(block[block.seed == "pooled"].delta.iloc[0]), 0.03, places=10)

    def test_paper_table_labels_recorded_v1(self):
        pooled = self.analyse(fake_results())[1]
        table = paper_table(pooled, CONFIG["v1_spec"]["recorded_v1"])
        self.assertIn("original V1 report, different split, not paired", table)
        self.assertEqual(table.count("\n"), 6)


class EndToEndRunnerTests(unittest.TestCase):
    """Drives the real runner path (score-once guard, tables, figures, manifest) on synthetic data."""

    def test_runner_path_end_to_end(self):
        import copy

        import run_a9
        from a8_outputs import write_manifest

        cohort, base, f0, states, keys = synthetic()
        config = copy.deepcopy(CONFIG)
        config["bootstrap"]["replicates"] = 25
        data = {"cohort": cohort, "keys": keys}
        frames = model_frames(base, states, f0)
        with tempfile.TemporaryDirectory() as tmp:
            guard_dir, out = Path(tmp) / "private", Path(tmp) / "out"
            out.mkdir()
            results = [run_a9._score_seed(seed, data, frames, config, FAST, guard_dir, "p" * 40) for seed in (11, 22)]
            for seed in (11, 22):
                self.assertEqual(read_guard(guard_dir, seed)["status"], "completed")
                with self.assertRaises(FileExistsError):
                    start_guard(guard_dir, seed, "p" * 40)
            for result in results:
                self.assertEqual(sum(result["audit"]["overlap"]["psu"].values()), 0)
                self.assertEqual(sum(result["audit"]["overlap"]["mother"].values()), 0)
                self.assertEqual(len(result["replicates"]["V2"]["roc_auc"]), 25)
            metrics = run_a9.test_metrics_table(results, "p")
            deltas = run_a9.paired_deltas_table(results, "p")
            pooled = run_a9.pooled_summary_table(results, "p")
            st = run_a9.statements(deltas, pooled, config, "reconciled")
            table = run_a9.paper_table(pooled, config["v1_spec"]["recorded_v1"])
            bins = pd.DataFrame([{"seed": r["seed"], "model": m, **row} for r in results for m, rows in r["bins"].items()
                                 for row in rows])
            for name, frame in (("m.csv", metrics), ("d.csv", deltas), ("p.csv", pooled), ("b.csv", bins)):
                run_a9.write_csv(frame, out / name)
            run_a9.write_json(run_a9.public_split_audit("p", {"n_psus": 1}, [r["audit"] for r in results],
                                                        {"status": "reconciled"}), out / "a9_split_audit.json")
            run_a9.write_json({"protocol_commit": "p", "statements": st}, out / "a9_statements.json")
            run_a9.fig_roc_pr(results, out / "roc.png")
            run_a9.fig_calibration(bins, out / "cal.png")
            run_a9.fig_forest(deltas, out / "forest.png")
            run_a9.write_text(run_a9._methods({
                "config": config, "statements": st, "paper_table": table, "deltas": deltas, "metrics": metrics,
                "audits": [r["audit"] for r in results], "v1": {"status": "reconciled", "reconciliation": {}},
                "guards": [{"seed": 11, "source": "x", "completed_utc": "t"}], "qa": [], "git": {"commit": "c"},
                "protocol_commit": "p", "generated": "t", "input_hash": "h"}), out / "methods.md")
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("x: 1\n", encoding="utf-8")
            manifest = write_manifest(out, experiment_id="A9", config=config, config_path=config_path,
                                      repo_root=_BASE.parent, seeds=[11, 22], split_manifest_hashes={})
            self.assertEqual(len(manifest["files"]), 10)
            self.assertIn("a9_split_audit.json", manifest["disclosure_scan"]["files_checked"])
            self.assertEqual(set(metrics.model), {"V2", "V2_LR", "F1_core", "V1_spec"})
            self.assertIn("V2 - V1_spec", set(deltas.comparison))


if __name__ == "__main__":
    unittest.main()
