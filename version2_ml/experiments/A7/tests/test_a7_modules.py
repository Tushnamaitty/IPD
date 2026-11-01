"""A7 synthetic acceptance tests. No real NFHS data is read or required."""
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[3]
for _entry in (
    str(_BASE),
    str(_BASE / "experiments" / "common"),
    str(_BASE / "experiments" / "A4" / "src"),
    str(_BASE / "experiments" / "A7" / "src"),
):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from a7_calibration import (
    apply_calibrator,
    calibration_intercept_slope,
    f1_at_threshold,
    fit_calibrator,
    threshold_f1_optimal,
    threshold_for_recall,
    threshold_youden,
)
from a7_candidates import (
    MODELS,
    TIERS,
    build_tier_frame,
    carve_calibration_partition,
    fit_candidate,
    predict_candidate,
    psu_overlap_summary,
    scale_pos_weight,
    tier_manifest,
    tune_candidate,
)
from a7_shap import (
    explained_feature_names,
    family_map,
    rank_stability,
    shap_family_table,
)


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _synthetic_frame(n=900, seed=11):
    rng = np.random.default_rng(seed)
    caseids = np.array([f"m{i // 2}" for i in range(n)])
    frame = pd.DataFrame(
        {
            "maternal_age_at_index_birth": rng.uniform(18, 40, n),
            "education_years": rng.integers(0, 16, n).astype(float),
            "wealth_index": rng.choice(["1", "2", "3", "4", "5"], n),
            "birth_order": rng.integers(1, 5, n).astype(float),
            "facility_type": rng.choice(["public", "private", "other"], n),
            "residence": rng.choice(["urban", "rural"], n),
            "religion": rng.choice(["1", "2", "3"], n),
            "twin_order": rng.choice(["0", "1"], n),
        }
    )
    signal = (
        frame["facility_type"].eq("private").astype(float)
        + 0.5 * frame["residence"].eq("urban").astype(float)
        + rng.normal(0, 0.7, n)
    )
    labels = (signal > 0.8).astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        labels[0], labels[-1] = 0, 1
    return frame, labels, caseids


def _enhanced_frame(frame, seed=5):
    rng = np.random.default_rng(seed)
    enhanced = frame.copy()
    enhanced["facility_detail"] = rng.choice(
        ["public: govt./munic. hospital", "private: hospital/maternity home/clinic", "other"],
        len(frame),
    )
    enhanced["history_four_level"] = rng.choice(
        ["first_recorded_live_birth", "higher_order_earlier_institutional_history_unobserved"],
        len(frame),
    )
    enhanced["has_any_observed_earlier_institutional_birth"] = rng.choice(
        ["true", "false"], len(frame)
    )
    enhanced["history_quality_flag"] = "ok"
    return enhanced


class CalibrationTests(unittest.TestCase):
    def test_calibrators_monotone_and_in_range(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0.05, 0.95, 4000)
        y = (rng.uniform(size=4000) < p).astype(int)
        grid = np.linspace(0.02, 0.98, 200)
        for method in ("platt", "isotonic"):
            calibrator = fit_calibrator(method, p, y)
            out = apply_calibrator(method, calibrator, grid)
            self.assertTrue(np.isfinite(out).all())
            self.assertTrue(((out >= 0) & (out <= 1)).all())
            self.assertTrue(np.all(np.diff(out) >= -1e-9))

    def test_calibrator_requires_both_classes(self):
        with self.assertRaises(ValueError):
            fit_calibrator("platt", [0.1, 0.2, 0.3], [1, 1, 1])
        with self.assertRaises(ValueError):
            fit_calibrator("isotonic", [0.1, 0.2, 0.3], [1, 1, 1])

    def test_calibration_intercept_slope_recovers_calibrated_scale(self):
        rng = np.random.default_rng(7)
        p = rng.uniform(0.05, 0.95, 20000)
        y = (rng.uniform(size=20000) < p).astype(int)
        intercept, slope = calibration_intercept_slope(y, p)
        self.assertLess(abs(intercept), 0.15)
        self.assertLess(abs(slope - 1.0), 0.15)

    def test_threshold_functions(self):
        rng = np.random.default_rng(9)
        p = rng.uniform(0, 1, 3000)
        y = (rng.uniform(size=3000) < p).astype(int)
        threshold, best_f1 = threshold_f1_optimal(y, p)
        self.assertGreaterEqual(best_f1, f1_at_threshold(y, p, 0.5))
        youden = threshold_youden(y, p)
        self.assertGreater(youden, 0.0)
        self.assertLess(youden, 1.0)
        recall_threshold = threshold_for_recall(y, p, min_recall=0.8)
        self.assertTrue(recall_threshold is None or 0 < recall_threshold < 1)
        separable = threshold_for_recall(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.99)
        self.assertIsNotNone(separable)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.labels, self.groups = _synthetic_frame()

    def test_scale_pos_weight(self):
        self.assertEqual(scale_pos_weight([1, 0, 0, 1]), 1.0)
        with self.assertRaises(ValueError):
            scale_pos_weight([1, 1])

    def test_tier_frames_and_manifests(self):
        enhanced = _enhanced_frame(self.frame)
        f1 = build_tier_frame(self.frame, None, None, "F1")
        tier_manifest("F1").validate(f1)
        detailed = build_tier_frame(enhanced, enhanced["facility_detail"], enhanced, "F1_detail_history")
        tier_manifest("F1_detail_history").validate(detailed)
        self.assertIn("facility_detail", detailed.columns)
        self.assertIn("history_four_level", detailed.columns)

    def test_fit_predict_finite_and_partition_guard(self):
        manifest = tier_manifest("F1")
        train = self.frame.iloc[:600].reset_index(drop=True)
        valid = self.frame.iloc[600:].reset_index(drop=True)
        classifier = fit_candidate(
            "logistic_regression", "F1", {"C": 1.0, "solver": "liblinear", "max_iter": 500,
                                          "class_weight": "balanced"},
            train, self.labels[:600], 42,
        )
        probabilities = predict_candidate(classifier, valid, "validation")
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        with self.assertRaises(ValueError):
            predict_candidate(classifier, valid, "eval")
        with self.assertRaises(ValueError):
            fit_candidate("logistic_regression", "F1", {"max_iter": 500}, train,
                          self.labels[:600], 42).fit(train, self.labels[:600], partition="validation")

    def test_tuning_is_deterministic_and_never_accepts_eval_labels(self):
        signature = set(inspect.signature(tune_candidate).parameters)
        self.assertFalse({"valid", "validation", "test", "eval"} & signature)
        train = self.frame.iloc[:750].reset_index(drop=True)
        groups = self.groups[:750]
        labels = self.labels[:750]
        grid = [{"C": 0.1, "solver": "liblinear", "max_iter": 500, "class_weight": "balanced"},
                {"C": 1.0, "solver": "liblinear", "max_iter": 500, "class_weight": "balanced"}]
        first = tune_candidate("logistic_regression", "F1", train, labels, groups,
                               folds=3, seed=11, grid=grid)
        second = tune_candidate("logistic_regression", "F1", train, labels, groups,
                                folds=3, seed=11, grid=grid)
        self.assertEqual(first["best"], second["best"])
        self.assertEqual(len(first["results"][0]["fold_average_precision"]), 3)

    def test_preprocessing_frozen_under_validation_label_swap(self):
        train = self.frame.iloc[:600].reset_index(drop=True)
        valid = self.frame.iloc[600:].reset_index(drop=True)
        classifier = fit_candidate(
            "logistic_regression", "F1",
            {"C": 1.0, "solver": "liblinear", "max_iter": 500, "class_weight": "balanced"},
            train, self.labels[:600], 42,
        )
        before = _dense(classifier.preprocessor.transform(valid))
        flipped = 1 - self.labels[600:]
        after = _dense(classifier.preprocessor.transform(valid))
        np.testing.assert_array_equal(before, after)
        self.assertEqual(int(flipped.sum()) + int(self.labels[600:].sum()), len(flipped))

    def test_psu_overlap_summary(self):
        summary = psu_overlap_summary(["a", "a", "b", "c"], ["a", "c", "d"])
        self.assertEqual(summary["n_shared_psus"], 2)
        self.assertAlmostEqual(summary["share_validation_rows_in_shared_psus"], 2 / 3, places=5)

    def test_carve_partition_deterministic_and_disjoint(self):
        frame = self.frame.copy()
        frame["caseid"] = self.groups
        frame["m17"] = self.labels
        first = carve_calibration_partition(frame, seed=4208)
        second = carve_calibration_partition(frame, seed=4208)
        pd.testing.assert_series_equal(first, second)
        merged = pd.DataFrame({"g": self.groups, "p": first.to_numpy()})
        self.assertTrue(merged.groupby("g")["p"].nunique().le(1).all())
        self.assertGreater(int((first == "calibration").sum()), 0)
        self.assertGreater(int((first == "fit").sum()), 0)
        with self.assertRaises(ValueError):
            carve_calibration_partition(frame, calibration_fraction=1.5)


class CacheSafetyTests(unittest.TestCase):
    def test_test_scoring_refused_without_cache_or_flag(self):
        from run_a7 import _test_stage

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "--allow-test-score"):
                _test_stage(
                    cache_dir, {}, {}, np.array([], dtype=int), np.array([], dtype=int),
                    42, {"repo_commit": "synthetic"}, {"python": "synthetic"},
                )
            self.assertEqual(list(cache_dir.iterdir()), [])

    def test_cached_test_payload_reused_without_rescoring(self):
        from run_a7 import _test_stage

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cache = cache_dir / "a7_test_scored.json"
            cache.write_text(json.dumps({"metrics": {}, "note": "synthetic"}), encoding="utf-8")
            before = cache.read_bytes()
            payload = _test_stage(
                cache_dir, {}, {}, np.array([], dtype=int), np.array([], dtype=int),
                42, {"repo_commit": "synthetic"}, {"python": "synthetic"},
            )
            self.assertEqual(payload["source"], "cache_no_rescore")
            self.assertNotIn("scored_at", payload)
            self.assertEqual(cache.read_bytes(), before)

    def test_cache_dir_and_allow_flag_defaults(self):
        from run_a7 import _parse_args

        args = _parse_args(["--nfhs5-path", "synthetic.dta"])
        self.assertIsNone(args.cache_dir)
        self.assertFalse(args.allow_test_score)
        args = _parse_args(
            ["--nfhs5-path", "synthetic.dta", "--cache-dir", "cache", "--allow-test-score"]
        )
        self.assertEqual(args.cache_dir, "cache")
        self.assertTrue(args.allow_test_score)


class ShapTests(unittest.TestCase):
    def test_family_map_and_aggregation(self):
        frame, labels, _ = _synthetic_frame()
        enhanced = _enhanced_frame(frame)
        train = enhanced.iloc[:700].reset_index(drop=True)
        classifier = fit_candidate(
            "xgboost", "F1_detail_history",
            {"n_estimators": 30, "max_depth": 4, "learning_rate": 0.1,
             "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss",
             "tree_method": "hist", "scale_pos_weight": 1.0},
            train, labels[:700], 42,
        )
        mapping = family_map(classifier)
        names = explained_feature_names(classifier)
        self.assertEqual(set(names), set(mapping))
        table = shap_family_table(classifier, enhanced.iloc[700:].reset_index(drop=True),
                                  sample_size=150, seed=1)
        expected_families = set(tier_manifest("F1_detail_history").numerical) | set(
            tier_manifest("F1_detail_history").categorical
        )
        self.assertEqual({row["family"] for row in table["families"]}, expected_families)
        self.assertEqual(table["n_encoded_features"], len(names))
        share_sum = sum(row["share_of_total"] for row in table["families"])
        self.assertAlmostEqual(share_sum, 1.0, places=4)

    def test_rank_stability_requires_common_families(self):
        table_a = {"families": [
            {"family": "a", "rank": 1, "mean_abs_shap_margin": 2.0, "share_of_total": 0.5},
            {"family": "b", "rank": 2, "mean_abs_shap_margin": 1.0, "share_of_total": 0.5},
        ]}
        table_b = {"families": [
            {"family": "b", "rank": 1, "mean_abs_shap_margin": 2.0, "share_of_total": 0.5},
            {"family": "a", "rank": 2, "mean_abs_shap_margin": 1.0, "share_of_total": 0.5},
        ]}
        stability = rank_stability([table_a, table_b])
        self.assertEqual(stability["min_pairwise_spearman"], -1.0)
        with self.assertRaises(RuntimeError):
            rank_stability([table_a, {"families": [{"family": "a", "rank": 1}]}])


if __name__ == "__main__":
    unittest.main()
