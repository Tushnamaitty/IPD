"""A2 runner contract tests: required matched arms, row matching, metric schema."""
import sys
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run  # noqa: E402
from A2_historical_context import STATE_RATE, CELL_RATE, SUPPORT_FEATURES  # noqa: E402

REQUIRED_COLUMNS = {"experiment", "wave", "source_domain", "target_domain", "cohort", "split", "model",
                    "feature_tier", "n_births", "n_mothers", "n_events", "prevalence", "roc_auc",
                    "average_precision", "brier", "calibration_intercept", "calibration_slope",
                    "threshold_provenance", "precision", "recall", "f1", "confidence_interval_status"}


class TestRunnerContract(unittest.TestCase):
    def test_required_non_cumulative_arms(self):
        base = set(run.arm_manifest("B0").numerical) | set(run.arm_manifest("B0").categorical)
        expected = {"B0+S": {STATE_RATE}, "B0+SX": {CELL_RATE}, "B0+SUP": set(SUPPORT_FEATURES)}
        for arm, extra in expected.items():
            m = run.arm_manifest(arm)
            self.assertEqual(set(m.numerical) | set(m.categorical), base | extra, arm)
        self.assertNotIn(STATE_RATE, run.arm_manifest("B0+SX").numerical)

    def test_baseline_has_no_state_or_context(self):
        names = set(run.arm_manifest("B0").numerical) | set(run.arm_manifest("B0").categorical)
        self.assertEqual(len(names), 8)
        self.assertFalse(any(n.startswith("hist_") or n == "nfhs5_state" for n in names))

    def test_matched_rows_guard(self):
        run.assert_matched_rows(np.arange(5), np.arange(5))
        with self.assertRaises(AssertionError):
            run.assert_matched_rows(np.arange(5), np.array([0, 1, 2, 3, 5]))
        with self.assertRaises(AssertionError):
            run.assert_matched_rows(np.arange(5), np.arange(4))

    def test_metric_row_schema(self):
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, 500)
        p = np.clip(0.3 + 0.2 * (y - 0.3) + rng.normal(0, 0.1, 500), 0.01, 0.99)
        base = {"experiment": "A2", "wave": "NFHS-5", "source_domain": "s", "target_domain": "t",
                "cohort": "c", "split": "validation", "model": "m", "feature_tier": "F3"}
        row = run.metric_row(base, y, p, np.arange(500) // 2, 0.4, run.threshold_provenance("validation"))
        self.assertTrue(REQUIRED_COLUMNS <= set(row))
        self.assertEqual(row["confidence_interval_status"], "NOT_APPROVED")
        self.assertEqual(row["n_mothers"], 250)
        self.assertIn("optimistic", row["threshold_provenance"])


if __name__ == "__main__":
    unittest.main()
