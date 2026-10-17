"""Synthetic tests for shared experiment helpers (no NFHS records)."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from exp_metrics import binary_metrics, calibration_intercept_slope, select_threshold_max_f1  # noqa: E402
from exp_provenance import (assert_disclosure_safe, suppress_small_cells, digest_frame,  # noqa: E402
                            scan_output_dir, write_json, SUPPRESSED)
from exp_cohort import build_birth_table  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_calibrated_scores_give_unit_slope_zero_intercept(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.02, 0.98, 200000)
        y = rng.binomial(1, p)
        cal = calibration_intercept_slope(y, p)
        self.assertAlmostEqual(cal["calibration_slope"], 1.0, delta=0.03)
        self.assertAlmostEqual(cal["calibration_intercept"], 0.0, delta=0.03)

    def test_overconfident_scores_give_slope_below_one(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.05, 0.95, 100000)
        y = rng.binomial(1, p)
        sharp = 1 / (1 + np.exp(-3 * np.log(p / (1 - p))))
        self.assertLess(calibration_intercept_slope(y, sharp)["calibration_slope"], 0.5)

    def test_constant_scores_not_identifiable(self):
        cal = calibration_intercept_slope([0, 1, 0, 1], [0.3] * 4)
        self.assertIsNone(cal["calibration_slope"])

    def test_unequal_arrays_rejected(self):
        with self.assertRaises(ValueError):
            calibration_intercept_slope([0, 1], [0.2, 0.4, 0.5])
        with self.assertRaises(ValueError):
            binary_metrics([0, 1, 1], [0.1, 0.9])
        with self.assertRaises(ValueError):
            select_threshold_max_f1([0, 1], [0.5])

    def test_threshold_selection_separable(self):
        y = np.array([0, 0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
        t = select_threshold_max_f1(y, p)
        self.assertEqual(binary_metrics(y, p, threshold=t)["f1"], 1.0)


class TestProvenance(unittest.TestCase):
    def test_suppression(self):
        t = pd.DataFrame({"unit": list("abcd"), "n": [1000, 40, 1000, 1000], "e": [300, 20, 5, 995],
                          "rate": [0.3, 0.5, 0.005, 0.995]})
        out = suppress_small_cells(t, n_col="n", event_col="e", value_cols=("rate",))
        self.assertEqual(list(out["suppressed"]), [False, True, True, True])
        self.assertEqual(out.loc[1, "rate"], SUPPRESSED)

    def test_disclosure_guard(self):
        with self.assertRaisesRegex(ValueError, "DISCLOSURE"):
            assert_disclosure_safe(pd.DataFrame({"caseid": ["x"], "auc": [0.8]}))
        with self.assertRaisesRegex(ValueError, "row-level"):
            assert_disclosure_safe(pd.DataFrame({"auc": np.zeros(6000)}))

    def test_output_scan(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"auc": [0.8]}).to_csv(Path(d) / "ok.csv", index=False)
            write_json({"a": {"n": 1}}, Path(d) / "ok.json")
            self.assertEqual(scan_output_dir(Path(d))["status"], "PASS")
            write_json({"a": [{"bidx": 1}]}, Path(d) / "bad.json")
            with self.assertRaisesRegex(ValueError, "DISCLOSURE"):
                scan_output_dir(Path(d))

    def test_digest_order_sensitive(self):
        f = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        self.assertNotEqual(digest_frame(f), digest_frame(f.iloc[::-1]))
        self.assertEqual(digest_frame(f), digest_frame(f.copy()))


def raw_rows():
    return pd.DataFrame({"caseid": ["  1 1 2", "  1 1 2", "  1 5 3"], "bidx": [1, 2, 1],
                         "b3": [1450, 1430, 1462], "b19": [10, 30, 5], "v008": [1460, 1460, 1467],
                         "v011": [1150, 1150, 1126], "v133": [12, 12, 16], "v190": [1, 1, 5],
                         "bord": [2, 1, 3], "m15": [21, 31, 96], "m17": [0, 1, 0], "v025": [1, 1, 2],
                         "v130": [1, 1, 2], "b0": [0, 0, 0], "v024": [19, 19, 7], "v005": [1e6, 1e6, 2e6],
                         "v001": [10, 10, 20], "v022": [1, 1, 2]})


class TestCohortTable(unittest.TestCase):
    def test_builds_restricted_table(self):
        out, flow = build_birth_table(raw_rows(), "NFHS-5", {"raw_rows": 3})
        self.assertEqual(flow["final_births"], 3)
        self.assertEqual(flow["final_mothers"], 2)
        self.assertEqual(list(out["sector"]), ["public", "private", "other"])
        self.assertFalse(flow["matches_expected"])

    def test_duplicate_birth_keys_rejected(self):
        bad = raw_rows()
        bad.loc[1, "bidx"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_birth_table(bad, "NFHS-5", {})

    def test_missing_mother_key_rejected(self):
        bad = raw_rows()
        bad.loc[0, "caseid"] = "  "
        with self.assertRaisesRegex(ValueError, "mother"):
            build_birth_table(bad, "NFHS-5", {})

    def test_invalid_outcome_counted_and_excluded(self):
        bad = raw_rows()
        bad.loc[2, "m17"] = 8
        out, flow = build_birth_table(bad, "NFHS-5", {})
        self.assertEqual(flow["invalid_or_missing_outcome_excluded"], 1)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
