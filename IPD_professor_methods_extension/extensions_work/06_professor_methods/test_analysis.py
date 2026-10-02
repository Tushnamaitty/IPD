"""Synthetic validation only: no NFHS observations or published estimates."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

spec = importlib.util.spec_from_file_location("methods_analysis", Path(__file__).with_name("analysis.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestMethods(unittest.TestCase):
    def test_crossfit_target_and_known_standardized_difference(self):
        rng = np.random.default_rng(71)
        n = 1600
        ageproxy = rng.normal(size=n)
        a = rng.binomial(1, expit(-.2 + .55*ageproxy))
        q0 = expit(-1.3 + .5*ageproxy)
        q1 = expit(-1.3 + .7 + .5*ageproxy)
        y = rng.binomial(1, np.where(a, q1, q0))
        df = pd.DataFrame({
            "birth_order": ageproxy, "wealth_index": rng.integers(1, 6, n),
            "education_years": rng.integers(0, 16, n),
            "residence": rng.choice(["urban", "rural"], n),
            "religion": rng.choice(["A", "B"], n),
            "social_group": rng.choice(["A", "B"], n),
            "twin_order": rng.choice(["single", "twin"], n),
            "state": rng.choice(["X", "Y"], n),
            "facility_type": np.where(a, "private", "public"),
            "csection": y, "respondent_id": np.repeat(np.arange(n//2), 2),
            "sample_weight_normalized": rng.uniform(.5, 1.5, n),
        })
        a, y, w, e, pred0, pred1, fold, blends = mod.crossfit(df, 2, 2, 42)
        self.assertTrue(np.isfinite(e).all())
        self.assertEqual(len(set(fold)), 2)
        self.assertTrue(all(len(set(fold[df.respondent_id == i])) == 1 for i in range(n//2)))
        for item in blends:
            for target in ("propensity", "outcome"):
                self.assertAlmostEqual(sum(item[target][k] for k in ("logistic", "tree")), 1)
        result = mod.estimate(a, y, w, e, pred0, pred1)
        self.assertAlmostEqual(result["tmle"]["target_score_private"], 0, places=9)
        self.assertAlmostEqual(result["tmle"]["target_score_public"], 0, places=9)
        self.assertTrue(0 < result["tmle"]["private_risk"] < 1)
        self.assertTrue(0 < result["tmle"]["public_risk"] < 1)
        truth = np.mean(q1-q0)
        self.assertLess(abs(result["tmle"]["risk_difference"]-truth), .08)
        ci = mod.bootstrap(a, y, w, e, pred0, pred1, df.respondent_id.to_numpy(),
                           rng.integers(0, 4, n), 5, 4)
        self.assertEqual(len(ci["tmle"]["risk_difference_ci95"]), 2)


if __name__ == "__main__":
    unittest.main()
