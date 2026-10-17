"""A5 synthetic tests: source-only fit, target-label invariance, FT schema, unseen levels."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from A5_temporal_transfer import (SourceOnlyTransferModel, AdaptationRecalibrator, zero_shot_pipeline,  # noqa: E402
                                  harmonize_core, harmonization_report, assert_ft_only, FT_NUM, FT_CAT,
                                  RESTRICTED_F0)

ALGOS = ("logistic_regression", "xgboost")


def births(wave, n=3000, seed=0, religions=("1", "2", "3"), twins=("0", "1")):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({
        "maternal_age_at_index_birth": rng.uniform(16, 45, n), "education_years": rng.integers(0, 18, n).astype(float),
        "birth_order": rng.integers(1, 6, n).astype(float),
        "wealth_index": rng.choice(list("12345"), n), "facility_type": rng.choice(["public", "private", "other"],
                                                                                  n, p=[.6, .38, .02]),
        "residence": rng.choice(["urban", "rural"], n), "religion": rng.choice(list(religions), n),
        "twin_order": rng.choice(list(twins), n, p=[0.97, 0.03] if len(twins) == 2 else None)})
    lp = -2 + 1.5 * (f.facility_type == "private") + 0.05 * f.education_years
    f["m17"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    f["wave"] = wave
    f["caseid"] = [f"{wave}-{i // 2}" for i in range(n)]
    f["source_partition"] = np.where(np.arange(n) % 5 == 0, "validation", "train")
    return f


class TestSourceOnlyFit(unittest.TestCase):
    def test_fit_refuses_target_and_mixed_waves(self):
        src, tgt = births("NFHS-4"), births("NFHS-5", seed=1)
        for algo in ALGOS:
            with self.assertRaisesRegex(ValueError, "LEAKAGE"):
                SourceOnlyTransferModel(algo).fit(tgt, tgt.m17, wave=tgt.wave, partition="train")
            mixed = pd.concat([src, tgt], ignore_index=True)
            with self.assertRaisesRegex(ValueError, "LEAKAGE"):
                SourceOnlyTransferModel(algo).fit(mixed, mixed.m17, wave=mixed.wave, partition="train")
            with self.assertRaisesRegex(ValueError, "LEAKAGE"):
                SourceOnlyTransferModel(algo).fit(src, src.m17, wave=src.wave, partition="validation")

    def test_threshold_refuses_target_rows(self):
        src, tgt = births("NFHS-4"), births("NFHS-5", seed=1)
        m = SourceOnlyTransferModel().fit(src, src.m17, wave=src.wave, partition="train")
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            m.select_threshold(tgt, tgt.m17, wave=tgt.wave, partition="validation")
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            m.select_threshold(src, src.m17, wave=src.wave, partition="test")

    def test_fit_called_only_with_source_rows_in_pipeline(self):
        src, tgt = births("NFHS-4"), births("NFHS-5", seed=1)
        seen = []
        original = SourceOnlyTransferModel.fit

        def spy(self, frame, y, *, wave, partition):
            seen.append(set(pd.Series(np.asarray(wave)).astype(str)))
            return original(self, frame, y, wave=wave, partition=partition)
        SourceOnlyTransferModel.fit = spy
        try:
            zero_shot_pipeline(src, tgt, algorithm="logistic_regression")
        finally:
            SourceOnlyTransferModel.fit = original
        self.assertEqual(seen, [{"NFHS-4"}])

    def test_frozen_model_cannot_refit_or_drift(self):
        src, tgt = births("NFHS-4"), births("NFHS-5", seed=1)
        m, _ = zero_shot_pipeline(src, tgt, algorithm="logistic_regression")
        with self.assertRaises(RuntimeError):
            m.fit(src, src.m17, wave=src.wave, partition="train")
        m.clf.estimator.coef_[0, 0] += 1.0  # simulate post-freeze tampering
        with self.assertRaisesRegex(RuntimeError, "LEAKAGE"):
            m.predict(tgt, wave="NFHS-5")

    def test_preprocessor_categories_come_from_source_only(self):
        src = births("NFHS-4", religions=("1", "2"))
        tgt = births("NFHS-5", seed=1, religions=("1", "2", "96"))
        for algo in ALGOS:
            m, _ = zero_shot_pipeline(src, tgt, algorithm=algo)
            self.assertEqual(m.fitted_categories()["religion"], ["1", "2"])


class TestTargetLabelInvariance(unittest.TestCase):
    def test_target_label_mutation_bitwise_identical(self):
        src, tgt = births("NFHS-4"), births("NFHS-5", seed=1)
        rng = np.random.default_rng(7)
        for algo in ALGOS:
            m0, p0 = zero_shot_pipeline(src, tgt, algorithm=algo)
            for mutated in (tgt.assign(m17=rng.permutation(tgt.m17.to_numpy())), tgt.assign(m17=1 - tgt.m17),
                            tgt.assign(m17=0), tgt.drop(columns="m17")):
                m1, p1 = zero_shot_pipeline(src, mutated, algorithm=algo)
                self.assertEqual(p1.tobytes(), p0.tobytes(), algo)
                self.assertEqual(m1.frozen_digest, m0.frozen_digest)
                self.assertEqual(m1.threshold, m0.threshold)

    def test_repeated_scoring_does_not_change_model(self):
        src, tgt = births("NFHS-4"), births("NFHS-5", seed=1)
        for algo in ALGOS:
            m, p = zero_shot_pipeline(src, tgt, algorithm=algo)
            for _ in range(3):
                self.assertEqual(m.predict(tgt, wave="NFHS-5").tobytes(), p.tobytes())
            self.assertEqual(m.digest(), m.frozen_digest)


class TestSchemaAndHarmonization(unittest.TestCase):
    def test_exactly_eight_ft_features(self):
        assert_ft_only(FT_NUM + FT_CAT)
        self.assertEqual(len(FT_NUM + FT_CAT), 8)
        with self.assertRaisesRegex(ValueError, "Restricted"):
            assert_ft_only(FT_NUM + FT_CAT + ("anc_visits",))
        with self.assertRaisesRegex(ValueError, "Restricted"):
            assert_ft_only(FT_NUM + FT_CAT[:-1] + ("maternal_age",))
        with self.assertRaisesRegex(ValueError, "exactly"):
            assert_ft_only(FT_NUM + FT_CAT[:-1])
        self.assertIn("anc_visits", RESTRICTED_F0)
        self.assertEqual(len(RESTRICTED_F0), 29)

    def test_extra_restricted_columns_are_ignored_not_used(self):
        src, tgt = births("NFHS-4"), births("NFHS-5", seed=1)
        src_x = src.assign(anc_visits=src.m17 * 10)  # perfectly leaky column must not be used
        _, p_clean = zero_shot_pipeline(src, tgt, algorithm="logistic_regression")
        _, p_extra = zero_shot_pipeline(src_x, tgt.assign(anc_visits=0), algorithm="logistic_regression")
        self.assertEqual(p_clean.tobytes(), p_extra.tobytes())

    def test_harmonize_dtypes_and_vocabulary(self):
        tgt = births("NFHS-5", seed=1)
        x = harmonize_core(tgt.assign(wealth_index=tgt.wealth_index.astype("category")), "NFHS-5")
        self.assertEqual(list(x.columns), list(FT_NUM + FT_CAT))
        self.assertTrue(all(x[c].dtype == float for c in FT_NUM))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            harmonize_core(tgt.assign(wealth_index="6"), "NFHS-5")
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            harmonize_core(tgt.assign(facility_type="home"), "NFHS-5")
        with self.assertRaisesRegex(ValueError, "Missing FT"):
            harmonize_core(tgt.drop(columns="religion"), "NFHS-5")
        with self.assertRaisesRegex(ValueError, "missing"):
            harmonize_core(tgt.assign(education_years=np.nan), "NFHS-5")

    def test_unobserved_target_categories_handled(self):
        src = births("NFHS-4", religions=("1", "2"), twins=("0", "1"))
        tgt = births("NFHS-5", seed=1, religions=("1", "2", "96"), twins=("0", "1", "2"))
        for algo in ALGOS:
            m, p = zero_shot_pipeline(src, tgt, algorithm=algo)
            self.assertTrue(np.isfinite(p).all() and ((p > 0) & (p < 1)).all())
        rep = harmonization_report(harmonize_core(src, "NFHS-4"), harmonize_core(tgt, "NFHS-5"))
        self.assertEqual(rep["categorical"]["religion"]["target_levels_unseen_in_source"], ["96"])
        self.assertEqual(rep["categorical"]["twin_order"]["target_levels_unseen_in_source"], ["2"])
        self.assertEqual(rep["categorical"]["religion"]["target_rows_with_unseen_level"],
                         int((tgt.religion == "96").sum()))


class TestAdaptation(unittest.TestCase):
    def test_overlapping_mothers_rejected(self):
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            AdaptationRecalibrator().fit([0.2, 0.7], [0, 1], adaptation_groups=["a", "b"],
                                         evaluation_groups=["b", "c"])

    def test_recalibration_fixes_shift_and_preserves_ranking(self):
        rng = np.random.default_rng(3)
        lp = rng.normal(-1, 1, 20000)
        y = rng.binomial(1, 1 / (1 + np.exp(-(lp + 0.5))))
        p = 1 / (1 + np.exp(-lp))
        for kind in ("intercept", "intercept_slope"):
            r = AdaptationRecalibrator(kind).fit(p[:10000], y[:10000], adaptation_groups=range(10000),
                                                 evaluation_groups=range(10000, 20000))
            q = r.transform(p[10000:])
            self.assertAlmostEqual(r.coef[0], 0.5, delta=0.08)
            self.assertTrue(np.array_equal(np.argsort(q, kind="stable"), np.argsort(p[10000:], kind="stable")))
        with self.assertRaises(ValueError):
            AdaptationRecalibrator("isotonic")


if __name__ == "__main__":
    unittest.main()
