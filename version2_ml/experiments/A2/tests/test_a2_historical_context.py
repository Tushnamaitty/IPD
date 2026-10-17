"""A2 synthetic tests: source exclusivity, fallback, label-mutation invariance.

Synthetic fixtures only; no NFHS records.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "common")]
from A2_historical_context import (HistoricalContextMapping, SmoothingSpec, build_a2_features,  # noqa: E402
                                   FEATURE_NAMES, CELL_RATE, STATE_RATE)


def source(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    units = rng.choice(["alpha", "beta", "gamma"], size=n, p=[0.5, 0.3, 0.2])
    sectors = rng.choice(["public", "private", "other"], size=n, p=[0.6, 0.38, 0.02])
    p = 0.1 + 0.3 * (sectors == "private") + 0.1 * (units == "beta")
    return pd.DataFrame({"wave": "NFHS-4", "source_unit": units, "sector": sectors,
                         "m17": rng.binomial(1, p), "sample_weight": rng.uniform(0.2, 3, n)})


def target(seed=1, n=2000):
    rng = np.random.default_rng(seed)
    units = rng.choice(["alpha", "beta", "gamma", "delta"], size=n)
    return pd.DataFrame({"wave": "NFHS-5", "caseid": [f"m{i // 2}" for i in range(n)],
                         "source_unit": pd.array(units, dtype="string"),
                         "sector": rng.choice(["public", "private", "other"], size=n),
                         "m17": rng.binomial(1, 0.3, n)})


def same_bits(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or not a.index.equals(b.index):
        return False
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f":
            if x.tobytes() != y.tobytes():
                return False
        elif not np.array_equal(x, y):
            return False
    return True


class TestSourceExclusivity(unittest.TestCase):
    def test_fit_rejects_nfhs5_rows(self):
        bad = target().assign(source_unit="alpha")
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            HistoricalContextMapping().fit(bad)

    def test_fit_rejects_mixed_waves(self):
        mixed = pd.concat([source(), target().assign(source_unit="alpha")], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            HistoricalContextMapping().fit(mixed)

    def test_fit_rejects_invalid_inputs(self):
        s = source()
        for broken, msg in [(s.drop(columns="m17"), "Missing source columns"),
                            (s.assign(m17=2), "0/1"),
                            (s.assign(sector="clinic"), "sector"),
                            (s.assign(source_unit=""), "source unit"),
                            (s.iloc[:0], "Empty")]:
            with self.assertRaisesRegex(ValueError, msg):
                HistoricalContextMapping().fit(broken)
        with self.assertRaisesRegex(ValueError, "weights"):
            HistoricalContextMapping(SmoothingSpec(weighted=True)).fit(s.assign(sample_weight=0.0))

    def test_frozen_and_order_of_calls(self):
        m = HistoricalContextMapping()
        with self.assertRaises(ValueError):
            m.transform(target())
        m.fit(source())
        with self.assertRaises(RuntimeError):
            m.fit(source())

    def test_invalid_spec(self):
        for kw in ({"alpha": -1}, {"n_min": 0}, {"alpha": float("nan")}):
            with self.assertRaises(ValueError):
                SmoothingSpec(**kw)


class TestTargetLabelInvariance(unittest.TestCase):
    """Adversarial: NFHS-5 outcomes must not influence the mapping or features."""

    def setUp(self):
        self.src = source()
        self.tgt = target()
        self.mapping, self.features = build_a2_features(self.src, self.tgt)
        self.hash = self.mapping.mapping_hash()

    def test_permuted_target_labels_bitwise_identical(self):
        rng = np.random.default_rng(99)
        for _ in range(5):
            mutated = self.tgt.assign(m17=rng.permutation(self.tgt["m17"].to_numpy()))
            m2, f2 = build_a2_features(self.src, mutated)
            self.assertEqual(m2.mapping_hash(), self.hash)
            self.assertTrue(same_bits(f2, self.features))

    def test_replaced_flipped_and_removed_target_labels(self):
        variants = [self.tgt.assign(m17=0), self.tgt.assign(m17=1),
                    self.tgt.assign(m17=1 - self.tgt["m17"]), self.tgt.drop(columns="m17"),
                    self.tgt.assign(m17=np.nan)]
        for mutated in variants:
            m2, f2 = build_a2_features(self.src, mutated)
            self.assertEqual(m2.mapping_hash(), self.hash)
            self.assertTrue(same_bits(f2, self.features))

    def test_heldout_partition_label_mutation(self):
        part = np.where(np.arange(len(self.tgt)) % 5 == 0, "test", "train")
        mutated = self.tgt.copy()
        mutated.loc[part == "test", "m17"] = 1 - mutated.loc[part == "test", "m17"]
        m2, f2 = build_a2_features(self.src, mutated)
        self.assertEqual(m2.mapping_hash(), self.hash)
        self.assertTrue(same_bits(f2, self.features))

    def test_transform_reads_only_unit_and_sector(self):
        minimal = self.tgt[["source_unit", "sector"]]
        self.assertTrue(same_bits(self.mapping.transform(minimal), self.features))
        noise = self.tgt.assign(extra=np.random.default_rng(3).normal(size=len(self.tgt)))
        self.assertTrue(same_bits(self.mapping.transform(noise), self.features))

    def test_transform_does_not_mutate_mapping_or_input(self):
        before = self.tgt.copy()
        for _ in range(3):
            self.mapping.transform(self.tgt)
        self.assertEqual(self.mapping.mapping_hash(), self.hash)
        pd.testing.assert_frame_equal(before, self.tgt)

    def test_source_label_change_does_change_mapping(self):
        """Positive control: the hash is sensitive to the data it SHOULD depend on."""
        flipped = self.src.assign(m17=1 - self.src["m17"])
        self.assertNotEqual(HistoricalContextMapping().fit(flipped).mapping_hash(), self.hash)

    def test_index_and_order_preserved(self):
        shuffled = self.tgt.sample(frac=1, random_state=4)
        out = self.mapping.transform(shuffled)
        self.assertTrue(out.index.equals(shuffled.index))
        self.assertTrue(same_bits(out.sort_index(), self.features.sort_index()))


class TestFallback(unittest.TestCase):
    def test_unseen_state_falls_back_to_global_and_sector(self):
        m = HistoricalContextMapping().fit(source())
        out = m.transform(pd.DataFrame({"source_unit": pd.array(["delta", None], dtype="string"),
                                        "sector": ["private", "public"]}))
        g = m.aggregate_tables()["global"]["rate"].iloc[0]
        sec = m.aggregate_tables()["sector"].set_index("sector")["rate"]
        self.assertTrue(np.allclose(out[STATE_RATE], g))
        self.assertAlmostEqual(out[CELL_RATE].iloc[0], sec["private"])
        self.assertAlmostEqual(out[CELL_RATE].iloc[1], sec["public"])
        self.assertTrue((out["hist_state_unsupported"] == 1).all())
        self.assertTrue((out["hist_cell_unsupported"] == 1).all())
        self.assertTrue((out["hist_state_support_log"] == 0).all())
        self.assertEqual(list(out["hist_cell_level"]), ["sector", "sector"])

    def test_singleton_positive_cell_is_not_its_label(self):
        s = pd.concat([source(), pd.DataFrame({"wave": ["NFHS-4"], "source_unit": ["tiny"], "sector": ["private"],
                                               "m17": [1], "sample_weight": [1.0]})], ignore_index=True)
        m = HistoricalContextMapping(SmoothingSpec(alpha=0, n_min=30)).fit(s)
        out = m.transform(pd.DataFrame({"source_unit": ["tiny"], "sector": ["private"]}))
        self.assertLess(out[CELL_RATE].iloc[0], 1.0)
        self.assertLess(out[STATE_RATE].iloc[0], 1.0)
        self.assertEqual(out["hist_cell_level"].iloc[0], "sector")
        cell = m.aggregate_tables()["cell"].set_index(["source_unit", "sector"])
        self.assertFalse(bool(cell.loc[("tiny", "private"), "supported"]))

    def test_zero_count_cell_with_alpha_zero_is_finite_fallback(self):
        s = source()
        s = s[~((s.source_unit == "gamma") & (s.sector == "other"))]
        m = HistoricalContextMapping(SmoothingSpec(alpha=0)).fit(s)
        out = m.transform(pd.DataFrame({"source_unit": ["gamma"], "sector": ["other"]}))
        st = m.aggregate_tables()["state"].set_index("source_unit")["rate"]
        self.assertAlmostEqual(out[CELL_RATE].iloc[0], st["gamma"])
        self.assertEqual(out["hist_cell_level"].iloc[0], "state")
        self.assertEqual(out["hist_cell_support_log"].iloc[0], 0.0)

    def test_unsupported_state_and_sector_fall_to_global(self):
        s = pd.DataFrame({"wave": "NFHS-4", "source_unit": ["a"] * 40 + ["b"] * 5,
                          "sector": ["public"] * 40 + ["other"] * 5, "m17": [0, 1] * 20 + [1] * 5})
        m = HistoricalContextMapping(SmoothingSpec(alpha=10, n_min=30)).fit(s)
        out = m.transform(pd.DataFrame({"source_unit": ["b", "b"], "sector": ["other", "private"]}))
        g = 25 / 45
        self.assertTrue(np.allclose(out[STATE_RATE], g))
        self.assertTrue(np.allclose(out[CELL_RATE], g))
        self.assertEqual(list(out["hist_cell_level"]), ["global", "global"])

    def test_smoothing_formula_hand_computed(self):
        s = pd.DataFrame({"wave": "NFHS-4", "source_unit": ["a"] * 60 + ["b"] * 40,
                          "sector": (["public"] * 30 + ["private"] * 30) + (["public"] * 40),
                          "m17": [0] * 27 + [1] * 3 + [0] * 15 + [1] * 15 + [0] * 32 + [1] * 8})
        m = HistoricalContextMapping(SmoothingSpec(alpha=20, n_min=30)).fit(s)
        g = 26 / 100
        r_a = (18 + 20 * g) / (60 + 20)
        r_cell = (15 + 20 * r_a) / (30 + 20)
        out = m.transform(pd.DataFrame({"source_unit": ["a"], "sector": ["private"]}))
        self.assertAlmostEqual(out[STATE_RATE].iloc[0], r_a, places=12)
        self.assertAlmostEqual(out[CELL_RATE].iloc[0], r_cell, places=12)

    def test_weighted_uses_kish_support(self):
        s = source()
        m = HistoricalContextMapping(SmoothingSpec(weighted=True)).fit(s)
        st = m.aggregate_tables()["state"].set_index("source_unit")
        sub = s[s.source_unit == "alpha"].sample_weight
        w = sub / s.sample_weight.mean()
        self.assertAlmostEqual(st.loc["alpha", "support"], w.sum() ** 2 / (w ** 2).sum(), places=6)
        self.assertLess(st.loc["alpha", "support"], st.loc["alpha", "n_births"])

    def test_unknown_target_sector_rejected(self):
        m = HistoricalContextMapping().fit(source())
        with self.assertRaisesRegex(ValueError, "sector"):
            m.transform(pd.DataFrame({"source_unit": ["alpha"], "sector": ["clinic"]}))
        with self.assertRaisesRegex(ValueError, "Missing target column"):
            m.transform(pd.DataFrame({"source_unit": ["alpha"]}))

    def test_all_features_finite_and_bounded(self):
        m, out = build_a2_features(source(), target())
        vals = out[list(FEATURE_NAMES)].to_numpy(float)
        self.assertTrue(np.isfinite(vals).all())
        self.assertTrue(out[[STATE_RATE, CELL_RATE]].stack().between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
