"""A6 synthetic adversarial tests: mother exclusivity, target-outcome isolation, sector policy."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "A3")]
from A6_transport import (SupportScreen, assert_mother_disjoint, mother_state_check, assert_sector_free,  # noqa: E402
                          sector_of, sector_overlap_policy, fit_predict_domain, loso_masks, sector_masks,
                          FT_NUM, FT_CAT, SECTOR_FREE_CAT)
from preprocessing import FeatureManifest  # noqa: E402
from A3_target_encoder import encode_partitions, TESpec, TE_CELL, LEVEL  # noqa: E402

FT = FeatureManifest(FT_NUM, FT_CAT, "FT")
SF = FeatureManifest(FT_NUM, SECTOR_FREE_CAT, "FT")


def births(n_mothers=2500, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 4, n_mothers)
    caseid = np.repeat([f"m{i:05d}" for i in range(n_mothers)], sizes)
    state_of = dict(zip(np.unique(caseid), rng.choice(["1", "2", "3", "4"], n_mothers)))
    n = len(caseid)
    m15 = rng.choice([21, 22, 31, 32, 96], n, p=[.35, .25, .2, .17, .03])
    f = pd.DataFrame({"caseid": caseid, "nfhs5_state": pd.Series(caseid).map(state_of).to_numpy(), "m15": m15,
                      "maternal_age_at_index_birth": rng.uniform(16, 45, n),
                      "education_years": rng.integers(0, 18, n).astype(float),
                      "birth_order": rng.integers(1, 6, n).astype(float),
                      "wealth_index": rng.choice(list("12345"), n), "residence": rng.choice(["urban", "rural"], n),
                      "religion": rng.choice(["1", "2", "3"], n), "twin_order": rng.choice(["0", "1"], n, p=[.97, .03])})
    f["sector"] = sector_of(m15)
    f["facility_type"] = f["sector"]
    lp = -1.8 + 1.4 * (f.sector == "private") + 0.04 * f.education_years + 0.3 * (f.nfhs5_state == "2")
    f["m17"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    mothers = np.unique(caseid)
    part = dict(zip(mothers, rng.choice(["train", "validation", "test"], len(mothers), p=[.7, .15, .15])))
    f["partition"] = pd.Series(caseid).map(part).to_numpy()
    return f[f.partition != "test"].reset_index(drop=True)


class TestMotherExclusivity(unittest.TestCase):
    def test_zero_mother_overlap_geographic(self):
        f = births()
        mother_state_check(f.caseid, f.nfhs5_state)
        for s in ("1", "2", "3", "4"):
            m = loso_masks(f.nfhs5_state, f.partition, s)
            src = m["source_train"] | m["source_val"]
            assert_mother_disjoint(f.caseid, src, m["target_all"], s)
            self.assertFalse(set(f.caseid[src]) & set(f.caseid[m["target_all"]]))
            self.assertTrue(m["target_all"].sum() > 0 and m["source_train"].sum() > 0)

    def test_overlap_detected_and_state_anomaly_rejected(self):
        f = births()
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            assert_mother_disjoint(f.caseid, f.index < 100, (f.index >= 50) & (f.index < 150))
        bad = f.copy()
        dup = bad.caseid.duplicated(keep=False)
        first = bad.index[dup][0]
        bad.loc[first, "nfhs5_state"] = "9"
        with self.assertRaisesRegex(ValueError, "more than one state"):
            mother_state_check(bad.caseid, bad.nfhs5_state)

    def test_sector_overlap_policy_and_zero_overlap(self):
        f = births()
        keep, rep = sector_overlap_policy(f.caseid, f.sector)
        self.assertGreater(rep["mothers_with_public_and_private_births"], 0)  # fixture has both-sector mothers
        self.assertFalse(f.sector[keep].eq("other").any())
        for src, tgt in (("public", "private"), ("private", "public")):
            m = sector_masks(f.sector, f.partition, src, tgt, keep)
            assert_mother_disjoint(f.caseid, m["source_train"] | m["source_val"], m["target_all"], f"{src}->{tgt}")
        g = f[keep].groupby("caseid").sector.nunique()
        self.assertTrue(g.eq(1).all())
        with self.assertRaises(ValueError):
            sector_overlap_policy(f.caseid, f.sector, policy="assign_latest")
        with self.assertRaises(ValueError):
            sector_masks(f.sector, f.partition, "public", "public", keep)


class TestTargetOutcomeIsolation(unittest.TestCase):
    def _mutations(self, y, target):
        rng = np.random.default_rng(1)
        out = []
        for mode in ("flip", "permute", "ones"):
            z = y.copy()
            z[target] = (1 - z[target]) if mode == "flip" else (rng.permutation(z[target]) if mode == "permute" else 1)
            out.append(z)
        return out

    def test_geographic_target_state_labels_unused(self):
        f = births()
        y = f.m17.to_numpy()
        m = loso_masks(f.nfhs5_state, f.partition, "2")
        score = {"source_val": m["source_val"], "target_all": m["target_all"]}
        for algo in ("logistic_regression", "xgboost"):
            ref, thr = fit_predict_domain(f, y, source_train=m["source_train"], source_val=m["source_val"],
                                          score=score, manifest=FT, algorithm=algo)
            for z in self._mutations(y, m["target_all"]):
                out, thr2 = fit_predict_domain(f, z, source_train=m["source_train"], source_val=m["source_val"],
                                               score=score, manifest=FT, algorithm=algo)
                self.assertEqual(thr, thr2)
                for k in score:
                    self.assertEqual(out[k].tobytes(), ref[k].tobytes())

    def test_sector_target_labels_unused_both_directions(self):
        f = births()
        y = f.m17.to_numpy()
        keep, _ = sector_overlap_policy(f.caseid, f.sector)
        for src, tgt in (("public", "private"), ("private", "public")):
            m = sector_masks(f.sector, f.partition, src, tgt, keep)
            score = {"target_all": m["target_all"]}
            ref, _ = fit_predict_domain(f, y, source_train=m["source_train"], source_val=m["source_val"],
                                        score=score, manifest=SF, algorithm="logistic_regression")
            for z in self._mutations(y, m["target_all"]):
                out, _ = fit_predict_domain(f, z, source_train=m["source_train"], source_val=m["source_val"],
                                            score=score, manifest=SF, algorithm="logistic_regression")
                self.assertEqual(out["target_all"].tobytes(), ref["target_all"].tobytes())

    def test_target_encoding_for_held_out_state_uses_source_only(self):
        f = births()
        m = loso_masks(f.nfhs5_state, f.partition, "3")
        rows = m["source_train"] | m["source_val"] | m["target_all"]
        sub = f[rows]
        part = np.where(m["source_train"][rows], "train", "other")
        _, te = encode_partitions(sub, sub.m17, sub.caseid, part, TESpec())
        held = sub.nfhs5_state.eq("3").to_numpy()
        self.assertTrue((te[LEVEL][held] == "sector").all())  # unseen state -> source sector fallback
        z = sub.m17.copy()
        z[held] = 1 - z[held]
        _, te2 = encode_partitions(sub, z, sub.caseid, part, TESpec())
        self.assertEqual(te[TE_CELL].to_numpy().tobytes(), te2[TE_CELL].to_numpy().tobytes())

    def test_fit_refuses_target_rows_in_source(self):
        f = births()
        m = loso_masks(f.nfhs5_state, f.partition, "1")
        bad = m["source_train"] | m["target_all"]
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            fit_predict_domain(f, f.m17, source_train=bad, source_val=m["source_val"],
                               score={"target_all": m["target_all"]}, manifest=FT, algorithm="logistic_regression")


class TestFeaturePolicy(unittest.TestCase):
    def test_sector_free_policy(self):
        assert_sector_free(SF)
        self.assertNotIn("facility_type", SF.categorical)
        with self.assertRaisesRegex(ValueError, "sector-free"):
            assert_sector_free(FT)

    def test_constant_source_feature_and_unseen_category(self):
        f = births()
        f.loc[f.sector.eq("public"), "residence"] = "urban"  # constant within public source
        f.loc[f.sector.eq("private") & (f.index % 7 == 0), "religion"] = "96"  # level absent from public source
        keep, _ = sector_overlap_policy(f.caseid, f.sector)
        m = sector_masks(f.sector, f.partition, "public", "private", keep)
        preds, _ = fit_predict_domain(f, f.m17, source_train=m["source_train"], source_val=m["source_val"],
                                      score={"target_all": m["target_all"]}, manifest=SF, algorithm="xgboost")
        p = preds["target_all"]
        self.assertTrue(np.isfinite(p).all() and ((p > 0) & (p < 1)).all())

    def test_support_screen_boundaries(self):
        s = SupportScreen()
        self.assertEqual(s.check([1] * 30 + [0] * 170)["status"], "ESTIMABLE")
        self.assertEqual(s.check([1] * 29 + [0] * 171)["status"], "NOT_ESTIMABLE")
        self.assertEqual(s.check([1] * 30 + [0] * 169)["status"], "NOT_ESTIMABLE")
        self.assertIn("negatives<30", s.check([1] * 190 + [0] * 10)["reasons"])


if __name__ == "__main__":
    unittest.main()
