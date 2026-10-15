"""Synthetic contract tests, not an empirical survey design validation."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from variance_prototype import kish_effective_n,validate_design,resample_psu_indices

class TestVariancePrototype(unittest.TestCase):
    def test_integer_weight_overflow_prevented(self):
        self.assertAlmostEqual(kish_effective_n([10_000_000,10_000_000]),2)
    def test_design_counts(self):
        d=validate_design([1,1,1,2,2],[1,1,2,1,1],[1]*5)
        self.assertEqual(d["singleton_strata"],1)
        self.assertEqual(d["n_psu_pairs"],3)
    def test_singleton_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError,"Singleton"):
            list(resample_psu_indices([1,1,2],[1,2,1],repetitions=3))
    def test_explicit_singleton_policy(self):
        samples=list(resample_psu_indices([1,1,2],[1,2,1],repetitions=5,singleton_policy="certainty"))
        self.assertEqual(len(samples),5)
        self.assertTrue(all(2 in s for s in samples))
    def test_seed_reproducible(self):
        args=([1,1,2,2],[1,2,3,4])
        a=list(resample_psu_indices(*args,repetitions=4,seed=12))
        b=list(resample_psu_indices(*args,repetitions=4,seed=12))
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))
if __name__=="__main__":unittest.main()
