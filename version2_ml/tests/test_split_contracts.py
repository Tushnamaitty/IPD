import unittest
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_contracts import grouped_partition, audit_partition, guard_wave_transfer


def demo():
    # Each mother has repeated births; both outcome classes present in every sizable subset.
    return pd.DataFrame(
        [(f"mom{i:03d}", int((i + j) % 2)) for i in range(100) for j in range(2)],
        columns=["caseid", "m17"])


class SplitTests(unittest.TestCase):
    def test_group_exclusive(self):
        d = demo()
        p = grouped_partition(d)
        report = audit_partition(d, p)
        self.assertEqual(report["status"], "GROUP_EXCLUSIVE_VALID")
        self.assertFalse(report["independent_final_test_certified"])

    def test_reproducible_after_shuffle(self):
        d = demo()
        a = grouped_partition(d)
        shuffled = d.sample(frac=1, random_state=10)
        b = grouped_partition(shuffled)
        x = dict(zip(d.caseid, a))
        y = dict(zip(shuffled.caseid, b))
        self.assertEqual(x, y)

    def test_reject_mother_leakage(self):
        d = demo()
        p = grouped_partition(d).copy()
        p.iloc[1] = "test" if p.iloc[0] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            audit_partition(d, p)

    def test_reject_missing_group(self):
        d = demo()
        d.loc[0, "caseid"] = None
        with self.assertRaises(ValueError):
            grouped_partition(d)

    def test_reject_invalid_outcome(self):
        d = demo()
        d.loc[0, "m17"] = 8
        with self.assertRaises(ValueError):
            grouped_partition(d)

    def test_wave_direction(self):
        x = guard_wave_transfer(pd.Series(["NFHS-4"]), pd.Series(["NFHS-5"]),
                                adaptation_ids=["a"], evaluation_ids=["b"])
        self.assertTrue(x["adaptation_disjoint_checked"])

    def test_wave_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            guard_wave_transfer(pd.Series(["NFHS-4"]), pd.Series(["NFHS-5"]),
                                adaptation_ids=["a"], evaluation_ids=["a"])

    def test_wrong_wave_rejected(self):
        with self.assertRaises(ValueError):
            guard_wave_transfer(pd.Series(["NFHS-5"]), pd.Series(["NFHS-4"]))


if __name__ == "__main__":
    unittest.main()
