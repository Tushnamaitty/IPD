"""A3 runner contract: the three mandatory arms are nested exactly as specified."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run  # noqa: E402
from A3_target_encoder import TE_CELL  # noqa: E402


def names(arm):
    m = run.arm_manifest(arm)
    return set(m.numerical) | set(m.categorical)


class TestA3Arms(unittest.TestCase):
    def test_three_required_arms_nested(self):
        i, ii, iii = (names(a) for a in run.REQUIRED)
        self.assertEqual(len(i), 8)
        self.assertIn("facility_type", i)  # ordinary sector indicator already in the core
        self.assertEqual(ii - i, {"nfhs5_state"})
        self.assertEqual(iii - ii, {TE_CELL})

    def test_no_te_in_reference_arms(self):
        for arm in ("i_core", "ii_state_sector_indicators"):
            self.assertFalse(any(n.startswith("te_") for n in names(arm)))

    def test_required_arms_registered(self):
        for arm in run.REQUIRED:
            self.assertIn(arm, run.ARMS)
            self.assertTrue(run.ARMS[arm]["role"].startswith("required"))


if __name__ == "__main__":
    unittest.main()
