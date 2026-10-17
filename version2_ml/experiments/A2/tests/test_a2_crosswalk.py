"""A2 crosswalk tests against the published NFHS-4/NFHS-5 v024 codebook labels."""
import sys
import unittest
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from state_crosswalk import (load_crosswalk, validate_crosswalk, nfhs4_unit_map, nfhs5_unit_map,  # noqa: E402
                             with_unseen, assign_units, canonical)

TABLE = HERE.parent / "state_crosswalk_v1.csv"
# Codebook value labels for v024 (public DHS recode documentation, no microdata).
N4 = {1: "andaman and nicobar islands", 2: "andhra pradesh", 3: "arunachal pradesh", 4: "assam",
      5: "bihar", 6: "chandigarh", 7: "chhattisgarh", 8: "dadra and nagar haveli", 9: "daman and diu",
      10: "goa", 11: "gujarat", 12: "haryana", 13: "himachal pradesh", 14: "jammu and kashmir",
      15: "jharkhand", 16: "karnataka", 17: "kerala", 18: "lakshadweep", 19: "madhya pradesh",
      20: "maharashtra", 21: "manipur", 22: "meghalaya", 23: "mizoram", 24: "nagaland", 25: "delhi",
      26: "odisha", 27: "puducherry", 28: "punjab", 29: "rajasthan", 30: "sikkim", 31: "tamil nadu",
      32: "tripura", 33: "uttar pradesh", 34: "uttarakhand", 35: "west bengal", 36: "telangana"}
N5 = {1: "jammu & kashmir", 2: "himachal pradesh", 3: "punjab", 4: "chandigarh", 5: "uttarakhand",
      6: "haryana", 7: "nct of delhi", 8: "rajasthan", 9: "uttar pradesh", 10: "bihar", 11: "sikkim",
      12: "arunachal pradesh", 13: "nagaland", 14: "manipur", 15: "mizoram", 16: "tripura",
      17: "meghalaya", 18: "assam", 19: "west bengal", 20: "jharkhand", 21: "odisha", 22: "chhattisgarh",
      23: "madhya pradesh", 24: "gujarat", 25: "dadra & nagar haveli and daman & diu", 27: "maharashtra",
      28: "andhra pradesh", 29: "karnataka", 30: "goa", 31: "lakshadweep", 32: "kerala", 33: "tamil nadu",
      34: "puducherry", 35: "andaman & nicobar islands", 36: "telangana", 37: "ladakh"}


class TestCrosswalk(unittest.TestCase):
    def setUp(self):
        self.table = load_crosswalk(TABLE)

    def test_versioned_table_validates(self):
        report = validate_crosswalk(self.table, N4, N5)
        self.assertEqual(report["status"], "CROSSWALK_VALID")
        self.assertEqual(report["nfhs5_codes"], 36)
        self.assertEqual(report["source_units"], 35)  # DNH+DD pooled
        self.assertFalse(report["numeric_codes_joined_directly"])

    def test_numeric_codes_differ_but_names_join(self):
        u4, u5 = nfhs4_unit_map(self.table), nfhs5_unit_map(self.table)
        self.assertEqual(u4[35], u5[19])  # West Bengal
        self.assertNotEqual(u4.get(19), u5[19])  # NFHS-4 code 19 is Madhya Pradesh

    def test_boundary_changes_explicit(self):
        u4, u5 = nfhs4_unit_map(self.table), nfhs5_unit_map(self.table)
        self.assertEqual(u4[8], u4[9])
        self.assertEqual(u5[25], u4[8])
        self.assertEqual(u5[37], u5[1])
        self.assertEqual(u5[37], u4[14])

    def test_missing_nfhs5_code_fails(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_crosswalk(self.table[self.table.nfhs5_code != "19"], N4, N5)

    def test_mislabelled_row_fails(self):
        bad = self.table.copy()
        bad.loc[bad.nfhs5_code == "19", "nfhs4_codes"] = "19"
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            validate_crosswalk(bad, N4, N5)

    def test_dta_label_drift_fails(self):
        drifted = {**N5, 19: "bengal west"}
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            validate_crosswalk(self.table, N4, drifted)

    def test_nfhs4_code_cannot_map_to_two_units(self):
        bad = self.table.copy()
        bad.loc[bad.nfhs5_code == "37", "source_unit"] = "ladakh_only"
        with self.assertRaisesRegex(ValueError, "two source units"):
            validate_crosswalk(bad, N4, N5)

    def test_uncovered_nfhs4_code_fails(self):
        bad = with_unseen(self.table, [24])  # Gujarat now unmapped -> NFHS-4 11 uncovered
        with self.assertRaisesRegex(ValueError, "not covered"):
            validate_crosswalk(bad, N4, N5)

    def test_ladakh_unseen_variant(self):
        variant = with_unseen(self.table, [37])
        validate_crosswalk(variant, N4, N5)
        units = assign_units(pd.Series([37, 1, 19]), nfhs5_unit_map(variant))
        self.assertTrue(pd.isna(units.iloc[0]))
        self.assertEqual(units.iloc[1], "jammu_and_kashmir_incl_ladakh")

    def test_unknown_code_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "absent from versioned table"):
            assign_units(pd.Series([19, 26]), nfhs5_unit_map(self.table))
        with self.assertRaisesRegex(ValueError, "absent"):
            assign_units(pd.Series([19, None]), nfhs5_unit_map(self.table))

    def test_canonical_aliases(self):
        self.assertEqual(canonical("NCT of Delhi"), "delhi")
        self.assertEqual(canonical("Jammu & Kashmir"), canonical("jammu and kashmir"))


if __name__ == "__main__":
    unittest.main()
