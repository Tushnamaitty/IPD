"""A4 facility-detail mapping tests. No real NFHS data is read or required."""
import sys
import unittest
from pathlib import Path

_BASE = Path(__file__).resolve().parents[3]
for _entry in (
    str(_BASE),
    str(_BASE / "experiments" / "common"),
    str(_BASE / "experiments" / "A4" / "src"),
):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from a4_facility_detail import (
    BROAD_CLASS,
    DETAIL_FEATURE,
    FACILITY_DETAIL_LABELS,
    INSTITUTIONAL_CODES,
    KNOWN_CODES,
    NONINSTITUTIONAL_HOME_CODES,
    crosswalk_rows,
    facility_broad_class,
    facility_detail,
)
from data_access import INSTITUTIONAL_CODES as SHARED_INSTITUTIONAL


class FacilityDetailMappingTests(unittest.TestCase):
    def test_mapping_covers_every_known_code(self):
        self.assertEqual(set(FACILITY_DETAIL_LABELS), set(KNOWN_CODES))
        self.assertEqual(len(KNOWN_CODES), 17)
        self.assertEqual(
            set(FACILITY_DETAIL_LABELS),
            set(NONINSTITUTIONAL_HOME_CODES) | set(INSTITUTIONAL_CODES),
        )

    def test_shared_institutional_rule_is_identical(self):
        self.assertEqual(set(INSTITUTIONAL_CODES), set(SHARED_INSTITUTIONAL))

    def test_verbatim_labels(self):
        self.assertEqual(FACILITY_DETAIL_LABELS[21], "public: govt./munic. hospital")
        self.assertEqual(FACILITY_DETAIL_LABELS[31], "private: hospital/maternity home/clinic")
        self.assertEqual(FACILITY_DETAIL_LABELS[96], "other")
        self.assertEqual(FACILITY_DETAIL_LABELS[11], "respondent's home")
        self.assertEqual(FACILITY_DETAIL_LABELS[24], "public: chc/rural hospital/block phc")

    def test_detail_mapping_values(self):
        result = facility_detail([21, 96, 10, 33])
        self.assertEqual(result.name, DETAIL_FEATURE)
        self.assertEqual(
            list(result),
            [
                "public: govt./munic. hospital",
                "other",
                "home",
                "ngo or trust hospital/clinic",
            ],
        )

    def test_broad_class_mapping(self):
        self.assertEqual(list(facility_broad_class([20, 31, 96, 13])),
                         ["public", "private", "other_unknown_sector", "home"])

    def test_unexpected_code_raises(self):
        with self.assertRaises(ValueError):
            facility_detail([21, 99])
        with self.assertRaises(ValueError):
            facility_broad_class([21, 5])

    def test_non_integer_and_non_numeric_raise(self):
        with self.assertRaises(ValueError):
            facility_detail([21.5])
        with self.assertRaises(ValueError):
            facility_detail(["public"])

    def test_missing_code_raises(self):
        with self.assertRaises(ValueError):
            facility_detail([21, None])

    def test_wave_guard(self):
        with self.assertRaises(NotImplementedError):
            facility_detail([21], wave="NFHS-4")
        with self.assertRaises(NotImplementedError):
            facility_broad_class([21], wave="NFHS-4")

    def test_crosswalk_rows(self):
        rows = crosswalk_rows()
        nfhs5 = [row for row in rows if row["wave"] == "NFHS-5"]
        pending = [row for row in rows if row["wave"] == "NFHS-4"]
        self.assertEqual(len(nfhs5), 17)
        self.assertEqual(len(pending), 1)
        codes = [row["m15_code"] for row in nfhs5]
        self.assertEqual(codes, sorted(codes))
        self.assertTrue(all(row["verification_source"] for row in nfhs5))


if __name__ == "__main__":
    unittest.main()
