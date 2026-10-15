"""Synthetic tests: no DHS/NFHS microdata involved."""
import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_contracts import MAPPINGS, audit_births


def fixture(wave="NFHS-5"):
    d = {
        "caseid": ["mother1", "mother1", "mother2", "mother3"],
        "bidx": [1, 2, 1, 1],
        "m15": [21, 31, 96, 11],
        "m17": [1, 0, 0, 1],
        "v008": [1500] * 4,
        "b3": [1499, 1440, 1500, 1499],
        "v005": [1000000] * 4,
        "v001": [1, 1, 2, 3],
        "v022": [10, 10, 11, 12],
    }
    index = 0 if wave == "NFHS-4" else 1
    for name, sources in MAPPINGS.items():
        d[sources[index]] = [0, 1, 8 if name.startswith("pregnancy_") else 0, None]
    return pd.DataFrame(d)


class ContractTests(unittest.TestCase):
    def test_expected_counts_and_boundary(self):
        r = audit_births(fixture(), "NFHS-5")
        self.assertEqual(r["status"], "DATA_CONTRACT_VALID")
        self.assertEqual(r["institutional_rows"], 3)
        self.assertEqual(r["recall_exactly_60"], 1)
        self.assertEqual(r["recall_0_59"], 2)

    def test_nfhs4_remapped_columns(self):
        r = audit_births(fixture("NFHS-4"), "NFHS-4")
        self.assertEqual(r["status"], "DATA_CONTRACT_VALID")
        self.assertEqual(r["seven_mapping_checks"]["pregnancy_convulsions"]["raw_column"], "s432")

    def test_duplicate_key_rejected(self):
        f = fixture()
        f.loc[1, "bidx"] = 1
        self.assertEqual(audit_births(f, "NFHS-5")["status"], "INVALID")

    def test_invalid_outcome_rejected(self):
        f = fixture()
        f.loc[0, "m17"] = 8
        self.assertEqual(audit_births(f, "NFHS-5")["status"], "INVALID")

    def test_invalid_mapping_code_rejected(self):
        f = fixture()
        f.loc[0, "s420a"] = 7
        self.assertEqual(audit_births(f, "NFHS-5")["status"], "INVALID")

    def test_missing_column_rejected(self):
        self.assertEqual(audit_births(fixture().drop(columns=["s420a"]), "NFHS-5")["status"], "INVALID")


if __name__ == "__main__":
    unittest.main()
