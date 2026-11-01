"""A4 observed-history counterexample tests. No real NFHS data is used."""
import sys
import unittest
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parents[3]
for _entry in (
    str(_BASE),
    str(_BASE / "experiments" / "common"),
    str(_BASE / "experiments" / "A4" / "src"),
):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from a4_history import (
    FIRST_RECORDED_LIVE_BIRTH,
    HAS_ANY_FEATURE,
    HISTORY_FEATURE,
    NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
    OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
    QUALITY_AMBIGUOUS,
    QUALITY_FEATURE,
    QUALITY_INCONSISTENT,
    QUALITY_MISSING_DATES,
    QUALITY_OK,
    UNKNOWN_HISTORY,
    UNOBSERVED_HISTORY,
    build_history_features,
    history_support_counts,
)

ROSTER_COLUMNS = ["caseid", "bidx", "b3", "b19", "m15", "m17"]
INDEX_COLUMNS = ["caseid", "bidx", "b3", "bord", "m15"]


def _roster(rows):
    return pd.DataFrame(rows, columns=ROSTER_COLUMNS)


def _index(rows):
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


class HistoryBuilderTests(unittest.TestCase):
    def _build(self, roster_rows, index_rows, **kwargs):
        return build_history_features(
            _roster(roster_rows), _index(index_rows), **kwargs
        )

    def test_single_birth_mother_is_first(self):
        features, diagnostics = self._build(
            [("m1", 1, 1000, 10, 21, 1)],
            [("m1", 1, 1000, 1, 21)],
        )
        row = features.iloc[0]
        self.assertEqual(row[HISTORY_FEATURE], FIRST_RECORDED_LIVE_BIRTH)
        self.assertEqual(row[HAS_ANY_FEATURE], "false")
        self.assertEqual(row[QUALITY_FEATURE], QUALITY_OK)
        self.assertEqual(diagnostics["n_index_rows"], 1)

    def test_prior_home_birth_is_unobserved_category(self):
        features, _ = self._build(
            [
                ("m1", 1, 1000, 30, 11, None),
                ("m1", 2, 1030, 5, 21, 0),
            ],
            [("m1", 2, 1030, 2, 21)],
        )
        self.assertEqual(features.iloc[0][HISTORY_FEATURE], UNOBSERVED_HISTORY)

    def test_prior_institutional_non_cs(self):
        features, _ = self._build(
            [
                ("m1", 1, 1000, 30, 21, 0),
                ("m1", 2, 1030, 5, 21, 0),
            ],
            [("m1", 2, 1030, 2, 21)],
        )
        row = features.iloc[0]
        self.assertEqual(row[HISTORY_FEATURE], NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS)
        self.assertEqual(row[HAS_ANY_FEATURE], "true")

    def test_prior_institutional_cs(self):
        features, _ = self._build(
            [
                ("m1", 1, 1000, 30, 31, 1),
                ("m1", 2, 1030, 5, 21, 0),
            ],
            [("m1", 2, 1030, 2, 21)],
        )
        self.assertEqual(features.iloc[0][HISTORY_FEATURE], OBSERVED_PREVIOUS_INSTITUTIONAL_CS)

    def test_higher_order_without_earlier_records(self):
        features, _ = self._build(
            [("m1", 1, 1000, 10, 21, 0)],
            [("m1", 1, 1000, 3, 21)],
        )
        self.assertEqual(features.iloc[0][HISTORY_FEATURE], UNOBSERVED_HISTORY)

    def test_same_month_twins_never_count_as_previous(self):
        roster_rows = [
            ("m1", 1, 1000, 10, 21, 1),
            ("m1", 2, 1000, 10, 21, 0),
        ]
        index_rows = [
            ("m1", 1, 1000, 1, 21),
            ("m1", 2, 1000, 2, 21),
        ]
        features, _ = self._build(roster_rows, index_rows)
        self.assertEqual(features.iloc[0][HISTORY_FEATURE], FIRST_RECORDED_LIVE_BIRTH)
        second = features.iloc[1]
        self.assertEqual(second[HISTORY_FEATURE], UNOBSERVED_HISTORY)
        self.assertEqual(second[HAS_ANY_FEATURE], "false")

    def test_duplicate_keys_raise(self):
        with self.assertRaises(ValueError):
            self._build(
                [
                    ("m1", 1, 1000, 10, 21, 0),
                    ("m1", 1, 1000, 10, 21, 0),
                ],
                [("m1", 1, 1000, 1, 21)],
            )

    def test_missing_dates_give_explicit_unknown(self):
        features, _ = self._build(
            [
                ("m1", 1, None, 30, 21, 0),
                ("m1", 2, 1030, 5, 21, 0),
            ],
            [
                ("m1", 2, 1030, 2, 21),
                ("m1", 1, None, 1, 21),
            ],
        )
        second_birth = features.iloc[0]
        self.assertEqual(second_birth[HISTORY_FEATURE], UNKNOWN_HISTORY)
        self.assertEqual(second_birth[HAS_ANY_FEATURE], "unknown")
        self.assertIn(QUALITY_MISSING_DATES, second_birth[QUALITY_FEATURE])
        first_birth = features.iloc[1]
        self.assertEqual(first_birth[HISTORY_FEATURE], FIRST_RECORDED_LIVE_BIRTH)
        self.assertIn(QUALITY_MISSING_DATES, first_birth[QUALITY_FEATURE])

    def test_missing_prior_outcome_is_ambiguous_not_negative(self):
        for coded in (None, 9):
            features, _ = self._build(
                [
                    ("m1", 1, 1000, 30, 21, coded),
                    ("m1", 2, 1030, 5, 21, 0),
                ],
                [("m1", 2, 1030, 2, 21)],
            )
            row = features.iloc[0]
            self.assertEqual(row[HISTORY_FEATURE], UNKNOWN_HISTORY)
            self.assertIn(QUALITY_AMBIGUOUS, row[QUALITY_FEATURE])
            self.assertEqual(row[HAS_ANY_FEATURE], "true")

    def test_invalid_outcome_code_raises(self):
        with self.assertRaises(ValueError):
            self._build(
                [
                    ("m1", 1, 1000, 30, 21, 5),
                    ("m1", 2, 1030, 5, 21, 0),
                ],
                [("m1", 2, 1030, 2, 21)],
            )

    def test_unexpected_facility_code_raises(self):
        with self.assertRaises(ValueError):
            self._build(
                [
                    ("m1", 1, 1000, 30, 99, 0),
                    ("m1", 2, 1030, 5, 21, 0),
                ],
                [("m1", 2, 1030, 2, 21)],
            )

    def test_temporal_leakage_future_and_same_month_changes(self):
        roster_rows = [
            ("m1", 1, 1000, 30, 21, 1),
            ("m1", 2, 1010, 5, 21, 0),
        ]
        index_rows = [
            ("m1", 1, 1000, 1, 21),
            ("m1", 2, 1010, 2, 21),
        ]
        baseline, _ = self._build(roster_rows, index_rows)
        mutated_rows = [
            ("m1", 1, 1000, 30, 21, 1),
            ("m1", 2, 1010, 5, 11, 1),
            ("m1", 3, 1000, 10, 21, 1),
        ]
        mutated, _ = self._build(mutated_rows, index_rows)
        self.assertEqual(baseline.iloc[0][HISTORY_FEATURE], FIRST_RECORDED_LIVE_BIRTH)
        self.assertTrue(
            baseline.iloc[0][[HISTORY_FEATURE, HAS_ANY_FEATURE, QUALITY_FEATURE]].equals(
                mutated.iloc[0][[HISTORY_FEATURE, HAS_ANY_FEATURE, QUALITY_FEATURE]]
            )
        )
        self.assertEqual(baseline.iloc[1][HISTORY_FEATURE], OBSERVED_PREVIOUS_INSTITUTIONAL_CS)

    def test_m15_96_rules(self):
        roster_rows = [
            ("m1", 1, 1000, 30, 96, 0),
            ("m1", 2, 1030, 5, 21, 0),
        ]
        index_rows = [("m1", 2, 1030, 2, 21)]
        included, _ = self._build(roster_rows, index_rows)
        self.assertEqual(
            included.iloc[0][HISTORY_FEATURE], NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS
        )
        self.assertEqual(included.iloc[0][HAS_ANY_FEATURE], "true")
        excluded, _ = self._build(
            roster_rows, index_rows, include_other_unknown_sector=False
        )
        self.assertEqual(excluded.iloc[0][HISTORY_FEATURE], UNOBSERVED_HISTORY)
        self.assertEqual(excluded.iloc[0][HAS_ANY_FEATURE], "false")

    def test_index_96_rows_rejected_under_exclusion_rule(self):
        with self.assertRaises(ValueError):
            self._build(
                [("m1", 1, 1000, 30, 96, 0), ("m1", 2, 1030, 5, 96, 0)],
                [("m1", 2, 1030, 2, 96)],
                include_other_unknown_sector=False,
            )

    def test_bord1_inconsistency_flagged(self):
        features, _ = self._build(
            [
                ("m1", 1, 1000, 30, 21, 1),
                ("m1", 2, 1030, 5, 21, 0),
            ],
            [("m1", 2, 1030, 1, 21)],
        )
        row = features.iloc[0]
        self.assertEqual(row[HISTORY_FEATURE], FIRST_RECORDED_LIVE_BIRTH)
        self.assertIn(QUALITY_INCONSISTENT, row[QUALITY_FEATURE])

    def test_outside_window_prior_is_ignored(self):
        features, _ = self._build(
            [
                ("m1", 1, 900, 70, 31, 1),
                ("m1", 2, 1030, 5, 21, 0),
            ],
            [("m1", 2, 1030, 2, 21)],
        )
        self.assertEqual(features.iloc[0][HISTORY_FEATURE], UNOBSERVED_HISTORY)

    def test_index_missing_from_roster_raises(self):
        with self.assertRaises(RuntimeError):
            self._build(
                [("m1", 1, 1000, 10, 21, 0)],
                [("m1", 9, 1200, 2, 21)],
            )

    def test_index_non_institutional_raises(self):
        with self.assertRaises(ValueError):
            self._build(
                [("m1", 1, 1000, 10, 21, 0)],
                [("m1", 1, 1000, 1, 11)],
            )

    def test_wave_guard(self):
        with self.assertRaises(NotImplementedError):
            self._build(
                [("m1", 1, 1000, 10, 21, 0)],
                [("m1", 1, 1000, 1, 21)],
                wave="NFHS-4",
            )

    def test_support_counts(self):
        features, _ = self._build(
            [
                ("m1", 1, 1000, 10, 21, 0),
                ("m1", 2, 1030, 5, 21, 1),
                ("m2", 1, 1000, 10, 21, 1),
            ],
            [
                ("m1", 1, 1000, 1, 21),
                ("m1", 2, 1030, 2, 21),
                ("m2", 1, 1000, 1, 21),
            ],
        )
        rows = history_support_counts(features, [0, 1, 1])
        first = next(
            row for row in rows
            if row["group"] == HISTORY_FEATURE and row["value"] == FIRST_RECORDED_LIVE_BIRTH
        )
        self.assertEqual(first["n_births"], 2)
        self.assertEqual(first["n_positive"], 1)
        with self.assertRaises(ValueError):
            history_support_counts(features, [0, 2, 1])


if __name__ == "__main__":
    unittest.main()
