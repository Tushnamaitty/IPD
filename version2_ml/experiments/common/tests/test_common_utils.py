"""Tests for experiments/common utilities. Synthetic fixtures only."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_COMMON = Path(__file__).resolve().parents[1]
_BASE = Path(__file__).resolve().parents[3]
for _entry in (str(_COMMON), str(_BASE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from aggregate_schema import (
    COLUMNS,
    NOT_APPLICABLE,
    build_row,
    suppress_small_count,
    write_rows,
)
from checksums import assert_matched_membership, membership_digest
from feature_builder import make_core_features
from manifests import feature_manifest_hash, split_manifest_hash, stable_hash
from synthetic import synthetic_birth_frame


class AggregateSchemaTests(unittest.TestCase):
    def test_columns_locked(self):
        self.assertEqual(len(COLUMNS), 31)
        self.assertEqual(COLUMNS[0], "experiment_id")
        self.assertEqual(COLUMNS[-1], "split_manifest_hash")
        self.assertIn("uncertainty_status", COLUMNS)
        self.assertIn("calibration_intercept", COLUMNS)

    def test_build_row_fills_not_applicable(self):
        row = build_row(experiment_id="A1", seed=42)
        self.assertEqual(tuple(row.keys()), COLUMNS)
        self.assertEqual(row["experiment_id"], "A1")
        self.assertEqual(row["wave"], NOT_APPLICABLE)
        with self.assertRaisesRegex(ValueError, "Unknown schema fields"):
            build_row(experiment_id="A1", mystery=1)
        with self.assertRaisesRegex(ValueError, "None"):
            build_row(experiment_id="A1", roc_auc=None)
        with self.assertRaisesRegex(ValueError, "NaN"):
            build_row(experiment_id="A1", roc_auc=float("nan"))

    def test_write_rows_locked_order(self):
        rows = [build_row(experiment_id="A1", seed=42)]
        with tempfile.TemporaryDirectory() as folder:
            path = write_rows(rows, Path(folder) / "out.csv")
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(COLUMNS))
                loaded = next(reader)
            self.assertEqual(loaded["experiment_id"], "A1")
            self.assertEqual(loaded["seed"], "42")

    def test_suppress_small_count(self):
        self.assertEqual(suppress_small_count(0), 0)
        self.assertEqual(suppress_small_count(9), "<10 (suppressed)")
        self.assertEqual(suppress_small_count(10), 10)
        self.assertEqual(suppress_small_count(9, threshold=5), 9)
        with self.assertRaises(ValueError):
            suppress_small_count(5, threshold=1)


class ManifestTests(unittest.TestCase):
    def test_stable_hash_deterministic_and_order_independent(self):
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))
        self.assertNotEqual(stable_hash({"a": 1}), stable_hash({"a": 2}))

    def test_feature_manifest_hash_sensitive(self):
        first = feature_manifest_hash(
            tier="F1", numerical=("a",), categorical=("b",), builder_id="x"
        )
        same = feature_manifest_hash(
            tier="F1", numerical=("a",), categorical=("b",), builder_id="x"
        )
        changed = feature_manifest_hash(
            tier="F2", numerical=("a",), categorical=("b",), builder_id="x"
        )
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)

    def test_split_manifest_hash_order_independent(self):
        pairs = [("m2", "train"), ("m1", "test"), ("m3", "validation")]
        first = split_manifest_hash(
            split_id="s", seed=42, fractions=(0.7, 0.15, 0.15), assignments=pairs
        )
        shuffled = split_manifest_hash(
            split_id="s",
            seed=42,
            fractions=(0.7, 0.15, 0.15),
            assignments=list(reversed(pairs)),
        )
        self.assertEqual(first, shuffled)
        reseeded = split_manifest_hash(
            split_id="s", seed=43, fractions=(0.7, 0.15, 0.15), assignments=pairs
        )
        self.assertNotEqual(first, reseeded)


class MembershipDigestTests(unittest.TestCase):
    def test_order_independent_and_length(self):
        keys = ["c|1", "a|2", "b|3"]
        digest = membership_digest(keys, seed=42)
        self.assertEqual(digest, membership_digest(reversed(keys), seed=42))
        self.assertEqual(len(digest), 64)
        self.assertNotEqual(digest, membership_digest(["c|1", "a|2"], seed=42))

    def test_matched_assertion(self):
        digest = membership_digest(["k1", "k2"], seed=42)
        assert_matched_membership(digest, digest)
        with self.assertRaisesRegex(ValueError, "MATCHED-ROW VIOLATION"):
            assert_matched_membership(digest, membership_digest(["k1"], seed=42))
        with self.assertRaises(ValueError):
            assert_matched_membership(digest)
        with self.assertRaises(ValueError):
            assert_matched_membership(digest, "")


class SyntheticFixtureTests(unittest.TestCase):
    def test_deterministic_and_valid(self):
        first = synthetic_birth_frame(n_mothers=25, seed=5)
        second = synthetic_birth_frame(n_mothers=25, seed=5)
        self.assertTrue(first.equals(second))
        self.assertEqual(first["caseid"].nunique(), 25)
        self.assertEqual(int(first.duplicated(["caseid", "bidx"]).sum()), 0)
        self.assertTrue(first["m17"].isin([0, 1]).all())
        self.assertEqual(set(first["m17"].unique()), {0, 1})
        self.assertTrue(first["b19"].between(0, 59).all())
        core = make_core_features(first, wave="NFHS-5")
        self.assertEqual(len(core), len(first))


if __name__ == "__main__":
    unittest.main()
