"""A1 synthetic acceptance tests. No real NFHS data is read or required."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[3]
for _entry in (
    str(_BASE),
    str(_BASE / "experiments" / "common"),
    str(_BASE / "experiments" / "A1" / "src"),
):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from a1_state_context import (
    A1_FEATURE_NAMES,
    A1_INTERACTION_FEATURE_NAMES,
    F1_CATEGORICAL,
    F1_FEATURES,
    F1_NUMERICAL,
    INTERACTION_FEATURE,
    STATE_FEATURE,
    STATE_LABELS,
    UNKNOWN_STATE_TOKEN,
    assert_external_context_specs,
    assert_unique_birth_keys,
    build_state_frame,
    map_states,
    support_screen,
)
from checksums import assert_matched_membership, membership_digest
from feature_builder import CORE_NAMES, make_core_features
from manifests import split_manifest_hash
from model_framework import AuthorizedClassifier
from preprocessing import FeatureManifest
from split_contracts import audit_partition, grouped_partition
from synthetic import synthetic_birth_frame


def _core_and_states(n_mothers=8, seed=7):
    births = synthetic_birth_frame(n_mothers=n_mothers, seed=seed)
    core = make_core_features(births, wave="NFHS-5")
    states = map_states(births["v024"])
    return core, states


def _manual_core():
    return pd.DataFrame(
        {
            "maternal_age_at_index_birth": [25.0],
            "education_years": [10.0],
            "wealth_index": ["3"],
            "birth_order": [2.0],
            "facility_type": ["public"],
            "residence": ["urban"],
            "religion": ["1"],
            "twin_order": ["0"],
        }
    )


class StateCodingTests(unittest.TestCase):
    def test_verified_table_is_the_nfhs5_36_units(self):
        self.assertEqual(len(STATE_LABELS), 36)
        self.assertEqual(
            set(STATE_LABELS), set(range(1, 26)) | set(range(27, 38))
        )
        self.assertEqual(len(set(STATE_LABELS.values())), 36)
        self.assertNotIn(UNKNOWN_STATE_TOKEN, set(STATE_LABELS.values()))

    def test_map_states_missing_and_unknown(self):
        mapped = map_states(pd.Series([1.0, np.nan]))
        self.assertEqual(mapped.iloc[0], "jammu & kashmir")
        self.assertEqual(mapped.iloc[1], UNKNOWN_STATE_TOKEN)
        with self.assertRaisesRegex(ValueError, "Unmapped"):
            map_states(pd.Series([999]))
        with self.assertRaisesRegex(ValueError, "integer"):
            map_states(pd.Series([1.5]))

    def test_map_states_row_order_independent(self):
        forward = map_states(pd.Series([9, 1, 21])).tolist()
        backward = map_states(pd.Series([21, 1, 9])).tolist()
        self.assertEqual(forward, list(reversed(backward)))


class FeatureContractTests(unittest.TestCase):
    def test_feature_orders_locked(self):
        self.assertEqual(F1_FEATURES, CORE_NAMES)
        self.assertEqual(A1_FEATURE_NAMES, CORE_NAMES + (STATE_FEATURE,))
        self.assertEqual(
            A1_INTERACTION_FEATURE_NAMES,
            CORE_NAMES + (STATE_FEATURE, INTERACTION_FEATURE),
        )

    def test_build_state_frame_appends_and_preserves_core(self):
        core, states = _core_and_states()
        frame = build_state_frame(core, states)
        self.assertEqual(tuple(frame.columns), A1_FEATURE_NAMES)
        self.assertEqual(tuple(core.columns), CORE_NAMES)
        interacted = build_state_frame(core, states, include_interaction=True)
        self.assertEqual(tuple(interacted.columns), A1_INTERACTION_FEATURE_NAMES)
        for state, residence, token in zip(
            interacted[STATE_FEATURE],
            interacted["residence"],
            interacted[INTERACTION_FEATURE],
        ):
            self.assertEqual(token, f"{state} | {residence}")

    def test_build_state_frame_rejects_bad_inputs(self):
        core, states = _core_and_states()
        poisoned = core.copy()
        poisoned["m17"] = 1
        with self.assertRaisesRegex(ValueError, "eight CORE_NAMES"):
            build_state_frame(poisoned, states)
        shuffled_states = states.iloc[::-1]
        with self.assertRaisesRegex(ValueError, "aligned"):
            build_state_frame(core, shuffled_states)
        bad_labels = states.astype("string").copy()
        bad_labels.iloc[0] = "atlantis"
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            build_state_frame(core, bad_labels)

    def test_no_target_dependent_context(self):
        core, states = _core_and_states()
        self.assertNotIn("m17", core.columns)
        frame = build_state_frame(core, states)
        births = synthetic_birth_frame(n_mothers=8, seed=7)
        births["m17"] = 1 - births["m17"]
        rebuilt_core = make_core_features(births, wave="NFHS-5")
        rebuilt = build_state_frame(rebuilt_core, map_states(births["v024"]))
        self.assertTrue(frame.equals(rebuilt))

    def test_unseen_state_ignored_train_only_onehot(self):
        manifest = FeatureManifest(
            F1_NUMERICAL, F1_CATEGORICAL + (STATE_FEATURE,), "F2"
        )
        train = build_state_frame(_manual_core(), pd.Series(["sikkim"]))
        valid = build_state_frame(_manual_core(), pd.Series(["ladakh"]))
        preprocessor = __import__("preprocessing").TrainOnlyPreprocessor(manifest)
        preprocessor.fit(train, partition="train")
        transformed_train = preprocessor.transform(train)
        transformed_valid = preprocessor.transform(valid)
        self.assertEqual(transformed_train.shape[1], transformed_valid.shape[1])
        self.assertEqual(transformed_valid.shape[0], 1)

    def test_row_order_independent_feature_join(self):
        core, states = _core_and_states(n_mothers=10)
        frame = build_state_frame(core, states)
        order = np.random.RandomState(3).permutation(len(core))
        expected = frame.iloc[order].reset_index(drop=True)
        shuffled = build_state_frame(
            core.iloc[order].reset_index(drop=True),
            states.iloc[order].reset_index(drop=True),
        )
        self.assertTrue(expected.equals(shuffled))


class UniquenessAndGateTests(unittest.TestCase):
    def test_duplicate_birth_keys_rejected(self):
        births = synthetic_birth_frame(n_mothers=10)
        duplicate = pd.concat([births, births.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Repeated birth keys"):
            assert_unique_birth_keys(duplicate)
        missing = births.copy()
        missing.loc[0, "caseid"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing birth key"):
            assert_unique_birth_keys(missing)

    def test_support_screen_gate_outcomes(self):
        good = pd.DataFrame(
            {
                "cell": ["good"] * 400,
                "y": [1] * 150 + [0] * 250,
            }
        )
        passing = support_screen(
            good["cell"], good["y"], min_rows=200, min_class=30
        )
        self.assertTrue(passing["gate_passed"])
        mixed = pd.concat(
            [
                good,
                pd.DataFrame({"cell": ["small"] * 50, "y": [1] * 5 + [0] * 45}),
            ],
            ignore_index=True,
        )
        failing = support_screen(
            mixed["cell"], mixed["y"], min_rows=200, min_class=30
        )
        self.assertFalse(failing["gate_passed"])
        self.assertIn("small", failing["reason"])
        empty = support_screen(pd.Series([], dtype="string"), pd.Series([], dtype=int), min_rows=200, min_class=30)
        self.assertFalse(empty["gate_passed"])
        self.assertEqual(empty["reason"], "no_cells")
        with self.assertRaises(ValueError):
            support_screen(
                pd.Series(["a", "a"]), pd.Series([1.0, np.nan]), min_rows=1, min_class=1
            )

    def test_external_context_provenance_gate(self):
        self.assertEqual(
            assert_external_context_specs([])["status"], "NO_EXTERNAL_CONTEXT_DECLARED"
        )
        with self.assertRaisesRegex(ValueError, "missing provenance fields"):
            assert_external_context_specs([{}])
        complete = {
            "variable": "x",
            "geo_level": "state",
            "source": "example",
            "reference_year": 2015,
            "coverage": "all states",
            "join_key": "v024",
            "post_outcome_timing_risk": "low",
            "missingness_plan": "explicit fallback",
            "fallback": "drop descriptor",
        }
        self.assertEqual(
            assert_external_context_specs([complete])["status"], "PROVENANCE_COMPLETE"
        )

    def test_no_fit_on_validation_partition(self):
        births = synthetic_birth_frame(n_mothers=40)
        core = make_core_features(births, wave="NFHS-5")
        classifier = AuthorizedClassifier(
            FeatureManifest(F1_NUMERICAL, F1_CATEGORICAL, "F1")
        )
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            classifier.fit(
                core, births["m17"], partition="validation"
            )


class MatchedSplitTests(unittest.TestCase):
    def _partitioned(self, seed=42):
        births = synthetic_birth_frame(n_mothers=150)
        partition = grouped_partition(
            births,
            group_col="caseid",
            target_col="m17",
            fractions=(0.7, 0.15, 0.15),
            seed=seed,
        )
        audit = audit_partition(births, partition, group_col="caseid", target_col="m17")
        self.assertEqual(audit["status"], "GROUP_EXCLUSIVE_VALID")
        return births, partition, audit

    def test_group_exclusive_split_and_matched_digests(self):
        births, partition, _ = self._partitioned()
        keys = (births["caseid"].astype("string") + "|" + births["bidx"].astype("string")).to_numpy()
        train_mask = partition.eq("train").to_numpy()
        digest = membership_digest(keys[train_mask], seed=42)
        self.assertEqual(digest, membership_digest(list(reversed(keys[train_mask])), seed=42))
        assert_matched_membership(digest, digest)
        truncated = membership_digest(keys[train_mask][:10], seed=42)
        with self.assertRaisesRegex(ValueError, "MATCHED-ROW VIOLATION"):
            assert_matched_membership(digest, truncated)

    def test_split_manifest_hash_row_order_independent(self):
        births, partition, _ = self._partitioned(seed=42)
        first = split_manifest_hash(
            split_id="s", seed=42, fractions=(0.7, 0.15, 0.15),
            assignments=zip(births["caseid"], partition),
        )
        order = np.random.RandomState(11).permutation(len(births))
        shuffled_births = births.iloc[order].reset_index(drop=True)
        shuffled_partition = grouped_partition(
            shuffled_births, group_col="caseid", target_col="m17",
            fractions=(0.7, 0.15, 0.15), seed=42,
        )
        second = split_manifest_hash(
            split_id="s", seed=42, fractions=(0.7, 0.15, 0.15),
            assignments=zip(shuffled_births["caseid"], shuffled_partition),
        )
        self.assertEqual(first, second)
        other_seed, other_partition, _ = self._partitioned(seed=43)
        third = split_manifest_hash(
            split_id="s", seed=43, fractions=(0.7, 0.15, 0.15),
            assignments=zip(other_seed["caseid"], other_partition),
        )
        self.assertNotEqual(first, third)


if __name__ == "__main__":
    unittest.main()
