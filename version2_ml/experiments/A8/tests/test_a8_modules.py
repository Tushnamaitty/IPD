"""A8 synthetic tests. No NFHS records are read or required."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[3]
for _entry in (
    _BASE,
    _BASE / "experiments" / "common",
    _BASE / "experiments" / "A1" / "src",
    _BASE / "experiments" / "A3",
    _BASE / "experiments" / "A4" / "src",
    _BASE / "experiments" / "A7" / "src",
    _BASE / "experiments" / "A8" / "src",
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from a1_state_context import STATE_FEATURE, map_states  # noqa: E402
from A3_target_encoder import TE_CELL  # noqa: E402
from a7_candidates import TIERS, carve_calibration_partition  # noqa: E402
from a8_arms import ARM_ORDER, added_feature_count, arm_frame, arm_manifest, encode_state_sector, te_key_frame  # noqa: E402
from a8_decision import decide  # noqa: E402
from a8_outputs import write_csv, write_manifest  # noqa: E402
from a8_pipeline import paired_deltas, run_seed  # noqa: E402
from split_contracts import grouped_partition  # noqa: E402

PARAMS = {
    "xgboost": {"n_estimators": 15, "max_depth": 3, "learning_rate": 0.2, "subsample": 0.8,
                "colsample_bytree": 0.8, "eval_metric": "logloss", "tree_method": "hist"},
    "logistic_regression": {"C": 1.0, "solver": "liblinear", "max_iter": 1000, "class_weight": "balanced"},
}
TE = {"alpha": 5, "n_min": 5, "inner_folds": 3, "inner_seed": 42}
RULE = {
    "model": "xgboost", "comparator": "H0", "adopt_arm_if": "test rule",
    "min_delta_roc_auc": 0.005, "min_delta_average_precision": 0.010, "tie_break": "H1",
    "tier_names": {"H0": "F1_detail_history", "H1": "F1_detail_history_state",
                   "H2": "F1_detail_history_state_sector_te", "H3": "F1_detail_history_state_plus_state_sector_te"},
}
STATE_CODES = (1, 9, 10, 27, 33)


def synthetic_cohort(n_mothers=900, seed=0):
    rng = np.random.default_rng(seed)
    births = rng.integers(1, 3, n_mothers)
    caseid = np.repeat([f"m{i:05d}" for i in range(n_mothers)], births)
    n = len(caseid)
    state = np.repeat(rng.choice(STATE_CODES, n_mothers), births)
    facility = rng.choice(["public", "private", "other"], n, p=(0.6, 0.35, 0.05))
    bord = rng.integers(1, 5, n)
    history = np.where(bord == 1, "first_recorded_live_birth",
                       rng.choice(["observed_previous_institutional_cs",
                                   "higher_order_no_observed_previous_institutional_cs",
                                   "higher_order_earlier_institutional_history_unobserved"], n))
    base = pd.DataFrame({
        "maternal_age_at_index_birth": rng.uniform(16, 45, n),
        "education_years": rng.integers(0, 21, n).astype(float),
        "wealth_index": rng.integers(1, 6, n).astype(str),
        "birth_order": bord.astype(float),
        "facility_type": facility,
        "residence": rng.choice(["urban", "rural"], n),
        "religion": rng.choice(["1", "2", "3"], n),
        "twin_order": "0",
        "facility_detail": np.where(facility == "public", "public: govt./munic. hospital",
                                    np.where(facility == "private", "private: hospital/maternity home/clinic", "other")),
        "history_four_level": history,
        "has_any_observed_earlier_institutional_birth": np.where(bord == 1, "false", "true"),
        "history_quality_flag": "ok",
    })
    state_effect = pd.Series(state).map({1: -0.8, 9: -0.4, 10: 0.0, 27: 0.5, 33: 0.9}).to_numpy()
    logit = -1.2 + 1.3 * (facility == "private") + 2.0 * (history == "observed_previous_institutional_cs") + state_effect
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cohort = pd.DataFrame({"caseid": caseid, "m17": y, "v024": state})
    return cohort, base, map_states(state).to_numpy()


def synthetic_split(cohort, seed=42):
    partition = grouped_partition(cohort, group_col="caseid", target_col="m17", fractions=(0.7, 0.15, 0.15), seed=seed)
    train = np.flatnonzero(partition.eq("train").to_numpy())
    carve = carve_calibration_partition(cohort.iloc[train].reset_index(drop=True), calibration_fraction=0.2, seed=4208)
    return {"fit": train[carve.eq("fit").to_numpy()], "calib": train[carve.eq("calibration").to_numpy()],
            "valid": np.flatnonzero(partition.eq("validation").to_numpy()),
            "test": np.flatnonzero(partition.eq("test").to_numpy())}


def run(cohort, base, states, split, y=None, models=("xgboost", "logistic_regression"), arms=ARM_ORDER):
    return run_seed(base, states, cohort["v024"].to_numpy(), cohort["m17"].to_numpy() if y is None else y,
                    cohort["caseid"].to_numpy(), fit_positions=split["fit"], calib_positions=split["calib"],
                    valid_positions=split["valid"], models=list(models), params=PARAMS, te_config=TE,
                    estimator_seed=42, arms=arms)


class ArmTests(unittest.TestCase):
    def test_manifests_extend_a7_tier(self):
        base = TIERS["F1_detail_history"]
        h0 = arm_manifest("H0")
        self.assertEqual(h0.numerical, tuple(base["numerical"]))
        self.assertEqual(h0.categorical, tuple(base["categorical"]))
        self.assertEqual(arm_manifest("H1").categorical[-1], STATE_FEATURE)
        self.assertEqual(arm_manifest("H2").numerical[-1], TE_CELL)
        self.assertIn(STATE_FEATURE, arm_manifest("H3").categorical)
        self.assertIn(TE_CELL, arm_manifest("H3").numerical)
        self.assertEqual([added_feature_count(a) for a in ARM_ORDER], [0, 1, 1, 2])

    def test_arm_frame_order_and_alignment(self):
        _, base, states = synthetic_cohort(50)
        frame = arm_frame(base, states, np.full(len(base), 0.2), "H3")
        manifest = arm_manifest("H3")
        self.assertEqual(list(frame.columns), list(manifest.numerical) + list(manifest.categorical))
        with self.assertRaises(ValueError):
            arm_frame(base, states, np.full(len(base) - 1, 0.2), "H2")
        with self.assertRaises(ValueError):
            arm_frame(base, states, np.full(len(base), np.nan), "H2")


class EncoderLeakageTests(unittest.TestCase):
    def setUp(self):
        self.cohort, self.base, _ = synthetic_cohort()
        self.split = synthetic_split(self.cohort)
        rows = np.concatenate([self.split["fit"], self.split["calib"], self.split["valid"]])
        self.rows = rows
        self.fit_mask = np.zeros(len(rows), dtype=bool)
        self.fit_mask[:len(self.split["fit"])] = True
        sub = self.base.iloc[rows].reset_index(drop=True)
        self.keys = te_key_frame(self.cohort["v024"].to_numpy()[rows], sub["facility_type"], sub.index)
        self.y = self.cohort["m17"].to_numpy()[rows]
        self.groups = self.cohort["caseid"].to_numpy()[rows]

    def encode(self, y, keys=None, groups=None, fit_mask=None):
        return encode_state_sector(self.keys if keys is None else keys, y,
                                   self.groups if groups is None else groups,
                                   self.fit_mask if fit_mask is None else fit_mask,
                                   alpha=TE["alpha"], n_min=TE["n_min"], n_splits=3, seed=42)

    def test_full_table_fitted_on_fit_pool_only(self):
        encoder, values = self.encode(self.y)
        self.assertEqual(encoder.full_table.tables["n"], int(self.fit_mask.sum()))
        self.assertTrue(np.isfinite(values).all())

    def test_permuting_held_out_labels_leaves_encoder_bit_identical(self):
        encoder, values = self.encode(self.y)
        mutated = self.y.copy()
        held = ~self.fit_mask
        mutated[held] = 1 - np.random.default_rng(1).permutation(mutated[held])
        encoder_m, values_m = self.encode(mutated)
        self.assertEqual(encoder.full_table.digest(), encoder_m.full_table.digest())
        self.assertEqual(values.tobytes(), values_m.tobytes())

    def test_removing_held_out_rows_leaves_encoder_bit_identical(self):
        encoder, values = self.encode(self.y)
        n_fit = int(self.fit_mask.sum())
        keys_fit = self.keys.iloc[:n_fit]
        encoder_r, values_r = self.encode(self.y[:n_fit], keys=keys_fit, groups=self.groups[:n_fit],
                                          fit_mask=np.ones(n_fit, dtype=bool))
        self.assertEqual(encoder.full_table.digest(), encoder_r.full_table.digest())
        self.assertEqual(values[:n_fit].tobytes(), values_r.tobytes())
        held = self.keys.iloc[n_fit:].reset_index(drop=True)
        self.assertEqual(values[n_fit:].tobytes(), encoder_r.transform(held)[TE_CELL].to_numpy().tobytes())

    def test_fit_labels_do_change_the_encoder(self):
        encoder, _ = self.encode(self.y)
        mutated = self.y.copy()
        mutated[:50] = 1 - mutated[:50]
        self.assertNotEqual(encoder.full_table.digest(), self.encode(mutated)[0].full_table.digest())


class PipelineLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort, cls.base, cls.states = synthetic_cohort()
        cls.split = synthetic_split(cls.cohort)
        cls.reference = run(cls.cohort, cls.base, cls.states, cls.split)

    def mutated(self, positions):
        y = self.cohort["m17"].to_numpy().copy()
        y[positions] = 1 - np.random.default_rng(3).permutation(y[positions])
        return y

    def test_validation_labels_do_not_reach_features_or_predictions(self):
        result = run(self.cohort, self.base, self.states, self.split, y=self.mutated(self.split["valid"]))
        self.assertEqual(result["te_digest"], self.reference["te_digest"])
        self.assertEqual(result["te_values"].tobytes(), self.reference["te_values"].tobytes())
        for key, probs in self.reference["probabilities"].items():
            self.assertEqual(result["probabilities"][key]["base"].tobytes(), probs["base"].tobytes(), key)
            self.assertEqual(result["probabilities"][key]["calibrated"].tobytes(), probs["calibrated"].tobytes(), key)

    def test_calibration_labels_reach_only_the_calibrator(self):
        result = run(self.cohort, self.base, self.states, self.split, y=self.mutated(self.split["calib"]))
        self.assertEqual(result["te_digest"], self.reference["te_digest"])
        self.assertEqual(result["te_values"].tobytes(), self.reference["te_values"].tobytes())
        changed = 0
        for key, probs in self.reference["probabilities"].items():
            self.assertEqual(result["probabilities"][key]["base"].tobytes(), probs["base"].tobytes(), key)
            changed += result["probabilities"][key]["calibrated"].tobytes() != probs["calibrated"].tobytes()
        self.assertGreater(changed, 0)

    def test_test_rows_are_never_read(self):
        y = self.cohort["m17"].to_numpy().copy()
        y[self.split["test"]] = 9  # invalid label: any use would raise
        base = self.base.copy()
        base.loc[self.split["test"], "facility_type"] = "never_seen"
        result = run(self.cohort, base, self.states, self.split, y=y, models=("xgboost",), arms=("H3",))
        ref = self.reference["probabilities"][("xgboost", "H3")]["calibrated"]
        self.assertEqual(result["probabilities"][("xgboost", "H3")]["calibrated"].tobytes(), ref.tobytes())

    def test_overlapping_positions_rejected(self):
        bad = dict(self.split, calib=np.concatenate([self.split["calib"], self.split["valid"][:3]]))
        with self.assertRaises(ValueError):
            run(self.cohort, self.base, self.states, bad, models=("xgboost",), arms=("H0",))

    def test_metric_rows_cover_every_model_and_arm(self):
        rows = pd.DataFrame(self.reference["rows"])
        self.assertEqual(len(rows), 8)
        self.assertTrue(rows[["roc_auc", "average_precision", "brier_score", "calibration_slope"]].notna().all().all())
        self.assertTrue(rows["threshold"].between(0.05, 0.95).all())


class DeltaAndDecisionTests(unittest.TestCase):
    def metrics(self, auc, ap, seeds=(42, 7, 2026, 99), digest="d"):
        rows = []
        for seed in seeds:
            for arm in ARM_ORDER:
                for model in ("xgboost", "logistic_regression"):
                    rows.append({"seed": seed, "model": model, "arm": arm, "valid_digest": digest,
                                 "roc_auc": 0.80 + auc.get(arm, 0.0), "average_precision": 0.60 + ap.get(arm, 0.0),
                                 "brier_score": 0.13, "calibration_intercept": 0.0, "calibration_slope": 1.0,
                                 "threshold": 0.3, "precision": 0.5, "recall": 0.6, "f1": 0.55})
        return pd.DataFrame(rows)

    def test_paired_delta_arithmetic(self):
        deltas = paired_deltas(self.metrics({"H1": 0.01}, {"H1": 0.02}))
        h1 = deltas[(deltas.arm == "H1") & (deltas.model == "xgboost")]
        self.assertTrue(np.allclose(h1["delta_roc_auc"], 0.01))
        self.assertTrue(np.allclose(h1["delta_average_precision"], 0.02))
        self.assertEqual(len(deltas), 4 * 2 * 3)

    def test_mismatched_rows_rejected(self):
        frame = self.metrics({}, {})
        frame.loc[frame.index[0], "valid_digest"] = "other"
        with self.assertRaises(ValueError):
            paired_deltas(frame)

    def decide(self, auc, ap):
        return decide(paired_deltas(self.metrics(auc, ap)), RULE, (42, 7, 2026, 99))

    def test_none_pass_keeps_f1_detail_history(self):
        decision = self.decide({"H1": 0.004, "H2": 0.02, "H3": 0.02}, {"H1": 0.02, "H2": 0.009, "H3": 0.0})
        self.assertEqual(decision["selected_tier"], "F1_detail_history")
        self.assertEqual(decision["passed_arms"], [])

    def test_fewest_features_wins(self):
        decision = self.decide({"H2": 0.006, "H3": 0.02}, {"H2": 0.011, "H3": 0.03})
        self.assertEqual(decision["selected_arm"], "H2")

    def test_tie_goes_to_h1(self):
        decision = self.decide({"H1": 0.006, "H2": 0.02, "H3": 0.02}, {"H1": 0.011, "H2": 0.03, "H3": 0.03})
        self.assertEqual(decision["selected_arm"], "H1")
        self.assertEqual(decision["passed_arms"], ["H1", "H2", "H3"])

    def test_one_failing_seed_blocks_adoption(self):
        frame = self.metrics({"H1": 0.01}, {"H1": 0.02})
        frame.loc[(frame.seed == 99) & (frame.arm == "H1") & (frame.model == "xgboost"), "roc_auc"] = 0.803
        decision = decide(paired_deltas(frame), RULE, (42, 7, 2026, 99))
        self.assertEqual(decision["selected_tier"], "F1_detail_history")
        self.assertFalse(decision["per_seed_deltas"]["H1"]["99"]["passes"])

    def test_lr_deltas_do_not_drive_decision(self):
        frame = self.metrics({}, {})
        lr = frame.model == "logistic_regression"
        frame.loc[lr & (frame.arm == "H1"), ["roc_auc", "average_precision"]] = [0.9, 0.8]
        self.assertEqual(decide(paired_deltas(frame), RULE, (42, 7, 2026, 99))["selected_arm"], "H0")

    def test_missing_seed_raises(self):
        with self.assertRaises(ValueError):
            decide(paired_deltas(self.metrics({}, {}, seeds=(42, 7, 2026))), RULE, (42, 7, 2026, 99))


class OutputTests(unittest.TestCase):
    def test_manifest_hashes_match_lf_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            write_csv(pd.DataFrame({"arm": ["H0", "H1"], "roc_auc": [0.8, 0.81]}), out / "a.csv")
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("x: 1\n", encoding="utf-8")
            manifest = write_manifest(out, experiment_id="T", config={"x": 1}, config_path=config_path,
                                      repo_root=_BASE.parent, seeds=[42], split_manifest_hashes={42: "h"})
            entry = manifest["files"][0]
            data = (out / "a.csv").read_bytes()
            self.assertNotIn(b"\r\n", data)
            self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(json.loads((out / "MANIFEST.json").read_text())["seeds"], [42])

    def test_manifest_refuses_row_level_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_csv(pd.DataFrame({"caseid": ["m1"], "probability": [0.2]}), out / "bad.csv")
            with self.assertRaises(ValueError):
                write_manifest(out, experiment_id="T", config={}, config_path=out / "bad.csv",
                               repo_root=_BASE.parent, seeds=[42], split_manifest_hashes={})

    def test_support_rows_suppress_small_counts(self):
        from run_a8 import _support_rows

        screen = {"cells": [{"cell": "a | urban", "n_rows": 250, "n_positive": 7, "n_negative": 243, "passes": False}]}
        row = _support_rows(screen, 10).iloc[0]
        self.assertEqual(row["n_train_positive"], "<10 (suppressed)")
        self.assertEqual(row["n_train_rows"], 250)


if __name__ == "__main__":
    unittest.main()
