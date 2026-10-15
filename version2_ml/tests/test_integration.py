"""IPD Version 2 Phase 3 Step 6 — synthetic end-to-end integration.

This test does not read licensed NFHS data, certify inferential validity, or
freeze Phase 3 for real-data analyses.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_contracts import MAPPINGS, audit_births
from split_contracts import grouped_partition, audit_partition, guard_wave_transfer
from preprocessing import FeatureManifest, TrainOnlyPreprocessor, map_seven
from model_framework import AuthorizedClassifier, evaluate_binary


def synthetic_births(n_mothers=140):
    records = []
    for i in range(n_mothers):
        for j in range(2):
            y = int((i * 7 + j * 11) % 5 in (0, 1))
            row = {
                "caseid": f"synthetic_{i:04d}", "bidx": j + 1,
                "m15": 21 if i % 2 == 0 else 31,
                "m17": y, "v008": 1500,
                "b3": 1500 - (j * 6 + 1),
                "v005": 1000000, "v001": i // 10 + 1,
                "v022": i // 20 + 1,
                "maternal_age_at_birth_demo": 19 + i % 22,
                "facility_demo": "public" if i % 2 == 0 else "private",
            }
            for name, pair in MAPPINGS.items():
                row[pair[1]] = int((i + j) % 2)
            records.append(row)
    return pd.DataFrame(records)


class Phase3IntegrationTests(unittest.TestCase):
    def test_full_pipeline_leakage_guards_and_evaluation(self):
        df = synthetic_births()
        contract = audit_births(df, "NFHS-5")
        self.assertEqual(contract["status"], "DATA_CONTRACT_VALID")

        splits = grouped_partition(df, fractions=(0.7, 0.15, 0.15), seed=42)
        verified = audit_partition(df, splits)
        self.assertEqual(verified["status"], "GROUP_EXCLUSIVE_VALID")
        self.assertFalse(verified["independent_final_test_certified"])

        mapped = map_seven(df, "NFHS-5")
        self.assertEqual(len(mapped), len(df))
        self.assertEqual(set(mapped["counselled_vaginal_bleeding"].unique()), {"no", "yes"})

        manifest = FeatureManifest(("maternal_age_at_birth_demo",), ("facility_demo",), "F1")
        train = df.loc[splits == "train"]
        valid = df.loc[splits == "validation"]
        test = df.loc[splits == "test"]
        self.assertTrue(len(train) and len(valid) and len(test))
        model = AuthorizedClassifier(manifest, algorithm="logistic_regression")
        model.fit(train, train["m17"], partition="train")
        pred = model.predict_proba(valid, partition="validation")
        metrics = evaluate_binary(valid["m17"], pred)
        self.assertEqual(metrics["n"], len(valid))
        self.assertFalse(metrics["inference_validated"])
        self.assertTrue(0 <= metrics["roc_auc"] <= 1)
        self.assertTrue(np.isfinite(metrics["brier_score"]))
        # Deliberately no test-set model selection or model promotion.
        self.assertGreater(len(test), 0)

    def test_fit_rejected_on_validation(self):
        df = synthetic_births()
        splits = grouped_partition(df)
        valid = df.loc[splits == "validation"]
        clf = AuthorizedClassifier(FeatureManifest(("maternal_age_at_birth_demo",), (), "F1"))
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            clf.fit(valid, valid["m17"], partition="validation")

    def test_wave_transfer_disjointness(self):
        report = guard_wave_transfer(pd.Series(["NFHS-4"] * 10),
                                     pd.Series(["NFHS-5"] * 8),
                                     adaptation_ids=["ad1", "ad2"],
                                     evaluation_ids=["ev1", "ev2"])
        self.assertTrue(report["adaptation_disjoint_checked"])
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            guard_wave_transfer(pd.Series(["NFHS-4"]),
                                pd.Series(["NFHS-5"]),
                                adaptation_ids=["same"], evaluation_ids=["same"])

    def test_transformed_validation_uses_training_median(self):
        manifest = FeatureManifest(("age",), ("facility",), "F1")
        train = pd.DataFrame({"age": [20.0, 30.0, None], "facility": ["public", "private", "public"]})
        val = pd.DataFrame({"age": [None], "facility": ["not_seen"]})
        processor = TrainOnlyPreprocessor(manifest).fit(train, partition="train")
        x = processor.transform(val)
        self.assertEqual(float(x[0, 0]), 25.0)


if __name__ == "__main__":
    unittest.main()
