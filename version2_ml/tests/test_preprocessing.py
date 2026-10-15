"""Synthetic checks only. Requires pandas, numpy and scikit-learn."""
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import FeatureManifest, TrainOnlyPreprocessor, map_seven, SEVEN_MAPPINGS


class FeaturePipelineTests(unittest.TestCase):
    def test_crosswave_mapping(self):
        for wave in ("NFHS-4", "NFHS-5"):
            cols = {code[wave]: [0, 1, 8 if name.startswith(("convulsions_during", "swelling_during")) else None]
                    for name, code in SEVEN_MAPPINGS.items()}
            mapped = map_seven(pd.DataFrame(cols), wave)
            self.assertEqual(mapped["counselled_vaginal_bleeding"].iloc[0], "no")
            self.assertEqual(mapped["convulsions_during_pregnancy"].iloc[2], "unknown")

    def test_unexpected_code_rejected(self):
        cols = {v["NFHS-5"]: [0] for v in SEVEN_MAPPINGS.values()}
        cols["s420a"] = [9]
        with self.assertRaises(ValueError):
            map_seven(pd.DataFrame(cols), "NFHS-5")

    def test_train_only_median_and_unseen_category(self):
        manifest = FeatureManifest(("maternal_age",), ("facility_type",), "F0")
        train = pd.DataFrame({"maternal_age": [20.0, 30.0, None],
                              "facility_type": ["public", "private", None]})
        heldout = pd.DataFrame({"maternal_age": [None, 999.0],
                                "facility_type": ["other_unseen", "private"]})
        pre = TrainOnlyPreprocessor(manifest)
        with self.assertRaises(ValueError):
            pre.fit(heldout, partition="test")
        pre.fit(train, partition="train")
        transformed = pre.transform(heldout)
        # First column is numeric; missing heldout age imputed using TRAIN median 25.
        self.assertEqual(float(transformed[0, 0]), 25.0)
        self.assertEqual(float(transformed[1, 0]), 999.0)
        self.assertEqual(transformed.shape[0], 2)

    def test_transform_before_fit_rejected(self):
        x = pd.DataFrame({"age": [20]})
        with self.assertRaises(ValueError):
            TrainOnlyPreprocessor(FeatureManifest(("age",), (), "F1")).transform(x)

    def test_outcome_or_design_feature_rejected(self):
        for col in ("m17", "v005", "caseid"):
            with self.assertRaises(ValueError):
                FeatureManifest((col,), (), "F0").validate(pd.DataFrame({col: [1]}))

    def test_duplicate_manifest_rejected(self):
        with self.assertRaises(ValueError):
            FeatureManifest(("age",), ("age",), "F1").validate(pd.DataFrame({"age": [20]}))


if __name__ == "__main__":
    unittest.main()
