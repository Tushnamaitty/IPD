import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_framework import AuthorizedClassifier, evaluate_binary, make_estimator
from preprocessing import FeatureManifest


def fixtures():
    n = 80
    frame = pd.DataFrame({
        "age_at_birth": [19 + i % 25 for i in range(n)],
        "facility": ["public" if i % 2 else "private" for i in range(n)],
    })
    y = np.array([int((i % 3) == 0 or i % 5 == 0) for i in range(n)])
    return frame, y


class ModelFrameworkTests(unittest.TestCase):
    def test_training_and_aggregate_metrics(self):
        x, y = fixtures()
        model = AuthorizedClassifier(FeatureManifest(("age_at_birth",), ("facility",), "F1"))
        model.fit(x.iloc[:60], y[:60], partition="train")
        p = model.predict_proba(x.iloc[60:], partition="validation")
        r = evaluate_binary(y[60:], p)
        self.assertGreaterEqual(r["roc_auc"], 0)
        self.assertLessEqual(r["roc_auc"], 1)
        self.assertFalse(r["inference_validated"])

    def test_reject_nontrain_fitting(self):
        x, y = fixtures()
        model = AuthorizedClassifier(FeatureManifest(("age_at_birth",), ("facility",), "F1"))
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            model.fit(x, y, partition="test")

    def test_prediction_before_fit(self):
        x, _ = fixtures()
        model = AuthorizedClassifier(FeatureManifest(("age_at_birth",), (), "F1"))
        with self.assertRaises(ValueError):
            model.predict_proba(x, partition="test")

    def test_metrics_known_perfect_predictions(self):
        r = evaluate_binary([0, 1, 0, 1], [.1, .9, .2, .8], threshold=.5)
        self.assertAlmostEqual(r["roc_auc"], 1.0)
        self.assertAlmostEqual(r["average_precision"], 1.0)
        self.assertAlmostEqual(r["precision"], 1.0)
        self.assertAlmostEqual(r["recall"], 1.0)
        self.assertAlmostEqual(r["f1"], 1.0)

    def test_metrics_bad_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_binary([0, 1], [0.1, 1.2])

    def test_single_class_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_binary([1, 1], [0.5, 0.9])

    def test_invalid_weights_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_binary([0, 1], [0.1, 0.9], sample_weight=[1, -1])

    def test_optional_models_are_explicit(self):
        with self.assertRaises(ValueError):
            make_estimator("not_a_model")


if __name__ == "__main__":
    unittest.main()
