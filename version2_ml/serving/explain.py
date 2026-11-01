"""Per-request SHAP explanation aggregated to user-facing fields.

TreeExplainer on the fitted XGBoost estimator in margin (log-odds) space, as in
A7. Encoded columns are mapped to their source columns with a7_shap.family_map
(fitted encoder categories), then source columns are merged into the fields
the user filled in. Contributions describe the uncalibrated model margin; Platt
scaling is monotone with slope close to 1, so signs and ranking carry over.
Attributions are predictive, not causal.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .predictor import Predictor, get_predictor
from .schemas import WEALTH_LABELS, PredictRequest

from a7_shap import explained_feature_names, family_map  # noqa: E402

TOP_K = 5

# source model column -> user-facing field
FIELD_BY_COLUMN = {
    "maternal_age_at_index_birth": "maternal_age_at_index_birth",
    "education_years": "education_years",
    "wealth_index": "wealth_index",
    "birth_order": "birth_order",
    "twin_order": "twin_order",
    "residence": "residence",
    "religion": "religion",
    "facility_type": "place_of_delivery",
    "facility_detail": "place_of_delivery",
    "history_four_level": "previous_csection_history",
    "has_any_observed_earlier_institutional_birth": "previous_csection_history",
    "history_quality_flag": "previous_csection_history",
}
FIELD_NAMES = {
    "maternal_age_at_index_birth": "Mother's age",
    "education_years": "Education",
    "wealth_index": "Household wealth",
    "birth_order": "Birth order",
    "twin_order": "Multiple birth",
    "residence": "Residence",
    "religion": "Religion",
    "place_of_delivery": "Place of delivery",
    "previous_csection_history": "Previous C-section",
}
PREVIOUS_CS_VALUES = {None: "First birth", "yes": "Yes", "no": "No", "not_sure": "Not sure"}


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:]


def display_values(request: PredictRequest, config: dict) -> dict[str, str]:
    labels = config["display_labels"]
    return {
        "maternal_age_at_index_birth": f"{request.maternal_age_at_index_birth} years",
        "education_years": f"{request.education_years} years",
        "wealth_index": WEALTH_LABELS[request.wealth_index],
        "birth_order": str(request.birth_order),
        "twin_order": _sentence_case(request.twin_order),
        "residence": labels["residence"][request.residence],
        "religion": _sentence_case(labels["religion"][request.religion]),
        "place_of_delivery": _sentence_case(labels["facility_detail"][request.facility_detail]),
        "previous_csection_history": PREVIOUS_CS_VALUES[request.previous_csection],
    }


class Explainer:
    def __init__(self, predictor: Predictor):
        import shap

        classifier = predictor.classifier
        self.predictor = predictor
        self.preprocessor = classifier.preprocessor
        self.explainer = shap.TreeExplainer(classifier.estimator)
        names = explained_feature_names(classifier)
        families = family_map(classifier)
        missing = sorted(set(names) - set(families))
        if missing:
            raise RuntimeError(f"Encoded features without a family mapping: {missing}")
        self.fields = np.array([FIELD_BY_COLUMN[families[name]] for name in names])

    def field_contributions(self, frame) -> dict[str, float]:
        values = np.asarray(self.explainer.shap_values(self.preprocessor.transform(frame)), dtype=float)
        if values.ndim != 2 or values.shape != (len(frame), len(self.fields)):
            raise RuntimeError("Unexpected SHAP value shape")
        row = values[0]
        return {field: float(row[self.fields == field].sum()) for field in FIELD_NAMES}

    def top_factors(self, request: PredictRequest, frame, k: int = TOP_K) -> list[dict]:
        contributions = self.field_contributions(frame)
        shown = display_values(request, self.predictor.config)
        ordered = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:k]
        factors = []
        for field, contribution in ordered:
            direction = "raises risk" if contribution > 0 else "lowers risk"
            factors.append({
                "feature": field,
                "label": f"{FIELD_NAMES[field]}: {shown[field]} - {direction}",
                "value": shown[field],
                "direction": direction,
                "contribution": round(contribution, 4),
            })
        return factors


@lru_cache(maxsize=1)
def get_explainer() -> Explainer:
    return Explainer(get_predictor())
