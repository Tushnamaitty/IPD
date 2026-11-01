"""PredictRequest -> one-row model frame in the trained feature order.

The request is first written back as a single NFHS-5 raw-code row so the
unchanged training transforms produce the features:
feature_builder.make_core_features (age, wealth, facility_type, residence,
religion, twin_order encodings) and a4_facility_detail.facility_detail. The
three history columns use the a4_history category constants directly.
"""
from __future__ import annotations

import pandas as pd

from .artifacts import load_config
from .schemas import PredictRequest

from a4_facility_detail import FACILITY_DETAIL_LABELS, facility_detail  # noqa: E402
from a4_history import (  # noqa: E402
    FIRST_RECORDED_LIVE_BIRTH,
    HAS_ANY_FALSE,
    HAS_ANY_FEATURE,
    HAS_ANY_TRUE,
    HISTORY_FEATURE,
    NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
    OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
    QUALITY_FEATURE,
    QUALITY_OK,
    UNOBSERVED_HISTORY,
)
from a7_candidates import build_tier_frame  # noqa: E402
from feature_builder import make_core_features  # noqa: E402

TIER = "F1_detail_history"
M15_BY_DETAIL = {label: code for code, label in FACILITY_DETAIL_LABELS.items()}
V025_BY_RESIDENCE = {"urban": 1, "rural": 2}
B0_BY_TWIN = {"single": 0, "multiple": 1}

# has_any follows a4_history: "true" iff an earlier institutional birth is observed.
HISTORY_BY_ANSWER = {
    (True, None): (FIRST_RECORDED_LIVE_BIRTH, HAS_ANY_FALSE),
    (False, "yes"): (OBSERVED_PREVIOUS_INSTITUTIONAL_CS, HAS_ANY_TRUE),
    (False, "no"): (NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS, HAS_ANY_TRUE),
    (False, "not_sure"): (UNOBSERVED_HISTORY, HAS_ANY_FALSE),
}


def _raw_row(request: PredictRequest) -> pd.DataFrame:
    # Maternal age = (b3 - v011) / 12 in make_core_features; v011 = 0 makes it exact.
    return pd.DataFrame([{
        "b3": request.maternal_age_at_index_birth * 12,
        "v011": 0,
        "v133": request.education_years,
        "v190": request.wealth_index,
        "bord": request.birth_order,
        "m15": M15_BY_DETAIL[request.facility_detail],
        "v025": V025_BY_RESIDENCE[request.residence],
        "v130": int(request.religion),
        "b0": B0_BY_TWIN[request.twin_order],
    }])


def facility_types(details) -> dict[str, str]:
    """facility_detail -> facility_type through the same make_core_features derivation."""
    details = list(details)
    raw = pd.DataFrame({
        "b3": 360, "v011": 0, "v133": 0, "v190": 1, "bord": 1, "v025": 1, "v130": 1, "b0": 0,
        "m15": [M15_BY_DETAIL[detail] for detail in details],
    })
    return dict(zip(details, make_core_features(raw, wave="NFHS-5")["facility_type"]))


def history_columns(request: PredictRequest) -> dict[str, str]:
    level, has_any = HISTORY_BY_ANSWER[(request.is_first_birth, request.previous_csection)]
    return {HISTORY_FEATURE: level, HAS_ANY_FEATURE: has_any, QUALITY_FEATURE: QUALITY_OK}


def to_frame(request: PredictRequest) -> pd.DataFrame:
    config = load_config()
    raw = _raw_row(request)
    core = make_core_features(raw, wave="NFHS-5")
    detail = facility_detail(raw["m15"])
    history = pd.DataFrame([history_columns(request)], dtype="string")
    frame = build_tier_frame(core, detail, history, TIER)[config["feature_order"]]
    vocabulary = config["vocabulary"]
    for column in config["categorical_features"]:
        value = str(frame.at[0, column])
        if value not in vocabulary[column]:
            raise ValueError(f"{column}={value!r} is not a category the model was trained on")
    return frame
