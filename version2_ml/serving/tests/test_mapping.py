"""Request -> model frame produces the exact strings and columns used in training."""
import pytest

from serving.mapping import to_frame
from serving.schemas import PredictRequest

from a4_facility_detail import FACILITY_DETAIL_LABELS
import a4_history

# (is_first_birth, previous_csection) -> (history_four_level, has_any, quality), as literal training strings
HISTORY_CASES = {
    (True, None): ("first_recorded_live_birth", "false", "ok"),
    (False, "yes"): ("observed_previous_institutional_cs", "true", "ok"),
    (False, "no"): ("higher_order_no_observed_previous_institutional_cs", "true", "ok"),
    (False, "not_sure"): ("higher_order_earlier_institutional_history_unobserved", "false", "ok"),
}
HISTORY_COLUMNS = ("history_four_level", "has_any_observed_earlier_institutional_birth", "history_quality_flag")


def _request(valid_body, **patch):
    return PredictRequest(**{**valid_body, **patch})


def test_literal_history_strings_are_the_a4_constants():
    assert {case[0] for case in HISTORY_CASES.values()} == {
        a4_history.FIRST_RECORDED_LIVE_BIRTH,
        a4_history.OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
        a4_history.NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS,
        a4_history.UNOBSERVED_HISTORY,
    }
    assert {a4_history.HAS_ANY_TRUE, a4_history.HAS_ANY_FALSE} == {"true", "false"}
    assert a4_history.QUALITY_OK == "ok"


@pytest.mark.parametrize("answer", list(HISTORY_CASES))
def test_history_mapping(valid_body, config, answer):
    first, previous = answer
    frame = to_frame(_request(valid_body, is_first_birth=first, previous_csection=previous,
                              birth_order=1 if first else 3))
    observed = tuple(str(frame.at[0, column]) for column in HISTORY_COLUMNS)
    assert observed == HISTORY_CASES[answer]
    for column, value in zip(HISTORY_COLUMNS, observed):
        assert value in config["vocabulary"][column]


def _expected_facility_type(detail: str) -> str:
    code = {label: code for code, label in FACILITY_DETAIL_LABELS.items()}[detail]
    if 20 <= code <= 27:
        return "public"
    if 30 <= code <= 33:
        return "private"
    return "other"


def test_facility_type_for_every_trained_facility_detail(valid_body, config):
    details = config["vocabulary"]["facility_detail"]
    assert len(details) == 11
    for detail in details:
        frame = to_frame(_request(valid_body, facility_detail=detail))
        assert frame.at[0, "facility_detail"] == detail
        assert frame.at[0, "facility_type"] == _expected_facility_type(detail), detail


def test_columns_equal_feature_order(valid_body, config):
    frame = to_frame(_request(valid_body))
    assert list(frame.columns) == config["feature_order"]
    assert len(frame.columns) == 12 and len(frame) == 1


def test_core_encodings_match_training(valid_body):
    frame = to_frame(_request(valid_body, maternal_age_at_index_birth=31, wealth_index=5,
                              twin_order="multiple", residence="rural", religion="96"))
    row = frame.iloc[0]
    assert row["maternal_age_at_index_birth"] == 31.0
    assert row["wealth_index"] == "5"  # make_core_features keeps wealth as a string rank
    assert row["twin_order"] == "1" and row["residence"] == "rural" and row["religion"] == "96"
