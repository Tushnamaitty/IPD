"""Every out-of-range, unknown or inconsistent input is rejected with HTTP 422."""
import pytest

INVALID = {
    "age below 15": {"maternal_age_at_index_birth": 14},
    "age above 49": {"maternal_age_at_index_birth": 50},
    "age not an integer": {"maternal_age_at_index_birth": 27.5},
    "education below 0": {"education_years": -1},
    "education above 20": {"education_years": 21},
    "wealth below 1": {"wealth_index": 0},
    "wealth above 5": {"wealth_index": 6},
    "birth order above 15": {"birth_order": 16},
    "birth order 0": {"birth_order": 0},
    "twin order unknown": {"twin_order": "triplets"},
    "twin order trained code": {"twin_order": "1"},
    "facility unknown": {"facility_detail": "spaceship"},
    "facility never in fit pool": {"facility_detail": "public sector"},
    "residence unknown": {"residence": "suburban"},
    "religion never in fit pool": {"religion": "7"},
    "religion label not code": {"religion": "hindu"},
    "previous_csection unknown": {"previous_csection": "maybe"},
    "is_first_birth not boolean": {"is_first_birth": "sometimes"},
    "extra field": {"caste": "x"},
}
MISMATCH = {
    "first birth with birth_order 2": ({"is_first_birth": True, "previous_csection": None}, "a first birth has birth_order 1"),
    "not first with birth_order 1": ({"birth_order": 1}, "set birth_order to 2 or more"),
    "not first without previous_csection": ({"previous_csection": None}, "previous_csection (yes/no/not_sure) is required"),
    "first birth with previous_csection": ({"is_first_birth": True, "birth_order": 1}, "must be omitted"),
}


def test_valid_body_is_accepted(client, valid_body):
    assert client.post("/predict", json=valid_body).status_code == 200


@pytest.mark.parametrize("name", list(INVALID))
def test_invalid_value_returns_422(client, valid_body, name):
    response = client.post("/predict", json={**valid_body, **INVALID[name]})
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("field", [
    "maternal_age_at_index_birth", "education_years", "wealth_index", "birth_order", "twin_order",
    "facility_detail", "residence", "religion", "is_first_birth",
])
def test_missing_field_returns_422(client, valid_body, field):
    body = dict(valid_body)
    del body[field]
    assert client.post("/predict", json=body).status_code == 422


@pytest.mark.parametrize("name", list(MISMATCH))
def test_first_birth_mismatch_returns_422_with_message(client, valid_body, name):
    patch, message = MISMATCH[name]
    response = client.post("/predict", json={**valid_body, **patch})
    assert response.status_code == 422
    assert message in response.text
