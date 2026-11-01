"""API endpoints return 200 with valid bodies, and /predict stays fast with SHAP."""
import time

import numpy as np

from serving.schemas import PredictResponse

LATENCY_CALLS = 200
LATENCY_WARMUP = 5
P95_LIMIT_MS = 300.0


def test_predict(client, valid_body):
    response = client.post("/predict", json=valid_body)
    assert response.status_code == 200
    body = PredictResponse.model_validate(response.json())
    assert 0.0 <= body.probability <= 1.0 and round(body.probability, 3) == body.probability
    assert len(body.top_factors) == 5
    assert all(f.label.endswith(f.direction) for f in body.top_factors)
    assert body.predicted_label == ("C-section likely" if body.probability >= body.threshold_used else "Vaginal delivery likely")


def test_health(client, config):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": config["model_version"], "model_sha256": config["model_sha256"]}


def test_options(client, config):
    response = client.get("/options")
    assert response.status_code == 200
    body = response.json()
    assert [item["value"] for item in body["facility_detail"]] == config["vocabulary"]["facility_detail"]
    assert {item["facility_type"] for item in body["facility_detail"]} == {"public", "private", "other"}
    assert [item["value"] for item in body["religion"]] == config["vocabulary"]["religion"]
    assert [item["value"] for item in body["wealth_index"]] == [1, 2, 3, 4, 5]
    assert [item["value"] for item in body["previous_csection"]] == ["yes", "no", "not_sure"]
    assert body["numeric"]["maternal_age_at_index_birth"] == {"min": 15, "max": 49, "description": "Mother's age at this birth, years"}
    assert all(item["label"] for key in ("facility_detail", "religion", "residence", "twin_order") for item in body[key])


def test_model_card(client):
    response = client.get("/model-card")
    assert response.status_code == 200
    body = response.json()
    assert body["model_card"].startswith("# Model card")
    assert {"roc_auc", "average_precision", "n_births"} <= set(body["metrics"])


def test_predict_latency_p95(client, valid_body):
    for _ in range(LATENCY_WARMUP):
        client.post("/predict", json=valid_body)
    timings = []
    for _ in range(LATENCY_CALLS):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_body)
        timings.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    p50, p95 = np.percentile(timings, [50, 95])
    print(f"\n/predict latency over {LATENCY_CALLS} sequential calls (SHAP included): "
          f"p50={p50:.1f} ms, p95={p95:.1f} ms, max={max(timings):.1f} ms")
    assert p95 < P95_LIMIT_MS
