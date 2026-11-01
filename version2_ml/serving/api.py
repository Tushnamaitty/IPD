"""FastAPI service for the V2 C-section probability model.

Run from the repo root:
    .venv\\Scripts\\python.exe -m uvicorn version2_ml.serving.api:app --port 8000

Request bodies are never logged; the access log records method, path and
status only.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .artifacts import load_config, model_dir
from .explain import PREVIOUS_CS_VALUES, get_explainer
from .mapping import facility_types
from .predictor import get_predictor
from .schemas import WEALTH_LABELS, PredictRequest, PredictResponse

MODEL_CARD = Path(__file__).resolve().parent / "model_card.md"
NUMERIC_FIELDS = ("maternal_age_at_index_birth", "education_years", "birth_order")
LOCALHOST_ORIGINS = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_explainer()  # loads and sha256-verifies the model, builds the SHAP explainer
    yield


app = FastAPI(
    title="IPD V2 C-section probability",
    version=load_config()["model_version"],
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=LOCALHOST_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:]


def _bounds(name: str) -> dict:
    field = PredictRequest.model_fields[name]
    bounds = {"min": None, "max": None, "description": field.description}
    for constraint in field.metadata:
        if getattr(constraint, "ge", None) is not None:
            bounds["min"] = constraint.ge
        if getattr(constraint, "le", None) is not None:
            bounds["max"] = constraint.le
    return bounds


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        result = get_predictor().predict(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    factors = get_explainer().top_factors(request, result.pop("frame"))
    return PredictResponse(**result, top_factors=factors)


@app.get("/health")
def health() -> dict:
    predictor = get_predictor()
    return {
        "status": "ok",
        "model_version": predictor.model_version,
        "model_sha256": predictor.config["model_sha256"],
    }


@app.get("/model-card")
def model_card() -> dict:
    metrics = json.loads((model_dir() / "metrics.json").read_text(encoding="utf-8"))
    return {"model_card": MODEL_CARD.read_text(encoding="utf-8"), "metrics": metrics}


@app.get("/options")
def options() -> dict:
    config = load_config()
    labels = config["display_labels"]
    facility_type_by_detail = facility_types(config["vocabulary"]["facility_detail"])

    def choices(values, mapping, case=_sentence_case):
        return [{"value": value, "label": case(mapping[value])} for value in values]

    return {
        "model_version": config["model_version"],
        "numeric": {name: _bounds(name) for name in NUMERIC_FIELDS},
        "wealth_index": [{"value": code, "label": label} for code, label in sorted(WEALTH_LABELS.items())],
        "twin_order": [{"value": "single", "label": "Single birth"}, {"value": "multiple", "label": "Multiple birth (twins or more)"}],
        "facility_detail": [
            {**item, "facility_type": facility_type_by_detail[item["value"]],
             "group": labels["facility_type"][facility_type_by_detail[item["value"]]]}
            for item in choices(config["vocabulary"]["facility_detail"], labels["facility_detail"])
        ],
        "residence": choices(config["vocabulary"]["residence"], labels["residence"]),
        "religion": choices(config["vocabulary"]["religion"], labels["religion"]),
        "previous_csection": [
            {"value": value, "label": label} for value, label in PREVIOUS_CS_VALUES.items() if value is not None
        ],
        "thresholds": config["thresholds"],
        "risk_bands": config["risk_bands"],
    }

