"""Request/response schemas. Categorical enums are built from config.json at import time."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import load_config

_CONFIG = load_config()
_VOCAB = _CONFIG["vocabulary"]
_LABELS = _CONFIG["display_labels"]

FacilityDetail = Literal[tuple(_VOCAB["facility_detail"])]
Religion = Literal[tuple(_VOCAB["religion"])]
Residence = Literal[tuple(_VOCAB["residence"])]
WEALTH_LABELS = {int(code): label.capitalize() for code, label in _LABELS["wealth_index"].items()}

DISCLAIMER = (
    "Research prototype trained on NFHS-5 (2019-21) survey data. The probability "
    "describes how often births with similar recorded characteristics were "
    "C-sections; it is not a clinical risk assessment and must not be used for "
    "individual care decisions."
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maternal_age_at_index_birth: int = Field(ge=15, le=49, description="Mother's age at this birth, years")
    education_years: int = Field(ge=0, le=20, description="Completed years of schooling")
    wealth_index: int = Field(
        ge=1, le=5,
        description="Household wealth quintile: " + ", ".join(f"{k}={v}" for k, v in sorted(WEALTH_LABELS.items())),
    )
    birth_order: int = Field(ge=1, le=15, description="Birth order of this child (1 = first)")
    twin_order: Literal["single", "multiple"] = Field(description="Single or multiple birth")
    facility_detail: FacilityDetail = Field(description="Place of delivery")
    residence: Residence
    religion: Religion = Field(
        description="NFHS-5 v130 code: " + ", ".join(f"{k}={v}" for k, v in _LABELS["religion"].items()),
    )
    is_first_birth: bool
    previous_csection: Optional[Literal["yes", "no", "not_sure"]] = Field(
        default=None,
        description="Was an earlier birth a C-section? Required only when is_first_birth is false",
    )

    @model_validator(mode="after")
    def _check_history(self):
        if self.is_first_birth:
            if self.birth_order != 1:
                raise ValueError(
                    f"is_first_birth is true but birth_order is {self.birth_order}; a first birth has birth_order 1"
                )
            if self.previous_csection is not None:
                raise ValueError("previous_csection must be omitted when is_first_birth is true")
        else:
            if self.birth_order == 1:
                raise ValueError("is_first_birth is false but birth_order is 1; set birth_order to 2 or more")
            if self.previous_csection is None:
                raise ValueError("previous_csection (yes/no/not_sure) is required when is_first_birth is false")
        return self


class Factor(BaseModel):
    feature: str
    label: str
    value: str
    direction: Literal["raises risk", "lowers risk"]
    contribution: float = Field(description="SHAP contribution in log-odds (margin) space")


class PredictResponse(BaseModel):
    probability: float = Field(ge=0, le=1, description="Calibrated probability of C-section, 3 decimals")
    risk_band: Literal["Low", "Medium", "High"]
    threshold_used: float
    predicted_label: Literal["C-section likely", "Vaginal delivery likely"]
    top_factors: list[Factor]
    model_version: str
    disclaimer: str = DISCLAIMER
