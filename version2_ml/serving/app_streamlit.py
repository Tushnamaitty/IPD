"""Streamlit front end for the V2 C-section probability API.

Run from the repo root:
    .venv\\Scripts\\python.exe -m streamlit run version2_ml\\serving\\app_streamlit.py

Calls the API at $API_URL (default http://localhost:8000). If the API is not
reachable, the same endpoint functions are imported and called in-process.
Inputs and results live only in the current page run; nothing is stored.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
DEMO_NOTE = "Research demo built on NFHS-5 survey data (2019-21). Not for clinical decisions."
RAISES_COLOR = "#d03b3b"
LOWERS_COLOR = "#0ca30c"
BAND_STYLE = {"Low": "green", "Medium": "orange", "High": "red"}
WEALTH_HINTS = {
    1: "bottom fifth of households in India by assets and housing",
    2: "second-lowest fifth of households",
    3: "middle fifth of households",
    4: "second-highest fifth of households",
    5: "top fifth of households (e.g. car, fridge, piped water)",
}


class LocalBackend:
    """In-process fallback: the API's own endpoint functions, no HTTP."""

    name = "local (API unreachable)"

    def __init__(self):
        base = Path(__file__).resolve().parents[1]
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        from serving import api
        from serving.schemas import PredictRequest

        self.api = api
        self.request_type = PredictRequest

    def options(self) -> dict:
        return self.api.options()

    def model_card(self) -> dict:
        return self.api.model_card()

    def predict(self, payload: dict) -> tuple[dict | None, list[str]]:
        from fastapi import HTTPException
        from pydantic import ValidationError

        try:
            return self.api.predict(self.request_type(**payload)).model_dump(), []
        except ValidationError as exc:
            return None, [error["msg"] for error in exc.errors()]
        except HTTPException as exc:
            return None, [str(exc.detail)]


@st.cache_data(ttl=300)
def _get_json(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


class HttpBackend:
    name = f"API {API_URL}"

    def options(self) -> dict:
        return _get_json("/options")

    def model_card(self) -> dict:
        return _get_json("/model-card")

    def predict(self, payload: dict) -> tuple[dict | None, list[str]]:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json(), []
        detail = response.json().get("detail", response.text)
        if isinstance(detail, list):
            return None, [item.get("msg", str(item)) for item in detail]
        return None, [str(detail)]


def _api_up() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


@st.cache_resource
def _local_backend() -> LocalBackend:
    return LocalBackend()


def get_backend():
    return HttpBackend() if _api_up() else _local_backend()


def _factor_chart(factors: list[dict]) -> alt.Chart:
    frame = pd.DataFrame(factors)
    frame["effect"] = frame["direction"].str.capitalize()
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4, height=18)
        .encode(
            x=alt.X("contribution:Q", title="Contribution to the score (log-odds)"),
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
            color=alt.Color(
                "effect:N",
                scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RAISES_COLOR, LOWERS_COLOR]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Factor"),
                alt.Tooltip("contribution:Q", title="Contribution", format="+.3f"),
            ],
        )
        .properties(height=40 * len(frame))
    )


def _sidebar(backend) -> None:
    with st.sidebar:
        st.header("Model")
        try:
            card = backend.model_card()
        except Exception as exc:  # noqa: BLE001 - sidebar must not break the form
            st.warning(f"Model card unavailable: {exc}")
            return
        metrics = card["metrics"]
        st.write(f"**Version:** `{metrics['model_version']}`")
        col_roc, col_pr = st.columns(2)
        col_roc.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        col_pr.metric("PR-AUC", f"{metrics['average_precision']:.3f}")
        st.caption(f"Validation partition, n = {metrics['n_births']:,} births.")
        if isinstance(backend, HttpBackend):
            st.markdown(f"[Model card (JSON)]({API_URL}/model-card)")
        with st.expander("Model card"):
            st.markdown(card["model_card"])
        st.caption(f"Backend: {backend.name}")


def _form(options: dict) -> dict:
    numeric = options["numeric"]
    left, right = st.columns(2)
    with left:
        age = st.number_input(
            "Mother's age at this birth (years)", key="age", value=25, step=1,
            min_value=numeric["maternal_age_at_index_birth"]["min"],
            max_value=numeric["maternal_age_at_index_birth"]["max"],
        )
        education = st.number_input(
            "Years of schooling completed", key="education", value=8, step=1,
            min_value=numeric["education_years"]["min"], max_value=numeric["education_years"]["max"],
        )
        wealth = {item["value"]: item["label"] for item in options["wealth_index"]}
        wealth_value = st.select_slider(
            "Household wealth", key="wealth", options=list(wealth), value=3,
            format_func=lambda value: wealth[value],
        )
        st.caption(f"{wealth[wealth_value]}: {WEALTH_HINTS[wealth_value]}.")
        residence = {item["value"]: item["label"] for item in options["residence"]}
        residence_value = st.radio(
            "Residence", key="residence", options=list(residence),
            format_func=residence.get, horizontal=True,
        )
        religion = {item["value"]: item["label"] for item in options["religion"]}
        religion_value = st.selectbox("Religion", key="religion", options=list(religion), format_func=religion.get)
    with right:
        facilities = sorted(options["facility_detail"], key=lambda item: (item["facility_type"] != "public", item["facility_type"], item["label"]))
        facility_label = {item["value"]: f"{item['group']} · {item['label']}" for item in facilities}
        facility_value = st.selectbox(
            "Place of delivery", key="facility", options=list(facility_label), format_func=facility_label.get,
        )
        twin = {item["value"]: item["label"] for item in options["twin_order"]}
        twin_value = st.radio("Single or multiple birth", key="twin", options=list(twin), format_func=twin.get, horizontal=True)
        first_birth = st.radio("Is this her first birth?", key="first_birth", options=["Yes", "No"], horizontal=True) == "Yes"
        if first_birth:
            birth_order = 1
            previous = None
            st.number_input("Birth order", key="birth_order_first", value=1, disabled=True)
        else:
            birth_order = st.number_input(
                "Birth order (2 = second child)", key="birth_order", value=2, step=1,
                min_value=2, max_value=numeric["birth_order"]["max"],
            )
            previous_options = {item["value"]: item["label"] for item in options["previous_csection"]}
            previous = st.radio(
                "Was any earlier birth a C-section?", key="previous", options=list(previous_options),
                format_func=previous_options.get, horizontal=True,
            )
    return {
        "maternal_age_at_index_birth": int(age),
        "education_years": int(education),
        "wealth_index": int(wealth_value),
        "birth_order": int(birth_order),
        "twin_order": twin_value,
        "facility_detail": facility_value,
        "residence": residence_value,
        "religion": religion_value,
        "is_first_birth": first_birth,
        "previous_csection": previous,
    }


def _result(result: dict) -> None:
    with st.container(border=True):
        band = result["risk_band"]
        col_p, col_band, col_label = st.columns([1, 1, 2])
        col_p.metric("Probability of C-section", f"{result['probability']:.1%}")
        col_band.markdown("**Risk band**")
        col_band.markdown(f":{BAND_STYLE[band]}-background[**{band}**]")
        col_label.markdown("**Prediction**")
        col_label.markdown(f"{result['predicted_label']}  \n"
                           f"<small>at threshold {result['threshold_used']:.2f}</small>", unsafe_allow_html=True)
        st.markdown("**What moved this estimate**")
        st.altair_chart(_factor_chart(result["top_factors"]), width="stretch")
        st.caption(DEMO_NOTE)


def main() -> None:
    st.set_page_config(page_title="IPD V2 C-section estimate", layout="wide")
    st.title("C-section probability estimate")
    st.caption("How often births with similar recorded characteristics in NFHS-5 were C-sections.")
    backend = get_backend()
    _sidebar(backend)
    try:
        options = backend.options()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load form options: {exc}")
        st.stop()
    payload = _form(options)
    if st.button("Estimate", type="primary", key="submit"):
        result, errors = backend.predict(payload)
        if errors:
            for message in errors:
                st.error(message)
        else:
            _result(result)


main()
