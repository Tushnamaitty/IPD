"""Capture demo screenshots of the Streamlit UI for three contrasting profiles.

Requires the API (port 8000) and the UI (port 8501) to be running, plus
Playwright (dev only, not in requirements.txt):
    .venv\\Scripts\\python.exe -m pip install playwright
    .venv\\Scripts\\python.exe -m playwright install chromium
    .venv\\Scripts\\python.exe version2_ml\\serving\\demo\\capture_demo.py

Profiles are invented examples, not survey records. Each screenshot's shown
probability is checked against a direct POST /predict for the same inputs.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import requests
from playwright.sync_api import Page, sync_playwright

UI_URL = "http://localhost:8501"
API_URL = "http://localhost:8000"
OUT_DIR = Path(__file__).resolve().parent
WEALTH_DEFAULT = 3

PROFILES = {
    "low": {
        "maternal_age_at_index_birth": 24, "education_years": 0, "wealth_index": 1,
        "birth_order": 3, "twin_order": "single", "facility_detail": "public: phc/additional phc",
        "residence": "rural", "religion": "1", "is_first_birth": False, "previous_csection": "no",
    },
    "medium": {
        "maternal_age_at_index_birth": 23, "education_years": 5, "wealth_index": 2,
        "birth_order": 1, "twin_order": "single", "facility_detail": "public: govt./munic. hospital",
        "residence": "rural", "religion": "1", "is_first_birth": True, "previous_csection": None,
    },
    "high": {
        "maternal_age_at_index_birth": 31, "education_years": 15, "wealth_index": 5,
        "birth_order": 2, "twin_order": "single", "facility_detail": "private: hospital/maternity home/clinic",
        "residence": "urban", "religion": "1", "is_first_birth": False, "previous_csection": "yes",
    },
}


def _settle(page: Page) -> None:
    page.wait_for_timeout(300)
    page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=30_000)
    page.wait_for_timeout(200)


def _number(page: Page, label: str, value: int) -> None:
    field = page.get_by_label(label, exact=True)
    field.fill(str(value))
    field.press("Enter")
    _settle(page)


def _radio(page: Page, group: str, option: str) -> None:
    page.get_by_role("radiogroup", name=group).get_by_text(option, exact=True).click()
    _settle(page)


def _select(page: Page, label: str, option: str) -> None:
    page.locator('[data-testid="stSelectbox"]').filter(has_text=label).click()
    page.get_by_role("option", name=option, exact=True).click()
    _settle(page)


def _fill(page: Page, profile: dict, options: dict) -> None:
    labels = {key: {str(item["value"]): item["label"] for item in options[key]}
              for key in ("residence", "religion", "twin_order", "previous_csection")}
    facility = next(item for item in options["facility_detail"] if item["value"] == profile["facility_detail"])
    _radio(page, "Is this her first birth?", "Yes" if profile["is_first_birth"] else "No")
    _number(page, "Mother's age at this birth (years)", profile["maternal_age_at_index_birth"])
    _number(page, "Years of schooling completed", profile["education_years"])
    slider = page.get_by_role("slider")
    slider.focus()
    step = profile["wealth_index"] - WEALTH_DEFAULT
    for _ in range(abs(step)):
        slider.press("ArrowRight" if step > 0 else "ArrowLeft")
    _settle(page)
    _radio(page, "Residence", labels["residence"][profile["residence"]])
    _select(page, "Religion", labels["religion"][profile["religion"]])
    _select(page, "Place of delivery", f"{facility['group']} · {facility['label']}")
    _radio(page, "Single or multiple birth", labels["twin_order"][profile["twin_order"]])
    if not profile["is_first_birth"]:
        _number(page, "Birth order (2 = second child)", profile["birth_order"])
        _radio(page, "Was any earlier birth a C-section?", labels["previous_csection"][profile["previous_csection"]])


def main() -> int:
    options = requests.get(f"{API_URL}/options", timeout=10).json()
    failures = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for name, profile in PROFILES.items():
            expected = requests.post(f"{API_URL}/predict", json=profile, timeout=30).json()
            # Streamlit scrolls an inner container, so full_page cannot extend past
            # the viewport; make the viewport tall enough for the whole result card.
            page = browser.new_page(viewport={"width": 1400, "height": 1450}, device_scale_factor=1.5)
            page.goto(UI_URL)
            page.get_by_role("button", name="Estimate").wait_for(timeout=60_000)
            _settle(page)
            _fill(page, profile, options)
            page.get_by_role("button", name="Estimate").click()
            page.wait_for_selector("text=Probability of C-section", timeout=60_000)
            _settle(page)
            page.wait_for_selector('[data-testid="stVegaLiteChart"] canvas, [data-testid="stVegaLiteChart"] svg', timeout=30_000)
            page.wait_for_timeout(800)
            shown = page.locator('[data-testid="stMetricValue"]').filter(has_text="%").first.inner_text()
            shown_value = float(re.sub(r"[^\d.]", "", shown)) / 100
            ok = abs(shown_value - expected["probability"]) < 0.0015
            failures += not ok
            path = OUT_DIR / f"demo_{name}.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"{name:7s} UI {shown:>6s}  API {expected['probability']:.3f} {expected['risk_band']:6s} "
                  f"{'match' if ok else 'MISMATCH'} -> {path.name}")
            page.close()
        browser.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
