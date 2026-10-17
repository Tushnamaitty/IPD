# A2 — Historical NFHS-4 state / facility-sector context

Status: **EXPLORATORY development** (feature tier F3). Not confirmatory; no confidence intervals.

Question: do historical NFHS-4 state and state×sector cesarean rates, built without any NFHS-5
outcome, improve NFHS-5 classification beyond the matched eight-core baseline?

## Layout

| File | Purpose |
|---|---|
| `A2_historical_context.py` | `HistoricalContextMapping`: fit on NFHS-4 only, frozen, SHA-256 hashed; `transform` reads only `source_unit` and `sector` |
| `state_crosswalk.py`, `state_crosswalk_v1.csv` | Versioned NFHS-5 → NFHS-4 state/UT crosswalk, validated against both DTA `v024` value-label tables |
| `config.yaml` | Predeclared smoothing, sensitivities, models, split, disclosure rules |
| `run.py` | Runner: crosswalk → NFHS-4 fit and hash → NFHS-5 load → matched ablation → aggregate outputs |
| `tests/` | Synthetic tests (no NFHS records) |
| `outputs/aggregate/` | Disclosure-checked aggregate outputs only |
| `../common/` | Shared loader, metrics and provenance helpers for A2/A3/A5/A6 |

## Run (from repo root)

```
python -m unittest discover -s version2_ml/experiments/A2/tests -p "test_*.py" -v
python -m unittest discover -s version2_ml/experiments/common/tests -p "test_*.py" -v
python version2_ml/experiments/A2/run.py              # validation partition only
python version2_ml/experiments/A2/run.py --evaluate-test   # scores the provisional test partition (logged in manifest)
```

Needs the authorized local files `data/raw/nfhs4/IABR74FL.DTA` and `data/raw/IABR7EFL.DTA`
(gitignored). A restricted birth-table cache is written to `data/processed/v2_experiments/` (gitignored).

## Design

- **Features:** `hist_state_cs_rate`, `hist_state_sector_cs_rate`, `hist_state_support_log`,
  `hist_cell_support_log`, `hist_state_unsupported`, `hist_cell_unsupported`.
- **Smoothing:** pseudo-count alpha=20 toward the parent rate, support threshold 30 births, unweighted
  (predeclared). Sensitivities: alpha 0/5/100, v005-weighted rates with Kish support, Ladakh with no source unit.
- **Fallback (code-fixed):** state feature: supported state → NFHS-4 global. State×sector feature:
  supported cell → supported state → supported sector → global. NFHS-5 prevalence is never used.
- **Crosswalk:** numeric `v024` codes are never joined across waves. Dadra & Nagar Haveli and Daman & Diu
  are pooled in NFHS-4 for the merged NFHS-5 UT; Ladakh uses the NFHS-4 Jammu & Kashmir unit.
- **Matched arms (non-cumulative):** B0 (eight core), B0+S, B0+SX, B0+SUP, B0+S+SX+SUP, R1 (B0 + NFHS-5
  state indicators, reference), N2 (historical cell rate used directly as a probability, nothing fitted).
  Logistic regression and XGBoost with fixed `model_framework.make_estimator` defaults.
- **Cohort sensitivities:** NFHS-5 CMC 0..59, excluding m15=96, excluding NFHS-5 births in or before the
  last NFHS-4 interview month.

## Leakage guards

- `fit` refuses any non-NFHS-4 row. Mappings are hashed before NFHS-5 is loaded and re-checked at the end.
- The runner re-fits the mapping with permuted, removed and all-positive NFHS-5 labels and requires
  bitwise-identical features (fails closed otherwise). Unit tests repeat this on synthetic data.
- Preprocessing and models are fitted on the NFHS-5 train partition only; thresholds on validation only.
- Outputs are scanned for restricted columns; small source cells (n<50 or <10 events/non-events) are suppressed.
