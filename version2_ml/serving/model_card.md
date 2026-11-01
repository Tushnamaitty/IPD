# Model card: IPD V2 C-section probability (v2.0-a7-f1dh-xgb-platt)

## Intended use

**Research demonstration only. Not a clinical tool.**

The model estimates how often institutional births with similar recorded
characteristics in India's NFHS-5 survey were delivered by caesarean section.
It is meant for teaching, exploring the V2 research results and discussing
population-level patterns. It is not an individual medical risk assessment and
must not inform the care, counselling or triage of any person.

## Model

The A7 candidate `xgboost__F1_detail_history`
(`version2_ml/experiments/A7`), refitted unchanged by
`version2_ml/serving/train_export.py`:

- XGBoost: `max_depth=4, n_estimators=200, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8, eval_metric=logloss,
  tree_method=hist`, `scale_pos_weight` = fit-pool negatives / positives,
  seed 42. Selected by mother-grouped 3-fold CV inside the training pool.
- Platt scaling fitted on a separate calibration pool (lowest validation Brier
  of uncalibrated / Platt / isotonic in A7).
- 12 model columns: 4 numeric (age, education, wealth, birth order) and 8
  categorical (facility type, residence, religion, twin order, facility
  detail and three observed-history columns).

## Data

- NFHS-5 (2019-21) birth recode `IABR7EFL`.
- Frozen C1 cohort: institutional births (m15 in 20-27, 30-33, 96) in the
  59 months before interview with a recorded delivery mode (m17 in 0/1):
  **201,311 institutional births to 158,429 mothers**, 44,742 C-sections
  (22.2%).
- Mother-grouped 70/15/15 split (seed 42), so no mother appears in two
  partitions. The 70% training partition is split again, by mother, into a
  fit pool (112,739 births) and a 20% calibration pool (28,181 births, seed
  4208).
- Unweighted analysis: survey weights, clustering and strata are not used.

## Inputs (10)

| # | User input | Model column(s) |
| --- | --- | --- |
| 1 | Mother's age at this birth, 15-49 years | `maternal_age_at_index_birth` |
| 2 | Years of schooling, 0-20 | `education_years` |
| 3 | Household wealth quintile, 1 Poorest - 5 Richest | `wealth_index` |
| 4 | Birth order, 1-15 | `birth_order` |
| 5 | Single or multiple birth | `twin_order` (multiple is encoded as "1st of multiple") |
| 6 | Place of delivery (11 facility types) | `facility_detail`, and `facility_type` derived from it |
| 7 | Urban or rural residence | `residence` |
| 8 | Religion (NFHS-5 codes seen in training) | `religion` |
| 9 | Is this the first birth? | history columns (see 10) |
| 10 | Was an earlier birth a C-section? yes / no / not sure (only if not first) | `history_four_level`, `has_any_observed_earlier_institutional_birth`, `history_quality_flag` |

History mapping: first birth -> *first recorded live birth*; yes -> *observed
previous institutional C-section*; no -> *earlier institutional birth(s),
none a C-section*; not sure -> *higher-order birth, earlier institutional
history unobserved* (in training this meant no earlier facility birth was
recorded in the survey window).

Inputs outside the trained vocabulary are rejected: facility "public
sector" / "private sector" without a facility type, religion code 7, and
ages, birth orders or education years outside the ranges above.

## Performance

### Validation (from `metrics.json`)

Validation partition, n = 30,227 births, 6,701 C-sections (prevalence
22.2%), calibrated probabilities. Reproduced exactly by `train_export.py`.

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.8298 |
| PR-AUC (average precision) | 0.6113 |
| Brier score | 0.1257 |
| Calibration intercept / slope | -0.016 / 0.990 |
| Default threshold 0.30 (F1-optimal) | precision 0.510, recall 0.673, F1 0.580 |
| Alternative threshold 0.19 (recall >= 0.80) | precision 0.429, recall 0.805, F1 0.560 |

### Development test, non-confirmatory: partition previously used in V1

Recorded once by A7 (`aggregate_model_comparison.csv`); never re-scored by the
serving code. n = 30,164 births.

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.8323 |
| PR-AUC | 0.6077 |
| Brier score | 0.1256 |

These rows were inspected during V1 work, so this is not an independent
test of V2.

### V1 vs V2

| | V1 baseline (XGBoost) | V2 (this model) |
| --- | --- | --- |
| Features | 36 (incl. BMI, ANC content, provider) | 12 (8 core + facility detail + 3 observed history) |
| Evaluation rows | V1 test, n = 40,263 | development test, n = 30,164 |
| ROC-AUC | 0.8037 | 0.8323 |
| PR-AUC | 0.5334 | 0.6077 |
| Brier (calibrated) | 0.1358 (isotonic) | 0.1256 (Platt) |
| Source | `outputs/tables/b2_model_performance.csv`, `outputs/final_tables/final_predictive_performance_table.csv` | A7 recording |

The two columns use different cohort definitions and evaluation rows, so this
is a descriptive comparison, not a matched one. A7's matched comparison on
identical rows is in `version2_ml/experiments/A7/outputs/`.

### Subgroups (A7 validation, calibrated)

| Subgroup | ROC-AUC | Calibration slope |
| --- | --- | --- |
| Public facilities | 0.799 | 0.994 |
| Private facilities | 0.713 | 1.029 |
| Urban | 0.786 | 1.009 |
| Rural | 0.834 | 0.989 |
| 27 supported states (range) | 0.694-0.889 | |

Discrimination is notably weaker within private facilities.

## Decision outputs

- Predicted label at the default threshold 0.30: "C-section likely" or
  "Vaginal delivery likely".
- Risk band from tertiles of calibrated validation probabilities: Low below
  about 0.068, Medium from 0.068 to 0.259, High from 0.259 up. Exact
  cut-points are in `config.json`. "High" is relative to this cohort: it
  starts at about a 26% probability.

## Explanations

The top 5 factors are per-request SHAP values (TreeExplainer, log-odds margin
of the uncalibrated model) summed into the user-facing inputs. Facility type
and detail together and previous C-section history dominate globally (about
41% and 21% of mean |SHAP|). Attributions describe what moved this prediction,
not causes.

## Limitations

- **Retrospective survey data.** Delivery mode, facility and history are
  mothers' recall at interview for births up to 5 years earlier; misreporting
  is possible and not corrected.
- **History observed only within the 5-year recall window.** Earlier births
  outside the survey's birth window (or in other surveys) are invisible, so
  "no previous C-section" means none observed in the window, not a lifetime
  negative. The "not sure" answer maps to the unobserved-history level.
- **BMI excluded.** Maternal height and weight are measured at interview time,
  after the birth, so BMI was not used, unlike V1.
- **Not causal.** Associations reflect where and to whom births happen
  (facility choice is itself linked to the outcome). Neither the probability
  nor the SHAP factors show the effect of changing an input.
- **No confidence intervals yet.** All metrics are unweighted point estimates
  without design-based (survey-weighted, clustered) intervals.
- **Probabilities may need recalibration on other waves.** Calibrated on
  NFHS-5 only; not validated on NFHS-4 or any external data, and C-section
  practice changes over time.
- Some encoding simplifications: "multiple birth" is scored as the first of
  a multiple birth; ages are whole years.
