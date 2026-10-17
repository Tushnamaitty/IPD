# A2 — Historical NFHS-4 state / facility-sector context: Methods and Results note

**Status: EXPLORATORY development result.** Provisional mother-grouped split; metrics are on the *validation* partition; the test partition was not scored. No confidence intervals or significance claims (survey-design estimator NOT approved).

## Methods

- **Source (NFHS-4):** 195,997 institutional births (institutional m15 in 20-27/30-33/96; v008-b3 in 0..59 (b19 unavailable)). Interviews 2015-01 to 2016-12. Labels are used only to build aggregate state, sector and state×sector cesarean rates.
- **Target (NFHS-5):** 201,311 births, 158,429 mothers (institutional m15 in 20-27/30-33/96; b19 in 0..59). Shared provisional mother-grouped 70/15/15 split (seed 42): train 140,920, validation 30,227, test 30,164.
- **Crosswalk:** explicit versioned table (`state_crosswalk_v1.csv`) validated against both DTA v024 value-label tables; numeric codes are never joined. DNH and DD are pooled for NFHS-4; Ladakh uses the NFHS-4 Jammu & Kashmir unit (sensitivity: no source unit).
- **Features:** smoothed historical state rate, state×sector rate, log source support and unsupported flags. Pseudo-count alpha=20, support threshold 30 births, unweighted (predeclared). Fallback: cell → state → sector → NFHS-4 global; the state feature falls back to the global rate. No NFHS-5 outcome enters the mapping; it is fitted and hashed before NFHS-5 is loaded.
- **Matched ablation:** B0 = eight common-core features; B0+S, B0+SX, B0+SUP each add one A2 block; B0+S+SX+SUP adds all; R1 adds ordinary NFHS-5 state indicators (reference). Same rows, split, train-only preprocessing and fixed default hyperparameters for logistic regression and XGBoost. N2 uses the historical state×sector rate directly as a probability (nothing fitted).
- **Thresholds:** max-F1 on validation, so validation threshold metrics are optimistic.

## Results (primary cohort, primary context, validation, unweighted)

| Model | Arm | ROC-AUC | ΔAUC vs B0 | AP | ΔAP vs B0 | Brier | Cal. slope | Cal. intercept |
|---|---|---|---|---|---|---|---|---|
| logistic_regression | B0 | 0.7729 | — | 0.4899 | — | 0.1431 | 1.006 | -0.006 |
| logistic_regression | B0+S | 0.7996 | +0.0267 | 0.5341 | +0.0442 | 0.1369 | 0.990 | -0.007 |
| logistic_regression | B0+SX | 0.8077 | +0.0348 | 0.5615 | +0.0716 | 0.1336 | 0.998 | -0.005 |
| logistic_regression | B0+SUP | 0.7855 | +0.0126 | 0.5018 | +0.0119 | 0.1408 | 1.001 | -0.007 |
| logistic_regression | B0+S+SX+SUP | 0.8076 | +0.0347 | 0.5626 | +0.0727 | 0.1334 | 0.998 | -0.004 |
| logistic_regression | R1 | 0.8064 | +0.0335 | 0.5539 | +0.0640 | 0.1340 | 0.992 | -0.003 |
| xgboost | B0 | 0.7743 | — | 0.4917 | — | 0.1427 | 1.088 | -0.007 |
| xgboost | B0+S | 0.8050 | +0.0307 | 0.5477 | +0.0561 | 0.1347 | 1.109 | -0.007 |
| xgboost | B0+SX | 0.8093 | +0.0350 | 0.5636 | +0.0719 | 0.1328 | 1.085 | -0.007 |
| xgboost | B0+SUP | 0.8013 | +0.0270 | 0.5492 | +0.0575 | 0.1357 | 1.169 | -0.008 |
| xgboost | B0+S+SX+SUP | 0.8096 | +0.0353 | 0.5641 | +0.0724 | 0.1328 | 1.090 | -0.006 |
| xgboost | R1 | 0.8021 | +0.0278 | 0.5462 | +0.0545 | 0.1358 | 1.226 | -0.007 |
| none_direct_rate | N2_hist_cell_rate_as_probability | 0.7837 | — | 0.5158 | — | 0.1410 | 1.038 | 0.308 |

Validation prevalence 0.2217 (n=30,227 births, 23,765 mothers). AP depends on this prevalence.

## Sensitivity summary (ΔAUC vs B0, validation, unweighted)

| Cohort | Context variant | Model | B0+S | B0+SX | B0+SUP | B0+S+SX+SUP |
|---|---|---|---|---|---|---|
| primary | primary | logistic_regression | +0.0267 | +0.0348 | +0.0126 | +0.0347 |
| primary | primary | xgboost | +0.0307 | +0.0350 | +0.0270 | +0.0353 |
| nfhs5_cmc_0_59 | primary | logistic_regression | +0.0270 | +0.0349 | +0.0127 | +0.0348 |
| nfhs5_cmc_0_59 | primary | xgboost | +0.0311 | +0.0352 | +0.0270 | +0.0355 |
| exclude_other_facility | primary | logistic_regression | +0.0269 | +0.0350 | +0.0127 | +0.0349 |
| exclude_other_facility | primary | xgboost | +0.0312 | +0.0352 | +0.0278 | +0.0355 |
| exclude_calendar_overlap | primary | logistic_regression | +0.0255 | +0.0328 | +0.0132 | +0.0328 |
| exclude_calendar_overlap | primary | xgboost | +0.0294 | +0.0332 | +0.0261 | +0.0334 |
| primary | alpha_0 | logistic_regression | +0.0267 | +0.0348 | +0.0126 | +0.0345 |
| primary | alpha_0 | xgboost | +0.0307 | +0.0351 | +0.0270 | +0.0353 |
| primary | alpha_5 | logistic_regression | +0.0267 | +0.0348 | +0.0126 | +0.0346 |
| primary | alpha_5 | xgboost | +0.0307 | +0.0351 | +0.0270 | +0.0353 |
| primary | alpha_100 | logistic_regression | +0.0269 | +0.0348 | +0.0126 | +0.0350 |
| primary | alpha_100 | xgboost | +0.0307 | +0.0348 | +0.0270 | +0.0351 |
| primary | weighted_v005 | logistic_regression | +0.0264 | +0.0341 | +0.0122 | +0.0342 |
| primary | weighted_v005 | xgboost | +0.0307 | +0.0340 | +0.0277 | +0.0346 |
| primary | ladakh_unseen | logistic_regression | +0.0263 | +0.0342 | +0.0132 | +0.0347 |
| primary | ladakh_unseen | xgboost | +0.0301 | +0.0344 | +0.0269 | +0.0349 |

Across all 72 A2 sensitivity comparisons, ΔAUC ranged from +0.0122 to +0.0355. The direction is consistent across cohorts and smoothing variants, but these are point differences on one provisional split without approved intervals: exploratory only.

## Reading the results

- **Versus ordinary state indicators (R1):** B0+SX minus R1 ROC-AUC = logistic_regression +0.0013; xgboost +0.0072. Historical NFHS-4 context carries state/sector information of similar discriminative value to NFHS-5-trained state indicators on this split. This does not show signal beyond state identity.
- **Support features act as state identifiers:** log source support takes one value per NFHS-4 source unit, so B0+SUP (especially for XGBoost) can learn state effects from NFHS-5 training labels. Its gain should not be read as an effect of support itself.
- **Wave shift (N2):** the NFHS-4 rate used directly as a probability has ROC-AUC 0.7837 but calibration intercept +0.308: NFHS-4 global rate 0.182 vs NFHS-5 validation prevalence 0.222. Historical rates rank state/sector cells usefully but under-predict NFHS-5 cesarean levels.
- **Calendar overlap:** B0+SX ΔAUC with NFHS-5 births in or before the last NFHS-4 interview month excluded vs primary cohort: logistic_regression +0.0328 vs +0.0348; xgboost +0.0332 vs +0.0350.

## Supported claims

- A2 features were built from NFHS-4 only; runtime and unit tests show NFHS-5 label permutation, replacement or removal leaves the mapping hash and every feature bitwise identical.
- The table reports how much historical context changes development-split discrimination and calibration relative to the matched eight-core baseline and to ordinary NFHS-5 state indicators (R1).

## Not supported / limitations

- Not causal: rates describe historical case mix and reporting by state and sector, not sector effects or medical necessity. Not prenatal prediction: features are retrospectively recorded.
- No confidence intervals or significance; one provisional split; old V1 test already examined.
- Survey clusters cross partitions (validation: 15,501 of 16,040 clusters also in train); mother grouping only.
- Calendar overlap: 70,510 NFHS-5 births occurred in or before the last NFHS-4 interview month; see the `exclude_calendar_overlap` row.
- State is interview residence, not verified delivery-facility state. Ladakh shares the NFHS-4 J&K source unit in the primary analysis.
- Weighted (v005) results are descriptive only; weights were not verified for this estimand.
- m15=96 ('other') births: all non-cesarean in the audit; see `exclude_other_facility`.

## Failure modes checked

- Fitting on NFHS-5 or mixed-wave rows is refused (LEAKAGE error). Unknown state codes or crosswalk/label drift stop the run. Unknown sector values are refused. Unsupported, zero-count and unseen cells use the fallback order and set flags. Ablation arms are checked for identical evaluation rows.
