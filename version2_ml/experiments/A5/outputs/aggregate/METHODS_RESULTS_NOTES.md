# A5 — NFHS-4 → NFHS-5 source-only temporal transfer: Methods and Results note

**Status: EXPLORATORY development result.** NFHS-5 test partition was not scored. No confidence intervals or significance claims.

## Methods

- **Source:** NFHS-4 195,997 births, 153,489 mothers (institutional m15 in 20-27/30-33/96; v008-b3 in 0..59 (b19 unavailable)); mother-grouped 70/15/15 source split (seed 42): train 137,214, validation 29,414, test 29,369 (train = fit; validation = threshold and primary-model choice; test = source holdout).
- **Target:** NFHS-5 201,311 births, 158,429 mothers (institutional m15 in 20-27/30-33/96; b19 in 0..59); shared split as A2/A3: train 140,920, validation 30,227, test 30,164.
- **Features:** exactly the eight common-core FT features; the 29 F0-only predictors (including the 22 pregnancy/ANC items) are rejected in code. v190 wealth quintiles are wave-relative ranks.
- **Source-only fitting:** preprocessing (median imputation, one-hot vocabulary), logistic regression / XGBoost (fixed defaults) and the max-F1 decision threshold all come from NFHS-4 only. Each model is frozen and SHA-256 hashed before NFHS-5 is loaded; the frozen file is saved under the gitignored `artifacts/A5/`. Primary model by NFHS-4 validation ROC-AUC: **xgboost**.
- **Comparisons:** (A) NFHS-4 validation and holdout; (B) frozen model zero-shot on the NFHS-5 shared validation partition (and all non-test NFHS-5 rows); (C) NFHS-5 within-wave model on the same validation rows — context only, not an isolated wave effect (reproduces A2 B0: PASS); (D) logistic recalibration (intercept, intercept+slope) fitted on NFHS-5 train mothers, evaluated on disjoint NFHS-5 validation mothers — target-supervised, reported separately from zero-shot.
- **Unseen categories:** levels inside the codebook vocabulary but absent from NFHS-4 training are encoded as all-zero one-hot; rates are in `shift_diagnostics.json`: wealth_index: none; facility_type: none; residence: none; religion: none; twin_order: none.

## Results

### All comparisons (validation, unweighted, primary cohort)

| Comparison | Model | n | Prevalence | ROC-AUC | AP | Brier | Cal. slope | Cal. intercept |
|---|---|---|---|---|---|---|---|---|
| A (source_validation) | logistic_regression | 29,414 | 0.1821 | 0.7747 | 0.4264 | 0.1266 | 1.000 | 0.008 |
| A (source_holdout) | logistic_regression | 29,369 | 0.1822 | 0.7785 | 0.4366 | 0.1254 | 1.017 | -0.010 |
| A (source_validation) | xgboost | 29,414 | 0.1821 | 0.7769 | 0.4281 | 0.1260 | 1.096 | 0.010 |
| A (source_holdout) | xgboost | 29,369 | 0.1822 | 0.7806 | 0.4336 | 0.1252 | 1.112 | -0.006 |
| B (target_validation) | logistic_regression | 30,227 | 0.2217 | 0.7711 | 0.4874 | 0.1457 | 1.007 | 0.303 |
| B (target_validation) | xgboost | 30,227 | 0.2217 | 0.7731 | 0.4890 | 0.1454 | 1.087 | 0.298 |
| C (target_validation) | logistic_regression | 30,227 | 0.2217 | 0.7729 | 0.4899 | 0.1431 | 1.006 | -0.006 |
| C (target_validation) | xgboost | 30,227 | 0.2217 | 0.7743 | 0.4917 | 0.1427 | 1.088 | -0.007 |
| D_intercept (target_validation) | logistic_regression | 30,227 | 0.2217 | 0.7711 | 0.4874 | 0.1434 | 1.007 | -0.006 |
| D_intercept_slope (target_validation) | logistic_regression | 30,227 | 0.2217 | 0.7711 | 0.4874 | 0.1434 | 1.005 | -0.006 |
| D_intercept (target_validation) | xgboost | 30,227 | 0.2217 | 0.7731 | 0.4890 | 0.1429 | 1.087 | -0.006 |
| D_intercept_slope (target_validation) | xgboost | 30,227 | 0.2217 | 0.7731 | 0.4890 | 0.1428 | 0.998 | -0.006 |

### NFHS-5 cohort sensitivities for B (validation)

| Cohort | Model | n | ROC-AUC | AP | Brier | Cal. intercept |
|---|---|---|---|---|---|---|
| primary | logistic_regression | 30,227 | 0.7711 | 0.4874 | 0.1457 | 0.303 |
| nfhs5_cmc_0_59 | logistic_regression | 29,994 | 0.7710 | 0.4871 | 0.1457 | 0.303 |
| exclude_other_facility | logistic_regression | 30,139 | 0.7702 | 0.4874 | 0.1461 | 0.303 |
| exclude_calendar_overlap | logistic_regression | 19,653 | 0.7742 | 0.5026 | 0.1469 | 0.323 |
| primary | xgboost | 30,227 | 0.7731 | 0.4890 | 0.1454 | 0.298 |
| nfhs5_cmc_0_59 | xgboost | 29,994 | 0.7729 | 0.4886 | 0.1455 | 0.299 |
| exclude_other_facility | xgboost | 30,139 | 0.7725 | 0.4892 | 0.1458 | 0.300 |
| exclude_calendar_overlap | xgboost | 19,653 | 0.7759 | 0.5019 | 0.1469 | 0.324 |

### Contrasts

| Model | Contrast | ΔAUC | ΔAP | ΔAP lift | ΔBrier |
|---|---|---|---|---|---|
| logistic_regression | B_target_validation_minus_A_source_validation | -0.0036 | +0.0610 | -0.143 | +0.0190 |
| logistic_regression | B_target_validation_minus_A_source_holdout | -0.0074 | +0.0508 | -0.198 | +0.0202 |
| logistic_regression | B_target_non_test_minus_A_source_holdout | -0.0069 | +0.0474 | -0.219 | +0.0207 |
| logistic_regression | C_within_target_minus_B_zero_shot | +0.0018 | +0.0025 | +0.011 | -0.0025 |
| logistic_regression | B_minus_A_validation_v005_descriptive | +0.0067 | +0.0486 | -0.238 | +0.0087 |
| logistic_regression | D_intercept_minus_B_zero_shot | +0.0000 | +0.0000 | +0.000 | -0.0022 |
| logistic_regression | D_intercept_slope_minus_B_zero_shot | +0.0000 | +0.0000 | +0.000 | -0.0022 |
| xgboost | B_target_validation_minus_A_source_validation | -0.0039 | +0.0609 | -0.145 | +0.0195 |
| xgboost | B_target_validation_minus_A_source_holdout | -0.0075 | +0.0554 | -0.174 | +0.0202 |
| xgboost | B_target_non_test_minus_A_source_holdout | -0.0058 | +0.0534 | -0.189 | +0.0204 |
| xgboost | C_within_target_minus_B_zero_shot | +0.0012 | +0.0026 | +0.012 | -0.0027 |
| xgboost | B_minus_A_validation_v005_descriptive | +0.0053 | +0.0498 | -0.235 | +0.0092 |
| xgboost | D_intercept_minus_B_zero_shot | +0.0000 | +0.0000 | +0.000 | -0.0026 |
| xgboost | D_intercept_slope_minus_B_zero_shot | +0.0000 | +0.0000 | +0.000 | -0.0026 |

## Post-execution verification

### 1. Adversarial target-label mutation test
- Synthetic `test_target_label_mutation_bitwise_identical`: **PASS** (NFHS-5 labels permuted, flipped, all-zero, removed: predictions, threshold and frozen model digest bitwise identical for both models).
- Runtime check on the real data (full source pipeline re-run with permuted / removed NFHS-5 labels): logistic_regression_target_labels_permuted **PASS**, logistic_regression_target_labels_removed **PASS**, xgboost_target_labels_permuted **PASS**, xgboost_target_labels_removed **PASS**.
- Synthetic suite overall: 13/13 passed.

### 2. NFHS-4 within-wave validation (A) vs zero-shot NFHS-5 (B)

| Model | Set | n births | Prevalence | ROC-AUC | AP | Brier | Cal. slope | Cal. intercept |
|---|---|---|---|---|---|---|---|---|
| logistic_regression | A: NFHS-4 validation | 29,414 | 0.1821 | 0.7747 | 0.4264 | 0.1266 | 1.000 | 0.008 |
| logistic_regression | A: NFHS-4 holdout | 29,369 | 0.1822 | 0.7785 | 0.4366 | 0.1254 | 1.017 | -0.010 |
| logistic_regression | B: NFHS-5 validation (zero-shot) | 30,227 | 0.2217 | 0.7711 | 0.4874 | 0.1457 | 1.007 | 0.303 |
| logistic_regression | B: NFHS-5 non-test (zero-shot) | 171,147 | 0.2222 | 0.7716 | 0.4840 | 0.1462 | 1.003 | 0.308 |
| xgboost (primary) | A: NFHS-4 validation | 29,414 | 0.1821 | 0.7769 | 0.4281 | 0.1260 | 1.096 | 0.010 |
| xgboost (primary) | A: NFHS-4 holdout | 29,369 | 0.1822 | 0.7806 | 0.4336 | 0.1252 | 1.112 | -0.006 |
| xgboost (primary) | B: NFHS-5 validation (zero-shot) | 30,227 | 0.2217 | 0.7731 | 0.4890 | 0.1454 | 1.087 | 0.298 |
| xgboost (primary) | B: NFHS-5 non-test (zero-shot) | 171,147 | 0.2222 | 0.7748 | 0.4870 | 0.1456 | 1.089 | 0.303 |

### 3. Temporal transfer penalty and prevalence shift

- **logistic_regression:** ΔROC-AUC (B − A, validation) = -0.0036 (loss); vs NFHS-4 holdout -0.0074. ΔAP = +0.0610; ΔBrier = +0.0190. Prevalence 0.1821 (NFHS-4) → 0.2217 (NFHS-5). AP lift over prevalence 2.341 → 2.199 (Δ -0.143). AP rose even though ROC-AUC fell.
- **xgboost:** ΔROC-AUC (B − A, validation) = -0.0039 (loss); vs NFHS-4 holdout -0.0075. ΔAP = +0.0609; ΔBrier = +0.0195. Prevalence 0.1821 (NFHS-4) → 0.2217 (NFHS-5). AP lift over prevalence 2.351 → 2.206 (Δ -0.145). AP rose even though ROC-AUC fell.
- **Why AP moves with prevalence:** a classifier with no skill has AP equal to the outcome prevalence, so a higher NFHS-5 cesarean prevalence raises AP mechanically even if ranking quality is unchanged or worse. AP lift (AP / prevalence) and ROC-AUC, which does not depend on prevalence, are the fairer cross-wave comparisons; raw ΔAP should not be read as better or worse transfer.
- **Calibration:** a positive NFHS-5 calibration intercept means the frozen NFHS-4 model under-predicts NFHS-5 cesarean risk on average (higher target prevalence); the slope shows whether risk spread is preserved.
- No intervals (survey-design estimator NOT approved); point differences only.

## Limitations

- Exploratory: one source split and one provisional target split; no intervals; V1 test previously examined.
- Wave differences mix calendar time, questionnaire changes, case mix, sampling frame and reporting; the penalty is not a causal effect of time and not a clinical validation.
- The C reference uses NFHS-5 labels and a different training population; B − C is context, not an isolated wave-transfer effect.
- D uses NFHS-5 labels (target supervision). Monotone logistic recalibration with a positive slope cannot change ROC-AUC; it changes calibration and thresholded metrics only.
- NFHS-4 and NFHS-5 birth calendars overlap; see `exclude_calendar_overlap`.
- Retrospective survey-recorded features; interview-time wealth/education/residence; not prenatal prediction. Weighted (v005) metrics are descriptive only.
