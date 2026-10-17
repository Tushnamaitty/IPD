# A3 — State × sector grouped out-of-fold target encoding: Methods and Results note

**Status: EXPLORATORY development result.** Provisional mother-grouped split; metrics are on the *validation* partition; the test partition was not scored. No confidence intervals or significance claims.

## Methods

- **Data:** NFHS-5 201,311 institutional births, 158,429 mothers (institutional m15 in 20-27/30-33/96; b19 in 0..59). Shared provisional mother-grouped split identical to A2: train 140,920, validation 30,227, test 30,164.
- **Encoding:** TE = (y + α·μ)/(n + α) for state × sector, state and sector, with μ the overall rate of the rows fitting the table. Levels with fewer than 30 births are unsupported. Fallback for state × sector: cell → state → sector → μ (flagged).
- **Grouped OOF:** outer-train mothers are split by StratifiedGroupKFold(3, shuffle, 42). Each fold is encoded by tables fitted on the other two folds (zero mother overlap asserted per fold); validation (and test) rows are encoded by one table fitted on 100% of outer train. Validation and test outcomes never enter any table.
- **α selection:** grid [0, 1, 5, 20, 50, 100, 300], minimum OOF log-loss of the state × sector encoding on outer-train rows only (ties → larger α). Selected α = 5 (primary). α = [5, 100] reported as exploratory sensitivity, not used for selection.
- **Arms:** (i) eight common-core features; (ii) (i) + NFHS-5 state indicators (the sector indicator, facility_type, is already in the core); (iii) (ii) + OOF state × sector TE. Exploratory: (ii) + all three TEs and the fallback flag; (i) + TE without state indicators. Logistic regression and XGBoost with fixed `make_estimator` defaults; train-only preprocessing; max-F1 threshold on validation.
- **Reproducibility:** arms (i)/(ii) vs A2 B0/R1 on the same split: PASS (max |difference| 0.0e+00).

## Results (primary cohort, validation, selected α)

| Model | Arm | ROC-AUC | AP | Brier | Cal. slope | Cal. intercept |
|---|---|---|---|---|---|---|
| logistic_regression | i_core | 0.7729 | 0.4899 | 0.1431 | 1.006 | -0.006 |
| logistic_regression | ii_state_sector_indicators | 0.8064 | 0.5539 | 0.1340 | 0.992 | -0.003 |
| logistic_regression | iii_indicators_plus_oof_te | 0.8102 | 0.5655 | 0.1327 | 0.991 | -0.005 |
| logistic_regression | x_indicators_plus_all_te | 0.8102 | 0.5655 | 0.1327 | 0.993 | -0.004 |
| logistic_regression | x_core_plus_oof_te_only | 0.8096 | 0.5636 | 0.1330 | 0.994 | -0.006 |
| xgboost | i_core | 0.7743 | 0.4917 | 0.1427 | 1.088 | -0.007 |
| xgboost | ii_state_sector_indicators | 0.8021 | 0.5462 | 0.1358 | 1.226 | -0.007 |
| xgboost | iii_indicators_plus_oof_te | 0.8101 | 0.5643 | 0.1326 | 1.080 | -0.007 |
| xgboost | x_indicators_plus_all_te | 0.8101 | 0.5644 | 0.1326 | 1.082 | -0.007 |
| xgboost | x_core_plus_oof_te_only | 0.8100 | 0.5641 | 0.1326 | 1.078 | -0.007 |

Validation prevalence 0.2217 (n=30,227 births). AP depends on prevalence.

## Matched deltas (validation)

| Cohort | α | Model | Comparison | ΔAUC | ΔAP | ΔBrier |
|---|---|---|---|---|---|---|
| exclude_other_facility | — | logistic_regression | ii_state_sector_indicators vs i_core | +0.03368 | +0.06409 | -0.00914 |
| exclude_other_facility | 5 | logistic_regression | iii_indicators_plus_oof_te vs ii_state_sector_indicators | +0.00370 | +0.01139 | -0.00130 |
| exclude_other_facility | 5 | logistic_regression | iii_indicators_plus_oof_te vs i_core | +0.03737 | +0.07548 | -0.01044 |
| exclude_other_facility | 5 | logistic_regression | x_indicators_plus_all_te vs ii_state_sector_indicators | +0.00379 | +0.01158 | -0.00133 |
| exclude_other_facility | 5 | logistic_regression | x_core_plus_oof_te_only vs i_core | +0.03684 | +0.07361 | -0.01014 |
| exclude_other_facility | 5 | logistic_regression | x_core_plus_oof_te_only vs ii_state_sector_indicators | +0.00316 | +0.00952 | -0.00100 |
| exclude_other_facility | — | xgboost | ii_state_sector_indicators vs i_core | +0.02841 | +0.05504 | -0.00702 |
| exclude_other_facility | 5 | xgboost | iii_indicators_plus_oof_te vs ii_state_sector_indicators | +0.00745 | +0.01769 | -0.00314 |
| exclude_other_facility | 5 | xgboost | iii_indicators_plus_oof_te vs i_core | +0.03586 | +0.07273 | -0.01016 |
| exclude_other_facility | 5 | xgboost | x_indicators_plus_all_te vs ii_state_sector_indicators | +0.00743 | +0.01777 | -0.00311 |
| exclude_other_facility | 5 | xgboost | x_core_plus_oof_te_only vs i_core | +0.03579 | +0.07255 | -0.01013 |
| exclude_other_facility | 5 | xgboost | x_core_plus_oof_te_only vs ii_state_sector_indicators | +0.00738 | +0.01752 | -0.00311 |
| nfhs5_cmc_0_59 | — | logistic_regression | ii_state_sector_indicators vs i_core | +0.03369 | +0.06452 | -0.00918 |
| nfhs5_cmc_0_59 | 5 | logistic_regression | iii_indicators_plus_oof_te vs ii_state_sector_indicators | +0.00374 | +0.01150 | -0.00131 |
| nfhs5_cmc_0_59 | 5 | logistic_regression | iii_indicators_plus_oof_te vs i_core | +0.03743 | +0.07602 | -0.01049 |
| nfhs5_cmc_0_59 | 5 | logistic_regression | x_indicators_plus_all_te vs ii_state_sector_indicators | +0.00380 | +0.01174 | -0.00134 |
| nfhs5_cmc_0_59 | 5 | logistic_regression | x_core_plus_oof_te_only vs i_core | +0.03683 | +0.07405 | -0.01017 |
| nfhs5_cmc_0_59 | 5 | logistic_regression | x_core_plus_oof_te_only vs ii_state_sector_indicators | +0.00313 | +0.00953 | -0.00099 |
| nfhs5_cmc_0_59 | — | xgboost | ii_state_sector_indicators vs i_core | +0.02833 | +0.05616 | -0.00708 |
| nfhs5_cmc_0_59 | 5 | xgboost | iii_indicators_plus_oof_te vs ii_state_sector_indicators | +0.00774 | +0.01704 | -0.00313 |
| nfhs5_cmc_0_59 | 5 | xgboost | iii_indicators_plus_oof_te vs i_core | +0.03608 | +0.07321 | -0.01021 |
| nfhs5_cmc_0_59 | 5 | xgboost | x_indicators_plus_all_te vs ii_state_sector_indicators | +0.00770 | +0.01701 | -0.00312 |
| nfhs5_cmc_0_59 | 5 | xgboost | x_core_plus_oof_te_only vs i_core | +0.03616 | +0.07320 | -0.01022 |
| nfhs5_cmc_0_59 | 5 | xgboost | x_core_plus_oof_te_only vs ii_state_sector_indicators | +0.00782 | +0.01703 | -0.00314 |
| primary | — | logistic_regression | ii_state_sector_indicators vs i_core | +0.03352 | +0.06402 | -0.00911 |
| primary | 5 | logistic_regression | iii_indicators_plus_oof_te vs ii_state_sector_indicators | +0.00376 | +0.01155 | -0.00131 |
| primary | 5 | logistic_regression | iii_indicators_plus_oof_te vs i_core | +0.03728 | +0.07556 | -0.01042 |
| primary | 5 | logistic_regression | x_indicators_plus_all_te vs ii_state_sector_indicators | +0.00377 | +0.01159 | -0.00133 |
| primary | 5 | logistic_regression | x_core_plus_oof_te_only vs i_core | +0.03671 | +0.07364 | -0.01012 |
| primary | 5 | logistic_regression | x_core_plus_oof_te_only vs ii_state_sector_indicators | +0.00319 | +0.00963 | -0.00101 |
| primary | — | xgboost | ii_state_sector_indicators vs i_core | +0.02779 | +0.05455 | -0.00690 |
| primary | 5 | xgboost | iii_indicators_plus_oof_te vs ii_state_sector_indicators | +0.00804 | +0.01807 | -0.00322 |
| primary | 5 | xgboost | iii_indicators_plus_oof_te vs i_core | +0.03583 | +0.07261 | -0.01012 |
| primary | 5 | xgboost | x_indicators_plus_all_te vs ii_state_sector_indicators | +0.00800 | +0.01818 | -0.00320 |
| primary | 5 | xgboost | x_core_plus_oof_te_only vs i_core | +0.03575 | +0.07248 | -0.01010 |
| primary | 5 | xgboost | x_core_plus_oof_te_only vs ii_state_sector_indicators | +0.00796 | +0.01794 | -0.00320 |
| primary | 100 | logistic_regression | iii_indicators_plus_oof_te vs ii_state_sector_indicators | +0.00402 | +0.01078 | -0.00130 |
| primary | 100 | logistic_regression | iii_indicators_plus_oof_te vs i_core | +0.03754 | +0.07480 | -0.01041 |
| primary | 100 | logistic_regression | x_indicators_plus_all_te vs ii_state_sector_indicators | +0.00409 | +0.01098 | -0.00133 |
| primary | 100 | logistic_regression | x_core_plus_oof_te_only vs i_core | +0.03669 | +0.07164 | -0.01002 |
| primary | 100 | logistic_regression | x_core_plus_oof_te_only vs ii_state_sector_indicators | +0.00317 | +0.00762 | -0.00091 |
| primary | 100 | xgboost | iii_indicators_plus_oof_te vs ii_state_sector_indicators | +0.00742 | +0.01480 | -0.00290 |
| primary | 100 | xgboost | iii_indicators_plus_oof_te vs i_core | +0.03522 | +0.06934 | -0.00980 |
| primary | 100 | xgboost | x_indicators_plus_all_te vs ii_state_sector_indicators | +0.00727 | +0.01443 | -0.00286 |
| primary | 100 | xgboost | x_core_plus_oof_te_only vs i_core | +0.03515 | +0.06848 | -0.00976 |
| primary | 100 | xgboost | x_core_plus_oof_te_only vs ii_state_sector_indicators | +0.00735 | +0.01394 | -0.00285 |

## Post-execution verification

### 1. Adversarial tests
- Synthetic A3 suite: 15/15 passed.
- Singleton test (`test_singleton_state_sector_does_not_output_own_label`): **PASS**. An artificial one-birth, one-positive state x sector cell never receives its own label (1.0) as its OOF encoding for alpha in {0, 1, 20} and n_min in {1, 30}; it falls back to the sector rate from other folds.
- Validation/test label mutation (`test_validation_test_label_mutation_bitwise_identical`): **PASS** (flip, permute, all-1, all-0: all encodings and the full-train table digest bitwise identical; alpha selection unchanged).
- Same check on the real NFHS-5 data at runtime: flipped **PASS**, permuted **PASS**.
- Other required tests: mother-group isolation **PASS**, unseen-category fallback **PASS**, training row-order reconstruction **PASS**.

### 2. Arm (iii) vs Arm (ii) (primary cohort, validation, selected alpha)
- **logistic_regression:** Arm (iii) outperformed Arm (ii) on ROC-AUC: ΔAUC = +0.00376, ΔAP = +0.01155 (n = 30,227 identical births). |ΔAUC| is below the provisional promotion reference of 0.005 (ΔAP reference 0.01).
- **xgboost:** Arm (iii) outperformed Arm (ii) on ROC-AUC: ΔAUC = +0.00804, ΔAP = +0.01807 (n = 30,227 identical births). |ΔAUC| is at or above the provisional promotion reference of 0.005 (ΔAP reference 0.01).
- No interval is attached to these deltas (survey-design estimator NOT approved); they are point differences on one provisional split.

## Limitations

- Exploratory: one provisional split, validation partition, no intervals; old V1 test already examined.
- OOF encodings for training rows come from tables fitted on two-thirds of outer train, while validation rows use the full-train table; this standard cross-fitting asymmetry can slightly change the scale the model learns.
- With state indicators already in arm (ii), target encoding mainly adds the state × sector interaction; tree models can learn part of that interaction from indicators alone.
- Survey clusters cross partitions; mothers do not. State is interview residence. No causal reading of state or sector; retrospective survey features, not prenatal prediction.
- Per-state TE has no entry for an unseen state; A6 must use the documented fallback.
