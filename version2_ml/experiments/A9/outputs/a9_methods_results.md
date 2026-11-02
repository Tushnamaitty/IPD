# A9 methods and results: final PSU-grouped evaluation

- protocol_commit: `0ce1d4884239ea35f75d37afc338550bf5dde52b` (PROTOCOL.md committed before any scoring)
- repo_commit at run: `0ce1d4884239ea35f75d37afc338550bf5dde52b`
- generated_at_utc: 2026-11-02T07:02:37+00:00
- input_sha256: 93267b17596929371dc3190d17af114271851f14d1f566678ffceda2db4cae6e
- FINAL_TIER: `F1_detail_history_state` (A8 arm H1, commit f03ba25)
- seeds: [11, 22, 33, 44, 55]; PSU (v021) grouped 70/15/15 stratified on PSU outcome rate; bootstrap B = 1000, seed 2027, PSU resampling within test
- V1_spec: reconciled

## Predeclared statements (wording fixed in PROTOCOL.md)

1. **V2 discrimination is superior to the V1 specification.** lower 95% bound of delta ROC-AUC (V2 - V1_spec) > 0 on 5 of 5 seeds (required >= 4); per-seed lower bounds: seed 11: +0.0448, seed 22: +0.0420, seed 33: +0.0411, seed 44: +0.0428, seed 55: +0.0413.
2. **V2 is well calibrated.** pooled slope 1.008 (required 0.9-1.1); pooled intercept -0.011 (required |.| <= 0.05).
3. **XGBoost and LR are equivalent.** pooled delta ROC-AUC (V2 - V2_LR) +0.0023, 95% CI [+0.0016, +0.0029] (margin ±0.01); per seed: seed 11: [+0.0012, +0.0042], seed 22: [+0.0004, +0.0031], seed 33: [+0.0010, +0.0038], seed 44: [+0.0008, +0.0034], seed 55: [+0.0010, +0.0039].

## Paper table (pooled = mean across 5 seeds; 95% CI from the across-seed mean of PSU bootstrap replicates)

| Model | Features | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier (95% CI) | Cal. slope | Cal. intercept |
| --- | --- | --- | --- | --- | --- | --- |
| V2 | F1_detail_history + state_ut (FINAL_TIER, 13) | 0.8487 (0.8457-0.8516) | 0.6477 (0.6396-0.6550) | 0.1194 (0.1179-0.1211) | 1.008 | -0.011 |
| V2_LR | F1_detail_history + state_ut (FINAL_TIER, 13) | 0.8464 (0.8434-0.8495) | 0.6425 (0.6346-0.6500) | 0.1204 (0.1187-0.1221) | 1.006 | -0.017 |
| F1_core | 8 core features (F1) | 0.7782 (0.7745-0.7820) | 0.4923 (0.4849-0.4997) | 0.1419 (0.1399-0.1439) | 1.024 | +0.000 |
| V1_spec | V1 F0_historical_36 (incl. BMI, ANC) | 0.8013 (0.7977-0.8048) | 0.5283 (0.5208-0.5360) | 0.1364 (0.1345-0.1384) | 1.015 | -0.008 |
| V1 (recorded) | original V1 report, different split, not paired | 0.8037 (not available) | 0.5334 (not available) | 0.1358 (not available) | not reported | not reported |

The V1 recorded row is the original V1 report (different split, mother-level random test set, not paired, no CI available); it is shown for orientation only.

## Paired deltas (test, ROC-AUC and PR-AUC)

| comparison | seed | metric | delta | 95% CI |
| --- | --- | --- | --- | --- |
| V2 - V1_spec | 11 | roc_auc | +0.0496 | [+0.0448, +0.0549] |
| V2 - V1_spec | 22 | roc_auc | +0.0473 | [+0.0420, +0.0523] |
| V2 - V1_spec | 33 | roc_auc | +0.0461 | [+0.0411, +0.0509] |
| V2 - V1_spec | 44 | roc_auc | +0.0475 | [+0.0428, +0.0521] |
| V2 - V1_spec | 55 | roc_auc | +0.0464 | [+0.0413, +0.0513] |
| V2 - V1_spec | pooled | roc_auc | +0.0474 | [+0.0449, +0.0500] |
| V2 - V1_spec | 11 | average_precision | +0.1246 | [+0.1126, +0.1363] |
| V2 - V1_spec | 22 | average_precision | +0.1193 | [+0.1082, +0.1307] |
| V2 - V1_spec | 33 | average_precision | +0.1134 | [+0.1015, +0.1246] |
| V2 - V1_spec | 44 | average_precision | +0.1241 | [+0.1128, +0.1352] |
| V2 - V1_spec | 55 | average_precision | +0.1156 | [+0.1035, +0.1267] |
| V2 - V1_spec | pooled | average_precision | +0.1194 | [+0.1136, +0.1252] |
| V2 - F1_core | 11 | roc_auc | +0.0716 | [+0.0666, +0.0767] |
| V2 - F1_core | 22 | roc_auc | +0.0728 | [+0.0671, +0.0783] |
| V2 - F1_core | 33 | roc_auc | +0.0684 | [+0.0631, +0.0737] |
| V2 - F1_core | 44 | roc_auc | +0.0713 | [+0.0666, +0.0757] |
| V2 - F1_core | 55 | roc_auc | +0.0684 | [+0.0631, +0.0735] |
| V2 - F1_core | pooled | roc_auc | +0.0705 | [+0.0679, +0.0730] |
| V2 - F1_core | 11 | average_precision | +0.1581 | [+0.1466, +0.1693] |
| V2 - F1_core | 22 | average_precision | +0.1523 | [+0.1408, +0.1639] |
| V2 - F1_core | 33 | average_precision | +0.1491 | [+0.1365, +0.1606] |
| V2 - F1_core | 44 | average_precision | +0.1625 | [+0.1512, +0.1736] |
| V2 - F1_core | 55 | average_precision | +0.1551 | [+0.1436, +0.1658] |
| V2 - F1_core | pooled | average_precision | +0.1554 | [+0.1500, +0.1609] |
| V2 - V2_LR | 11 | roc_auc | +0.0027 | [+0.0012, +0.0042] |
| V2 - V2_LR | 22 | roc_auc | +0.0017 | [+0.0004, +0.0031] |
| V2 - V2_LR | 33 | roc_auc | +0.0024 | [+0.0010, +0.0038] |
| V2 - V2_LR | 44 | roc_auc | +0.0021 | [+0.0008, +0.0034] |
| V2 - V2_LR | 55 | roc_auc | +0.0024 | [+0.0010, +0.0039] |
| V2 - V2_LR | pooled | roc_auc | +0.0023 | [+0.0016, +0.0029] |
| V2 - V2_LR | 11 | average_precision | +0.0070 | [+0.0041, +0.0099] |
| V2 - V2_LR | 22 | average_precision | +0.0032 | [+0.0003, +0.0061] |
| V2 - V2_LR | 33 | average_precision | +0.0043 | [+0.0016, +0.0073] |
| V2 - V2_LR | 44 | average_precision | +0.0061 | [+0.0034, +0.0090] |
| V2 - V2_LR | 55 | average_precision | +0.0052 | [+0.0024, +0.0080] |
| V2 - V2_LR | pooled | average_precision | +0.0052 | [+0.0039, +0.0065] |

## Per-seed test metrics (V2)

| seed | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier | slope | intercept | thr F1 | F1 | thr rec80 | recall | precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 0.8446 (0.8382-0.8504) | 0.6450 (0.6313-0.6593) | 0.1221 | 0.988 | -0.017 | 0.31 | 0.5983 | 0.20 | 0.8050 | 0.4605 |
| 22 | 0.8479 (0.8417-0.8538) | 0.6443 (0.6310-0.6580) | 0.1192 | 0.996 | -0.018 | 0.28 | 0.5997 | 0.19 | 0.8241 | 0.4529 |
| 33 | 0.8492 (0.8430-0.8554) | 0.6474 (0.6337-0.6621) | 0.1192 | 0.997 | -0.024 | 0.33 | 0.5983 | 0.20 | 0.8093 | 0.4584 |
| 44 | 0.8524 (0.8468-0.8581) | 0.6513 (0.6376-0.6637) | 0.1175 | 1.030 | -0.043 | 0.31 | 0.6016 | 0.21 | 0.8136 | 0.4574 |
| 55 | 0.8494 (0.8436-0.8552) | 0.6506 (0.6367-0.6643) | 0.1191 | 1.030 | +0.049 | 0.29 | 0.6028 | 0.20 | 0.8031 | 0.4628 |

## Split audit

| seed | train births/PSUs | validation births/PSUs | test births/PSUs | test prevalence | PSU overlap | mother overlap |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 140,763/20,702 | 30,483/4,436 | 30,065/4,437 | 0.2260 | 0 | 0 |
| 22 | 141,031/20,702 | 30,211/4,436 | 30,069/4,437 | 0.2204 | 0 | 0 |
| 33 | 140,864/20,702 | 30,501/4,436 | 29,946/4,437 | 0.2215 | 0 | 0 |
| 44 | 140,996/20,702 | 30,090/4,436 | 30,225/4,437 | 0.2195 | 0 | 0 |
| 55 | 140,735/20,702 | 30,071/4,436 | 30,505/4,437 | 0.2219 | 0 | 0 |

## V1_spec reconciliation

- status: reconciled
- parquet_rows: 201311
- cohort_rows: 201311
- matched_keys: 201311
- cohort_only_keys: 0
- parquet_only_keys: 0
- duplicate_parquet_keys: 0
- outcome_mismatch: 0
- facility_mismatch: 0

## Test scoring log

- seed 11: cache_no_rescore (guard completed 2026-11-02T06:35:40+00:00)
- seed 22: cache_no_rescore (guard completed 2026-11-02T06:41:45+00:00)
- seed 33: cache_no_rescore (guard completed 2026-11-02T06:46:52+00:00)
- seed 44: cache_no_rescore (guard completed 2026-11-02T06:53:13+00:00)
- seed 55: cache_no_rescore (guard completed 2026-11-02T06:59:59+00:00)

## Quality checks

| check | status | detail |
| --- | --- | --- |
| protocol_files_unchanged_since_protocol_commit | PASS | 0ce1d48 |
| synthetic_a9_tests | PASS | 24 synthetic tests OK |
| input_sha256_matches_a7 | PASS | 93267b1759692937 |
| cohort_and_history_frozen | PASS | 201311/158429/44742; 6276 |
| every_mother_in_exactly_one_psu | PASS | 29575 PSUs |
| v1_spec_parquet_reconciled | PASS | reconciled |
| zero_psu_and_mother_overlap_all_seeds | PASS | asserted per seed before fitting |
| guard_files_completed | PASS | test_scored_11.json, test_scored_22.json, test_scored_33.json, test_scored_44.json, test_scored_55.json |

## Disclosure and limitations

- Unweighted primary analysis; weighted (v005) metrics in `a9_weighted_sensitivity.csv` are descriptive only.
- Percentile CIs from a PSU cluster bootstrap within each test partition. They ignore the survey's stratification and are not design-based standard errors.
- The NFHS-5 outcomes were inspected in V1 and in A1-A8 development work; PSU-grouped test partitions are new, but the data are not an independent external validation.
- Aggregate outputs only; counts below 10 suppressed; no keys, rows or per-birth predictions written.
