# A8: history + state context together (decides FINAL_TIER)

**Question.** On the frozen NFHS-5 C1 cohort and A7's evaluation contract,
does adding state context to `F1_detail_history` add reliable predictive
value?

## Arms (identical rows within each split, validation partition only)

| Arm | Features | Added |
| --- | --- | --- |
| H0 | `F1_detail_history` (A7 reference) | 0 |
| H1 | H0 + `state_ut` categorical (A1 verbatim v024 labels) | 1 |
| H2 | H0 + `te_state_sector` (A3 grouped OOF state x sector target encoding) | 1 |
| H3 | H0 + `state_ut` + `te_state_sector` | 2 |

## Predeclared decision rule

Written into `config.yaml` (`decision_rule`) and here **before any real-data
run**:

- Adopt an arm over H0 only if, **for XGBoost**, delta ROC-AUC >= +0.005
  **and** delta AP >= +0.010 versus H0 on identical rows, on **all 4 seeds**
  (42, 7, 2026, 99).
- If several arms pass, choose the one with the fewest added features; ties
  go to H1.
- If none pass, `FINAL_TIER = F1_detail_history`.

Logistic-regression deltas are reported but do not enter the rule.

## Design

- **Cohort:** the A7 data path (`data_access.read_dta_subset`,
  `build_c1_cohort`, `make_core_features`, A4 `facility_detail` and
  `build_history_features`). The input sha256, the cohort counts
  (201,311 / 158,429 / 44,742) and the history reproduction (6,276) are
  asserted.
- **Splits:** `split_contracts.grouped_partition`, mother-grouped 70/15/15,
  seeds 42 (frozen), 7, 2026 and 99 (A7 robustness seeds). For seed 42 the
  run asserts:
  - the frozen split hash;
  - the A1 train and validation membership digests;
  - the A7 fit-pool and calibration-pool digests.
- **Test partition:** never indexed, predicted or scored.
- **Calibration:** A7's 20% mother-grouped carve of each seed's training pool
  (seed 4208). Platt scaling (`a7_calibration`) is fitted on the carve and
  applied to validation.
- **Models (fixed, no tuning):** A7's selected parameters, read from
  `A7/outputs/a7_run_summary.json` and asserted equal to the config.
  - XGBoost: `max_depth=4, n_estimators=200, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, logloss, hist`, with
    `scale_pos_weight` = fit-pool negatives / positives.
  - LR: `C=1, liblinear, max_iter=1000, class_weight=balanced`.
  - The estimator seed equals the split seed, as in `run_a7`.
- **Target encoding:** A3 `encode_partitions` (alpha 5, n_min 30, 3 inner
  mother-grouped folds, seed 42). The keys are the v024 code and
  `facility_type`, as in A3.
  - The encoder's only labelled partition is the **fit pool**: fit rows get
    OOF encodings, and calibration and validation rows get the 100% fit-pool
    table.
  - Calibration and validation labels therefore never reach the encoder, and
    the calibration rows are encoded the same way as validation rows.
- **Metrics (validation, calibrated probabilities):**
  - ROC-AUC, AP and Brier from `exp_metrics.binary_metrics`;
  - calibration intercept and slope from `a7_calibration`, matching A7's
    recorded columns;
  - F1 at the A7 F1-optimal threshold grid.

  Paired deltas versus H0 are computed per seed and model, and require
  identical validation membership digests.
- **Gate:** H0 XGBoost on seed 42 must reproduce A7 validation ROC-AUC
  0.8298 +/- 0.003 and AP 0.6113 +/- 0.005, or the run stops before writing
  anything.
- **Runtime leakage checks (seed 42, H3, XGBoost):** permuting and flipping
  validation labels, and separately calibration labels, must leave the
  encoder digest, the encoded features and the base predictions
  bit-identical. Validation-label mutation must also leave the calibrated
  predictions identical.
- **Also reported:**
  - the state x residence support screen (A1 D9 gate: >= 200 training rows
    and >= 30 per class, seed-42 training rows);
  - SHAP family shares for the selected arm (`a7_shap.shap_family_table`,
    5,000 validation rows, seed 4209).

## Run

```powershell
$env:NFHS5_PATH = "C:\path\to\IABR7EFL.DTA"
& .venv\Scripts\python.exe version2_ml\experiments\A8\src\run_a8.py
```

The runner first runs the A8 synthetic suite and stops if it fails.

## Tests

```powershell
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\A8\tests -p "test_*.py" -v
```

The tests are synthetic only (no NFHS records). They cover:
- arm manifests and column order;
- the encoder's fit-pool-only table;
- held-out label permutation and removal leaving the encoder, features and
  predictions bit-identical;
- calibration labels reaching only the calibrator;
- test rows never being read;
- overlap guards;
- paired-delta arithmetic and the matched-row guard;
- every branch of the decision rule;
- LF-hashed manifests, refusal of row-level columns, and small-count
  suppression.

## Outputs (`outputs/`, aggregate only)

- `a8_ablation.csv`: Section-8 schema (as A7 `aggregate_model_comparison.csv`),
  one row per seed x model x arm.
- `a8_paired_deltas.csv`: arm minus H0 for every metric, per seed and model.
- `a8_decision.json`: `selected_tier`, `rule`, `passed_arms`,
  `per_seed_deltas`.
- `a8_support_screen.csv`: state x residence training support (counts < 10
  suppressed).
- `a8_shap_families.csv`: SHAP family shares for the selected arm.
- `a8_methods_results.md`, `fig_a8_deltas.png`.
- `MANIFEST.json`: sha256 of each output, git commit, config hash, software
  versions, seeds, split manifest hashes, seed-42 membership digests and the
  QA log.

Disclosure: no keys, rows or per-birth predictions; counts below 10
suppressed; unweighted point estimates without design-based intervals.
