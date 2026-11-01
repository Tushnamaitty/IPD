# A7: matched multi-model comparison, calibration, SHAP and robustness

Phase-3 experiment of the A1/A4/A7 rework. One predeclared evaluation contract
on the frozen C1 cohort and the provisional mother-grouped split:

- candidates: logistic regression, XGBoost, CatBoost on `F1` (8 core features)
  and on `F1_detail_history` (core + verbatim facility detail + observed
  earlier-institutional-birth history), six candidates total;
- every candidate is fitted on identical fit-pool rows and scored on identical
  validation keys; membership digests are asserted against the frozen A1
  train/validation digests and the split manifest hash must match the frozen
  value;
- tuning (StratifiedGroupKFold, three folds, average precision) runs only
  inside the training pool and its payload is cached under `outputs/private/`;
- calibrators (Platt, isotonic) are fitted only on a deterministic 20 percent
  mother-grouped carve of the training pool (seed 4208); the variant with the
  lowest validation Brier is selected per candidate (tie order: uncalibrated,
  Platt, isotonic);
- thresholds: primary 0.50 plus exploratory F1-optimal, Youden and
  recall >= 0.80 thresholds selected on validation only;
- the development test partition is scored once per recording; the cache guard
  `outputs/private/a7_test_scored.json` prevents rescoring. The original
  recording is the committed scoring; the later verification rescoring is
  disclosed under Deviations and caveats. The test is labelled
  development-test because of documented V1 exposure;
- SHAP: TreeExplainer in margin space on a seeded 5000-row validation sample,
  encoded columns aggregated back to source families via the fitted encoder
  categories, ranks compared across grouped fold models and supported
  subgroups (>= 200 rows and >= 30 rows per class);
- robustness: three repeated grouped splits (seeds 7, 2026, 99, XGBoost on both
  tiers), m15=96 exclusion with inherited partition, most-recent-birth-per-
  mother subset, facility/residence supported subgroups, pooled supported
  states, and a PSU (v021) overlap check;
- F0 reconciliation: if `--f0-parquet` is supplied and the processed parquet
  matches the frozen cohort key-for-key and label-for-key, the 36-feature
  historical XGBoost is fitted with the A7 reference config (not an exact V1
  reproduction); otherwise it is reported as not run.

## Run

```powershell
& .venv\Scripts\python.exe version2_ml\experiments\A7\src\run_a7.py `
    --nfhs5-path "<local>\IABR7EFL.DTA" `
    --f0-parquet "<local>\df_model_v1.parquet"
```

Optional flags: `--skip-shap`, `--skip-robustness`, `--force-retune`,
`--seed`, `--out-dir`, `--cache-dir`, `--allow-test-score`. The runner refuses
to score the development test partition when no `a7_test_scored.json` exists in
the cache directory; `--allow-test-score` is required to acknowledge a new
scoring.

## Provenance and verification disclosure

- The committed outputs are the original 2026-10-17 recording and were not
  re-recorded for the global-environment verification.
- A global-environment verification rerun reproduced all tables and figures and
  the CatBoost validation metrics with zero value differences.
- The development test partition was scored twice on this machine (the original
  recording and one sandbox verification run). The values were identical and no
  decision depended on the second scoring.
- Results remain non-confirmatory because the same NFHS-5 outcomes were
  inspected during V1 work (development-test exposure).

## Deviations and caveats

- **Threshold.** Thresholded metrics (precision, recall, F1) use the fixed
  prespecified 0.50 threshold, not a validation-selected threshold. This is a
  deviation from handoff sections 4 and 7. They are therefore not comparable
  across calibration states; ROC-AUC, AP and Brier are the appropriate
  comparisons.
- **Calibration encoding.** `aggregate_model_comparison.csv` has no
  `calibration_method` column. For each of the six model/tier candidates, its
  three validation rows are in fixed order: uncalibrated, Platt, isotonic.
  This was verified against `calibration_summary.csv` by matching Brier scores.
- **Facility SHAP.** `F1_detail_history` contains both `facility_type` and
  `facility_detail`; combined facility share in SHAP is 0.4148 overall (41.5%;
  folds 0.4114-0.4188; urban 0.4011, rural 0.4177) for XGBoost. The tier's
  ROC-AUC/AP uplift bundles facility detail with observed history and is
  decomposed only in A4's matched ablations, not in A7.
- **F0.** The F0 row is uncalibrated, uses raw probabilities at 0.50, is not
  comparable to calibrated rows, and is a reconciled 36-feature reference
  fitted with A7 parameters, not a reproduction of the V1 score.
- **Test scoring.** The original note stated the test was scored once; that
  statement referred to the recording run, but the test partition was scored a
  second time in a sandbox verification run on 2026-10-19. Values were
  identical, no decision depended on it, and the committed outputs remain the
  original recording.
- **Model comparisons.** Between-model differences are smaller than
  split-to-split variation for ROC-AUC in both tiers (0.0033-0.0056 versus
  0.0075-0.0081) and for AP in `F1_detail_history` (0.0066 versus 0.0128).
  The `F1` AP gap for logistic regression (~0.026) is larger than the
  repeated-split AP range (~0.009). No model is declared superior.
- **History quality.** `history_quality_flag` is constant in validation (all
  validation rows are `ok`; the 14 ambiguous cohort rows fall outside
  validation), so its SHAP share is 0.0000 and it contributes no validation
  split signal.
- **Robustness rows.** Repeated-split, m15=96-exclusion and
  most-recent-birth lines use uncalibrated raw XGBoost probabilities;
  m15=96 and most-recent-birth are `F1`-only. Subgroup and pooled-state lines
  are `F1_detail_history`-only and use the selected Platt-calibrated
  probabilities. All prediction lines were evaluated at threshold 0.50,
  although the reported ROC-AUC, AP and Brier do not depend on it; Brier scores
  mix calibration states and are not directly comparable across line types.
  `psu_overlap` is not a prediction line.

## Tests

```powershell
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\A7\tests -v
```

Synthetic-only unit tests cover calibrator monotonicity and range, intercept/
slope recovery, threshold behaviour, carve determinism and group exclusivity,
tuning determinism and absence of evaluation-label parameters, partition
guards, preprocessing invariance to held-out labels, SHAP family
reconciliation and aggregation, rank stability and PSU overlap summarisation.

## Outputs (aggregate only)

- `aggregate_model_comparison.csv` (locked schema): all candidates x
  calibration methods on validation plus selected-method development-test rows
  and F0 rows if reconciled;
- `calibration_summary.csv`: Brier, intercept, slope and thresholds per variant;
- `a7_tuning_summary.csv`: grouped-CV results and selected configuration;
- `a7_robustness.csv`: repeated splits, sensitivity, most-recent-birth,
  subgroups, PSU overlap;
- `shap_feature_family_summary.csv` and `a7_shap_stability.csv`;
- `a7_reconciliation.csv`: F0 match counts and PSU overlap;
- `fig_a7_roc_pr.png`, `fig_a7_calibration.png`, `fig_a7_shap_stability.png`;
- `a7_methods_results.md`, `a7_run_summary.json`.

Disclosure: positive counts below 10 are suppressed; all metrics are
unweighted point estimates without design-based intervals; the development
test is not an independent test.
