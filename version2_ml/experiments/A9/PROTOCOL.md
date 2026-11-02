# A9 protocol: final evaluation of V2 with confidence intervals

Status: **predeclared**. This file is committed before any A9 code is run on
NFHS-5 data and before any test partition is scored. The commit that last
changed this file (the "protocol commit") is recorded in every A9 output. The
runner refuses to score if this file differs from the protocol commit. Any
deviation found later is reported as a deviation, never edited in here.

## 1. Question

On cluster (PSU) grouped splits with a predeclared analysis, does V2
(`FINAL_TIER`) beat the V1 36-feature specification, and by how much, with 95%
confidence intervals?

## 2. Data and cohort

- NFHS-5 birth recode `IABR7EFL.DTA`, read via the A7 data path
  (`data_access.read_dta_subset` with `v021` and `v005`, `build_c1_cohort`).
  The input sha256 must equal the A7 recording.
- Frozen C1 cohort, asserted: 201,311 births, 158,429 mothers, 44,742
  C-sections. History reproduction asserted (6,276 observed previous
  institutional C-sections).
- `FINAL_TIER = F1_detail_history_state`, which is A8 arm H1, selected by
  A8's predeclared rule (commit `f03ba25`): `F1_detail_history` plus
  `state_ut` (A1 verbatim v024 labels). The features are built with A8's
  `a8_arms` (H1) on the A7 tier frame.

## 3. Splits

- **Grouping unit:** the survey cluster (PSU) `v021`, as used by A7's PSU
  overlap check. Asserted before splitting: every mother lies in exactly one
  PSU, so PSU grouping is also mother-exclusive.
- **Fractions:** 70/15/15 train/validation/test by PSU.
- **Stratification:** on the PSU-level outcome rate, using five fixed strata:
  rate = 0; (0, 0.2]; (0.2, 0.4]; (0.4, 0.6]; (0.6, 1]. Within each stratum,
  PSUs are ordered by `sha256(f"{seed}|{psu}")` (the hashing scheme of
  `split_contracts.grouped_partition`). The first 70% of PSUs go to train,
  the next 15% to validation and the rest to test.
- **Seeds:** 11, 22, 33, 44, 55. These are new and were never used before.
- **Assertions for every seed:**
  - zero PSU overlap and zero mother overlap between partitions;
  - all three partitions contain both outcome classes;
  - `split_contracts.audit_partition` passes with both `v021` and `caseid` as
    the group.
- **Recorded per seed:** the split manifest hash
  (`manifests.split_manifest_hash` over (PSU, partition) pairs), membership
  digests, sizes and prevalence.
- **Calibration carve:** the A7 carve (`a7_candidates.carve_calibration_partition`,
  20%, seed 4208) applied to each seed's training pool, **grouped by PSU**
  (`group_col="v021"`). Models are fitted on the fit pool, and Platt
  calibrators on the carve.

## 4. Test-partition rule

- Each seed's test partition is scored **exactly once**. Guard file:
  `A9/outputs/private/test_scored_<seed>.json` (git-ignored).
- The guard is written with status `started` before the first test
  prediction, and set to `completed` with the seed's aggregate results
  (metrics, bootstrap summaries, calibration bins, ROC/PR curves on a fixed
  grid) once scoring finishes.
- A later run with a completed guard reuses the stored aggregates and never
  re-scores. A guard left at `started` blocks the seed until a human decides.
- No row-level data or per-birth predictions are ever written, including to
  `private/`.

## 5. Models (fixed recipe, no tuning)

All four are fitted on each seed's fit pool and Platt-calibrated
(`a7_calibration`) on the PSU-grouped carve. The estimator seed equals the
split seed. XGBoost `scale_pos_weight` = fit-pool negatives / positives
(`a7_candidates.scale_pos_weight`).

| Model | Algorithm | Features | Parameters |
| --- | --- | --- | --- |
| V2 | XGBoost | FINAL_TIER (13 columns) | A7 selected: max_depth 4, n_estimators 200, learning_rate 0.05, subsample 0.8, colsample_bytree 0.8, logloss, hist |
| V2_LR | Logistic regression | FINAL_TIER | A7 selected: C 1.0, liblinear, max_iter 1000, class_weight balanced |
| F1_core | XGBoost | 8 core features (A7 tier `F1`) | as V2 |
| V1_spec | XGBoost | F0_historical_36 | A7 reference config: max_depth 6, n_estimators 200, learning_rate 0.05, subsample 0.8, colsample_bytree 0.8, logloss, hist |

**V1_spec input** is `data/processed/df_model_v1.parquet`. It is reconciled
key-for-key with the cohort exactly as `run_a7._f0_stage` does:
- equal row counts, every key matched, no cohort-only, parquet-only or
  duplicate keys;
- zero outcome mismatches (`csection` vs `m17`) and zero facility mismatches
  (`delivery_place_code` vs `m15`).

If the file is missing or reconciliation fails, V1_spec is marked **not run**.
V2 is then compared only with V1's recorded numbers (ROC-AUC 0.8037, PR-AUC
0.5334, Brier 0.1358), labelled descriptive and not paired, and the
superiority statement reads "not shown to be superior". V1_spec keeps V1's
feature definitions as they are, including BMI and ANC items.

## 6. Thresholds (selected on validation only)

- **F1-optimal:** `a7_calibration.threshold_f1_optimal` on calibrated
  validation probabilities (grid 0.05-0.95, step 0.01).
- **Recall >= 0.80:** `a7_calibration.threshold_for_recall(..., 0.8)` on
  calibrated validation probabilities.
- Both thresholds are fixed before test scoring and are never re-selected on
  test.

## 7. Metrics (test partition, per model and seed, calibrated probabilities)

- **Discrimination and accuracy:** ROC-AUC, average precision (PR-AUC) and
  Brier (`exp_metrics.binary_metrics`).
- **Calibration:** intercept and slope (`a7_calibration.calibration_intercept_slope`,
  the joint logistic fit used by A7).
- **Thresholded:** precision, recall and F1 at the F1-optimal threshold and
  at the recall >= 0.80 threshold.
- **Weighted sensitivity** (`v005 / 1e6`): ROC-AUC, AP, Brier and weighted
  prevalence as point estimates. These are descriptive only and enter no
  statement.

## 8. Uncertainty

- **Cluster bootstrap within each seed's test partition:** resample test PSUs
  with replacement, B = 1,000. The generator is
  `numpy.random.default_rng(2027)`, initialised fresh for each seed. Each
  replicate's rows are all births of the drawn PSUs, with multiplicity.
- **Paired:** the same replicate draws are used for every model in a seed,
  and for the paired differences V2 - V1_spec, V2 - F1_core and V2 - V2_LR.
- **CIs:** percentile 95% intervals (2.5th and 97.5th percentiles) for every
  metric and every paired difference. A replicate where a metric is undefined
  (one outcome class) is dropped for that metric, and the count is reported.
- **Across seeds:**
  - mean, min and max of each metric;
  - **pooled estimate = mean across the 5 seeds**;
  - **pooled 95% CI** = percentile interval of the across-seed mean of
    replicate b (b = 1..1000). This is valid because seeds' bootstrap draws
    are generated independently.
  - Seed-level CIs are also reported.

## 9. Predeclared statements (wording fixed; only numbers are filled in)

1. **"V2 discrimination is superior to the V1 specification"** only if the
   lower 95% bound of delta ROC-AUC (V2 - V1_spec) is > 0 on at least 4 of 5
   seeds. Otherwise: **"V2 discrimination is not shown to be superior to the
   V1 specification"**.
2. **"V2 is well calibrated"** if the pooled calibration slope is in
   [0.9, 1.1] and |pooled intercept| <= 0.05. Otherwise: **"V2 is not shown to
   be well calibrated"**.
3. **"XGBoost and LR are equivalent"** if the pooled 95% CI of delta ROC-AUC
   (V2 - V2_LR) lies entirely within [-0.01, +0.01]. Otherwise: **"XGBoost and
   LR are not shown to be equivalent"**. Per-seed CIs are reported alongside.

## 10. Outputs (A9/outputs/, aggregate only)

- `a9_test_metrics.csv`: per model x seed, with CIs.
- `a9_paired_deltas.csv`: per comparison x seed, plus pooled, with CIs.
- `a9_pooled_summary.csv`: per model, the mean/min/max across seeds and the
  pooled CI.
- `a9_split_audit.json`: sizes, prevalence, overlap = 0 and hashes per seed.
- `a9_calibration_bins.csv`: 10 quantile bins of calibrated test
  probability, per model x seed. Cells with fewer than 10 births, events or
  non-events are suppressed.
- `a9_weighted_sensitivity.csv`: descriptive.
- `fig_a9_roc_pr.png`, `fig_a9_calibration.png`, `fig_a9_forest.png`.
- `a9_methods_results.md`, containing the statements above with numbers and
  the paper table: Model | Features | ROC-AUC (95% CI) | PR-AUC (95% CI) |
  Brier (95% CI) | Cal. slope | Cal. intercept, with pooled values for V2,
  V2_LR, F1_core and V1_spec, plus the row "original V1 report, different
  split, not paired" (0.8037 / 0.5334 / 0.1358).
- `MANIFEST.json`: sha256 of each output, git commit, protocol commit, config
  hash, software versions, seeds and split manifest hashes.
- Every CSV/JSON output carries the protocol commit hash.

## 11. Revision log

- **r1** (`94a6506`, 2026-11-02): the original protocol.
- **r2** (2026-11-02, before any A9 code had run on NFHS-5 data and before
  any test scoring): `config.yaml` line `guard` quoted
  (`pattern: "test_scored_{seed}.json"`), because the unquoted braces were
  invalid YAML and the file could not be parsed. There is no change to any
  analysis choice, seed, model, metric, threshold or statement. The protocol
  commit recorded in outputs is the commit of this revision.

## 12. Disclosure

Unweighted primary analysis. No keys, rows or per-birth predictions are
written or printed. Counts below 10 are suppressed (MIN_CELL = 10). Results
are point estimates with cluster-bootstrap percentile intervals. Those
intervals do not account for the survey's stratified design and are not
design-based standard errors.
