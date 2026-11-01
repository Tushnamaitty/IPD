# Final review bundle index - A1/A4/A7

Index for handoff section 12, items 1-7. All pointers are repository-relative
and refer to files that exist on branch `gaurav-v2-A1-A4-A7-r3`. No data values
or machine-specific paths are included. The bundle is not pushed and no pull
request has been opened.

## Provenance note

- Recorded `repo_commit` SHAs in A1/A4/A7 outputs and manifests refer to the
  local, unpushed r3 history (`gaurav-v2-A1-A4-A7-r3`); they do not identify
  this squashed branch.
- The original squashed tree was identical to the r3 tip tree
  (`74f637ff769d8e02e006c8cca039705a47b9c025`). The final tracked tree differs
  from r3's only by `REVIEW_BUNDLE.md`, which carries this note; a file cannot
  record the hash of the tree that contains it.
- A1 and A4 outputs were re-recorded on 2026-10-19 under the approved global
  environment. A7 outputs are the original 2026-10-17 recording; the
  2026-10-19 global-environment rerun was verification-only and did not change
  any committed value.

Status legend: COMPLETE (NFHS-5, aggregate-only), PARTIAL, NOT RUN,
NON-IDENTIFIABLE (outside this work), OPEN.

## 1. A1 matched ablation and provenance - COMPLETE (NFHS-5)

- `version2_ml/experiments/A1/outputs/a1_matched_ablation.csv`
- `version2_ml/experiments/A1/outputs/a1_interaction_gate.csv`
- `version2_ml/experiments/A1/outputs/a1_support_table.csv`
- `version2_ml/experiments/A1/outputs/a1_run_summary.json`
- `version2_ml/experiments/A1/outputs/a1_methods_results.md`
- `version2_ml/experiments/A1/outputs/fig_a1_roc_pr.png`
- `version2_ml/experiments/A1/outputs/fig_a1_calibration.png`
- `version2_ml/experiments/A1/outputs/MANIFEST.json`
- `version2_ml/experiments/A1/README.md`
- `version2_ml/experiments/A1/TEST_COMMAND_LOG.md`

Notes: recorded run `e6f9c79`, timestamp 2026-10-19T04:59:43Z. The A1 test
partition is NOT RUN by design; evaluation is validation-only and the test is
deferred to A7 (see `ISSUE_LOG.md` S-01).

## 2. A4 four-category history and facility-detail sensitivities - COMPLETE (NFHS-5); NFHS-4 NOT RUN

- `version2_ml/experiments/A4/outputs/a4_matched_ablations.csv`
- `version2_ml/experiments/A4/outputs/a4_sensitivity_m15_96.csv`
- `version2_ml/experiments/A4/outputs/a4_history_support.csv`
- `version2_ml/experiments/A4/outputs/a4_facility_crosswalk.csv`
- `version2_ml/experiments/A4/outputs/a4_run_summary.json`
- `version2_ml/experiments/A4/outputs/a4_methods_results.md`
- `version2_ml/experiments/A4/outputs/fig_a4_roc_pr.png`
- `version2_ml/experiments/A4/outputs/fig_a4_roc_pr_sensitivity.png`
- `version2_ml/experiments/A4/outputs/fig_a4_history_distribution.png`
- `version2_ml/experiments/A4/outputs/MANIFEST.json`
- `version2_ml/experiments/A4/README.md`
- `version2_ml/experiments/A4/TEST_COMMAND_LOG.md`

Notes: recorded run `2eda72c`, timestamp 2026-10-19T09:43:24Z. NFHS-4
facility-detail harmonization is NOT RUN because the authorized NFHS-4 birth
recode is unavailable (see `ISSUE_LOG.md` I-06 and S-03).

## 3. A7 model comparisons, calibration, SHAP and robustness - COMPLETE (NFHS-5, original recording)

- `version2_ml/experiments/A7/outputs/aggregate_model_comparison.csv`
- `version2_ml/experiments/A7/outputs/calibration_summary.csv`
- `version2_ml/experiments/A7/outputs/a7_tuning_summary.csv`
- `version2_ml/experiments/A7/outputs/a7_robustness.csv`
- `version2_ml/experiments/A7/outputs/shap_feature_family_summary.csv`
- `version2_ml/experiments/A7/outputs/a7_shap_stability.csv`
- `version2_ml/experiments/A7/outputs/a7_reconciliation.csv`
- `version2_ml/experiments/A7/outputs/a7_run_summary.json`
- `version2_ml/experiments/A7/outputs/a7_methods_results.md`
- `version2_ml/experiments/A7/outputs/fig_a7_roc_pr.png`
- `version2_ml/experiments/A7/outputs/fig_a7_calibration.png`
- `version2_ml/experiments/A7/outputs/fig_a7_shap_stability.png`
- `version2_ml/experiments/A7/outputs/MANIFEST.json`
- `version2_ml/experiments/A7/README.md`
- `version2_ml/experiments/A7/TEST_COMMAND_LOG.md`

Notes: outputs are the original 2026-10-17 recording (recorded run `d668549`)
and were not re-recorded. The global-environment verification reproduced all
tables, figures and CatBoost validation metrics with zero value differences;
the development test was scored twice on this machine with identical values
and no decision dependence. Results remain non-confirmatory because of V1 test
exposure.

F0 reconciliation scope: `a7_reconciliation.csv` compares only the processed
36-feature F0 input parquet against the frozen C1 cohort. It checks key-for-key
identity on `respondent_id|birth_index` and label-for-key identity on the
`csection`/`m17` outcome and the `delivery_place_code`/`m15` facility code. The
recorded result is `reconciled_and_fitted`: every parquet row matched a cohort
key, with zero cohort-only keys, zero parquet-only keys, zero duplicate parquet
keys, zero outcome mismatches and zero facility-code mismatches. It does not
compare model predictions, coefficients, V1 splits, V1 preprocessing,
calibration or V1 reported scores. The `psu_overlap_v021` rows in the same file
are a separate overlap check, not part of the F0 reconciliation.

`Exact V1 F0 reproduction NOT RUN` means no V1 pipeline was executed or
matched: the V1 model specification, training-sample construction, split,
calibration and reported F0 scores were not reproduced. The F0 block is a
36-feature reference fitted with the A7 configuration on the reconciled
parquet and evaluated on A7's partitions. It is a reconciled reference only,
not an exact V1 reproduction (see `ISSUE_LOG.md` S-06).

## 4. Test and command logs - COMPLETE

- `version2_ml/experiments/A1/TEST_COMMAND_LOG.md`
- `version2_ml/experiments/A4/TEST_COMMAND_LOG.md`
- `version2_ml/experiments/A7/TEST_COMMAND_LOG.md`
- `version2_ml/experiments/WORKLOG.md`
- `version2_ml/experiments/ISSUE_LOG.md`

Notes: the latest full synthetic suite is 114 tests OK (43 base, 10 common, 15
A1, 30 A4, 16 A7), exit 0 per suite. Exact commands and run dates are in the
logs. The A7 double-scoring incident and its mitigation are recorded in
`WORKLOG.md` and `ISSUE_LOG.md` I-01.

## 5. Aggregate privacy-reviewed results and graphics - COMPLETE (aggregate-only)

All committed A1/A4/A7 outputs under:

- `version2_ml/experiments/A1/outputs/`
- `version2_ml/experiments/A4/outputs/`
- `version2_ml/experiments/A7/outputs/`

Provenance and integrity:

- `version2_ml/experiments/A1/outputs/MANIFEST.json`
- `version2_ml/experiments/A4/outputs/MANIFEST.json`
- `version2_ml/experiments/A7/outputs/MANIFEST.json`

Notes: outputs are aggregate tables, point-estimate figures and provenance
JSON only. Counts below 10 are suppressed; no row-level, key-level or
per-birth predictions are exported by A1/A4/A7. The pre-existing tracked
row-level V1 files are NON-IDENTIFIABLE to this work and require coordinator
handling (see `ISSUE_LOG.md` E-01). Third-party absolute paths in
`extensions_work/` and `notebooks/` pre-date this branch (see
`ISSUE_LOG.md` E-02).

## 6. Exact methods assumptions and scientific limitations - COMPLETE

- `version2_ml/experiments/A1/README.md`
- `version2_ml/experiments/A4/README.md`
- `version2_ml/experiments/A7/README.md`
- `version2_ml/experiments/A1/outputs/a1_methods_results.md`
- `version2_ml/experiments/A4/outputs/a4_methods_results.md`
- `version2_ml/experiments/A7/outputs/a7_methods_results.md`
- `version2_ml/experiments/WORKLOG.md`
- `version2_ml/experiments/ISSUE_LOG.md`

Notes: unweighted point estimates, no design-based intervals, no significance
or independent-confirmatory claims. Fit is train-only with mother-exclusive
partitions; thresholds are predeclared or validation-selected as documented;
unknown history is never recoded as negative. A1/A4 are validation-only; A7 is
development-test and non-confirmatory. NFHS-4 is absent. MIN_CELL = 10
suppression applies to every committed output.

## 7. Failed experiments or unresolved decisions - OPEN

- `version2_ml/experiments/ISSUE_LOG.md`
- `version2_ml/experiments/WORKLOG.md`

NOT RUN:

- NFHS-4 facility-detail harmonization and any NFHS-4 cross-wave sensitivity
  (authorized NFHS-4 birth recode unavailable).
- A1 and A4 test-partition evaluation (deferred to the single A7 evaluation).
- Exact V1 36-feature reproduction (A7 F0 is a reconciled reference only).

NON-IDENTIFIABLE / outside this work:

- Pre-existing tracked row-level V1 files, which pre-date the branch and were
  not created or modified by A1/A4/A7.

OPEN decisions:

- Validation-only reporting for A1/A4 (S-01).
- MIN_CELL = 10 suppression sign-off (S-02).
- Whether NFHS-4 cross-wave work is required (S-03).
- Acceptance of manifests as supplementary-table provenance (S-04).
- A7 non-confirmatory and two-scoring disclosure language (S-05).
- A7 F0 reconciliation labeling (S-06).
- Handling of pre-existing row-level V1 files and third-party absolute paths
  (E-01, E-02).
