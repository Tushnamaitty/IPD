# Issue log - A1/A4/A7 (NFHS-5 rework)

Scope: implementation, verification and reporting issues for experiments A1,
A4 and A7 on branch `gaurav-v2-A1-A4-A7-r3`. This log contains no data values
and no machine-specific paths. Repository-relative file names are used only as
pointers. Status values: RESOLVED, BLOCKED, ESCALATED, OPEN.

## 1. Blockers and incidents

### I-01 - A7 development-test partition scored twice - RESOLVED, DISCLOSED

- What happened: a global-environment verification run on 2026-10-19 scored the
  development test partition a second time in a temporary sandbox.
- Cause: the verification copied the tuning/test caches one directory level
  above the runner's cache location. The runner resolves its cache as
  `<out-dir>/private`; the sandbox therefore started without a cache and took
  the scoring branch.
- Impact: second scoring only. The committed outputs remain the original
  2026-10-17 recording; the recorded private cache was not modified; no
  sandbox-derived output was committed; no reported decision depended on the
  second scoring. All values were identical to the recorded values, and the
  tuning payload was byte-identical.
- Mitigation: `run_a7.py` now accepts a separate `--cache-dir` and refuses to
  score the test partition when no cache exists unless `--allow-test-score` is
  passed. Synthetic tests cover the refusal and cache reuse. The incident and
  the hash of every sandbox artifact were recorded before the sandbox was
  deleted.
- Disclosure: `A7/README.md`, `A7/outputs/MANIFEST.json` and
  `WORKLOG.md`. The result remains non-confirmatory because of V1 exposure.

### I-02 - A4 history-distribution figure call site - RESOLVED

- What happened: the committed A4 figure call passed the full cohort history
  frame instead of the validation-masked rows.
- Impact: the pre-fix call would have failed a length check (cohort rows versus
  validation rows). The committed outputs were produced after the fix was
  applied in the working tree; a fixed-code rerun was pixel-identical
  (mean absolute pixel error 0.0000) and no table values changed.
- Resolution: call-site-only fix in `run_a4.py`; documented in `WORKLOG.md`.

### I-03 - Local virtual environment unusable - RESOLVED

- What happened: the repository virtual environment failed at import time with
  a DLL load error for `_philox`, blocked by an Application Control policy.
- Impact: package versions pinned for the original recording could not be used.
- Resolution: all current runs use the approved global Python environment. The
  virtual environment was not repaired and Application Control was not
  modified. The single approved package addition was CatBoost (already the
  recorded version), verified by import and by reproduced A7 validation
  metrics.
- Note: the global environment differs from the original recording environment
  for numpy, pandas, scikit-learn, xgboost, matplotlib and shap. Every
  re-recorded output records the versions used at run time.

### I-04 - Manifest hash rule mismatch - RESOLVED

- What happened: the first manifest generation hashed working-tree bytes. On
  Windows with automatic line-ending conversion, text files were hashed in
  CRLF form while the repository stores LF, so manifest hashes did not match
  committed blobs.
- Resolution: A1/A4/A7 manifests were regenerated on 2026-10-19 from committed
  git blobs. Text files are hashed in repository-normalized LF form; PNG files
  are hashed as raw bytes. The rule is recorded inside each `MANIFEST.json`
  under `hash_rule`.
- Constraint honored: no output table or figure was regenerated.

### I-05 - CatBoost availability - RESOLVED

- What happened: CatBoost was required for the A7 candidate set but was not
  importable in the broken virtual environment.
- Resolution: installed in the approved global environment (recorded version
  1.2.10) and exercised on real data. Its validation metrics reproduced the
  recorded values with zero differences. Its development-test metrics were not
  used for any decision beyond the disclosed verification.

### I-06 - NFHS-4 birth recode unavailable - BLOCKED, EXTERNAL

- What happened: the authorized NFHS-4 birth recode (`IABR74FL.DTA`) is not
  present in the local authorized data area.
- Impact: A4 NFHS-4 facility-detail label harmonization and any NFHS-4
  cross-wave sensitivity are NOT RUN. The NFHS-4 mapping intentionally raises
  rather than guessing labels.
- Resolution: requires an authorized local copy and label verification. Not an
  automatic prerequisite for the isolated NFHS-5 A1/A4/A7 results.

## 2. Unresolved scientific questions

### S-01 - A1/A4 validation-only evaluation

A1 and A4 score the development validation partition only; the test partition
is deferred to the single frozen A7 evaluation. Scientific sign-off is needed
that validation-only reporting for A1/A4 is acceptable, and that the deferral
to A7 is stated explicitly in the manuscript.

### S-02 - MIN_CELL suppression sign-off

The suppression threshold is MIN_CELL = 10 (A7 also uses subgroup minima of
200 rows and 30 rows per class). Coordinator sign-off is needed for the exact
suppression rule, its rendering (`<10 (suppressed)`), and its application to
all supplementary tables and figures.

### S-03 - NFHS-4 absence and cross-wave sensitivity

A decision is needed on whether A4 NFHS-4 facility-detail harmonization or any
cross-wave comparison is required for the paper. If required, an authorized
NFHS-4 file and label-verification time must be provided; until then the item
is NOT RUN.

### S-04 - Supplementary-table provenance via manifests

The A1/A4/A7 manifests now provide SHA-256 and size for every committed output
under a documented committed-blob rule. Confirmation is needed that this
manifest mechanism is the accepted provenance route for supplementary tables,
and that no separate archival copy is required.

### S-05 - A7 non-confirmatory status and disclosure

A7 must remain labeled development-test and non-confirmatory because the same
NFHS-5 outcomes were inspected during V1 work. Confirmation is needed that the
two-scoring disclosure (I-01) is sufficient manuscript language and that no
independent-test or confirmatory claim is made.

### S-06 - A7 F0 reconciliation labeling

The A7 F0 reconciliation checks only key and label identity between the
processed 36-feature parquet and the frozen C1 cohort: the
`respondent_id|birth_index` key, the `csection`/`m17` outcome and the
`delivery_place_code`/`m15` facility code. It does not reproduce V1
preprocessing, model, split, calibration or reported scores; the F0 model is a
36-feature reference fitted with the A7 configuration. "Exact V1 F0
reproduction NOT RUN" means no V1 pipeline execution or prediction/score
matching was attempted. Confirmation is needed that this distinction is stated
wherever F0 rows are reported.

### S-07 - Validation-selected threshold requirement

A7's aggregate thresholded metrics are computed at the fixed 0.50 threshold,
whereas handoff sections 4 and 7 specify a validation-selected threshold. A
decision is needed on whether fixed-0.50 reporting is acceptable or whether
validation-selected threshold metrics must be reported alongside it. Until
then, precision, recall and F1 at 0.50 are labeled as fixed-threshold
descriptives and are not compared across calibration states.

### S-08 - Calibration-method column in the shared schema

`aggregate_model_comparison.csv` encodes calibration variants by row order
(uncalibrated, Platt, isotonic per model/tier) and has no explicit
`calibration_method` column. A decision is needed, coordinated with Rifa, on
adding that column to the shared aggregate schema. Until then, the row order is
documented and verified against `calibration_summary.csv`.

### S-09 - F1-replace-facility tier

A7 has no tier that replaces `facility_type` with `facility_detail` while
omitting observed history; `F1_detail_history` bundles facility detail and
history. A decision is needed on whether the paper requires an
F1-replace-facility tier to separate those contributions, or whether A4's
matched ablations are sufficient.

### S-10 - F1_detail_history sensitivity coverage

The m15!=96 exclusion and most-recent-birth-per-mother sensitivities were run
on F1 only, so they do not test the F1_detail_history tier. Should both
sensitivities be run on F1_detail_history? Doing so would require a new
approved A7 real-data run, with new outputs subject to the same test-scoring
controls.

## 3. Coordinator escalation items

### E-01 - Pre-existing tracked row-level V1 files - ESCALATED

Three row-level files pre-date this branch and are tracked on all branches:
`outputs/tables/b2_test_predictions_logistic_regression.csv`,
`outputs/tables/b2_test_predictions_xgboost.csv` and
`outputs/tables/b3_record_level_standardization.csv`. They were not created,
modified or removed by A1/A4/A7 work. They are outside this bundle and require
a coordinator decision on retention, privacy review and history handling.
They are non-identifiable to the A1/A4/A7 outputs.

### E-02 - Pre-existing third-party absolute paths - ESCALATED

Machine-specific absolute paths belonging to another contributor remain in
`extensions_work/` and `notebooks/`. They pre-date this branch and were not
introduced by A1/A4/A7. They are untouched pending coordinator ownership and
cleanup direction. No such path is present in the A1/A4/A7 code, outputs or
documentation changed by this branch.

### E-03 - Data authorization and retention

Authorized local NFHS-5 data, the processed parquet used for F0 reconciliation,
and any future NFHS-4 file remain outside the repository. Confirmation is
needed that access, retention and deletion follow the project's DHS
authorization.

## 4. Verification pointers

- `WORKLOG.md`: chronological commands, reproduction evidence and incident
  record.
- `ISSUE_LOG.md`: this file.
- `REVIEW_BUNDLE.md`: handoff section 12 index.
- `A1/outputs/MANIFEST.json`, `A4/outputs/MANIFEST.json`,
  `A7/outputs/MANIFEST.json`: committed-blob hashes and run provenance.
- `A1/TEST_COMMAND_LOG.md`, `A4/TEST_COMMAND_LOG.md`,
  `A7/TEST_COMMAND_LOG.md`: exact test and run commands.
