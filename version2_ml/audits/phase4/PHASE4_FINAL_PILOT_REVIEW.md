# IPD Version 2 — Phase 4 Final Pilot Review
Date: 2026-10-15
Status: CONDITIONAL PASS FOR EXPERIMENT DEVELOPMENT; NOT APPROVAL FOR CONFIRMATORY INFERENCE

## Evidence checked
- NFHS-5 broad institutional cohort: 201,311 births, 158,429 mothers.
- Conservative eight-feature F1/FT builder passed real-data row preservation on NFHS-4 195,997 and NFHS-5 201,311.
- Mother-exclusive provisional 70/15/15 split passed: 140,920 train, 30,227 validation, 30,164 test. This is a development split, not an independently untouched confirmatory holdout.
- All 36 states/UTs present in each split. This does NOT establish out-of-state generalization.
- F0 notebook baseline XGBoost closely reproduced: ROC-AUC 0.8035825, average precision 0.5328407, F1 0.5477801; notebook reference approximately 0.8033, 0.5330, 0.5469. Artifact-level provenance comparison still pending.
- Historical held-out fold: 40,263 births, 31,673 mothers.
- Descriptive unweighted versus weighted performance: ROC-AUC 0.80358 vs 0.80272; AP 0.53284 vs 0.55406; Brier 0.18236 vs 0.19078.
- Kish effective sample size calculation reconciled at 20,482.431256 (floating point); 2,685 observed strata, 19,078 stratum-PSU pairs, 78 singleton-PSU strata in held-out subset.
- 517 births coded as other facility (m15=96), all coded non-cesarean in m17. Interpretation requires questionnaire/coding review; exclude in prespecified sensitivity.
- One state/UT had fewer than 200 eligible births overall. Statewise performance disclosure and stability thresholds are needed.

## Experiment readiness
A1 State/area context: READY FOR DEVELOPMENT with within-state information sources and leakage restrictions; no spatial independence from random split.
A2 Historical NFHS-4 state/facility context: READY FOR DEVELOPMENT with source-time and target-outcome leakage rules; validate wave definitions.
A3 State x sector target encoding: READY FOR DEVELOPMENT, grouped out-of-fold encoding on training folds only; no validation/test target leakage.
A4 Facility detail and observed previous cesarean history: CONDITIONAL DEVELOPMENT; observed prior history is incomplete and not equal to lifetime history. Require routing/timing checks.
A5 NFHS-4-only to NFHS-5 temporal transfer: CONDITIONAL DEVELOPMENT; conservative eight common features available, source-only preprocessing and no NFHS-5 tuning.
A6 State/sector holdout transport: CONDITIONAL DEVELOPMENT; separately construct domain-held-out splits and report data support; random split is inadequate for this.
A7 Models, calibration, SHAP and robustness: READY FOR DEVELOPMENT; preprocessing and threshold selection train/development-only. Avoid unsupported individual clinical causation.

## Gates before publication
1. Reproduce and lock precise original F0 transformation, split and test membership artifacts; evaluate F0/F1 only on matched births.
2. Finalize feature timing, questionnaire equivalence and category code decisions, especially 22 eligibility-limited pregnancy/ANC variables.
3. Verify original survey sampling design and domain/subpopulation definition.
4. No survey-design confidence intervals or significance tests until estimator and singleton-stratum policy are validated.
5. Training / development and test independence must be stated accurately; old inspected V1 test is not virgin holdout.
6. Do not commit DHS microdata, record-level predictions, mother IDs, PSU keys, or unsafe fine-grained counts.
7. Mark state/sector context and facility as retrospective survey-recorded predictors, not prospective medical predictors.

## Decision
Phase 4 PILOT CLOSED CONDITIONALLY. Proceed to teammate experiment implementation (A1-A7), using the conservative common feature builder and provisional splits for software development. Do not claim statistically significant model differences, independent validation, or fully design-based uncertainty. Final manuscript evidence requires matched-model and methodological review.
