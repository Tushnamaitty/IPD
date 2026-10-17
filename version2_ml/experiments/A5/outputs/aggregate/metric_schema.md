# A5 metric schema

`metrics.csv`: one row per (comparison, split, model, cohort, weighting).

| Column | Meaning |
|---|---|
| comparison | A = NFHS-4 source reference; B = frozen NFHS-4 model zero-shot on NFHS-5; C = NFHS-5 within-wave reference (context only); D_intercept / D_intercept_slope = target-supervised recalibration of B |
| wave, source_domain, target_domain | evaluation wave; wave whose labels trained the model |
| split | source_validation (threshold chosen here), source_holdout, target_validation (shared NFHS-5 split, matched to C), target_non_test (NFHS-5 train+validation), target_test (only with --evaluate-test) |
| cohort | primary, or NFHS-5 sensitivity (nfhs5_cmc_0_59, exclude_other_facility, exclude_calendar_overlap) |
| target_supervision | none (A, B), target_train_labels_within_wave (C), target_adaptation_labels_counted (D) |
| weighting | unweighted or v005_descriptive (descriptive point metric only) |
| n_births, n_mothers, n_events, prevalence | denominators |
| roc_auc, average_precision, brier, calibration_intercept, calibration_slope | point metrics; AP is prevalence-dependent |
| threshold, threshold_provenance, precision, recall, f1 | A/B: threshold frozen from NFHS-4 validation |
| confidence_interval_status | NOT_APPROVED |

`transfer_penalties.csv`: first minus second for each named contrast; `ap_lift` = AP / prevalence.
`calibration_bins.csv`: deciles of predicted risk (n, mean predicted risk, observed rate).
`shift_diagnostics.json`: harmonization, unseen levels, composition and feature distributions by wave.
