# A2 metric schema

`metrics.csv` has one row per (cohort, context_variant, split, model, arm, weighting).

| Column | Meaning |
|---|---|
| experiment, wave, source_domain, target_domain | A2; NFHS-5 evaluation; NFHS-4 is the context source only |
| cohort | `primary` (NFHS-5 b19 0..59 institutional) or a cohort sensitivity |
| context_variant | historical mapping variant (`primary` alpha=20,n_min=30 unweighted; alpha grid; `weighted_v005`; `ladakh_unseen`) |
| split | `validation` (default) or `test` (only with --evaluate-test) |
| model / arm / arm_role / feature_tier | fixed-hyperparameter estimator and matched feature arm |
| weighting | `unweighted` or `v005_descriptive` (descriptive point metric; design not validated) |
| n_births, n_mothers, n_events, prevalence | evaluation denominators (unweighted) |
| roc_auc | primary metric |
| average_precision | prevalence-dependent; read with `prevalence` |
| brier | mean squared probability error |
| calibration_intercept | logistic recalibration intercept with logit(p) as offset (slope fixed at 1) |
| calibration_slope | slope of logit(p) in logistic recalibration; null if not identifiable |
| threshold, threshold_provenance, precision, recall, f1 | max-F1 threshold chosen on validation only |
| confidence_interval_status | always NOT_APPROVED (survey-design estimator not validated) |
| result_status | EXPLORATORY |

`ablations.csv`: delta = arm minus B0 on identical rows (same cohort, split, model, weighting). No paired CI.
