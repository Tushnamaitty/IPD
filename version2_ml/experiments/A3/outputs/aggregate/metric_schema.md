# A3 metric schema

`metrics.csv`: one row per (cohort, te_alpha, split, model, arm).

| Column | Meaning |
|---|---|
| experiment, wave, source_domain, target_domain | A3; NFHS-5; TE fitted on NFHS-5 outer-train labels |
| cohort | `primary` (NFHS-5 b19 0..59 institutional), `exclude_other_facility`, `nfhs5_cmc_0_59` |
| te_alpha, alpha_selected | smoothing pseudo-count for TE arms; `alpha_selected` false = exploratory alpha sensitivity |
| split | `validation` (default) or `test` (only with --evaluate-test) |
| model, arm, arm_role, feature_tier | fixed-hyperparameter estimator; matched arms i / ii / iii plus exploratory |
| n_births, n_mothers, n_events, prevalence | evaluation denominators |
| roc_auc | primary metric |
| average_precision | prevalence-dependent |
| brier, calibration_intercept, calibration_slope | probability accuracy; logistic recalibration diagnostics |
| threshold, threshold_provenance, precision, recall, f1 | max-F1 threshold chosen on validation only |
| confidence_interval_status | NOT_APPROVED |

`ablations.csv`: delta = arm minus comparator on identical rows. `alpha_selection.csv`: OOF log-loss of the
state x sector encoding on outer-train rows for each alpha. `te_support_summary.csv`: births by encoding
fallback level and a histogram of training-cell support (no cell names).
