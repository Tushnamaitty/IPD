# Extension 2 - Exploratory sector-contrast heterogeneity

**Run completed 27 September 2026.** The 200,794 analytic birth records were separated by respondent
into development (120,553 records; 94,855 respondents) and held-out test (80,241 records; 63,237
respondents). All fitting, encoding and predicted-group threshold selection used development only.
The study compares observed public and private institutional births after adjustment; it does not
estimate the causal effect of moving a woman between sectors.

## Method and limitations

Stage 1 refits propensity and outcome nuisance models within development, rather than reusing
full-cohort predictions. Stage 2 fits a survey-weighted random forest to AIPW pseudo-outcomes using
wealth, education, residence, social group, state and twin order as moderators. Development pseudo-outcomes use respondent-grouped cross-fitted nuisance predictions within
development. Separate nuisance models fitted on all development records score the held-out test set.

Whole-variable permutation importance was evaluated on 10,000 held-out records with five repetitions.
The baseline weighted pseudo-outcome prediction R-squared was **0.00609**, indicating very weak
individual predictive performance. State was highest (R-squared drop 0.01213, SD 0.00220), followed
by wealth (0.00091) and residence (0.00067). State importance is for the whole variable, not a
particular state's share of explained heterogeneity or evidence of a mechanism.

Five intended development quantile positions had a repeated cut point due to tied predictions.
Removing it and applying the thresholds to test yielded **four unequal ordered groups**, Q1-Q4,
not literal quintiles. The CSV value `cate_predicted_quintile` and notebook column `cate_quintile`
are legacy internal names for these ordered groups.

## Held-out results

Risk differences (RD) and confidence limits are in percentage points. Marginal group intervals
use 500 within-stratum PSU bootstrap replicates, holding the fitted models fixed.

| Predicted group | Test n | Mean predicted contrast | Realized adjusted RD | Marginal 95% CI |
|---|---:|---:|---:|---:|
| Q1 | 22,475 | 18.65 | 17.58 | 15.39-19.48 |
| Q2 | 39,688 | 30.16 | 29.23 | 27.29-31.02 |
| Q3 | 2,334 | 30.46 | 44.56 | 37.28-51.87 |
| Q4 | 15,744 | 41.26 | 40.46 | 37.62-43.03 |

Realized means are **not monotonic** (Q3 exceeds Q4), and Q3 is small. The directly paired
Q4-minus-Q1 RD difference was **22.88 pp** (conditional PSU-bootstrap 95% CI 19.28-26.19).
This does not establish reliable individual-level ranking or a smooth gradient.

| Residence | Test n | Realized adjusted RD | Marginal 95% CI |
|---|---:|---:|---:|
| Urban (coded 1) | 17,663 | 23.09 | 20.48-25.59 |
| Rural (coded 2) | 62,578 | 30.28 | 28.97-31.64 |

The directly paired **rural-minus-urban RD difference was 7.20 pp** (conditional PSU-bootstrap
95% CI 4.46-10.10). This describes an adjusted association within the one held-out split; it
does not explain why sector contrasts vary by residence. Social-group RDs range from 26.90 to
29.00 pp across coded groups 1-4. Those rows cover 76,047 records; social group is missing for
4,194 test records. No direct paired social-group contrasts were tested here.

## Files and use

- `notebook.ipynb`: completed notebook with the current run and corrected documentation.
- `outputs/heterogeneity_individual_or_binned_summary.csv`: group/subgroup sizes, RD, RR and
  marginal conditional bootstrap intervals.
- `outputs/heterogeneity_headline_contrasts.csv`: paired rural-minus-urban and Q4-minus-Q1
  RD differences and conditional intervals.
- `outputs/heterogeneity_modifier_summary.csv`: held-out whole-variable permutation importance.
- `outputs/heterogeneity_distribution.png`, `outputs/heterogeneity_subgroups.png`: diagnostics.
- `outputs/heterogeneity_metadata.json`: split, methods and limitations.

Inputs: `data/processed/df_model_v2.parquet`, the frozen primary result in
`outputs/final_tables/final_aipw_overall_table.csv`, and `extensions_work/shared` code. Run from
`extensions_work/02_causal_heterogeneity`. Older backups `notebook_before_review_fix.ipynb`
and `outputs.before_review_fix/` belong to the earlier analysis and must not be reported.

## Reporting rule

These are **exploratory single-split findings about an adjusted association**. Bootstrap intervals
condition on both fitted stages and exclude full model-fitting and split uncertainty. With weak
pseudo-outcome prediction, tied/unequal groups and nonmonotonic realized estimates, do not claim
personalized effects, a smooth gradient or a specific state mechanism. The frozen overall
private-public analysis remains the primary result.
