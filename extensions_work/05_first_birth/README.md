# Extension 5 — First-delivery birth-record sector comparison

## Research question and interpretation

Among birth records from a woman's first delivery, what is the survey-weighted, covariate-standardized difference in Cesarean risk between private and public institutional facilities? This is an associational sector contrast. Restricting to first deliveries rules out a previous Cesarean, but other clinical indications and selection factors may remain unmeasured.

## Cohort and descriptive results

The parent public/private NFHS-5 analytic cohort has 200,794 birth records. Delivery order is derived from the recorded child birth order and multiple-birth order: for `twin_order > 1`, `delivery_order = birth_order - twin_order + 1`; otherwise `delivery_order = birth_order`. The analysis includes all birth records with `delivery_order == 1`. Both children of a first twin delivery contribute records; the unit is a **birth record**, not a unique delivery.

There are 83,158 first-delivery birth records (41.41% of the parent cohort), 732 more than the old `birth_order == 1` rule. None were excluded for missing `age_at_first_birth`.

| Sector | Birth records | Raw Cesarean prevalence | Survey-weighted Cesarean prevalence |
| --- | ---: | ---: | ---: |
| Public | 58,797 | 17.40% | 18.22% |
| Private | 24,361 | 50.91% | 51.63% |

These prevalences are descriptive; the adjusted estimates below come from the separately fitted AIPW model.

## Adjusted analysis and diagnostics

The respondent-grouped, cross-fitted AIPW model is refit in this cohort with eight covariates: `age_at_first_birth`, `wealth_index`, `education_years`, `residence`, `religion`, `social_group`, `twin_order`, and `state`. Raw `birth_order` is excluded because it describes child position in a multiple delivery, and `delivery_order` is constant within the subgroup.

The within-cohort propensity/balance diagnostic reported 0 of 60 expanded covariates with absolute standardized mean difference above 0.1 after weighting. The effective sample sizes after weighting were 10,125 private and 22,136 public; the observed fitted propensity range was 0.0049–0.9740. This diagnostic uses its own fitted propensity model and should not be described as a direct diagnostic of each out-of-fold AIPW prediction.

**Adjusted private risk:** 49.72%  
**Adjusted public risk:** 20.76%  
**Private minus public risk difference:** 28.96 percentage points (95% PSU-bootstrap interval 27.90–30.03)  
**Risk ratio:** 2.395 (95% PSU-bootstrap interval 2.318–2.476)

The frozen full-cohort estimate is 28.47 percentage points (95% interval 27.67–29.24) and a risk ratio of 2.726 (95% interval 2.657–2.803). The first-delivery point estimate is 0.49 percentage points higher. These nested-cohort estimates have no joint confidence interval for their difference here, so this comparison is descriptive. The first-delivery result shows that the adjusted sector gap remains large in this subgroup; it does not identify a causal facility effect.

The completed notebook saved the subgroup estimate, 500 PSU-cluster bootstrap replicates, a side-by-side plot against the frozen full-cohort estimate, and metadata. Bootstrap replicates resample PSUs with the fitted nuisance predictions held fixed, without refitting the nuisance models; the interval should be described with that scope. The subgroup is nested within the full cohort, so its side-by-side intervals do not establish a statistical difference between the two estimates. Such a comparison would require joint resampling.

## Inputs and outputs

Inputs: `data/processed/df_model_v2.parquet` and the frozen primary `outputs/final_tables/final_aipw_overall_table.csv` (paths resolved by `shared/config.py`). Run `notebook.ipynb` from this extension directory in repository context.

Outputs in `outputs/`:

- `first_birth_cohort_summary.csv`
- `first_birth_overlap_balance.csv`
- `first_birth_aipw_summary.csv`
- `first_birth_bootstrap_summary.csv`
- `first_birth_comparison_plot.png`
- `first_birth_metadata.json`

## Limitations

The first-delivery restriction removes prior-Cesarean history as a potential confounder, while fetal compromise, labor complications, and other indications may still differ by sector. A multiple delivery has more birth records than a singleton delivery, so these estimates are not unique-delivery estimates. The bootstrap holds nuisance predictions fixed and the comparison with the full cohort is descriptive. No causal effect of choosing a facility sector is identified here.
