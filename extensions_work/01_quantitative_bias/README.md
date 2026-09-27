# Extension 1 — Quantitative Bias Analysis (QBA)

**Status: illustrative scenario analysis, not evidence-calibrated.** The primary AIPW estimate is held fixed. The prevalence and risk-ratio inputs for unmeasured confounders remain hypothetical and unsourced. These calculations do not establish that the sector contrast is robust to realistic unmeasured confounding, nor do they identify a corrected AIPW estimate.

## Question and inputs

How would the 28.47 percentage-point, survey-weighted, covariate-standardized private–public Cesarean risk difference move under specified hypothetical unmeasured-confounding scenarios?

The notebook reads:

- `outputs/final_tables/final_aipw_overall_table.csv` for the frozen primary estimate (no AIPW refitting).
- `data/processed/df_model_v2.parquet` for `facility_type`, `birth_order`, `twin_order` and `sample_weight_normalized`. These respondent-level columns yield the common analytic population's weighted later-delivery share.

Run `notebook.ipynb` from `extensions_work/01_quantitative_bias` so its relative import of `../shared/config.py` resolves correctly.

## Method and assumptions

For an assumed binary unmeasured factor, the illustrative bias factor is

`BF = [1 + p1 × (RR_UY − 1)] / [1 + p0 × (RR_UY − 1)]`,

where `p0` and `p1` are assumed public and private scenario prevalences and `RR_UY` is the assumed factor–outcome risk ratio. The notebook divides the primary risk ratio by `BF`. It then holds the primary public risk fixed to translate that ratio to a risk difference. The displayed difference is a **fixed-public-risk translation**, not a separately estimated corrected risk difference. The simple bias model assumes a binary factor and a constant outcome risk ratio across sectors; it does not account for the full AIPW estimation procedure or its sampling uncertainty.

For **prior Cesarean history**, the prevalence assumptions apply among women with a prior *delivery*. `birth_order` and `twin_order` distinguish a second child at a first multiple delivery from a later delivery. The notebook scales the hypothetical within-parity prevalences by the **common survey-weighted later-delivery share, 0.58030987 (58.031%)**, calculated from 200,794 analytic births. Separately observed weighted later-delivery shares are 51.241% for private and 60.957% for public births. The common-share scaling illustrates a standardized contrast; it is an additional simplifying assumption, not an identified bias correction.

The second factor, **unmeasured obstetric severity**, is an illustrative composite. It is not one validated diagnosis with a measured prevalence or association.

## Regenerated results (27 September 2026)

The frozen reference is a private risk of 44.9644%, public risk of 16.4946%, risk difference of **28.4698 percentage points**, and risk ratio of **2.7260** among **200,794** births. The risk-ratio confidence-bound input closest to the null is 2.6566. These are the notebook's existing reference values; the QBA does not refit their models.

The deterministic grid has **550 rows**; the summary has **10 rows**. All scenario prevalence gaps and risk ratios below are hypothetical.

| Factor | Scenario | Assumed prevalence gap in whole cohort | Assumed RR_UY | Illustrative RD, fixed public risk (pp) |
|---|---|---:|---:|---:|
| Prior Cesarean history | Mild | 2.902 pp | 2.0 | 27.2823 |
| Prior Cesarean history | Moderate | 6.964 pp | 4.0 | 21.8444 |
| Prior Cesarean history | Strong | 11.606 pp | 6.0 | 14.9394 |
| Obstetric severity composite | Mild | 3.000 pp | 1.5 | 27.8305 |
| Obstetric severity composite | Moderate | 8.000 pp | 2.5 | 24.1184 |
| Obstetric severity composite | Strong | 15.000 pp | 3.5 | 17.7640 |

At the maximum tested prevalence gaps (17.409 pp for prior Cesarean history and 20.000 pp for the severity composite), the analytic tipping risk ratios are approximately **33.02** and **28.88**, respectively. Both lie beyond their plotted risk-ratio grids; they are mathematical scenario boundaries, not claims about clinical plausibility.

The 20,000 uniform Monte Carlo draws per factor explore only arbitrarily selected, unsourced parameter ranges. Prior Cesarean history yielded a median illustrative RD of **21.4203 pp** (2.5th–97.5th percentiles **10.3453–28.2841**); the severity composite yielded **23.3606 pp** (**13.5043–28.3453**). All draws within those chosen ranges remained positive. These percentages and percentiles are **not confidence intervals or probabilities about the true effect**; primary-estimate sampling uncertainty is excluded.

The point-estimate E-value recomputes to **4.8951** (rounded to 4.90), an arithmetic cross-check against the previously reported value. It does not validate the hypothetical prevalence or risk-ratio inputs.

## Output files

- `outputs/qba_scenario_results.csv` — 550 deterministic scenarios with within-target and whole-cohort inputs.
- `outputs/qba_summary.csv` — representative scenarios, analytic tipping points and Monte Carlo summaries.
- `outputs/qba_heatmap.png` — illustrative fixed-public-risk translations over the tested grids. It labels a zero-difference tipping point as outside the plotted domain when appropriate.
- `outputs/assumptions_sources.csv` — source-status table; hypothetical values remain explicitly unsourced.

## Checks and limitations

The notebook checks parameter bounds, reproduces the frozen reference when the assumed prevalence gap is zero or `RR_UY = 1`, and verifies the E-value arithmetic. The corrected later-delivery share and all four output files were regenerated on 27 September 2026.

Before publication use, replace hypothetical inputs with defensible, documented parameters for clearly defined factors, considering population, facility sector, parity, time period and the reported association scale. Reassess the simplifying bias model and the translation to a risk difference. The composite severity scenario should be separated into named conditions if evidence permits. The current results support an **illustrative sensitivity exercise only**.
