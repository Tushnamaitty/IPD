# Extension 1 - Residual confounding sensitivity benchmark

**Status:** Repaired notebook run on the frozen primary table on the project
laptop; generated outputs independently checked on 28 September 2026.
The earlier prevalence-based outputs must not be presented as results of
this version.

## Why the method changed

The earlier notebook applied separate observed public and private
later-delivery shares to a hypothetical prior-Cesarean prevalence. Birth order
is already included among the measured adjustment variables of the primary
AIPW analysis. The procedure changed the sensitivity result even when the
hypothetical prevalence was identical within later deliveries, so it mixed
measured parity composition into a calculation described as *residual*
confounding. A pooled share is not a general fix: a prevalence-based
correction to a standardized estimate would need covariate-stratum-specific
assumptions and adequate risk information. The earlier formula and
fixed-public-risk "corrected risk difference" were removed.

## Current analysis

`notebook.ipynb` reads the original, frozen AIPW risk ratio and its lower
confidence limit from `outputs/final_tables/final_aipw_overall_table.csv`.
It makes a deterministic grid of two hypothetical residual-confounding
strengths, both defined conditional on the measured variables:

- `RR_AU`: maximum sector–unmeasured-factor association.
- `RR_UY`: maximum unmeasured-factor–Cesarean association.

It calculates the Ding–VanderWeele bounding factor
`RR_AU * RR_UY / (RR_AU + RR_UY - 1)` and the corresponding E-values.
The grid's risk ratios are *hypothetical lower bounds under a causal
interpretation*, not corrected estimates or new confidence intervals.
The two strength parameters are unsourced and cannot be attributed to prior
Cesarean history or an obstetric-severity composite. A causal effect of
changing a woman's facility sector has not been established.

Methods: Ding and VanderWeele, *Epidemiology* 2016,
DOI [10.1097/EDE.0000000000000457](https://doi.org/10.1097/EDE.0000000000000457);
VanderWeele and Ding, *Annals of Internal Medicine* 2017,
DOI [10.7326/M16-2607](https://doi.org/10.7326/M16-2607).

## Verified run

The output metadata reports 200,794 analytic births, reference RD 28.469798
percentage points, and reference RR 2.726007 (lower confidence limit
2.656604). The point-estimate E-value is 4.895133 and the lower-limit
E-value is 4.754446. All 841 distinct grid rows and seven selected summary
rows passed independent arithmetic checks; the plot is legible. The results
are still hypothetical sensitivity benchmarks, not evidence-calibrated
corrections or causal claims.

## Reproduce

1. Back up the existing `extensions_work/01_quantitative_bias` folder.
2. Put this folder at `extensions_work/01_quantitative_bias` and open the
   notebook with that folder as the working directory.
3. Confirm `extensions_work/shared/config.py` and the frozen
   `outputs/final_tables/final_aipw_overall_table.csv` exist. The notebook
   does not read raw NFHS data or refit the primary model.
4. Restart the kernel, run all cells, then save the notebook. The run writes
   `qba_scenario_results.csv`, `qba_summary.csv`, `qba_heatmap.png`,
   `assumptions_sources.csv`, and `qba_metadata.json` to `outputs/`.
5. Confirm the printed reference matches the frozen analysis
   (n=200,794, RD approximately 28.47 pp, RR approximately 2.726), that
   the point E-value is about 4.90, and that the grid has 841 rows. At
   `RR_AU=RR_UY=1`, the bound must equal the reference RR. Inspect the
   heatmap for legible labels and a white null-bound contour.

## Manuscript decision

Ask the supervisor whether a well-defined causal estimand exists for the
broad sector contrast before using this benchmark as a robustness claim.
If specific prior-Cesarean or obstetric-severity scenarios are desired,
first define the factor and obtain independently verified, relevant
conditional association estimates. Do not revive the old unsourced
prevalence grid or report its output as a calibrated analysis.
