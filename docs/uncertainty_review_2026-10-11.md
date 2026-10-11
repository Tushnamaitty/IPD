# Survey uncertainty review — 11 October 2026

**Status: reviewed existing records; qualified; not independently certified or rerun.**

## Current national narrow results
`extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json` declares joint fitted-model/reference Taylor linearization with with-replacement ultimate-PSU approximation. Primary singleton handling: `adjust` (grand-mean centering); `average` and `zero` sensitivity alternatives also saved. CIs are marginal, approximate, and conditional on fixed ridge `1e-5`, preprocessing/vocabulary/support choices; no finite-population correction, lower-stage variance, or complete exact PPS/systematic multistage variance.

The national point and interval values in the 3 October PDF agree at displayed precision with the national summary. The state-wise review also reports national point estimate reproduction essentially at floating-point precision and reproduces reported descriptive standard errors in its audit file.

## State-wise uncertainty
`extensions_work/17_statewise_descriptive/outputs/statewise_descriptive/statewise_summary.json` reports joint survey domain influence calculations; state-wise intervals are unadjusted marginal t/Wald estimates, with sparse-support and zero-variance/boundary outcome flags. No multiple-test correction, temporal state-change inference or cross-state ranking. Wave geography labels need manual review where boundaries changed. Completion status still says `completed_descriptive_outputs_require_review`.

## Household/PSU frame and policy-period qualification
`extensions_work/03_nfhs4_nfhs5_temporal/outputs/policy_period_state_roster_uncertainty/household_uncertainty_review.json` reports `completed_approximate_intervals_require_output_review` and `raw_data_run_review_required: true`. Corrected key: (state `v024/hv024`, stratum `v022/hv022`, PSU `v021/hv021`). Household record yields **30170** observed PSU units vs **30198** published completed fieldwork units, difference **28**, unresolved. In active period strata, household frame 14623 vs birth file 14612 PSU units; eight singleton strata remain. Earlier legacy frame (14990 PSU, 1314 strata) is explicitly `legacy_design_equivalent_for_this_cohort: false`, and the file flags corrected-state-key rerun required. **Do not substitute older bootstrap/design intervals without matching scope and provenance.**

The period result uses 11 selected common states, support restrictions and separated-cell profiling (23 rows); no FPC. Some earlier uncertainty attempts failed from separation/conditioning; only explicitly reviewed corrected-output estimates belong in a manuscript, labeled exploratory. The October 3 PDF uses five period point gaps and conditional intervals; confirm these against the corrected CSV before final inclusion.

## Missingness
October 7 PDF labels 30-imputation sensitivity approximate, within/between variance via Rubin/Barnard-Rubin but survey PSU exponential draws are not a guaranteed proper posterior scheme; MAR is unverifiable. Reproduction of baseline rows does not validate CI coverage. Imputation did not rerun every exploratory context model.

## Decision
The above limitations are reported, **not fixed by documentation**. No evidence from the inspected national numerical summaries suggests a headline point estimate arithmetic mismatch. Approval of exact design variance and uncertainty reporting is still a statistical sign-off task (potentially involving locally rerun model-refit PSU resampling or survey-design expert review). Professor Murthy permitted manuscript drafting, not unconditional certification of all intervals.

### Required before definitive methods/significance statements
1. Identify the exact primary CI file/version for each table and figure.
2. Review corrected state-stratum-PSU household-frame output against older figures and reports.
3. State precisely which uncertainty elements are conditional; acknowledge no FPC/lower-stage frame information.
4. Verify the complete local inputs and model refits independently if claiming exact reproducibility.
