# Publication uncertainty decision — 11 October 2026

**Decision: CONDITIONAL APPROVAL FOR MANUSCRIPT DRAFTING; NOT FINAL INDEPENDENT STATISTICAL SIGN-OFF.** This is a documented technical review, not an endorsement from an independent survey statistician, the data custodians, or Professor Murthy. No NFHS restricted microdata were rerun.

## Review scope
Inspected the committed national Taylor interval table; shared-reference change intervals; statewise primary and design-support tables; corrected state-household policy-period interval and contrast files; associated design summary statuses; October 3/7/9 user-supplied research PDFs.

### Presentation rule (all sections)
Use exact cohort/estimand/source in caption. Label confidence intervals **approximate marginal 95% intervals** under a with-replacement ultimate-PSU Taylor linearization, with singleton `adjust` primary and alternate centering sensitivity when applicable. Describe fixed ridge, preprocessing, reference, support and state crosswalk conditioning. No exact PPS/systematic/FPC or lower-stage design variance claim; no full uncertainty from all model-selection steps, no unverified coverage guarantee. No causal claims.

## Decisions by result family

| Result family | Status for manuscript drafting | Reason / remaining validation |
|---|---|---|
| National observed and adjusted narrow gaps | **USE WITH EXPLICIT APPROXIMATION LABEL** | National CSV has all primary `adjust` estimates and CIs, consistent at displayed precision with October 3 PDF; alternative singleton methods documented. Independent raw-data/design verification outstanding. |
| Shared-reference NFHS-4 versus NFHS-5 gaps and changes | **USE WITH EXPLICIT APPROXIMATION LABEL** | Exact source in `comment15_intervals.csv`; reference differs from national own-wave; conditional cross-wave uncertainty. Do not infer policy effects or direct differences of unrelated references. |
| State-wise public/private descriptive proportions and differences | **USE WITH SPARSE/BOUNDARY FLAGS AND QUALIFIED CIs** | 432 rows (2 waves × 2 cohorts × 36 states × 3 metrics); `statewise_design_support.csv` available. Some proportion Wald intervals exceed [0,100] or are withheld. No multiple comparison claims or cross-state ranking tests. |
| First-delivery descriptive/state analysis | **USE AS SUBGROUP** | Includes first-delivery twins, not woman-level singleton estimand; sparse/undefined NFHS-5 Ladakh private first-delivery comparison. |
| Social-group imputation sensitivity | **SUPPLEMENT AS APPROXIMATE SENSITIVITY** | 30 imputed draws, approximately survey-weighted; MAR untestable; does not establish proper multiple imputation or exact coverage. |
| Residence/wealth subgroup changes | **EXPLORATORY SUPPLEMENT** | Use reported Holm adjusted p-values for the ten tests; unadjusted CIs cannot be read as multiplicity-adjusted significance. |
| Policy-period contrasts | **EXPLORATORY ONLY; VERIFY CORRECTED KEY BEFORE REUSE** | Corrected state-stratum-PSU household frame file reports 30170 observed vs 30198 published PSU units and earlier frame `legacy_design_equivalent_for_this_cohort=false`. Corrected interval CSV matches October 3 five displayed gaps, but unresolved design and raw-run review statuses prevent unqualified definitive inference. |
| Historical macro-policy/context analyses | **DESCRIPTIVE SENSITIVITY ONLY** | Sparse source linkage/selection and state-level context do not permit policy causal inference. |
| Federated experiment | **COMPUTATIONAL PROTOTYPE ONLY** | Numerical parity is not privacy protection, remote federation or full survey design validation. |

## Checks performed on saved records
- National gap estimates and both ends of national CIs agree with October 3 PDF at displayed precision.
- Statewise summary records reproduction of national weighted descriptive estimates and SEs.
- Shared-reference CSV includes `all_narrow, minimum=1, overall, NFHS-5_minus_NFHS-4` point **4.66164354**, CI **[3.33097443,5.99231265]** pp.
- Corrected policy-period `adjust` intervals include A 21.3165 [17.7941,24.8390], B 23.6803 [21.3551,26.0054], C 23.7149 [21.9539,25.4758], D 25.8371 [22.0482,29.6260], E 24.3642 [20.8923,27.8361], matching October 3 at two decimals.
- Statewise retained sparse and withheld interval flags; out-of-bounds Wald proportion intervals require explicit labeling.

## What is NOT signed off
1. Verification of exact survey design against official sampling frame and primary sampling variance with no omitted sampled units.
2. Independent rerun of microdata from NFHS-4/5 and the complete table/CI pipeline, including corrected household state keys.
3. Empirical calibration/coverage of conditional Taylor or imputation intervals under actual complex sampling and model-fit uncertainty.
4. Final manuscript numerical/text-to-table proof and any publication decision requiring an external statistical reviewer.

## Recommended next action
Manuscript drafting **may proceed using the qualified national, shared-reference and state descriptive estimates**. For submission, seek survey-design expert confirmation or local microdata-based model-refit PSU resampling as a validation/sensitivity audit, with records of exact code, inputs, and any differences. Do not mark exact design uncertainty 'resolved' merely because the aggregate outputs are internally consistent.
