# Locked candidate manuscript results inventory — 11 October 2026

**Freeze designation:** candidate manuscript results inventory (aggregate-verified, NOT a new raw-data rerun). Repository base reviewed: `8082ede`; README update subsequently committed. All differences in percentage points (pp), private minus public.

## Primary study
Exposure: NFHS raw delivery-place `m15=21` government/municipal hospital vs `m15=31` private hospital/maternity home/clinic. Outcome `m17`, delivery by cesarean. Date/recall filter `0 <= v008-b3 < 60` months. Descriptive and covariate-standardized **associations only**.

| Wave | Population | n (public / private) | Observed gap (95% CI), pp | Adjusted gap (95% CI), pp |
|---|---|---|---|---|
| NFHS-4 | All narrow births | 114263 (62351 / 51912) | 22.417 [21.411,23.423] | 19.166 [18.196,20.135] |
| NFHS-4 | First delivery | 52851 | 23.616 [22.245,24.988] | 21.208 [19.825,22.591] |
| NFHS-5 | All narrow births | 115197 (67336 / 47861) | 25.155 [24.270,26.040] | 23.757 [22.848,24.666] |
| NFHS-5 | First delivery | 51854 | 25.617 [24.384,26.850] | 25.237 [23.946,26.528] |

Source: `extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json`. The first-delivery category includes first-delivery twins. Intervals approximate marginal Taylor-t; fixed model structure and uncertainty caveats apply.

## Shared-reference temporal comparisons (NOT same target as national results)
Source: `extensions_work/12_narrow_scope/outputs/comment15_shared_reference/comment15_summary.json`, `Research_Update_2026-10-03` p.2. Equal-wave pooled supported reference, 33 comparable state/UT geographies: all births NFHS-4 19.05, NFHS-5 23.71, change +4.66 [3.33,5.99] pp; first delivery NFHS-4 21.01, NFHS-5 25.01, change +4.00 [2.08,5.92] pp. The national own-wave gaps **must not** be directly subtracted to recreate these estimates.

## State descriptive (principal graphic + supplemental tables)
Source: `extensions_work/17_statewise_descriptive/outputs/statewise_descriptive/statewise_primary_results.csv`; `statewise_summary.json`; `IPD_Statewise_Brief_09_10_26` p.1-2. Two survey waves; 36 reported state/UT entries each. For all narrow births, positive private-public point gaps in all reported entries; for first deliveries, 36/36 positive NFHS-4, 34/35 estimable positive NFHS-5; Ladakh lacks private first-delivery births. Sparse samples and withheld intervals must be retained as flags. **Unadjusted proportions**; no causal, ranking or cross-wave-state change test. State boundaries are wave-specific.

## Imputation sensitivity
Source: `IPD_Research_Update_2026-10-07` p.1-2; scripts `IPD_missingness_audit.py`, `IPD_social_group_imputation.py`, `IPD_rural_urban_imputation.py`. Social-group missingness roughly 5%; 30 approximate survey-weighted imputations per wave; max reported absolute change of national adjusted gap 0.018 pp and of residence-specific gap 0.026 pp. Assumes MAR conditionally; intervals approximate, not full design-proper multiple imputation.

## Exploratory / supplement only
- Residence/wealth effect-modification contrasts: `extensions_work/12_narrow_scope/outputs/comment15_heterogeneity_change/`, `comment15_multiplicity_sensitivity/`. Holm adjustment for ten exploratory tests; no universal wealth trend.
- Policy period: `extensions_work/03_nfhs4_nfhs5_temporal/outputs/policy_period_state_roster_uncertainty/`; conditional design intervals only, unresolved frame limitations, not policy effects.
- Historical state context: `extensions_work/04_health_system_context/outputs/historical_state_context/`, `narrow_west_bengal_sensitivity/`; incomplete linkage and no policy effect.
- Federated: `extensions_work/11_state_federated/outputs/federated_extension/`; local computational numerical pilot, **not remote or privacy-validated**.

## Archived / never promote silently into primary
Original NFHS-5 n=201311 ML prediction, broad-sector n=200794 AIPW/TMLE, E-values, historical causal-heterogeneity/prediction-score work. Contrasts and sources differ from the primary narrow study; use only with explicit historical/secondary labeling.

## Freeze conditions
Before manuscript citation freeze, reconcile each chosen figure/table against its CSV/JSON, review CI and PSU assumptions, confirm all flags and provenance, and rerun locally using licensed DHS microdata if independent computation verification is required. No microdata were available to this aggregate audit.
