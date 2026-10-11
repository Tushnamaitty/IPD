# IPD — Remaining Technical Verification Checklist
**Created:** 11 October 2026  
**Status:** PENDING — these are recommended publication-quality verification tasks, not new experiments explicitly requested by Professor Murthy.

## 1. Independently verify national and cross-wave results
- [ ] Load original licensed NFHS-4 and NFHS-5 Birth Recode `.DTA` files locally (not on public GitHub).
- [ ] Independently reconstruct narrow cohorts (`m15=21` vs `m15=31`, `m17`, valid dates, `0 <= v008-b3 < 60` months).
- [ ] Check expected cohort sizes: NFHS-4 all 114263, first delivery 52851; NFHS-5 all 115197, first delivery 51854.
- [ ] Recalculate weighted public/private cesarean proportions and adjusted national gaps.
- [ ] Check expected adjusted gaps: NFHS-4 all 19.166 pp, first delivery 21.208 pp; NFHS-5 all 23.757 pp, first delivery 25.237 pp.
- [ ] Independently verify shared-reference NFHS-4/5 comparisons (different reference from own-wave national results).
- [ ] Produce a PASS/FAIL comparison report with predefined tolerances and differences explained.
**Code:** Yes; reuse current scripts and add a small *independent* checker. Simply rerunning the same pipeline is reproducibility, not independent method verification.

## 2. Independently verify state/UT results
- [ ] Recompute survey-weighted public/private cesarean percentages and private-minus-public gaps for each state/UT, both survey waves, all narrow births and first deliveries.
- [ ] Verify standard errors/CIs or document their approximation, sample counts, sparse flags, omitted/undefined intervals and wave-specific geography mappings.
- [ ] Check national totals against state-wise results, and compare all 36 reported entries per wave with `extensions_work/17_statewise_descriptive/outputs/statewise_descriptive/`.
- [ ] Create a PASS/FAIL validation report (separate implementation where practicable).
**Code:** Yes; reuse `IPD_statewise_descriptive.py` and write a small independent cross-check.

## 3. Investigate and document the NFHS-5 28-PSU discrepancy
- [ ] Confirm whether the published 30198 completed-fieldwork PSU count and 30170 observed household-recode units measure the same concept; compare exact state/stratum/PSU identifiers.
- [ ] Check discrepancies state by state, missing IDs, roster exclusions, and active analysis strata.
- [ ] Assess consequences for uncertainty estimates and whether a defensible alternative observed frame changes them.
- [ ] Resolve via supporting survey documentation if possible; otherwise retain explicit limitation without assuming the 28 units exist as observed sample records.
**Code:** Yes, at least diagnostic runs. Build on `IPD_nfhs5_roster_count_diagnostic.py` and corrected policy-period household-frame scripts.

## 4. Perform final survey-design sensitivity / model-refit verification
- [ ] Review `v005`, `v021`, `v022`, state keys and full observed design roster.
- [ ] Specify a stratified PSU-resampling method compatible with available design information; document singletons and zero-domain PSUs.
- [ ] Resample within appropriate strata, **refit** adjustment models per replicate, restandardize, and calculate interval estimates.
- [ ] Compare with existing approximate joint fitted-model/reference Taylor intervals for primary national results and, if feasible, key shared-reference contrasts.
- [ ] Investigate material CI differences; do not imply a bootstrap recovers unavailable finite-population corrections or exact multistage variance.
- [ ] Obtain expert statistical review if required to validate survey-design choices.
**Code:** Yes, likely substantial new computation. Alternative is independent survey-design expert review; not both necessarily required.

## 5. Freeze verified manuscript outputs
- [ ] Choose definitive point estimates, CIs, cohorts, references and figure/table source CSV/JSON versions.
- [ ] Record data/input script checksums, dependencies, commit SHA and analysis commands without exposing DHS microdata.
- [ ] Build a machine-readable verification manifest with source paths and PASS/FAIL/QUALIFIED statuses.
- [ ] Mark broad-sector, original predictive ML, E-value and old preliminary interval outputs historical/secondary so they cannot replace primary narrow estimates.
- [ ] Check manuscript numbers/captions and supplementary tables against the final manifest.
**Code:** Minimal; small manifest/checker script is helpful.

## Order and completion rules
Do **1 → 2 → 3 → 4 → 5**. Manuscript drafting can proceed concurrently, but do not call full independent statistical sign-off complete until outstanding design issues are resolved or transparently qualified and the final results have been checked.

### Relevant existing repo records
- `docs/manuscript_results_inventory_2026-10-11.md`
- `docs/uncertainty_review_2026-10-11.md`
- `docs/consistency_audit_2026-10-11.md`
- `docs/publication_uncertainty_decision_2026-10-11.md`

**Data rule:** NFHS source records remain local and must not be committed.
