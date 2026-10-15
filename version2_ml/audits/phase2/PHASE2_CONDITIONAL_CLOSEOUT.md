# IPD V2 — Phase 2 Conditional Closeout

Date: 2026-10-15. **Phase 2 preliminary data audits are complete; full feature and inference sign-offs are NOT complete.**

Evidence from user-run local audits:
- Raw institutional NFHS-4 195,997, NFHS-5 201,311, binary outcomes on those rows.
- NFHS-5 b19<60 includes all 201,311; CMC0..59 includes 199,677; NFHS-4 b19 absent, CMC0..59 yields 195,997.
- All 36 V1 feature raw-code candidates identified, but 22 original pregnancy/ANC fields have substantial birth-position-based missingness. Seven have different earlier-birth availability across waves.
- Index-age CMC construction passed aggregate plausibility and rounding checks; v201 is interview-time and can exceed indexed birth order.
- Seven corrected source mappings and remaining 29 raw codes present; construct equivalence is not established simply by shared code labels.
- Conservative 8-feature F1/FT real-data builder produced 195,997/201,311 complete rows with no builder errors.

**Conditional approval:** use the conservative 8-feature F1/FT builder in an exploratory Phase 4 pilot, subject to local tests and provenance. Keep original F0 36-feature definition for faithful reproduction; matched F0/F1 performance comparison requires same birth records and split.

Outstanding:
1. Reproduce V1 F0 transformations and baseline model.
2. Verify questionnaire/time definitions of core and restricted features.
3. Empirically validate mother-exclusive splits and domain support.
4. Freeze split strategy and independent-holdout claim (V1 test previously seen).
5. Design-valid weighted metrics and proposed paired stratified PSU bootstrap not validated.

**No statistical or independent-test claims authorized.** Do not upload DHS microdata or row-level predictions to GitHub.
