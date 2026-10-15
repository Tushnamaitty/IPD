# IPD V2 — Phase 2 Sign-off Review (15 October 2026)

**Branch:** `version2-ml-development`  
**Decision:** **CONDITIONAL / NOT FULLY SIGNED OFF.** Existing aggregate audits establish important facts, but cannot establish questionnaire equivalence, source-time availability, or completed feature construction.

## Decisions approved from existing evidence

1. **Historical cohort definitions are reconciled.** NFHS-4 broad institutional births 195,997; NFHS-5 201,311. Public/private-only cohorts 195,366 and 200,794. Do not substitute Paper 1 narrow-hospital cohort.
2. **The NFHS-5 recall discrepancy is localized.** Exactly 1,634 broad-institution births at completed month 60, none greater than 60; NFHS-4 has zero at 60 and above. Keep V1 historical cohort (201,311) intact. For future matched FT comparisons, `0..59` months is a **candidate** (NFHS-5 n=199,677), not approved as the official primary rule until survey eligibility reviewed. Compare F0/F1 on identical records.
3. **Seven corrected source-code pairs are approved as *candidate mappings*, not as fully equivalent feature constructs.** NFHS-4 `s419a..s419e` -> NFHS-5 `s420a..s420e`; NFHS-4 `s432/s433` -> NFHS-5 `s434/s435`. Observed 0/1 codes align and 8 is an explicit unknown response for latter pair. Preserve system missing separate from 0, do not make 8 equal 0.
4. **Historical V1 F0 stays unchanged for reproducibility.** It is not a prospective at-birth feature set. This does not confer leakage-free status.
5. **F1 replacement policy approved for experimental evaluation, not feature certification:** derive age-at-birth from `b3-v011` after CMC plausibility check; investigate use of birth-order/index parity rather than interview `v201`; exclude/interrogate post-birth BMI `v445` in temporally restricted sensitivity; prior-CS information only from actually observed earlier births, *unknown is not no*.

## Blocking final sign-offs

| Gate | Required evidence | Current status |
|---|---|---|
| Official recall eligibility | NFHS-4 and NFHS-5 BR recode/questionnaire 60-month boundary inclusion explanation, wave-level comparison | **UNRESOLVED** |
| Seven-variable semantic/skip harmonization | Questionnaire item text and routing / missing-by-eligibility universe for both surveys | **UNRESOLVED** |
| All 36 features | Original preprocessing equivalence and wave-specific value definitions/missingness/eligibility | **UNRESOLVED** |
| F1 age/parity creation | Executed mother/birth/date validation with boundary/duplicated multiple-birth audit and proof of source timing | **UNRESOLVED** |
| Previous-CS experimental variable | Birth sequence/previous institutional coverage, multiple births and ambiguous prior delivery handling | **UNRESOLVED** |
| Splits and inferential procedures | Real-data group overlap, domain support, survey-weighted variance and paired stratified PSU-bootstrap design review | **PHASE 4, UNRESOLVED** |

## Phase 2 closure criterion

Phase 2 can be declared **documentation and diagnostics complete**, but **NOT an unconditional data/feature freeze**. For operational progress, Phase 4 real-data pilot can start while the above issues are addressed. No team experiments or confirmatory test claims until freezes are approved.

**Do not publicly commit microdata, respondent IDs, per-row predictions, small sensitive cells or unreviewed licensed questionnaire extracts.**

## Next local check

Run `version2_ml/audits/phase2/phase2_final_feature_checks.py` in the local IPD working tree. It outputs **aggregate JSON only** and confirms the candidate month-based age-at-index and index-birth parity constraints for both waves. Even a green result cannot establish questionnaire eligibility or survey-bootstrap validity.
