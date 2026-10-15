# NFHS five-year cohort rule — evidence-based provisional decision (2026-10-15)

The user-run aggregate audit `official_five_year_eligibility.json` reports:

| Item | NFHS-4 | NFHS-5 |
|---|---:|---:|
| Broad institutional births | 195,997 | 201,311 |
| b19 present | No | Yes |
| v008-b3 in 0..59 | 195,997 | 199,677 |
| v008-b3 exactly 60 | 0 | 1,634 |
| b19 in 0..59 | Not available | 201,311 |
| b19 unavailable/invalid | Not applicable | 0 |
| Records CMC=60 yet b19<60 | Not applicable | 1,634 |

**Primary, provisional eligibility policy:** use `b19<60` where this DHS-7 child-age-in-months variable exists and passes range checks (NFHS-5); use `0 <= v008-b3 < 60` for NFHS-4 without b19. On the observed cohorts, this retains NFHS-4 n=195,997 and NFHS-5 n=201,311.

**Required interpretation:** The different boundary results reflect measure definitions, not proof that any record is erroneous. The earlier 0..59 CMC NFHS-5 subset (n=199,677) is a prespecified sensitivity. Preserve full historical Version 1 cohort for reproduction, and keep matched evaluation rows across model comparisons.

**Scope:** Local aggregate counts establish implementation compatibility for these data. They do **not** certify remaining feature harmonization, questionnaire universe, exact official design interpretation beyond consulted DHS guidance, test independence or survey bootstrap/statistical inference.

**Next Phase 2 gate:** audit and approve the remaining non-seven harmonized predictors and finalized F1/FT feature manifests. Do not declare Phase 2 feature freeze until completed.
