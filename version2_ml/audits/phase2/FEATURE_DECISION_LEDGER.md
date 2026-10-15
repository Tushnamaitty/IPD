# IPD Version 2 — Phase 2 feature decision ledger (2026-10-15)

**Evidence status:** Candidate modeling decisions based on local `remaining29_feature_audit.json`, `questionnaire_skip_audit.json`, age/parity audits, and original V1 feature inventory. **Not a final semantic or timing certification.** This is a retrospective survey-recorded-characteristics prediction task, not necessarily pre-delivery prediction.

## Feature policy and disposition

Definitions:
- **CORE-CANDIDATE**: candidate for the broad full-institutional-birth primary F1 and FT; still requires codebook/value cleaning.
- **RECENT-BIRTH-ONLY**: structural missingness when `bidx>1`; eligible as most-recent-birth sensitivity only, pending questionnaire routing confirmation.
- **HISTORICAL-ONLY**: preserve in original F0 but do not include in timing-restricted F1/FT.
- **REVIEW**: not certified; a defensible fallback is to exclude until coding and measurement timing checked.

| Original named predictor | Raw source | Primary F1/FT disposition | Rationale / required sign-off |
|---|---|---|---|
| maternal_age | v012 | CORE-CANDIDATE *replace* by derived index-age | Age at interview not index birth; derived `(b3-v011)/12` passed aggregate age consistency |
| education_years | v133 | CORE-CANDIDATE | Complete both waves; measured at interview; retrospective-only caveat |
| wealth_index | v190 | CORE-CANDIDATE | Survey-relative quintiles, not absolute comparable wealth; interview time |
| birth_order | bord | CORE-CANDIDATE | Complete both waves; order of live birth, not previous pregnancies |
| total_children_ever_born | v201 | HISTORICAL-ONLY | Interview time includes births subsequent to index for >20% |
| preceding_birth_interval | b11 | REVIEW | Often structurally absent for first births; confirm special codes |
| age_at_first_birth | v212 | REVIEW | Complete raw field but implausibly young minima and historical timing require checks |
| bmi | v445 | HISTORICAL-ONLY in timing-restricted model | Interview-time measurement, sentinel 9998 present; retrospective sensitivity only |
| first_anc_timing | m13 | RECENT-BIRTH-ONLY | No nonmissing for older births, specials to 98 |
| anc_visits | m14 | RECENT-BIRTH-ONLY | No nonmissing for older births, specials to 98 |
| facility_type | m15 | CORE-CANDIDATE | Observed delivery facility; context for descriptive retrospective task, not known pre-selection |
| residence | v025 | CORE-CANDIDATE | Complete, interview-time address may differ from delivery |
| religion | v130 | CORE-CANDIDATE | Complete, codebook category equivalence needed |
| social_group | s116 | REVIEW | ~4–5% missing and 8 category; ascertain codes/skip |
| twin_order | b0 | CORE-CANDIDATE | Complete, exact coding and time point to verify |
| anc_doctor | m2a | RECENT-BIRTH-ONLY | No older-birth coverage |
| anc_nurse_midwife | m2b | RECENT-BIRTH-ONLY | No older-birth coverage |
| anc_traditional_attendant | m2g | RECENT-BIRTH-ONLY | No older-birth coverage |
| anc_community_worker | m2h | RECENT-BIRTH-ONLY | No older-birth coverage |
| anc_anganwadi_worker | m2i | RECENT-BIRTH-ONLY | No older-birth coverage |
| anc_asha_worker | m2j | RECENT-BIRTH-ONLY | No older-birth coverage |
| anc_other_provider | m2k | RECENT-BIRTH-ONLY | No older-birth coverage |
| no_anc_provider | m2n | RECENT-BIRTH-ONLY | No older-birth coverage |
| health_insurance | v481 | REVIEW | Interview-time current coverage; may differ at delivery |
| weight_measured_during_pregnancy | m42a | RECENT-BIRTH-ONLY | No older-birth coverage; conditional ANC skip |
| bp_measured_during_pregnancy | m42c | RECENT-BIRTH-ONLY | No older-birth coverage; conditional ANC skip |
| urine_sample_during_pregnancy | m42d | RECENT-BIRTH-ONLY | No older-birth coverage; conditional ANC skip |
| blood_sample_during_pregnancy | m42e | RECENT-BIRTH-ONLY | No older-birth coverage; conditional ANC skip |
| told_about_pregnancy_complications | m43 | RECENT-BIRTH-ONLY | No older-birth coverage; conditional ANC skip |
| counselled_vaginal_bleeding | NFHS4 s419a / NFHS5 s420a | RECENT-BIRTH-ONLY | Differing availability for older births |
| counselled_convulsions | s419b / s420b | RECENT-BIRTH-ONLY | Same |
| counselled_prolonged_labour | s419c / s420c | RECENT-BIRTH-ONLY | Same |
| counselled_severe_abdominal_pain | s419d / s420d | RECENT-BIRTH-ONLY | Same |
| counselled_high_blood_pressure | s419e / s420e | RECENT-BIRTH-ONLY | Same |
| convulsions_during_pregnancy | s432 / s434 | RECENT-BIRTH-ONLY | Different older-birth eligibility, `8` = unknown |
| swelling_during_pregnancy | s433 / s435 | RECENT-BIRTH-ONLY | Same |

**Correction to prior conversational count:** The non-seven set has **15** most-recent-birth ANC features: 2 timing/visits, 8 provider, 4 checkups, 1 complications counselling. Together with seven corrected items there are **22 birth-position-restricted candidates**, not just seven. These are *not automatically invalid*: they are candidates for explicitly scoped most-recent-birth sensitivity analyses, after routing verification.

## Primary and secondary models

- **F0 historical (36 named features):** Preserve source code/transformation behavior; historical full NFHS-5 cohort n=201,311. For any performance claim about F1 improvement, retrain matched F0 on *identical* records/splits.
- **F1 broad primary conservative candidate:** index-age, education, wealth, birth order, broad facility, residence, religion, twin indicator = **8 candidate constructs**. Before modeling, audit coding and temporality; do not claim available antepartum/pregnancy-time just because a field exists.
- **FT broad cross-wave candidate:** same 8 constructs with wave-specific code/label harmonization. `v190` wealth is rank within wave; use careful transport interpretation.
- **Most-recent birth (`bidx=1`) sensitivity:** evaluate added 22 ANC/counselling/pregnancy variables with verified eligible denominators and explicit missing vs not asked vs unknown. Restrict both waves identically.
- **Exploratory additive group:** preceding interval, maternal age at first birth, social group, health insurance, and retrospective BMI (only labeled as post-delivery characteristics). Do not automatically combine.

## Remaining sign-off gates

1. Codebook meanings, value/special codes, and questionnaire universes for *every retained feature*, especially age/wealth/facility/twin coding.
2. Local concrete feature-build tests reproducing F0 then implementing validated F1/FT, with no targets/IDs in predictor columns.
3. Cohort/fold matching and temporal interpretation of interview-time variables.
4. Phase 4: actual mother-exclusive splits, model comparison, survey-aware design inference and 2,000 bootstrap replicate methodology validation.

**Current decision:** conservative model-specification planning **documented**, not a completed data/feature freeze or evidence of predictive adequacy. No raw NFHS microdata/prediction records may be published.
