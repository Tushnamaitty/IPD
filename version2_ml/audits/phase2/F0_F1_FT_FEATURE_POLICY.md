# IPD V2 — F0, F1, FT feature manifest policy (provisional)

**2026-10-15 — NOT a final freeze.** Historical source: V1 `notebooks/03_machine_learning_preprocessing.ipynb`. The 36 named predictors were previously audited; this document records feature policies, not a new empirical validation.

## F0 — original Version 1 (36 named predictors, historical reproducibility)

**10 numeric:** `maternal_age` (`v012`, age at interview), `education_years` (`v133`), `wealth_index` (`v190`), `birth_order` (`bord`), `total_children_ever_born` (`v201`, interview time), `preceding_birth_interval` (`b11`), `age_at_first_birth` (`v212`), `bmi` (`v445`, interview-time), `first_anc_timing` (`m13`), `anc_visits` (`m14`).

**5 categorical:** `facility_type` (`m15` broad grouping), `residence` (`v025`), `religion` (`v130`), `social_group` (`s116`), `twin_order` (`b0`).

**21 binary:** `anc_doctor` (`m2a`), `anc_nurse_midwife` (`m2b`), `anc_traditional_attendant` (`m2g`), `anc_community_worker` (`m2h`), `anc_anganwadi_worker` (`m2i`), `anc_asha_worker` (`m2j`), `anc_other_provider` (`m2k`), `no_anc_provider` (`m2n`), `health_insurance` (`v481`), `weight_measured_during_pregnancy` (`m42a`), `bp_measured_during_pregnancy` (`m42c`), `urine_sample_during_pregnancy` (`m42d`), `blood_sample_during_pregnancy` (`m42e`), `told_about_pregnancy_complications` (`m43`), five counselling features (`s420a-e` in NFHS-5), `convulsions_during_pregnancy` (`s434`), and `swelling_during_pregnancy` (`s435`).

F0 is **frozen as a historical specification only**; model results must be reproduced, and F0/F1 effects only compared on **matched births**.

## F1 — NFHS-5 leakage-reviewed *candidate*

Start from F0. Replace interview-time `maternal_age` with `maternal_age_at_index_birth=(b3-v011)/12` (or integer completed age with explicit representation choice). Drop `total_children_ever_born` (`v201`) because it reflects interview-time reproductive history; retain `birth_order` (`bord`) rather than duplicate with `bord-1`. Keep `twin_order` (`b0`) but verify the codebook interpretation. **Interview-time BMI `v445` is NOT temporally safe merely because measured:** primary temporally restricted F1 should exclude it, and evaluate its inclusion as a labeled retrospective sensitivity, unless its collection timing can be justified. Likewise evaluate other interview-time fields and pregnancy care variables for time validity / survey routing. Facility detail and observed previous-CS are **A4 experimental extras**, not automatic baseline F1 predictors.

Seven counselling/complication variables can be provisionally present for NFHS-5 F1 **only with explicit eligibility/missingness interpretation**. Structural missingness must never be imputed to `no` or counted as patient-level missing-at-random. With whole-cohort F1, conduct absence/most-recent-birth sensitivity and avoid inappropriate pregnancy-time claims.

## FT — conservative NFHS-4→NFHS-5 transport candidates

**Exclude from primary FT (seven structural skip-risk features):**
`counselled_vaginal_bleeding`, `counselled_convulsions`, `counselled_prolonged_labour`, `counselled_severe_abdominal_pain`, `counselled_high_blood_pressure`, `convulsions_during_pregnancy`, `swelling_during_pregnancy`.

Reason: In broad institutional records, all seven are missing for all NFHS-4 `bidx>1` births; in NFHS-5, some earlier births have valid responses. Restrict most-recent-birth subsets for these seven as a **planned sensitivity**, not an automatically comparable whole-birth feature set.

The **remaining 29 historic F0 predictors are only candidates**, NOT certified as comparable constructs across waves. FT must apply the same leakage-conscious time policy as F1: exclude interview-time `v201`, substitute correctly derived maternal age at delivery, and conservatively exclude interview-time `v445` from prospective-timing comparisons. This means the eventual FT primary feature count can be **lower than 29**. Check each remaining feature's question wording, skips, data availability, coding and wave-specific population definition before an explicit FT whitelist is frozen.

`v190` is survey-relative wealth quintile; never interpret as constant absolute wealth across waves. Geographic/state crosswalk and `s116` missingness require review. Preserve training-only preprocessing and avoid using NFHS-5 outcomes to tune a NFHS-4 source-only model.

## Cohort policy

NFHS-5 V1 broad institutional n=201,311; NFHS-4 n=195,997. NFHS-5 has exactly 1,634 births at `v008-b3=60`, 0 above 60. Candidate common 0–59-month eligibility yields NFHS-5 n=199,677, NFHS-4 n=195,997, but final rule needs official recode verification. Never attribute performance differences to model changes when test rows differ.

## Approval gates

- F0: **historical definition captured; reproduction pending**.
- F1: **timing/coding candidate manifest; questionnaire/timing audit and reproducible build pending**.
- FT: **conservative exclusion rule approved; remaining features and eligibility not frozen**.
- Cohorts: **reconciled; official 60-month rule unresolved**.
- Confirmatory test + survey/paired bootstrap: **Phase 4 pending**.

Do not publish licensed DHS microdata or per-row predictions.
