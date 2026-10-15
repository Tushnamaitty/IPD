# Version 2 ML — Phase 2 Audit Consolidation and Handoff

**Date:** 15 October 2026  
**Status:** Code-level and selected local aggregate checks complete; **NOT a complete Phase 2 data/feature freeze or statistical sign-off**.  
**Scope:** Retrospective cesarean/non-cesarean prediction in NFHS institutional births; maintain Version 1 question.  
**Upstream protocol:** [Version 2 research protocol](../../../docs/version2_ml_research_protocol_2026-10-15.md).

## 1. Evidence register and provenance

- Original V1 predictor definition and grouping: `notebooks/01_data_exploration.ipynb`, `notebooks/03_machine_learning_preprocessing.ipynb`; historical model `outputs/metadata/b2_model_metadata.json`.
- NFHS-4/5 codebook and cohort audit: user-run `Extension3_crossround_codebook_audit.py` outputs `nfhs4_raw_audit.json`, `nfhs5_raw_audit.json`, `crossround_variable_audit.csv`, `crossround_audit_summary.json`.
- Expanded 36-predictor raw metadata audit: user-run `IPD_version2_36_feature_audit.py` outputs `feature_inventory_by_wave.csv`, `crosswave_36_feature_flags.csv`, `expanded_audit_summary.json`.
- Corrected seven mapping metadata check: `seven_mapping_metadata_audit.json`.
- Corrected seven mapping institutional row check: `seven_mappings_row_counts.csv`, `step9_harmonization_summary.json`.
- Interview-time/history audit: `step10_timing_history_counts.csv`, `step10_timing_history_summary.json`.
- Recall eligibility audit: `recall_bucket_counts.csv`, `step11_recall_summary.json`.
- Source reports were shared in conversation for review; they are **not automatically published to public GitHub**, particularly because of data licensing/rare-cell disclosure. Figures below are aggregate observations, not trained-model results.

## 2. Reconciled survey cohorts

| Measure | NFHS-4 | NFHS-5 |
|---|---:|---:|
| All BR rows | 1,315,617 | 1,274,250 |
| Broad institutional births (public/private/other including `m15=96`) | **195,997** | **201,311** |
| Public | 141,028 | 150,299 |
| Private | 54,338 | 50,495 |
| Other institution | 631 | 517 |
| Public/private-only | 195,366 | 200,794 |
| Cesareans in public/private-only | 35,671 | 44,742 |
| Unique mothers in broad institution cohort, from expanded audit | 153,489 | 158,429 |

Both expected-count checks passed. Outcome `m17` in audited institutional rows was binary with no missing outcomes. **Do not confuse** broad Version 1 ML institutional cohort with Professor Murthy's narrow `m15=21/31` epidemiology cohort.

## 3. Recall-window boundary and proposed cohort policy

Local recall audit used `v008 - b3` in completed months:

| Interval | NFHS-4 broad institutional | NFHS-5 broad institutional |
|---|---:|---:|
| 0–59 months | 195,997 | **199,677** |
| Exactly 60 months | 0 | **1,634** |
| >60 months | 0 | 0 |
| Missing/negative | 0 | 0 |

The 1,634 NFHS-5 boundary births are **not proven erroneous**. The 0–59 versus 0–60 distinction is an eligibility/design choice requiring survey-recode documentation.

**Provisional policy:** preserve the full original **201,311** NFHS-5 cohort for faithful historical Version 1 reproduction. Define a **proposed** 0–59-month harmonized analysis cohort for F1/transport, with NFHS-5 n=199,677 and NFHS-4 n=195,997, **pending** survey eligibility documentation and impact analysis. Conduct 60-month-included sensitivity. **All matched F0/F1/enhanced model comparisons must use identical test births**; cohort restriction must not masquerade as a model improvement. The baseline and other models for a matched V2 comparison need to be rerun on the same chosen cohort.

## 4. All 36 original predictors and cross-wave mappings

The original NFHS-5 data/model code identified **36 predictors: 10 numeric, 5 categorical, 21 binary**. Thirty-one names were observed in NFHS-4 under their NFHS-5 raw codes; five counselling variables were absent by *name*, and two pregnancy-complication variable names referred to different concepts. Further metadata/row checks supplied candidate correct mappings for all seven:

| Original feature | NFHS-4 | NFHS-5 | Evidence status |
|---|---|---|---|
| Counselling vaginal bleeding | `s419a` | `s420a` | Present, same 0/1 metadata codes; row code audit passed |
| Counselling convulsions | `s419b` | `s420b` | Ditto |
| Counselling prolonged labour | `s419c` | `s420c` | Ditto |
| Counselling severe abdominal pain | `s419d` | `s420d` | Ditto |
| Counselling high blood pressure | `s419e` | `s420e` | Ditto |
| Convulsions during pregnancy | `s432` | `s434` | Present, 0/1/8 metadata and observed-code compatibility |
| Swelling during pregnancy | `s433` | `s435` | Ditto |

**Conclusion:** 36/36 original predictors now have plausible **source-variable candidates** across waves; **36/36 are not yet certified semantically harmonized**. Check questionnaire wording, reference pregnancy, eligibility/skip-universe and transformations before the final FT whitelist. The mapping audit was metadata-focused; the follow-up checked aggregate observed codes, not every substantive questionnaire property.

Seven-variable Step 9 system-missing percentages among broad institutional births (reported earlier): five counselling variables ~32.95% NFHS-4 / ~23.60% NFHS-5; convulsions and swelling ~24.16% NFHS-4 / ~19.95% NFHS-5. Differential skip/eligibility is a concern, not automatically a data error. Map `0=no`, `1=yes`; for complication fields retain `8=don't know` as a distinct unknown/missing category and never interpret as a complication. Missing is **not** no.

Other overlapping variables audited previously: `m15`, `m17`, `v024`, `v025`, `v130`, `v133`, `v190`, `s116`, `bord`, `b0`, survey design fields. State code numbering differs; harmonized state crosswalk yielded 35 mapped geographies, pending full geography changes/support review. `v190` wealth quintiles refer to wave-relative construction, not identical absolute wealth. Survey identifiers and weights are wave-specific.

## 5. Feature timing, historical and previous-CS audit

The Step 10 aggregate audit identified:

| Diagnostic | NFHS-4 | NFHS-5 |
|---|---:|---:|
| `v201` children-ever-born at interview > `bord` birth order | 24.16% | 22.70% |
| Interview age differs by >1 year from estimated age at index birth | 68.11% | 69.00% |
| BMI recorded | 98.75% | 97.38% |
| Earlier institutional BR birth observed | 20.96% | 20.60% |
| Earlier cesarean observed | 5,313 | 6,276 |
| `bord>1`, but no earlier institutional birth observed | 36.99% | 38.36% |

**Interpretation:** `v201` and `v012` are interview-time measures; they may encode information beyond the indexed delivery. `v445` BMI is also measured at interview, not necessarily during pregnancy. Prior-CS observability is limited to earlier recorded institutional births within available BR history; *absence of observed earlier CS does not demonstrate none ever*. Counts are descriptive and do not verify complete patient history or clinical timing.

**Provisional F0/F1 policy:**
- **F0 historical:** reproduce original 36 features and original full eligible institutional cohort, only as a **historical benchmark**.
- **F1 leakage-reviewed:** use **maternal age at birth** derived and verified from birth CMC and maternal DOB; replace `v201` with an index-birth reproductive-history/parity representation, carefully handling multiple births; document interview-time BMI as retrospective and sensitivity-test excluding it; keep original broad `facility_type`.
- **A4 add-on:** detailed facility categories and a **four-level observed-history variable**: first birth; earlier CS observed; earlier births observed with no CS recorded in *observed* history; unknown/incomplete history. Never call the third “no previous CS ever,” and retain timing/coverage indicators and limitations.
- **Both F0 and F1 must be compared on the same births** when attributing performance changes to the feature-definition revision. The original F0 full-cohort reproduction is separate from matched F0 evaluation on a harmonized/restricted cohort.

All replacements require **row-level test coverage, derivation checks and interpretation approval** prior to formal feature freeze.

## 6. What is verified versus unresolved

**Code/aggregate verified:** historical 36 predictor names and mappings; old group-split mechanism (`StratifiedGroupKFold` by mother/respondent); full and public/private institutional counts; broad outcome coding; candidate seven remappings from metadata and observed categories; interview-time vs index-birth timing concerns; exact 60-month boundary concentration.

**Not yet verified:** full questionnaire/skips and harmonized 36-feature *constructs*; whether age/parity replacements are correctly computed and valid in each wave; full previous-CS completeness; implementation of model splits; untouched final-test independence; weighted performance measures; paired intervals; robust domain inference; external country feasibility.

### Critical statistical-validation blocker — do not forget

The **proposed paired stratified-PSU bootstrap with 2,000 targeted resamples** has **not** been statistically validated. Before any confirmatory interpretation, audit PSU/stratum/weight choice, design domain handling, singleton strata, replicate weights, class-empty resamples, survey-specific sampling and appropriate fixed-model uncertainty interpretation. Do **not** mark inference validated merely because `v005/v001/v022` are present or because a bootstrap script runs. Avoid publishing raw NFHS row predictions or sensitive rare-cell tables.

### Critical test-set blocker

V1's earlier test set was already examined. A newly reshuffled holdout from that same source data is not automatically prospectively independent of the team's prior model-design experience. Use a documented development/final-test partition, distinguish exploratory from genuinely independent confirmation, and avoid reusing target-wave labels for NFHS-4-only transfer fitting.

## 7. Phase 3 handoff instructions

Proceed with **code-only shared infrastructure now**: config loading/validation; feature dictionary/crosswave mapping registry; safe cohort and column contracts; group-overlap and leakage guards; model/preprocessing runners; output schemas; synthetic fixture tests. Do not claim full baseline replication or design-validated intervals before real-data execution and independent review.

Required sign-off gates before handing teammate A1–A7 packages:
1. **Data freeze:** approved F0, F1 and FT feature lists with timing/skip treatment; common eligibility and cohort manifests.
2. **Split freeze:** zero mother overlap and correct isolation of model-development and holdout targets; reproducible hashed assignments.
3. **Baseline reproduction:** comparable original-feature XGBoost matched test metrics; valid feature ablations and provenance.
4. **Statistical validation:** justified survey-aware variance/interval procedure and reported inference limitations.
5. **Privacy/licensing:** no DHS raw individual data or per-row prediction tables in public repository.
6. **Pilot:** synthetic/local smoke tests and reviewer QA results.

## 8. Phase 2 status and next work

**Phase 2 is substantially progressed, but not technically complete.** Steps 1–11 contributed code review and local aggregate audits. This consolidation closes the **documentation** milestone only.

**Immediate next step:** Phase 3 shared infrastructure implementation, starting with config loading and data-contract validators, tested on synthetic fixtures. Resume questionnaire-equivalence, row-level feature transformations and design-based inference audits in parallel **before the teammate handoff**. This document is a handoff, **not a release authorization to train the final models**.
