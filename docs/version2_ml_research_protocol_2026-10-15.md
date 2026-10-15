# Version 2 — Cesarean ML Research Protocol (Phase 1)

**Date:** 15 October 2026  
**Repository:** Tushnamaitty/IPD  
**Status:** Planning freeze only. **No new modeling experiments or NFHS microdata audit have been performed for Version 2.** Phase 2–4 must complete technical validation before distributing teammate work.

## 1. Locked primary research question

**Can machine-learning models predict whether an observed institutional delivery was cesarean or non-cesarean using NFHS survey-recorded characteristics?**

This is **the same core retrospective binary prediction question as Version 1**, as required by the project guide. Version 2 seeks (i) better predictive discrimination and probability quality using additional features/algorithms and (ii) evidence that any gain remains reliable across survey waves, regions and facility sectors. These are secondary extensions, not a replacement question.

**Outcome:** recorded cesarean vs non-cesarean (`m17`, verified separately in each survey).  
**Setting:** retrospective classification of recorded institutional deliveries, not prenatal clinical prediction or clinical necessity.  
**Sources:** NFHS-5 development/evaluation, NFHS-4 wave-transfer evaluation; optional additional DHS country only after compatibility/permissions audit.  
**Unit:** birth record, clustered by mother and PSU; review twins and repeated births.  
**Population:** conditional on institutional delivery, not all pregnant women. Reconstruct eligible n from the actual code; historical Version 1 reported 201,311 records and 36 features.

## 2. Separate-paper boundary

Professor Murthy's epidemiology manuscript estimates survey-weighted public/private sector cesarean **inequalities**, including adjusted gaps and wave/state descriptions; this paper develops and validates **record-level ML predictions**. They share NFHS data and possibly facility/state context but **must not reuse** epidemiology figures, standardized gap estimates, manuscript prose or test-set-derived outcome statistics as independent ML results. Using shared data is not by itself plagiarism. Cite relevant concurrent/published work, credit actual contributors, agree authorship, and obey NFHS/DHS licensing. The professor's narrow `m15=21/31` cohort is **not** the historical ML institutional-delivery cohort.

## 3. Targeted literature verification (15 October 2026)

Independently checked primary pages:

1. **Rony et al. 2026, medRxiv preprint**, Bangladesh MICS *elective* cesarean prediction: random, cluster and leave-one-division-out validation, between-wave transfer and calibration. **Direct threat:** wave/geographic validation is not itself novel. Different target and survey from Indian NFHS. https://doi.org/10.64898/2026.08.12.26360275
2. **Zoha et al. 2025, PLOS Global Public Health**, Bangladesh DHS cesarean prediction ML. **Threat:** this outcome and survey-ML genre is established. https://doi.org/10.1371/journal.pgph.0005494
3. **Austin et al. 2016, Journal of Clinical Epidemiology**, established geographic/temporal model validation. https://doi.org/10.1016/j.jclinepi.2016.05.007
4. **Debray et al. 2023, BMJ, TRIPOD-Cluster**, reporting standards for clustered prediction. https://doi.org/10.1136/bmj-2022-071018
5. **TRIPOD+AI 2024, BMJ**, regression/ML prediction reporting. https://doi.org/10.1136/bmj-2023-078378

**Precision of novelty claim:** This was targeted verification, **not a systematic review**. Do not claim “first NFHS study” or that any transportability method is newly invented. The **potential** gap is a unified, leakage-controlled comparison of individual/context information, matched predictive improvements and penalties across multiple transport axes, plus interpretable correction/adaptation. Novelty scores suggested by Claude/Anara are subjective and contingent on results. Update the full-text literature review before submission.

## 4. Part A — Seven approved experiment work packages

| ID | Study | Planned work | Purpose |
|---|---|---|---|
| A1 | Geographic/area features | Test state and reliably sourced, time-valid area descriptors against matched original feature set; audit geographic availability | Potential discrimination gain |
| A2 | Historical NFHS-4 context | Build lagged state/sector prevalence predictors from legally accessible **historical training** records only; never incorporate evaluation labels | Potential discrimination gain; leakage sensitivity |
| A3 | State × sector target encoding | Smoothed target encoding fit in **mother-grouped out-of-fold** development folds; prespecified fallback for unseen combinations | Potential discrimination gain |
| A4 | Facility detail and prior C-section | Harmonize broad/detailed `m15`; reconstruct only earlier births' `m17`; separate first birth, observed yes/no and unknown | Potential discrimination gain |
| A5 | Between-wave transfer and recalibration | Tune on NFHS-4 alone, test on NFHS-5; audit birth calendar overlap; frozen model vs disjoint-sample recalibration | Wave transport; probability improvement, not guaranteed ROC-AUC gain |
| A6 | Geographic and sector transfer | Leave-one-state/region-out; public↔private transfer with sector-compatible features and no unseen-state-ID misuse | Generalization and transfer |
| A7 | Model tuning, robustness and explainability | Compare LR, XGBoost, CatBoost on matched splits; tune, check discrimination/calibration, subgroup reliability and SHAP stability | Potential discrimination/calibration gain and evaluation |

A1–A4 and A7 include **candidate model-improvement work**. A5 and A6 are **validation/adaptation**; a larger experiment count does not guarantee higher accuracy.

**Ownership when Phase 5 begins:** Advanced teammate A2+A3+A5+A6; other teammate A1+A4+A7. The second teammate's prior-CS reconstruction and inference methods receive advanced QC. Both must follow the same shared cohort, splits and output contracts.

## 5. Novelty roadmap — not all mandatory standalone experiments

| ID | Extension | Status |
|---|---|---|
| N1 | Individual-only vs context-only vs combined feature-layer comparisons | Priority; coordinate with A1–A4/A7 |
| N2 / E6 | Simple state/sector/prevalence benchmark (“map beats model”) | Priority; training-source aggregates only; compare fairly on target |
| N7 | Unified **feature tier × transport axis** penalty matrix | Priority; define metrics before opening holdouts, populate after A5/A6 |
| N5 | Few-shot target-local recalibration with disjoint adaptation/evaluation samples | Priority conditional on transferable model and adequate target labels |
| N3 / E1 | Label shift estimation/correction, with assumption checks | Conditional; seek calibration/Brier gains, **not monotone ROC-AUC gain** |
| N6 | One-country external DHS validation | Conditional on compatible outcome/population/features and licensing |
| N4 / E3 | Rashomon (near-equal source models) disagreement as label-free transport warning | Exploratory high-risk; benchmark against simple uncertainty |
| E2 | Covariate versus residual shift decomposition | Merge into N7 if assumptions defensible |
| E4 | Design-aware conformal transport coverage | Optional statistical-method extension |
| E5 | Audit of published survey-CS ML models | **Defer**: potential separate paper |

Define N7 estimands early, but calculate final summaries later. N4/N6 are not essential if infeasible or uninformative. No novelty score or success is guaranteed.

## 6. Hypotheses, metrics, improvement and failure criteria

**Primary objective:** improve held-out retrospective cesarean prediction compared with the **original-feature XGBoost rerun on the identical locked NFHS-5 population and observations**.

- **Primary predictive metric:** ROC-AUC, maintaining continuity with V1. **Key secondary:** average precision/PR-AUC (always report prevalence). Required probability-quality guards: Brier score, calibration intercept and slope. Also report sensitivity, precision, recall, F1 at validation-chosen threshold.
- **Historical-only V1 references:** ROC-AUC **0.8033**, PR-AUC **0.5330**, F1 **0.5469**. Do **not** compare a new cohort's metric directly to these as a treatment effect.
- **Primary hypothesis H1:** one enhanced configuration selected using *development data only* has higher independent holdout ROC-AUC than matched original-feature XGBoost (paired difference ΔAUC > 0). If untouched independent test unavailable, call this exploratory rather than confirmatory.
- **H2:** model improvement yields positive matched Δaverage precision without material deterioration in calibration, with paired uncertainty reported; secondary.
- **H3:** discrimination and calibration changes are heterogeneous across wave, state/region and facility-sector transfers; report axes transparently, never cherry-pick.
- **H4:** locally fitted recalibration improves probability accuracy/calibration on **disjoint** target evaluation records; not necessarily discrimination.

**Development advancement rule (provisional; not significance threshold):** consider configurations with ≥0.005 absolute grouped-validation ΔROC-AUC **or** ≥0.01 absolute ΔAP and without an important Brier/calibration regression for integration; revisit these pragmatic cutoffs **before** final test access once baseline variability is measured. Favor simpler well-calibrated models for near ties. **Publication superiority requires** a favorable matched independent-holdout effect estimate, a defensible uncertainty interval, clinical/practical size interpretation and multiplicity transparency; do not declare success purely from a threshold or a nominal p-value.

**No guaranteed improvement:** every negative/null result is documented. No tuning, feature selection or threshold choosing with final test labels. One prespecified final chosen enhanced model versus original-feature XGBoost is the primary confirmatory comparison. Other algorithm/feature-tier contrasts are exploratory unless their multiplicity-controlled family is locked beforehand.

**Transport penalty:** for metric M and predeclared axis, M(target) − M(source validation), reporting source/target prevalence and metric dependence on prevalence. A monotone label-prior correction cannot on its own improve ROC-AUC rankings. Local recalibration uses separate adaptation and evaluation records.

## 7. Mandatory shared protocol — technical freeze pending

1. Audit exact original **36 feature names**, transformations, cohort code, original test usage and whether outcomes/test data affected earlier tuning. If original test “burned,” construct and document new partition or downgrade independence claim.
2. Harmonize NFHS-4/5 variable definitions, `m15` categories, `m17` codes, state boundaries, birth calendar year/recall windows and missingness. Distinguish all institutional deliveries from Professor Murthy's narrow hospital cohort.
3. Mother-grouped development/validation/test folds; assess PSU-disjoint robustness. Survey weighting, strata and PSU resampling must be explicitly justified and expert-reviewed; naive row bootstrap is not design-valid.
4. Generate previous-CS history **only from truly earlier births** and label unobserved history **unknown**; audit recode availability by wave.
5. Train-only feature generation: target encodings, imputation, standardization, historical rate aggregates, threshold selection, hyperparameters and calibration. NFHS-5 outcome labels cannot guide NFHS-4-only temporal model training.
6. For unseen-state tests, a state categorical ID has no learned state effect: use a state-free baseline or valid externally observable context, not arbitrary held-out-state target encoding.
7. For public↔private transfer, exclude source-constant or destination-unavailable context variables as required for a fair source-to-target test; no causal interpretation of sector.
8. No claim of clinical necessity or early-pregnancy prediction: retrospective features may be unavailable before delivery.
9. Declare all eligibility criteria, subgroup minimum counts, confidence interval methods and test access controls before confirmatory evaluation.

## 8. Prediction and reproducibility contracts

**Aggregate metrics table:** cohort/survey/split/holdout/model/config/feature tier, n, events, prevalence, ROC-AUC, AP, Brier, calibration intercept and slope, F1/recall/precision, CI method and paired differences.

**Restricted per-birth predictions (not public GitHub):** privacy-safe birth key; opaque mother grouping key; true label; predicted probability; survey weight; stratum; PSU; wave; state; facility sector; date bucket; fold; model/config identifiers; threshold source.

**Code artifacts to commit:** data dictionaries; feature provenance and availability audit; split and model configurations; software versions; deterministic seeds; CLI runners; small synthetic fixtures; tests; *aggregate-only* metrics/figures/manifests. **Never publicly push licensed DHS microdata or row-level predictions without legal/privacy permission.**

## 9. Phases, responsibilities and gates

- **Phase 1 (this document):** primary question fixed; targeted literature verified; experiments and novelty candidates scoped; hypotheses, metrics, improvement criteria and confirmatory vs exploratory status documented. **Planning complete; empirical claims NOT verified.**
- **Phase 2:** verify V1 source code/data, original 36 features, sample counts, labels, data harmonization and leakage.
- **Phase 3:** common splits, model runner, preserved holdouts, survey-aware measurement, reproduced baseline and shared output interface.
- **Phase 4:** reproducible pilot and leakage/unit tests, reviewed survey inference and signed-off starter package.
- **Phase 5:** two complete teammate manuals with exact file inputs, algorithms, outputs, acceptance tests, deadlines and authorship/data access agreement.
- **Phase 6–7:** parallel implementation with continuous checks.
- **Phase 8:** integrate selected features, select solely from development evidence, open final test once, lock outputs.
- **Phase 9:** report transparently following TRIPOD+AI/TRIPOD-Cluster as applicable; separate Paper 2 from the Murthy epidemiology manuscript.

**Unresolved before teammate handoff:** source-data access and licenses, whether original test is reusable, actual cross-wave feature compatibility, prior-CS observability, district identifiers, harmonized state boundaries, defensible resampling and subgroup sample sizes, independent final split, exact external-country feasibility. These are **Phase 2–4 blockers**, not reasons to invent results now.

## 10. Phase 1 sign-off

**Phase 1 research planning documented; approved for Phase 2 data/code audit, NOT yet for final model training or teammate assignment.** The fixed primary prediction question remains unchanged; advanced analysis is explicitly secondary. Literature evidence supports relevance but **not a “first-ever” novelty claim**, and neither accuracy nor novelty gains are assured.
