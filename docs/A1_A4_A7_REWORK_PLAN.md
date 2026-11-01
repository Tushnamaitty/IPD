# A1/A4/A7 rework plan (Phase 0c — awaiting owner approval)

Date: 2026-10-17 · Branch: `gaurav-v2-A1-A4-A7-r2` (base `318f160`)
Owner: Gaurav · Scope: A1, A4, A7 only. A2/A3/A5/A6 = interface coordination only.
Source of truth: `GAURAV_IPD_V2_TECHNICAL_HANDOFF.pdf` (15 Oct 2026), protocol
`docs/version2_ml_research_protocol_2026-10-15.md`, phase-2/4 audit documents,
`version2_ml/configs/*.yaml`. Discrepancies: `docs/A1_A4_A7_DISCREPANCY_LOG.md`.
Prior attempt: archived at `backup/prior-attempt-2026-10-17` /
tag `prior-attempt-2026-10-17` — not built on.

## 1. Frozen scientific decisions (from handoff + verified evidence)

- Outcome: m17 ∈ {0,1} among broad institutional births; no claims of necessity/causation.
- Cohort C1 (verified, D1): m15 ∈ {20–27, 30–33, 96} & b19 ∈ 0–59 & m17 ∈ {0,1}
  = 201,311 births / 158,429 mothers. CMC 0–59 sensitivity (199,677) labeled.
- Unit (caseid, bidx); uniqueness asserted with hard errors; mothers never split.
- No survey-design CIs, no PSU bootstrap, no significance or confirmatory claims.
- All fitting (imputation, encoding, scaling, calibration, thresholds, tuning,
  scale_pos_weight) on training/development rows only; mother-exclusive assertions
  with hard errors. Unknown never relabeled as negative.
- Real-data results labeled NOT RUN wherever the pipeline step was not executed.

## 2. Interfaces reused from `version2_ml/` (verified by line-level reading)

| Interface | Reuse decision | Notes / mismatch |
|---|---|---|
| `feature_builder.make_core_features` + `CORE_NAMES` | Reuse as F1 core (8 features) | Validates allowed m15 codes incl. 96; keeps special codes explicit; no imputation. |
| `split_contracts.grouped_partition`, `audit_partition` | Reuse | 70/15/15 provisional; mother-exclusive hard error; coarse per-mother outcome-rate stratification — documented, not altered. |
| `preprocessing.FeatureManifest`, `TrainOnlyPreprocessor` | Reuse | `handle_unknown="ignore"` for unseen state; `__MISSING__` tokens preserved (never zero for unknown). |
| `model_framework.AuthorizedClassifier`, `evaluate_binary`, `make_estimator` | Reuse | Shared XGBoost defaults (100/3) differ from A7 handoff spec (200/6/lr .05/sub .8/col .8): A7 passes **explicit** params; no shared-module edits. `evaluate_binary` returns calibration intercept/slope = None → A7 adds a separate, documented calibration module. |
| `data_contracts.audit_births` | Reuse for aggregate QA only | Its INSTITUTIONAL set already includes 96 (consistent with verified C1). |
| `validate_configs.py` | Reuse unchanged | Config warnings remain; A7 records the frozen split in its own config. |
| New shared utilities (schema, hashes, checksums) | **New** | Proposed under `version2_ml/experiments/common/`; schema coordinated with the A2/A3/A5/A6 contributor (handoff §8). Interface request, no code from them imported. |

## 3. File layout (new, aggregate-only outputs; nothing row-level committed)

```
version2_ml/experiments/
  common/
    aggregate_schema.py     # Section-8 column contract; missing -> "not_applicable"
    manifests.py            # SHA-256 feature/split manifest hashing
    checksums.py            # privacy-safe row-membership digests (memory only)
    synthetic.py            # shared synthetic fixtures (no real data)
    README.md
  A1/
    src/a1_state_context.py # state coding table (v024 labels), train-only one-hot,
                            #   optional state x residence behind support gate (D9)
    tests/test_a1_state_context.py
    config.yaml             # seeds, features, split id, output paths
    README.md
    outputs/                # aggregate CSV/JSON + figures only
  A4/
    src/a4_facility_detail.py   # verified m15 -> readable label map (both waves)
    src/a4_history.py           # four-level observed-history builder (roster-first)
    tests/test_a4_facility_detail.py
    tests/test_a4_history.py    # incl. temporal-leakage counterexamples
    config.yaml
    README.md
    outputs/
  A7/
    src/a7_cohort.py        # frozen cohort/split assembly + manifest hashes
    src/a7_models.py        # LR/XGB/CatBoost CLIs with explicit params
    src/a7_tuning.py        # mother-grouped CV inside training only
    src/a7_calibration.py   # disjoint grouped calibration subset; slope/intercept
    src/a7_shap.py          # group aggregation to source constructs + rank stability
    src/a7_robustness.py    # repeated grouped splits, m15!=96, subgroups
    tests/test_a7_*.py
    config.yaml
    README.md
    outputs/                # aggregate_model_comparison.csv, calibration_summary.csv,
                            # shap_feature_family_summary.csv, robustness summaries, figures
```

Ancillary docs: `docs/A1_A4_A7_DISCREPANCY_LOG.md`, per-phase notes under
`version2_ml/experiments/{A1,A4,A7}/README.md`, cleanup log in `docs/CLEANUP_LOG.md`.

## 4. Shared output schema (handoff §8; single source in `common/aggregate_schema.py`)

Exact columns, consistent names/types:
`experiment_id, wave, cohort_rule, evaluation_design, split_id, fold,
source_domain, target_domain, model, feature_tier, baseline_id, n_births,
n_mothers, n_events, prevalence, roc_auc, average_precision, brier_score,
calibration_intercept, calibration_slope, threshold, precision, recall, f1,
weighted_descriptive_flag, uncertainty_status, repo_commit, seed,
software_versions, feature_manifest_hash, split_manifest_hash`.
Rules: absent values = explicit `"not_applicable"` (never zero-filled);
`uncertainty_status` states "not_validated_point_estimate_only";
`repo_commit` = branch SHA; software_versions = recorded exact versions.
Matched comparisons carry identical `split_manifest_hash` plus a
row-membership checksum digest (privacy-safe).

## 5. Phase 1 — A1 (state/area context)

Inputs: frozen C1 cohort + 8-core F1. Steps: (1) build the state/UT coding
table from NFHS-5 value labels (verified 36 codes, D1); one-hot train-only.
(2) Matched ablation on identical rows/folds/model: F1 vs F1+state;
optional F1+state+residence interaction only if the D9 screen passes.
(3) No external area data is available with full provenance → fallback is
categorical state + residence; documented as such. No outcome-derived context.
Tests (handoff §5): unseen state handled; row-order-independent joins;
repeated birth keys rejected; no target-dependent context; missing mapping →
explicit fallback; feature count/order locked; no fit on test rows; matched row
hashes + group-exclusive splits proven.
Outputs: matched ablation CSV (schema), metrics JSON, coverage/support table,
figures, Methods/Results text tied to actual outputs, limitations.

## 6. Phase 2 — A4 (detailed facility + four-level observed history)

Facility: verify m15 value labels in both waves' metadata before mapping
(NFHS-5 labels verified already, D1); `facility_detail` categorical labels;
broad `facility_type` kept for the broad-vs-detailed comparison; m15 != 96
sensitivity. History (roster-first): build timeline from ALL eligible births
(b19 0–59, including home deliveries) before restricting to index births;
strictly earlier calendar month only (same-month twins excluded); bord and bidx
orderings guarded; duplicate keys/missing m17/missing dates → explicit
UNKNOWN + quality flag; four-level variable frozen only after the D2 decision;
add `has_any_observed_earlier_institutional_birth` + quality flag; v201 never used.
Ablations on matched births: F1; +facility detail; +history; +both; broad vs
detailed; m15 != 96 sensitivity (consistent index/history rule per D2).
Leakage counterexample tests: alter future/current/same-month m17 (and m15) →
earlier index history unchanged; one-birth mother; prior home birth; prior
institutional non-CS; prior institutional CS; older unobserved births;
same-month twins; duplicate keys; missing dates; unexpected facility code.
Outputs: mapping/crosswalk table, support tables (disclosure-safe), matched
ablations, sensitivity, manifests, figures, methods/limitations note.

## 7. Phase 3 — A7 (models, calibration, SHAP, robustness)

Single frozen cohort/split (D10). Tuning via `StratifiedGroupKFold` (3 folds)
inside training only; scale_pos_weight from training only. LR/XGB/CatBoost on
identical validation keys with recorded versions/params (CatBoost pending
install, D8). Calibration: model-fit set = train minus grouped calibration
subset; calibrator fitted on the disjoint calibration subset only; threshold
chosen on validation only; test scored once after freeze, labeled
development-test (V1 exposure disclosed). SHAP: tree/linear explainers,
encoded columns aggregated back to source constructs, rank stability across
grouped seeds/folds + predeclared subgroup support; probabilities vs log-odds
treatment recorded. Robustness: repeated grouped splits, with/without m15=96,
subgroups (sector, urban/rural, state) only at adequate support. F0 matched
re-run only per D7 (else NOT RUN).
Acceptance tests (handoff §9): partition disjointness; train-only fitting;
swapped validation/test labels cannot change preprocessing/tuning; manifest
match; identical row count/order for paired comparisons; calibration sees only
calibration labels; threshold from validation only; SHAP aggregation reconciles
dimensionality; probabilities finite in [0,1]; seed reproducibility; null
results retained.

## 8. Cross-cutting rules (applied in every phase)

- Privacy: no row-level data, identifiers, predictions or models committed;
  aggregate outputs with small-cell suppression (<10); `.gitignore` guards added.
- Provenance: repo SHA, wave, cohort rule, split id/seed, n/mothers/events,
  feature sources, model params, threshold source, weight treatment in outputs.
- Per-phase deliverable format per handoff §10 (code paths, run commands, test
  output, cohort/missingness audits, aggregate CSV/JSON, figures, manifests
  with hashes, limitations, Methods/Results text, unresolved questions).
- STOP for owner review after each phase.

## 9. Open questions for owner (blocking items highlighted)

1. **D2 m15=96 history semantics** — blocks A4 freeze (recommendation: include
   as institutional/unknown-sector, sensitivity excludes consistently).
2. D7 F0 matched comparison attempt — yes/no.
3. D8 CatBoost install approval (local venv, recorded version).
4. D9 support screen 200 rows / 30 per class for the state × residence gate.
5. A7 test-set use: confirm single frozen test evaluation labeled
   development-test (V1 exposure) is expected now, or defer test scoring until
   the coordinator approves a confirmatory protocol.
6. Split fractions: confirm 70/15/15 provisional freeze for A1/A4/A7 work.

## 10. Sequencing

A1 isolated → A4 (after D2 answered) → A7 (with verified A1/A4 modules only after
their isolated ablations pass). Each phase ends with a review checkpoint.

## 11. Approval record (2026-10-17)

- Plan approved as written.
- D2: include m15 = 96 prior births as observed institutional history with
  unknown sector; m15 != 96 sensitivity applied consistently.
- D7: attempt key-level matched F0 reconciliation; F0 metrics only if exact row
  matching is proven, otherwise NOT RUN (F1-only comparisons).
- A7 test policy: score the single frozen test partition once after all choices
  freeze; label development-test with V1 exposure disclosed.
- Split: freeze provisional 70/15/15 mother-exclusive split (seed 42) with
  split manifest hash.
- D8: install CatBoost into the project venv, record exact version; failure is
  recorded and CatBoost marked NOT RUN.
- D9: support screen ≥200 training rows and ≥30 rows per outcome class per
  interaction cell; if failed, A1 reports state + residence without interaction.

## 12. Execution completion record (2026-10-17)

All three phases are implemented, executed on the authorized local NFHS-5 data,
and committed on branch `gaurav-v2-A1-A4-A7-r3` (never main):

- Phase 1 (A1): commits `235fca7`, `aed6252`; interaction gate NOT RUN per D9.
- Phase 2 (A4): commits `33a28a3`, `75f6904`, `7891204`.
- Phase 3 (A7): commits `d9f1bb3`, `de98145`, `d668549` (code), `f7d5089`
  (aggregate outputs, `TEST_COMMAND_LOG.md`, discrepancy-log Phase 3 record).
- Verification: 111 unit tests green; determinism reruns byte-identical at
  each phase; the frozen test partition was scored exactly once in A7 and is
  labelled development-test (V1 exposure).
- The approved split (70/15/15 mother-exclusive, seed 42) stays frozen under
  the recorded split manifest hash; the split_id string retains the
  `provisional` suffix for continuity, and any future change of fractions or
  seed requires a new split_id and fresh digests.
- Post-rework open items for the coordinator: none blocking; paper-level
  Methods/Results drafting consumes the Phase 1-3 records as-is.
