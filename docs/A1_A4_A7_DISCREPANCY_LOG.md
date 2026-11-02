# A1/A4/A7 discrepancy log — handoff vs repository vs local NFHS-5 data

Date: 2026-10-17
Branch: `gaurav-v2-A1-A4-A7-r2` (base `318f160`)
Data authorization: NFHS-5 local use authorized by project owner (2026-10-17).
All real-data values below are **aggregate counts** with positive cells <10
suppressed. No row-level data, identifiers or predictions appear here.
Local file: `IABR7EFL.DTA` (outside the repository; never committed).

Status legend: RESOLVED (evidence closes the item), OPEN (awaiting owner decision),
PLANNED (resolved procedurally by the rework plan).

---

## D1 — Cohort counts in the handoff: verified against local data — RESOLVED

**Handoff claims check:** 201,311 births / 158,429 mothers (NFHS-5); b19<60 rule;
1,634 births at CMC=60; 517 births m15=96 all non-cesarean; 82,607 first live
births; 6,276 index births with observed prior institutional CS.

**Independent aggregate verification (read-only, 2026-10-17):**

| Definition | Count |
|---|---:|
| m15 ∈ {20–27, 30–33, 96} & b19 ∈ 0–59 & m17 ∈ {0,1} | **201,311** |
| Same, mothers (distinct caseid) | **158,429** |
| Same, but CMC (v008−b3) ∈ 0–59 instead of b19 | **199,677** |
| Institutional births at CMC = 60 | **1,634** (0 above 60) |
| Institutional rows with b19 missing | 0 |
| m15 = 96 | **517**, all m17 = 0 |
| bord == 1 | **82,607** |
| Duplicate (caseid, bidx); missing caseid/bidx/b3/v024 in cohort | 0 |
| m17 distribution | 0: 156,569; 1: 44,742 |
| Observed prior institutional CS at index births (own aggregate reconstruction) | **6,276** |

Also verified from DTA value labels: m15 = {21,22,23,24,25,26,27, 31,32,33, 96}
present in data; parent codes 20 and 30 absent (labels exist, no rows), matching
the handoff. v024 has 36 distinct states/UTs, 0 missing; labels match the
official 36 state/UT names.

**Root cause of the prior attempt's 200,794 / 6,267 mismatch (now explained):**
the first archived pilot run used an index cohort excluding m15 = 96 (517 rows)
= 200,794, and reported 6,267 prior-CS index births; its own second run
(`pilot_cohort201311`) corrected to 201,311 and 6,276, which equals this
independent count. The earlier numbers are artifacts of inconsistent 96
handling, not data facts.

**Not yet run:** the provisional split counts 140,920 / 30,227 / 30,164.
Requires the frozen split algorithm and will be audited in A7 with a split
manifest hash. Label: NOT RUN until then.

**Decision:** adopt the handoff rule (b19 ∈ 0–59 primary, CMC 0–59 as labeled
sensitivity) since both counts reproduce exactly.

**Update (Phase 1 A1, 2026-10-17):** the provisional mother-grouped 70/15/15
split (seed 42) was executed on the frozen cohort and audited: train 140,920 /
validation 30,227 / test 30,164 rows; group-exclusive assertions passed; the
split manifest hash is recorded in
`version2_ml/experiments/A1/outputs/a1_run_summary.json` (test partition not
scored). Freezing the split record remains an A7 task.

## D2 — m15 = 96 prior births in the four-level history — RESOLVED (2026-10-17)

**Owner decision:** include m15 = 96 prior births in the history timeline as
observed institutional births with unknown sector (option a, recommended).
The m15 != 96 sensitivity applies consistently to index and history rows.
Freeze this rule in A4 code and outputs.

**Issue:** the broad cohort includes m15 = 96 (517 births, handoff §3), but the
prior A4 module silently excluded 96 from the history timeline while the
base-branch phase-5 audit included it. Because all 96 births have m17 = 0,
the OBSERVED_PREVIOUS_INSTITUTIONAL_CS category is unaffected either way.

**New aggregate evidence:** 94 index births have a strictly earlier 96-facility
birth (m17 = 0) and no other strictly earlier 20–27/30–33 birth in the survey
window. Under "exclude 96 from history" they are categorized
HIGHER_ORDER_..._UNOBSERVED; under "include 96 as institutional/unknown-sector"
they are categorized HIGHER_ORDER_NO_OBSERVED_PREVIOUS_INSTITUTIONAL_CS.
No 96 birth has m17 = 1, and no index birth with a prior 96 CS exists.

**Options considered:** (a) include 96 as institutional with unknown sector,
apply the m15 != 96 sensitivity consistently to index and history:
preserves symmetry with the index-cohort definition; (b) exclude 96 from
history entirely (prior behavior): creates an asymmetric cohort/history
definition; (c) decide after A4 writes the full support table.

**Update (Phase 2 A4, 2026-10-17):** implemented as option (a) in
`a4_history.py`; the sensitivity excludes 96 from index and history
consistently via an inherited mother partition. Support (main rule): first
birth 82,607; higher-order no observed prior CS 35,195; higher-order history
unobserved 77,219; observed previous institutional CS 6,276; unknown 14.
Sensitivity levels shift by the removed 96 rows only.

## D3 — Prior A4 module did not implement the full handoff contract — RESOLVED

Prior `a4_observed_history.py` omitted the required
`has_any_observed_earlier_institutional_birth` feature, duplicated facility
detail mapping inconsistently (module excluded 96; runner mapped it), and had
no temporal-leakage counterexample tests. All prior modules are archived; the
rework plan specifies the full contract.

**Update (Phase 2 A4, 2026-10-17):** the full contract is implemented in
`version2_ml/experiments/A4/` (four levels + `has_any...` + quality flag,
verbatim facility crosswalk, temporal-leakage and corner-case tests: 30 tests
pass) and the real-data run reproduced both Phase-0 feasibility counts exactly
(82,607 first recorded live births; 6,276 observed previous institutional CS).

## D4 — Shared module mutation by prior attempt — RESOLVED (reverted)

Prior attempt changed `version2_ml/model_framework.py` XGBoost defaults
(100/3 → 200/6 + subsample/colsample) inside shared code. Reverted by cleanup;
A7 will pass explicit, recorded per-model hyperparameters from its own config.

## D5 — Data path convention mismatch — RESOLVED (Phase 3, 2026-10-17)

Base-branch audit scripts expect `data/raw/...` under the repository root; the
authorized data lives outside the repository. No absolute paths may be
committed. A1/A4/A7 runners will take explicit CLI arguments
(`--nfhs5-path`, `--data-root`); base scripts are left untouched.

**Update (Phase 1, 2026-10-17):** implemented for A1 via the explicit
`--nfhs5-path` CLI in `run_a1.py`; a pattern scan of committed outputs and code
found no absolute/local paths (input identity recorded as file name, size and
SHA-256 only).

**Update (Phase 3, 2026-10-17):** A4 and A7 use the same explicit CLI
convention (`--nfhs5-path`, optional `--f0-parquet`), and pattern scans of all
committed A4/A7 outputs were clean of local paths; base scripts were not
modified.

## D6 — Prior "results" were not matched-comparison evidence — RESOLVED (discarded)

First/second archived pilot outputs report unverified point metrics without
row-membership checksums, calibration, SHAP, tuning or leakage counterexample
tests. They are preserved only in the archive refs and are non-evidence. The
rework regenerates everything with matched-row proofs.

## D7 — F0 baseline artifact provenance — RESOLVED (Phase 3, 2026-10-17)

`df_model_v1.parquet` exists locally; the old `artifacts/` folder is absent.
The handoff permits matched F0 comparisons "only when original processed data
and same cohort are available", with the historical split labeled separately.
**Owner decision (2026-10-17):** attempt key-level cohort reconciliation of
`df_model_v1.parquet` against the frozen cohort in A7; report F0 matched
metrics only if exact row matching is demonstrated, otherwise report F0
comparisons as NOT RUN and use F1-only comparisons.

**Update (Phase 3 A7, 2026-10-17):** reconciliation was exact — 201,311/201,311
keys matched with zero cohort-only, parquet-only, duplicate, outcome or
facility mismatches (`a7_reconciliation.csv`). The 36-feature historical
XGBoost (A7 reference config, explicitly not an exact V1 hyperparameter
reproduction) was fitted on the identical A7 fit pool and scored on the
identical validation/test keys, with results labelled
`f0_reconciled_historical36_a7_reference_params`. Validation ROC-AUC 0.8021 /
AP 0.5349; the single development-test scoring gave ROC-AUC 0.8035 / AP 0.5313,
consistent with the disclosed V1 nb04 figures (test AUC 0.8035 / AP 0.5333).

## D8 — Missing dependencies for A7 — RESOLVED (Phase 3, 2026-10-17)

`catboost` is not installed (and absent from `requirements.txt`);
`statsmodels` is absent. Proposal: install `catboost` into the project venv
and record the exact version in every output; implement calibration
intercept/slope with an explicitly documented single-variable logistic fit
(scikit-learn) rather than adding statsmodels. CatBoost remains "NOT RUN" if
installation fails and the failure is recorded.

**Update (Phase 3 A7, 2026-10-17):** `catboost` 1.2.10 was installed into the
project venv and is recorded in every A7 output via `software_versions`;
calibration intercept/slope use the documented single-variable logistic fit
(`LogisticRegression(C=inf, solver=lbfgs)` on log-odds, sklearn 1.9.1);
`statsmodels` was not needed.

## D9 — state × residence support gate (A1) — RESOLVED (approved 2026-10-17)

The handoff allows the interaction "only if groups do not become tiny and
unstable" but defines no threshold. Proposal: adopt a preliminary feasibility
screen of ≥200 training rows and ≥30 rows per outcome class per interaction
cell (consistent with the base phase-5 audit screen), applied on training rows
only, with the final gate recorded in the feature manifest. If the screen
fails, A1 reports categorical state and residence without interaction and says
so.

**Update (Phase 1 A1, 2026-10-17):** the screen was applied to real training
rows and the interaction gate failed (Chandigarh: 134 training rows; several
state × residence cells, e.g. Goa rural/urban, miss the class-count minimum).
The interaction is reported NOT RUN; the reported fallback is F1 vs F1+state.
Full cell table: `version2_ml/experiments/A1/outputs/a1_interaction_gate.csv`.

## D10 — Split and matched-comparison contract — RESOLVED (executed Phase 3, 2026-10-17)

- Frozen cohort: m15 ∈ {20–27, 30–33, 96}, b19 ∈ 0–59, m17 ∈ {0,1}, unique
  (caseid, bidx) asserted with hard error.
- Mother-exclusive provisional 70/15/15 via `split_contracts.grouped_partition`
  (seed recorded); fractions remain provisional per `splits.yaml` until frozen
  in A7 config.
- Matched comparisons proven by privacy-safe row-membership checksums
  (SHA-256 digests over sorted birth keys, digests only, computed in memory).
- Grouped CV for tuning: `StratifiedGroupKFold` (config: 3 folds) entirely
  inside training; calibration on a disjoint grouped subset carved from the
  training pool; threshold selection on validation only; test touched once
  after freeze and labeled development-test (V1 exposure disclosed).

**Update (Phase 1 A1, 2026-10-17):** the provisional split, matched-arm
membership digests, manifest hashes, model parameters and software versions are
recorded in `version2_ml/experiments/A1/outputs/a1_run_summary.json`; the test
partition remains unscored. Freezing the split in the A7 config is pending.

**Update (Phase 2 A4, 2026-10-17):** A4 re-derived the split independently and
asserts equality with the frozen A1 split hash and train/validation digests;
the m15 != 96 sensitivity inherits the mother-level partition (audited). Test
partition still unscored.

**Update (Phase 3 A7, 2026-10-17):** A7 executed the full contract at commit
`d668549`: tuning and calibrator fitting strictly inside the training pool,
thresholds selected on validation only, and the test partition scored exactly
once behind the `outputs/private/a7_test_scored.json` guard and labelled
`test_development_only` in the aggregate outputs. The split remains labeled
provisional; A7 is the last pre-freeze consumer, and any future change of
fractions or seed must define a new `split_id` and fresh digests rather than
reusing these matched-comparison constants.

---

## Phase 2 (A4) execution record — 2026-10-17

- Resolved in code: D2 (option a), D3 (full contract), D5 (same explicit
  `--nfhs5-path` CLI; committed outputs scanned clean of local paths).
- Facility detail: labels are verbatim NFHS-5 `m15` value metadata read
  2026-10-17 (code 96 = "other"; codes 20/30 are sector-only labels). NFHS-4
  stays pending local `IABR74FL.DTA` label verification and is recorded as such
  in `a4_facility_crosswalk.csv`.
- Timeline window: consistent with frozen C1 (`b19` 0–59); `b3` is used only
  for strict ordering. Data-quality note: 1,634 index births carry a `b3` date
  earlier than the nominal recode window (dead-child age-vs-date reporting);
  retained under the frozen rule and reported as an aggregate count.
- Ambiguity semantics: 14 index births have a prior institutional birth with a
  coded-missing outcome (8/9/blank) and no observed CS; they receive explicit
  `unknown_history` plus `history_quality_flag =
  ambiguous_prior_institutional_outcome`, never a negative. Zero
  missing-date or bord-inconsistency records in the frozen cohort.
- Matched ablations (validation, point estimates, seed 42; commit `33a28a3`):
  F1 rows reproduce A1 exactly (XGB ROC-AUC 0.7764 / AP 0.4939 / Brier 0.1930;
  LR 0.7722 / 0.4853 / 0.1941). Additions: detailed facility XGB +0.0150 AUC;
  observed history XGB +0.0426 AUC, AP +0.1107; both XGB +0.0538 AUC,
  AP +0.1180. The m15 != 96 sensitivity leaves conclusions unchanged.
- Outputs: `version2_ml/experiments/A4/outputs/` (aggregate-only; cells <10
  suppressed).

---

## Phase 3 (A7) execution record — 2026-10-17

Contract executed at commit `d668549` (code commits `d9f1bb3`, `de98145`,
`d668549`; branch `gaurav-v2-A1-A4-A7-r3`). Runner: `run_a7.py` with stages
tuning (cached), candidates, calibration, single test scoring (cache guard),
F0 reconciliation, SHAP, robustness. All supervision (tuning, calibrators,
threshold selection) stayed inside the training pool; validation labels
selected calibration variants only; the development test was scored once.

- Test/verification totals: 111 unit tests green (43 base, 10 common, 15 A1,
  30 A4, 13 A7; synthetic-only), `py_compile` clean; determinism rerun
  produced byte-identical SHA-256 for all 10 CSV/PNG outputs.
- Shared-row contract: split hash equals the frozen value; train/validation
  membership digests equal the frozen A1 digests; fit pool 112,739 rows
  (digest `1f30baed...`), calibration carve 28,181 rows (digest `6e50f6f0...`,
  seed 4208, mother-exclusive), validation 30,227, test 30,164 (digest
  `c55ccbc9...`).
- Tuning (grouped 3-fold, average precision, fit pool only): XGB selected
  depth 4 (200 trees, lr 0.05); CatBoost selected base 300/6/0.05; LR C=1.0.
- Validation (selected calibration, point estimates): CatBoost F1 0.7775 /
  0.4970 / 0.1418 (Platt); CatBoost detail+history 0.8306 / 0.6131 / 0.1254;
  XGB F1 0.7768 / 0.4957 / 0.1420; XGB detail+history 0.8298 / 0.6113 /
  0.1257; LR F1 0.7719 / 0.4710 / 0.1433 (isotonic); LR detail+history 0.8273 /
  0.6065 / 0.1263. Platt selected for 5/6 candidates; post-calibration slopes
  0.97-1.00 and intercepts near 0.
- Development-test single scoring (selected method): CatBoost detail+history
  0.8332 / 0.6089; XGB detail+history 0.8323 / 0.6077; LR detail+history
  0.8289 / 0.6012; F1 arms 0.772-0.779 ROC-AUC. F0 historical-36 reference:
  0.8035 / 0.5313 (validation 0.8021 / 0.5349) — matched F0 metrics reported
  after exact key-level reconciliation (D7).
- Calibration thresholds (validation, exploratory): F1-optimal 0.26-0.30 with
  F1 0.50-0.58; recall >= 0.80 attainable only at thresholds <= 0.19.
- SHAP (XGB detail+history, margin space, seeded 5,000-row validation sample):
  top families facility_detail, facility_type, history_four_level,
  birth_order, maternal age, wealth; rank stability across grouped fold models
  min Spearman 0.937; urban 0.958; rural 1.000.
- Robustness: three repeated grouped splits (seeds 7/2026/99) AP 0.487-0.496
  (F1) and 0.603-0.616 (detail+history); m15 != 96 exclusion 0.4945; most-
  recent-birth subset 0.5091; supported subgroups public 0.454 vs private
  0.699 AP (heterogeneity flagged); 27 supported states pooled AP 0.6132.
- PSU overlap (v021): 96.8% of validation rows fall in sampled PSUs also seen
  in training — an expected consequence of the mother-level (not cluster-level)
  split; recorded as a design limitation, no design-based intervals are
  claimed anywhere.
- Disclosure: all positive counts < 10 suppressed; outputs scanned clean of
  local paths and row-level identifiers; the development test remains
  non-independent (V1 exposure).
- Outputs: `version2_ml/experiments/A7/outputs/` (aggregate-only).
