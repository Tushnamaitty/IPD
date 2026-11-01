# A1 - State/area context (Phase 1, approved rework plan)

Question (handoff section 5): do location-related features add reliable
incremental predictive value to the retrospective eight-core F1 baseline on
matched NFHS-5 births? Prediction, not ecological causation.

## Status

- Implemented and covered by synthetic acceptance tests; real-data run is
  local-only, authorized (2026-10-17), aggregate outputs only.
- No external area descriptor with validated year/geography/coverage
  provenance is available, so the handoff fallback applies: categorical state
  and residence. Residence is already inside F1, so the "F1 + state and
  residence" fallback equals F1 + `state_ut`.
- The test partition is NOT scored here; it is deferred to the single frozen
  A7 evaluation (rework plan section 7). Evaluation here uses the development
  validation partition only.
- The state/UT coding table is verbatim from the authorized NFHS-5 DTA value
  metadata (36 units; code 26 absent because Dadra & Nagar Haveli and
  Daman & Diu are merged at code 25). State codes are never numeric ordinals.

## Input contract

`--nfhs5-path` must point to the authorized local NFHS-5 birth recode (a
`.DTA` file outside the repository). Required raw columns:
`caseid, bidx, b3, b19, v008, v011, m15, m17, v024, v025, v130, v133, v190,
bord, b0`. The frozen cohort C1 (m15 in {20-27, 30-33, 96}, b19 in 0-59,
m17 in {0, 1}, unique (caseid, bidx)) must reproduce 201,311 births /
158,429 mothers or the runner hard-fails.

## Run

```
python version2_ml/experiments/A1/src/run_a1.py --nfhs5-path "<authorized-dir>/IABR7EFL.DTA"
python -m unittest discover -s version2_ml/experiments/A1/tests -v
```

## Outputs (`outputs/`, aggregate-only, committed)

- `a1_matched_ablation.csv` - Section-8 schema rows (F1 / F2 per model, plus
  explicit `not_run_support_screen_failed` rows when the D9 interaction gate
  fails).
- `a1_support_table.csv` - per-state training/validation support with counts
  below 10 suppressed.
- `a1_interaction_gate.csv` - D9 training-row support screen per
  state x residence cell.
- `a1_run_summary.json` - provenance: commit, parameter record, split audit,
  manifest hashes, membership digests, software versions, warnings,
  limitations. No absolute paths and no row-level values.
- `fig_a1_roc_pr.png`, `fig_a1_calibration.png` - aggregate point-estimate
  figures.
- `a1_methods_results.md` - Methods/Results text generated from the actual run.

## Rules honored

Matched arms use identical rows/order (row-membership digests asserted);
one-hot state encoding is fitted on training rows only
(`handle_unknown="ignore"`); D9 gate (>=200 training rows and >=30 rows per
class per cell) runs on training rows only; threshold is predeclared at 0.50
(not optimized); point estimates only, no intervals or significance claims;
small cells (<10) suppressed; no identifiers or predictions written.

## Open items (owner review)

- Interpretation of the F2 uplift before A4 begins: state identity may partly
  encode local response prevalence; wording stays predictive-only and
  unseen-state validation is A6 scope.
- Split fractions remain provisional until the A7 freeze.
- This phase deliverable is uncommitted; commit/push awaits owner approval.
- Verification details (exact commands, test counts, run stdout, input
  SHA-256): `TEST_COMMAND_LOG.md`.
