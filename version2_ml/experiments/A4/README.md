# A4 - Observed delivery history and facility detail

Matched ablations on the frozen C1 NFHS-5 cohort testing whether detailed
facility type (`m15` mapped to verbatim audited labels), observed earlier
delivery history, or both add incremental value over the eight-core F1
baseline. The provisional mother-grouped split, split hash and membership
digests are asserted against the frozen A1 values, so all arms run on identical
rows in identical order.

## Semantics (frozen)

Four-level `history_four_level` for index births, built only from strictly
earlier calendar months of the same mother's survey-window timeline:

- `first_recorded_live_birth` - `bord == 1`.
- `observed_previous_institutional_cs` - at least one strictly earlier
  institutional birth with `m17 == 1`.
- `higher_order_no_observed_previous_institutional_cs` - strictly earlier
  institutional birth(s) observed, none `m17 == 1` (observed negative within
  recall, not a lifetime negation).
- `higher_order_earlier_institutional_history_unobserved` - no strictly
  earlier institutional record (includes earlier noninstitutional and
  out-of-recall births).

Missing/ambiguous records produce `unknown_history` plus
`history_quality_flag` tokens; unknown is never converted to negative.
Same-month twins never count as previous deliveries. Home deliveries stay in
the in-memory timeline as observed noninstitutional records. The timeline is
never exported.

## Files

- `src/a4_facility_detail.py` - verbatim NFHS-5 `m15` label mapping and
  crosswalk rows (NFHS-4 pending local verification).
- `src/a4_history.py` - timeline builder and four-level derivation with strict
  coding guards and quality flags.
- `src/run_a4.py` - CLI runner: matched ablations (F1, F1_detail,
  F1_detail_replace, F1_history, F1_detail_history) for XGBoost and logistic
  regression on validation, plus the `m15 != 96` sensitivity.
- `tests/` - synthetic counterexamples: single birth, prior home birth, prior
  institutional non-CS/CS, higher-order unobserved, same-month twins,
  duplicate keys, missing dates, missing prior outcome, unexpected facility
  code, temporal leakage (future/current/same-month edits), m15=96 rules.
- `config.yaml` - frozen cohort, split, arms, models and feasibility
  expectations.

## Run

```
.venv\Scripts\python.exe version2_ml\experiments\A4\src\run_a4.py ^
  --nfhs5-path "<authorized-dir>\IABR7EFL.DTA"
```

Tests:

```
.venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\A4\tests
```

## Outputs (aggregate-only, <10 suppressed)

`a4_matched_ablations.csv`, `a4_sensitivity_m15_96.csv`,
`a4_facility_crosswalk.csv`, `a4_history_support.csv`, `a4_run_summary.json`,
`a4_methods_results.md`, `fig_a4_roc_pr.png`,
`fig_a4_roc_pr_sensitivity.png`, `fig_a4_history_distribution.png`.

## Gates addressed

- Strictly earlier observed deliveries, same-month twins excluded.
- Four category meanings explicitly distinct; unknown never negative.
- `m15` detail mapping with verification source; `m15 != 96` sensitivity.
- Feasibility reproduction of the Phase-0 audit (82,607 first births; 6,276
  observed previous institutional CS) with hard failure on mismatch.
- Matched rows proven by frozen split hash and membership digests.
