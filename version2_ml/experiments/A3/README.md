# A3 — State × sector grouped out-of-fold target encoding

Status: **EXPLORATORY development** (feature tier F4). Not confirmatory; no confidence intervals.

Question: does a smoothed, leakage-controlled NFHS-5 training-label encoding of state × sector add
signal beyond ordinary state and sector indicators?

## Layout

| File | Purpose |
|---|---|
| `A3_target_encoder.py` | `GroupedOOFTargetEncoder` (OOF for outer train, full-train table for validation/test), `RateTable`, `select_alpha`, `encode_partitions` |
| `config.yaml` | Predeclared inner CV, alpha grid, support threshold, cohorts, models |
| `run.py` | Runner: synthetic tests → NFHS-5 → shared split → inner folds → alpha selection → encoding → 3-arm ablation |
| `tests/test_a3_leakage.py` | Synthetic adversarial tests (mother isolation, singleton, label mutation, unseen fallback, row order, …) |
| `tests/test_a3_runner_contract.py` | The three mandatory arms are nested exactly as specified |
| `outputs/aggregate/` | Disclosure-checked aggregate outputs |

## Run (from repo root)

```
python -m unittest discover -s version2_ml/experiments/A3/tests -p "test_*.py" -v
python version2_ml/experiments/A3/run.py                  # validation partition only
python version2_ml/experiments/A3/run.py --evaluate-test  # also scores the provisional test partition
```

Requires the authorized local `data/raw/IABR7EFL.DTA` (gitignored); reuses the restricted cache in
`data/processed/v2_experiments/`. The runner re-runs the synthetic adversarial suite first and stops if any test fails.

## Design

- **Outer split:** the shared mother-grouped 70/15/15 split (`split_contracts.grouped_partition`, seed 42),
  identical to A2. Arms (i)/(ii) must reproduce A2 B0/R1 exactly, or the run fails.
- **Inner cross-fitting:** `StratifiedGroupKFold(3, shuffle=True, random_state=42)` on outer-train mothers.
  Each fold is encoded by tables fitted on the other two; zero mother overlap is asserted per fold.
  Validation/test rows use one table fitted on 100% of outer train.
- **Smoothing:** TE = (y + α·μ)/(n + α), μ = overall rate of the fitting rows; support threshold 30 births.
  Fallback: state × sector cell → state → sector → μ (flagged with `te_cell_fallback`).
- **α:** chosen from {0, 1, 5, 20, 50, 100, 300} by OOF log-loss of the state × sector encoding on outer-train
  rows only (ties → larger α). α = 5 and 100 are exploratory sensitivities.
- **Arms:** (i) eight common-core features; (ii) (i) + NFHS-5 state indicators (`facility_type` in the core is
  the ordinary sector indicator); (iii) (ii) + OOF state × sector TE. Exploratory: (ii) + all TEs; (i) + TE only.
  Fixed `make_estimator` logistic regression and XGBoost across arms.
- **Unsupported splits** raise `NotEstimable` (`NOT ESTIMABLE`) instead of relaxing mother grouping.
