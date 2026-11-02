# A9: final evaluation of V2 with confidence intervals

**Question.** On cluster (PSU) grouped splits with a predeclared analysis,
does V2 (`FINAL_TIER = F1_detail_history_state`, from A8) beat the V1
36-feature specification, and by how much, with 95% CIs?

The full analysis plan is in [PROTOCOL.md](PROTOCOL.md), committed before any
scoring. Protocol commit: `0ce1d48` (r2; r1 `94a6506`).

## Design in one paragraph

- **Splits:** five new PSU (`v021`) grouped 70/15/15 splits (seeds 11, 22,
  33, 44, 55), stratified on the PSU outcome rate. Zero PSU and zero mother
  overlap is asserted.
- **Models:** four fixed-recipe models, each fitted on the fit pool and
  Platt-calibrated on a PSU-grouped 20% carve:
  - V2: XGBoost, `FINAL_TIER`;
  - V2_LR: logistic regression, `FINAL_TIER`;
  - F1_core: XGBoost, 8 core features;
  - V1_spec: XGBoost, the V1 36-feature parquet, reconciled key-for-key.
- **Thresholds:** selected on validation.
- **Testing:** each test partition is scored once behind a guard file.
- **Uncertainty:** a PSU cluster bootstrap (B = 1,000, seed 2027), paired
  across models. Pooled values are the mean across seeds, with CIs from the
  across-seed mean of replicates.
- **Statements:** three, with fixed wording.

## Run

```powershell
$env:NFHS5_PATH = "C:\path\to\IABR7EFL.DTA"       # data/processed/df_model_v1.parquet must exist for V1_spec
& .venv\Scripts\python.exe version2_ml\experiments\A9\src\run_a9.py
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\A9\tests -p "test_*.py" -v
```

A re-run reuses the completed guard files in `outputs/private/` (git-ignored,
aggregates only) and never re-scores a test partition.

## Outputs (`outputs/`)

- `a9_test_metrics.csv`, `a9_paired_deltas.csv`, `a9_pooled_summary.csv`.
- `a9_split_audit.json`, `a9_calibration_bins.csv`.
- `a9_weighted_sensitivity.csv` (descriptive only).
- `a9_statements.json` and `a9_methods_results.md` (statements and paper
  table).
- `fig_a9_roc_pr.png`, `fig_a9_calibration.png`, `fig_a9_forest.png`.
- `MANIFEST.json`.

Every CSV/JSON output carries the protocol commit. See `TEST_COMMAND_LOG.md`
for the run history, which includes a crash at the manifest step that was
fixed without re-scoring.
