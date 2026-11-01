# A1 test and command log (Phase 1)

Date: 2026-10-17 | Branch: `gaurav-v2-A1-A4-A7-r2` (base `318f160`)
Environment: project `.venv`, Python 3.13.11 (versions recorded in
`outputs/a1_run_summary.json`).

## Unit tests (synthetic only, no real data)

| Command | Result |
|---|---|
| `python -m unittest discover -s version2_ml/experiments/common/tests` | Ran 10 tests - OK |
| `python -m unittest discover -s version2_ml/experiments/A1/tests` | Ran 15 tests - OK |
| `python -m unittest discover -s version2_ml/tests -p "test_*.py"` | Ran 43 tests - OK |

## Real-data run (authorized local NFHS-5; aggregate outputs only)

```
python version2_ml/experiments/A1/src/run_a1.py --nfhs5-path "<authorized-dir>/IABR7EFL.DTA"
```

Input file identity (absolute paths never recorded):
`IABR7EFL.DTA`, 2,417,715,435 bytes, SHA-256
`93267b17596929371dc3190d17af114271851f14d1f566678ffceda2db4cae6e`.

Observed stdout summary (abridged):

```
A1 completed. Cohort: 201311 births / 158429 mothers; events 44742.
Split: train 140920 rows / 110900 mothers; validation 30227 rows; test partition NOT RUN.
Interaction gate passed: False (insufficient_support: andaman & nicobar islands | urban; chandigarh | rural; ...
  xgboost F1: ROC-AUC 0.7764, AP 0.4939, Brier 0.1930
  xgboost F2: ROC-AUC 0.8127, AP 0.5690, Brier 0.1761
  logistic_regression F1: ROC-AUC 0.7722, AP 0.4853, Brier 0.1941
  logistic_regression F2: ROC-AUC 0.8068, AP 0.5495, Brier 0.1786
```

The run was repeated after a logging-only change (deprecated explicit L2
penalty argument omitted; library default still L2); metrics, digests and
hashes were identical, confirming seed-42 determinism. All tables and figures
in `outputs/` are from the latest run.

## Re-record under the approved global environment (2026-10-19)

The project `.venv` used for the recording above is no longer usable on this
machine (Application Control blocked the numpy `_philox` DLL); repair is not
permitted, and the global Python is the approved environment. A1 was re-run to
re-record all outputs with honest provenance.

Environment: global Python 3.13.11 (numpy 2.4.6, pandas 3.0.5, scikit-learn
1.9.0, xgboost 3.4.0, matplotlib 3.11.1, pyreadstat 1.3.5, PyYAML 6.0.3).

```
python version2_ml\experiments\A1\src\run_a1.py ^
  --nfhs5-path "<authorized-dir>\IABR7EFL.DTA"
```

Result: exact reproduction — cohort 201,311 births / 158,429 mothers / 44,742
events; split 140,920 / 30,227 / 30,164; interaction gate unchanged; test
partition NOT RUN; all six ablation rows value-identical to the 2026-10-17
recording. `a1_support_table.csv` and `a1_interaction_gate.csv` are
byte-identical; `a1_run_summary.json` and `a1_matched_ablation.csv` differ only
in `repo_commit` (`e6f9c79`), `timestamp_utc` and software versions; figures
are pixel-identical (PNG bytes differ only via matplotlib container metadata).
All tables and figures in `outputs/` are now from this approved-environment
run.
