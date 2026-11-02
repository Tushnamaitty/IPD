# A7 test and verification command log

Environment: `.venv` Python 3.13.11 (pandas 3.0.6, numpy 2.5.4, scikit-learn
1.9.1, xgboost 3.4.1, shap 0.54.1, catboost 1.2.10, pyreadstat 1.3.6,
scipy 1.18.1, matplotlib 3.11.2, PyYAML 6.0.3). Run date: 2026-10-17, branch
`gaurav-v2-A1-A4-A7-r3`, runner commit `d668549`.

## 1. Compile and unit tests (all green before the real run)

```powershell
& .venv\Scripts\python.exe -m py_compile version2_ml\experiments\A7\src\a7_calibration.py `
    version2_ml\experiments\A7\src\a7_candidates.py version2_ml\experiments\A7\src\a7_shap.py `
    version2_ml\experiments\A7\src\run_a7.py version2_ml\experiments\common\data_access.py `
    version2_ml\experiments\A7\tests\test_a7_modules.py
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\tests
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\common\tests
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\A1\tests
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\A4\tests
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\A7\tests
```

Result: 111 tests OK (43 base, 10 common, 15 A1, 30 A4, 13 A7). The A7 suite is
synthetic-only and covers calibrator monotonicity/range, both-class guards,
intercept/slope recovery, thresholds, carve determinism and group exclusivity,
tuning determinism without evaluation-label parameters, partition guards,
preprocessing invariance to held-out labels, SHAP family reconciliation and
aggregation, rank stability and PSU overlap summarisation.

## 2. Smoke run (skips SHAP/robustness; validates all data stages)

```powershell
& .venv\Scripts\python.exe version2_ml\experiments\A7\src\run_a7.py `
    --nfhs5-path "<local>\IABR7EFL.DTA" `
    --f0-parquet "<local>\df_model_v1.parquet" `
    --skip-shap --skip-robustness
```

Outcome: tune/candidates/calibration/validation/F0 stages completed; F0
reconciliation exact (201,311/201,311 keys, zero mismatches); development test
scored once and cached. Two fixes were committed before the final run
(`de98145` history feasibility key, `d668549` primary threshold config key).

## 3. Full real-data run (caches reused; SHAP + robustness enabled)

```powershell
& .venv\Scripts\python.exe version2_ml\experiments\A7\src\run_a7.py `
    --nfhs5-path "<local>\IABR7EFL.DTA" `
    --f0-parquet "<local>\df_model_v1.parquet"
```

Outcome: exit 0. Tuning loaded from cache; development test NOT rescored
(`cache_no_rescore`, original scoring timestamp preserved).

## 4. Output verification

- Schema: `aggregate_model_comparison.csv` has the 31 locked columns and 26
  rows (18 validation, 6 development-test, 2 reconciled F0).
- Leak scan over committed outputs for local paths and identifier names:
  clean (only matches were the `missing_caseid` count field and the branch
  name in provenance).
- Suppression: all positive counts < 10 rendered as `<10 (suppressed)`; no
  row-level or key-level data exported.
- Determinism: SHA-256 of all 10 committed CSV/PNG outputs byte-identical
  across the full run and an immediate rerun.
