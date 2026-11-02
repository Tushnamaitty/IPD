# A8 test and run command log

Environment: `.venv` Python 3.13.7 (numpy 2.5.4, pandas 3.0.6, scikit-learn
1.9.1, xgboost 3.4.1, shap 0.54.1, scipy 1.18.1, matplotlib 3.11.2, PyYAML
6.0.3, pyreadstat 1.3.7), Windows 11. Branch `v2-final-eval` created from
`merged-all-experiments` at `eff1c1b`. Run date 2026-11-02.

## 1. Baseline suites (before any A8 code)

```powershell
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\tests -p "test_*.py"
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\common\tests -p "test_*.py"
# and the same for experiments\A1\tests ... experiments\A7\tests
```

Result: 200 tests OK (43 base, 23 common, 15 A1, 35 A2, 15 A3, 30 A4, 13 A5,
10 A6, 16 A7).

## 2. A8 synthetic suite

```powershell
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\A8\tests -p "test_*.py" -v
```

Result: 22 tests OK (no NFHS records). The runner re-runs this suite before
touching data, and it passed again at run time.

## 3. Recorded real-data run

```powershell
$env:NFHS5_PATH = "<local>\IABR7EFL.DTA"
& .venv\Scripts\python.exe version2_ml\experiments\A8\src\run_a8.py
```

Exit 0 after 354 s; `outputs/MANIFEST.json` was created 2026-11-02T04:59:10Z.
All QA gates passed:
- input sha256;
- cohort counts and the 6,276 history reproduction;
- the seed-42 frozen split hash, the A1 train/validation digests and the A7
  fit/calibration digests;
- H0 XGBoost seed 42 reproducing A7 (ROC-AUC 0.8298, AP 0.6113);
- seven real-data label-mutation leakage checks.

The predeclared decision rule was written into `config.yaml` and `README.md`
before this run.

### Code provenance

The run used the A8 code as untracked files on top of `eff1c1b`, because one
commit after the run was requested. `MANIFEST.json` therefore records git
commit `eff1c1b` (the base). The A8 code that produced the outputs is pinned
by these sha256 values, taken over LF-normalized bytes immediately after the
run. They equal the blobs committed together with the outputs.

| sha256 | file |
| --- | --- |
| 0a7996af94df1cc487caa5b43b17c6c69eb804dd158e7c57e8fa03eae48da38c | config.yaml |
| 06878de94e3fb19dbd93ba0afb9f1a02c11113a0d1f3ef92e813491b1f04af1b | src/a8_arms.py |
| e2f6f91e72bc9a1cd47d0f86da8c2592071ad4e64bc4950fa5c07183bf298e43 | src/a8_decision.py |
| 7b244f1cc9ee7ba189832aa7b1be355a5175fa17865f6c8df47eae678734db07 | src/a8_outputs.py |
| 25b48b1a1372cdd6349a6b3a4f48b838c09e9efaf8b0187a55466fa98a257b4d | src/a8_pipeline.py |
| 1582bbb6d2cfe2ad90c3cc8f18d9d469be6da5ee84b8edd0a7ec9869d51dfb6a | src/run_a8.py |
| bae07061b53cfb6cdb89cb30453f8344803dbf00e5c260b3cdf5d6a588c6393e | tests/test_a8_modules.py |
