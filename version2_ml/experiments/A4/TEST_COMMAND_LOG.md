# A4 test and command log

Date: 2026-10-17 | Branch: `gaurav-v2-A1-A4-A7-r3`
Environment: `.venv` Python 3.13.11 (pandas 3.0.6, numpy 2.5.4, scikit-learn
1.9.1, xgboost 3.4.1, matplotlib 3.11.2, pyreadstat 1.3.6, PyYAML 6.0.3).

## Unit tests (synthetic only, no real data read)

```
python -m unittest discover -s version2_ml\experiments\A4\tests      -> Ran 30 tests ... OK
python -m unittest discover -s version2_ml\experiments\A1\tests      -> Ran 15 tests ... OK
python -m unittest discover -s version2_ml\experiments\common\tests  -> Ran 10 tests ... OK
python -m unittest discover -s version2_ml\tests                     -> Ran 43 tests ... OK
```

A4 cases: single-birth mother; prior home birth; prior institutional
non-CS/CS; higher-order with no earlier records; same-month twins; duplicate
keys (hard error); missing dates; missing/coded-missing prior outcome;
unexpected facility code (hard error); temporal leakage via future,
current-birth and same-month label edits; m15=96 include/exclude rules;
bord==1 inconsistency flag; out-of-window prior ignored; index-missing-from-
roster hard error; support-count masking; wave guards.

## Real-data run (authorized local NFHS-5 file; aggregate-only outputs)

```
.venv\Scripts\python.exe version2_ml\experiments\A4\src\run_a4.py ^
  --nfhs5-path "<authorized-dir>\IABR7EFL.DTA"
```

Recorded results (commit 33a28a3):
- Cohort 201,311 births / 158,429 mothers / 44,742 events (frozen C1).
- Feasibility reproduced: first recorded live births 82,607; observed
  previous institutional CS 6,276 (both match the Phase-0 audit).
- Split hash equals the frozen A1 split hash; train/validation membership
  digests equal the frozen A1 digests; hard errors otherwise.
- F1 rows reproduce A1: XGBoost ROC-AUC 0.7764 / AP 0.4939 / Brier 0.1930;
  logistic regression 0.7722 / 0.4853 / 0.1941.
- Best arms: F1_detail_history (XGBoost 0.8302 / 0.6119 / 0.1685;
  logistic regression 0.8273 / 0.6070 / 0.1697).

Determinism: a second full run produced identical metrics and byte-identical
CSV/PNG outputs (SHA-256 compared); only timestamp fields in the JSON/Markdown
differ.

## Re-record under the approved global environment (2026-10-19)

The project `.venv` used for the recording above is no longer usable on this
machine (Application Control blocked the numpy `_philox` DLL); repair is not
permitted, and the global Python is the approved environment. A4 was re-run to
re-record all outputs with honest provenance.

Environment: global Python 3.13.11 (numpy 2.4.6, pandas 3.0.5, scikit-learn
1.9.0, xgboost 3.4.0, matplotlib 3.11.1, pyreadstat 1.3.5, PyYAML 6.0.3).

```
python version2_ml\experiments\A4\src\run_a4.py ^
  --nfhs5-path "<authorized-dir>\IABR7EFL.DTA"
```

Result: exact reproduction — cohort 201,311 / 158,429 / 44,742; feasibility
82,607 first births and 6,276 observed prior institutional CS; frozen split
hash and membership digests match A1; all ten matched arms value-identical to
the 2026-10-17 recording; `a4_history_support.csv` and
`a4_facility_crosswalk.csv` byte-identical; figures pixel-identical. JSON and
CSV provenance columns now record commit `2eda72c`, the 2026-10-19 timestamp and
the global-environment versions. NFHS-4 facility-detail harmonization remains
pending local `IABR74FL.DTA` label verification (unchanged).

## Disclosure and path hygiene

- Cells below 10 suppressed in `a4_history_support.csv`, masked in
  `a4_run_summary.json` and the Methods note (e.g. positives in the 14-row
  `unknown_history` level).
- Committed outputs scanned for local paths/user names: none.
