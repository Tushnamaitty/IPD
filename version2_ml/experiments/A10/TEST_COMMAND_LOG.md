# A10 test and run command log

Environment: `.venv` Python 3.13.7 (numpy 2.5.4, pandas 3.0.6, scikit-learn
1.9.1, xgboost 3.4.1, scipy 1.18.1, matplotlib 3.11.2, PyYAML 6.0.3,
pyreadstat 1.3.7), Windows 11, branch `v2-final-eval` at `f2a90c7`.
Run date 2026-11-02.

## 1. Decision on prediction source

A9 stored no per-birth predictions or model state. The A8 -> A9 -> A10
approval sequence made a same-process run impossible. The user chose a
**verified deterministic re-derivation** of A9's V2 test predictions over
validation-only or stopping (see README).

## 2. Synthetic suite

```powershell
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\A10\tests -p "test_*.py" -v
```

Result: 15 tests OK (no NFHS records). They cover:
- subgroup cut points and partitions;
- the support-screen and flag edges;
- precision and recall at the threshold;
- deterministic bootstrap, gap arithmetic, and "not estimable" rows with
  suppression;
- the re-derivation gate: it passes on a genuine A9 guard, and stops on a
  1e-15 metric change, a changed split hash or a changed threshold;
- test labels never reaching predictions;
- an end-to-end output and manifest path.

The runner executes this suite first.

## 3. Recorded run

```powershell
$env:NFHS5_PATH = "<local>\IABR7EFL.DTA"
& .venv\Scripts\python.exe version2_ml\experiments\A10\src\run_a10.py
```

- **Exit 0.** `MANIFEST.json` was created 2026-11-02T12:25:57Z.
- **Re-derivation:** all 5 seeds verified exactly (55 V2 test metrics,
  thresholds, split manifest hashes and train/validation/test membership
  digests). A `rederivations` entry was appended to each A9 guard
  (07:13:06-07:13:46Z). The A9 guards' status and results are unchanged.
- **Bootstrap:** 56 subgroups x 1,000 replicates, about 85 min of CPU.
  Wall-clock time was 5 h 14 min because the machine slept or throttled
  background work between about 13:20 and 17:50 local time. The process
  paused and resumed, and no work was lost or repeated.
- The README section on interpreting the MISCALIBRATED flag (joint-fit
  intercept vs calibration-in-the-large) was added after the run. No output
  or flag was changed.

### Code provenance

The code ran as untracked files on top of `f2a90c7` (recorded in
`MANIFEST.json`). These sha256 values (LF bytes) equal the committed blobs.

| sha256 | file |
| --- | --- |
| dce9e56af1b78a37a9b74da7e28678de1752a0ef461eeb18beca04094d290d6c | config.yaml |
| bb33f46184d7a2796c530456e28e4113b0b4d3adadcb52cd13a1d50cce458ae4 | src/a10_analysis.py |
| f01f969b3b72ffdfc3fb2366d186556b02aebc177b27db48e212a0dd1ce37f2c | src/a10_rederive.py |
| 47d251d438dfef8a364cd0c346f6b65a0f330b03b1b5050ce37f51ddc6f61fad | src/a10_subgroups.py |
| 89ab7b0900fc284676519e075f71b0c4a9cd8699a85564e016140279e2f8b5b6 | src/run_a10.py |
| 2b8a98e2749db72ecfcc0867ecb974da29c9a220ac62581940b0a0c6e70728d7 | tests/test_a10_modules.py |
