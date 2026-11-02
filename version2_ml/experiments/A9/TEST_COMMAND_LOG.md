# A9 test and run command log

Environment: `.venv` Python 3.13.7 (numpy 2.5.4, pandas 3.0.6, scikit-learn
1.9.1, xgboost 3.4.1, scipy 1.18.1, matplotlib 3.11.2, PyYAML 6.0.3,
pyreadstat 1.3.7, pyarrow 26.0.0), Windows 11, branch `v2-final-eval`.
Run date 2026-11-02.

## 1. Protocol commits (before any A9 code ran on NFHS-5 data)

| Commit | Content |
| --- | --- |
| `94a6506` | r1: `PROTOCOL.md` and `config.yaml` |
| `0ce1d48` | r2: `config.yaml` guard pattern quoted (it was invalid YAML) and a revision note added. No change to any analysis choice. **This is the protocol commit recorded in every output.** |

The runner checks that `PROTOCOL.md` and `config.yaml` are byte-identical
to the protocol commit and refuses to run otherwise.

## 2. Synthetic suite

```powershell
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\A9\tests -p "test_*.py" -v
```

Result: 24 tests OK (no NFHS records). They cover:
- PSU nesting, zero PSU and mother overlap, determinism, fractions;
- V1 reconciliation, including missing-file, outcome-mismatch and
  duplicate-key failures;
- the guard lifecycle;
- bootstrap pairing, determinism and NaN handling, and bin suppression;
- test-label and validation-label leakage at the model, threshold and
  prediction level;
- every statement branch, pooling arithmetic and the paper table;
- an end-to-end run of the runner's scoring and reporting path with
  temporary guards.

The runner executes this suite before reading data.

## 3. Real-data runs

```powershell
$env:NFHS5_PATH = "<local>\IABR7EFL.DTA"
& .venv\Scripts\python.exe version2_ml\experiments\A9\src\run_a9.py
```

**Run 1 (scoring run), 2026-11-02, about 32 min.**
- Every seed's test partition was scored once. The guard files in
  `outputs/private/` record start and completion times (UTC):
  - 11: 06:29:51 to 06:35:40
  - 22: 06:36:26 to 06:41:45
  - 33: 06:42:21 to 06:46:52
  - 44: 06:47:29 to 06:53:13
  - 55: 06:54:00 to 06:59:59
- All tables and figures were written. The run then **failed at the final
  `MANIFEST.json` step**: the shared disclosure scan rejected
  `a9_split_audit.json` because one key was named `psu`. That key held
  overlap *counts* (all zero), not PSU codes.

**Fix (output formatting only).** `run_a9.public_split_audit` renames the
overlap keys to `between_partitions_by_cluster` and
`between_partitions_by_mother`. The end-to-end test now writes and scans that
file. The scoring path (`_score_seed`, `a9_models`, `a9_metrics`) was not
changed.

**Run 2 (reporting run), 2026-11-02.**
- All 5 seeds were read from their completed guards (`cache_no_rescore`).
  **No test partition was scored again.** The guard files' sha256 and
  modification times were identical before and after run 2.
- Exit 0. `MANIFEST.json` was created 07:02:37Z. The committed outputs are
  from run 2.

### Code provenance

The code ran as untracked files on top of protocol commit `0ce1d48`, which
`MANIFEST.json` records as both git commit and protocol commit. These sha256
values (LF bytes) pin the code that produced the committed outputs (run 2).
They equal the committed blobs.

| sha256 | file |
| --- | --- |
| 81bec99ceed3b49daea8444d1d6d048f7d584a6688b04b9eb389d838c5881e9c | PROTOCOL.md |
| 5d6b621fd0873c223e751e0b75157d05174362ab540ff61cb51dcfdf3746b921 | config.yaml |
| a9af414c216e399956f006b773b49f25acb5adecc2075c3cfa90c7f82dfcbd92 | src/a9_guard.py |
| 45cdb6dcb6fcd272d3235f48b1d5bc73784a79d75f16d5e02e6baf296d026a92 | src/a9_metrics.py |
| 74adfe275fb6f30111d21738b05f8f805355ea29cb82f550c0b93c15acb5b2fd | src/a9_models.py |
| a68bbda383747019c351af4dc954ac3915d5c0ff231235994bd857ae1fd78aee | src/a9_report.py |
| 9509db92cbf0324a1f21e22d443baa808833801f7f985f1c6b8013a1e5e5b480 | src/a9_splits.py |
| 13d750df1b20a7b82ef955fd0edbe5dedc7e49c872034e62dad9694d5f13501a | src/a9_v1.py |
| e76ae08385aba199b9e02c22891ca96d2d3925af73813b07bdc4b1ba84a16995 | src/run_a9.py |
| 6e972f7049fd6c2e4c6b688cf821af0604481303fbcc58a0b962c3acd3942995 | tests/test_a9_modules.py |

In run 1, `run_a9.py` and the test file differed from these only by the
`public_split_audit` change described above.
