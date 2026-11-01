# Cleanup log — prior A1/A4/A7 attempt archived, clean rework branch created

Date: 2026-10-17
Base commit: 318f160 (`origin/version2-ml-development`)
New working branch: `gaurav-v2-A1-A4-A7-r2` (tracks `origin/version2-ml-development`)
Backup refs (local only, never pushed):
- commit `25cdaf3` on branch `gaurav-v2-A1-A4-A7`
- branch `backup/prior-attempt-2026-10-17`
- annotated tag `prior-attempt-2026-10-17`

## What was found

The prior AI attempt produced **no commits, no stashes, no tags** and was never
pushed to any remote. All of its work existed only as staged, unstaged and
untracked files in the working tree on branch `gaurav-v2-A1-A4-A7`, which
otherwise pointed at the same commit as `origin/version2-ml-development`.

## What was archived (class A, untrusted)

Archived via commit `25cdaf3` (22 files):

- `A1_A4_A7_HANDOFF.md`, `A1_A4_A7_IMPLEMENTATION_NOTES.md` (prior attempt summaries)
- `run_a1_a4_a7.py`, `preflight_nfhs5.py`, `preflight_features_nfhs5.py`
- `version2_ml/a1_state_context.py`, `version2_ml/a4_observed_history.py`,
  `version2_ml/a7_model_comparison.py`
- `version2_ml/tests/test_a1_a4_a7.py`, `version2_ml/tests/test_runner_safety.py`
- modifications to `version2_ml/model_framework.py` (silent XGBoost default change),
  `version2_ml/tests/test_feature_builder.py`, `version2_ml/tests/test_model_framework.py`
- untracked pilot outputs: `version2_ml/experiments/pilot/*` and
  `version2_ml/experiments/pilot_cohort201311/*` (8 aggregate JSON files)
- `version2_ml/audits/phase3/outputs/config_validation.json` (regenerable)

None of these files exist on `gaurav-v2-A1-A4-A7-r2`. They remain recoverable
from the backup refs above; nothing was pushed or rewritten anywhere.

## What was kept

- All raw/processed local data outside the repository tree (never touched).
- All tracked base-branch content: `docs/`, `notebooks/`, `outputs/`, `reports/`,
  `src/`, `extensions_work/`, `comment10_followup/`,
  `historical_state_context_data/`, `version2_ml/` modules, audits, configs, tests.
- Local virtual environment (ignored).
- `main` and `version2-ml-development` untouched; both still match their origin refs.

## .gitignore additions

- `version2_ml/audits/*/outputs/` — regenerable audit outputs stay untracked.
- `version2_ml/experiments/**/predictions/` and `version2_ml/experiments/**/private/`
  — per-birth predictions and private row-level files can never be committed.
Existing guards for `data/`, `artifacts/`, `*.dta`, `*.parquet`, `*.pkl`,
`*.joblib` verified present.

## Verification on `gaurav-v2-A1-A4-A7-r2` (as-is results)

- `git status`: clean (before the deliberate `.gitignore` edit and new docs below).
- `git diff origin/version2-ml-development --stat`: empty.
- Tests on clean base (pristine worktree of 318f160): 43 tests, OK.
- Tests on r2 before any new content: 43 tests, OK.
- `python version2_ml/validate_configs.py`: `PLANNING_CONFIG_VALID`, 0 errors,
  6 pre-existing warnings (outcome coding, survey variable selection, replicate
  weights, singleton strata, split proportions, unvalidated inference). These are
  pre-existing and are not silently fixed here.

Note: the polluted tree ran 62 tests; the extra 19 came from the archived prior
attempt and its modifications to shared modules/tests.
