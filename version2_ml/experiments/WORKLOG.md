# WORKLOG — A1 / A4 / A7 rework (running log)

Owner: Gaurav (implementation). Log maintained by the assistant under Gaurav's instructions.
Scope: NFHS-5 experiments A1, A4, A7 only. No raw DHS data, row-level records, or
per-birth outputs may appear in this file or in git; aggregate counts and hashes only.

## Status at Phase 1 start (2026-10-19)

- Branch: `gaurav-v2-A1-A4-A7-r3`, HEAD `a7ef6e3`, working tree clean.
- Archived untrusted attempt: branch `gaurav-v2-A1-A4-A7` (`25cdaf3`, tag
  `prior-attempt-2026-10-17`) plus `backup/prior-attempt-2026-10-17` — must not be
  touched, renamed, or pushed.
- Phase 0 audit report: `version2_ml/experiments/AUDIT_REPORT.md` (local, kept
  untracked pending Gaurav's decision).
- Audit verdict: the r3 rework is reusable; A1 ~90%, A4 ~85% (NFHS-4 blocked), A7 ~90%.

## Decisions recorded 2026-10-19 (from Gaurav, Phase 1 authorization)

- Q1 branch: continue on `gaurav-v2-A1-A4-A7-r3`. Do not touch/rename/push the archived
  `gaurav-v2-A1-A4-A7`. Nothing is pushed without Gaurav's explicit approval.
- Q2 environment: use the global Python; do NOT repair `.venv`; do NOT disable, bypass,
  or add exclusions to Application Control. Exactly one approved install:
  `pip install catboost` into the global Python, then verify `import catboost`.
  If blocked or failing: mark CatBoost in A7 NOT RUN with the reason and report; do not
  substitute another model. Record all package versions actually used.
- Pre-existing tracked row-level V1 files (audit finding C2): no modify/delete/untrack/
  history rewrite; Gaurav escalates to the coordinator. Only obligation here: ensure no
  new file of this kind is added on this branch (check `.gitignore` and `git status`
  before every commit) and keep these files listed here.

## Environment (Phase 1)

- Global Python: the approved CPython 3.13.11 executable, installed outside the
  repository.
- Project `.venv`: BROKEN as of 2026-10-19 —
  `ImportError: DLL load failed while importing _philox: An Application Control policy
  has blocked this file.` Not repaired per Q2.
- Recorded A1/A4/A7 outputs (commits `235fca7` / `33a28a3` / `d668549`) were produced
  under the `.venv` package set; comparisons in Phase 1 are made against those recorded
  outputs, never assumed equal.
- Package versions actually used in Phase 1 are recorded in the results section below
  and in each rerun's `*_run_summary.json`.

## Known pre-existing issues (no action per decisions)

- Tracked row-level V1 files on all branches including `main` (pre-existing, escalated):
  - `outputs/tables/b2_test_predictions_logistic_regression.csv`
  - `outputs/tables/b2_test_predictions_xgboost.csv`
  - `outputs/tables/b3_record_level_standardization.csv`
- `extensions_work/` committed outputs contain third-party absolute paths;
  pre-existing, out of scope, not modified.
- Stale ignored bytecode from the archived attempt (harmless):
  `__pycache__/preflight_features_nfhs5.cpython-313.pyc`,
  `__pycache__/run_a1_a4_a7.cpython-313.pyc`.

## Phase 1 checklist (approved 2026-10-19)

- [x] 1. Create this WORKLOG with audit findings, decisions, environment.
- [x] 2. Fix committed absolute paths (`A4/README.md:51`, `A4/TEST_COMMAND_LOG.md:28`)
      and scan the branch for other absolute/user-specific paths.
- [x] 3. Global Python: unit tests, config validation, cheapest recorded baseline;
      compare against recorded outputs and report differences.
- [x] 4. Checkpoint commit; show git status, test output, comparison; then STOP.

## Phase 1 results

### Step 2 — path hygiene (2026-10-19)

- Fixed: `A4/README.md` run example and `A4/TEST_COMMAND_LOG.md` now read
  `--nfhs5-path "<authorized-dir>\IABR7EFL.DTA"` (matches the A1 convention).
- Branch scan (`git grep` over tracked files): no local username or user-profile
  paths remain in the A1/A4/A7 code, outputs or documentation. All remaining
  absolute paths are pre-existing, other-owner files, not modified per decisions:
  `extensions_work/**` and `notebooks/**` note paths from another machine. Listed
  here for the coordinator escalation.

### Step 3 — environment, tests, baseline rerun (2026-10-19)

- CatBoost: the single approved install (`python -m pip install catboost`) succeeded
  into the global Python; `import catboost` verified OK (no Application Control
  block). Installed: catboost 1.2.10, plotly 7.1.0, graphviz 0.21.
  A7 CatBoost is available; no NOT RUN status needed.
- Package versions actually used in Phase 1: Python 3.13.11, numpy 2.4.6,
  pandas 3.0.5, scikit-learn 1.9.0, xgboost 3.4.0, shap 0.52.0, pyreadstat 1.3.5,
  PyYAML 6.0.3, pyarrow 25.0.0, matplotlib 3.11.1, catboost 1.2.10.
- Unit tests (global Python): 43 base / 10 common / 15 A1 / 30 A4 / 13 A7 = 111 OK;
  exit code 0 for every suite.
- Config validation: exit 0 (`PLANNING_CONFIG_VALID`); artifact written to the
  git-ignored `version2_ml/audits/phase3/outputs/config_validation.json`.
- Cheapest recorded baseline rerun: A1 validation-only runner (no CatBoost, no SHAP,
  test partition not scored), real DTA read read-only, outputs redirected to a temp
  directory so committed outputs were not touched:
  `python version2_ml/experiments/A1/src/run_a1.py --nfhs5-path "<authorized-dir>\IABR7EFL.DTA" --out-dir "<temp>\phase1_baseline\A1"`.
- Comparison vs recorded outputs (identical scientific content):
  - Cohort 201,311 births / 158,429 mothers / 44,742 events; split 140,920 train /
    30,227 validation / 30,164 test; interaction gate False with the same failing
    cells; split and feature manifest hashes identical.
  - All 6 rows of `a1_matched_ablation.csv` byte-identical (xgboost F1 0.7764/0.4939/
    0.1930, F2 0.8127/0.5690/0.1761; logistic F1 0.7722/0.4853/0.1941,
    F2 0.8068/0.5495/0.1786); support table (36 rows) and interaction gate (72 rows)
    identical; run summary has 0 non-dynamic differences.
  - Only expected metadata differs: repo_commit (`a7ef6e3` vs recorded code commit
    `235fca7`), timestamp, and package versions (numpy 2.4.6 vs 2.5.4, pandas 3.0.5
    vs 3.0.6, scikit-learn 1.9.0 vs 1.9.1, xgboost 3.4.0 vs 3.4.1, matplotlib 3.11.1
    vs 3.11.2).
- Conclusion: A1 reproduces exactly under the approved global environment; no metric
  drift detected.

### Step 4 — checkpoint

- Committed in this checkpoint: this WORKLOG plus the two A4 path fixes. No output
  files were regenerated or changed. `AUDIT_REPORT.md` intentionally remains
  untracked pending Gaurav's decision.
- Pre-commit privacy check: `git status` shows only the two A4 doc edits and the two
  new untracked files; no data, predictions, or row-level files added; `.gitignore`
  unchanged and still covers data, parquet/dta, experiment `private/`, `predictions/`
  and audit outputs. The pre-existing tracked V1 row-level files were not touched.
- Status: STOPPED per Phase 1 gate. Proposed first experiment after `CONTINUE`: A1
  (frozen code, exact reproduction already demonstrated; cheapest path to a
  re-recorded, environment-approved deliverable), then A4, then A7.

## Additional Phase 1 tasks (requested 2026-10-19)

### 1. A4 fix `7891204` — what changed and when outputs were produced

- Diff: one call-site edit only (+4/-1) in `run_a4.py`: the history-distribution
  figure now receives `history_feats.loc[main_block["valid_mask"]]` (validation rows)
  instead of the full-cohort `history_feats`. No history/facility logic, no model
  code, no table content touched.
- Why it was required: in that function `levels` (full cohort, 201,311) was framed
  against `y_valid` (30,227) — the pre-fix call would have raised a length-mismatch
  error. The fix repairs that and scopes the figure to validation only.
- Sequence evidence: recorded A4 run timestamp `2026-10-17T05:37:08Z` (=11:07:08 IST);
  outputs committed `75f6904` at 11:07:43 IST; fix committed `7891204` at 11:08:29 IST.
  The run succeeded, therefore the working tree already contained the fixed call at
  run time; the fix commit was bookkeeping 46 s later. Empirically confirmed: the
  committed `fig_a4_history_distribution.png` is **pixel-identical** (pixel MAE
  0.0000) to the figure regenerated by the current (fixed) code in the global
  environment; only PNG container bytes differ due to matplotlib 3.11.1 vs 3.11.2.
- A4 re-run (global env, outputs to temp dir; committed outputs NOT overwritten):
  - `a4_run_summary.json`: 0 non-dynamic differences.
  - `a4_matched_ablations.csv`: 0 value differences (all 10 arms).
  - `a4_sensitivity_m15_96.csv`: 0 value differences excluding provenance columns.
  - `a4_history_support.csv`, `a4_facility_crosswalk.csv`: byte-identical.
  - Figures: all pixel-identical to committed (PNG hashes differ only via matplotlib
    version metadata).
  - Conclusion: committed A4 outputs were produced with the fix applied; old-vs-new
    values are identical; nothing needs replacing.

### 2. A7 test-set safety

- `outputs/private/a7_test_scored.json` and `a7_tuning.json`: not deleted, edited, or
  regenerated; test partition NOT re-scored. Hashes recorded at audit time:
  `a7_test_scored.json` sha256 `adb4febc...ad2bceb8`;
  `a7_tuning.json` sha256 `f4718a15...709e732`. Test scoring timestamp:
  `2026-10-17T06:06:37Z` (=11:36:37 IST), source `computed`.
- Files changed after scoring (git history): `d668549` (only `_candidate_stage`
  threshold read: `evaluation.threshold` -> `thresholds.primary`; value identical,
  `thresholds.primary: 0.50`; test stage scores with a fixed `0.5` and the cached
  payload predates this commit anyway), `f7d5089` (outputs/log only), `a7ef6e3`
  (docs, requirements.txt, one docstring word). `A7/config.yaml` (tuning grids,
  candidate tiers, calibration settings) last modified `d9f1bb3` (11:21:42 IST),
  before scoring; `A7/tests/` last modified `d9f1bb3`. No commit after scoring
  changed candidate definitions, hyperparameter grids, or tuning logic.
- Tuning/selection provenance (code): `_tuning_stage` receives only the train fit
  frame, outcomes, mothers and grids (`run_a7.py:275-288`), called with
  `f1_frame.iloc[fit_positions]`, `y_fit`, `groups_fit` (`run_a7.py:1344-1348`);
  `fit_positions`/`calib_positions` are carved from `train_positions` only
  (`run_a7.py:1314-1320`); calibrators fitted on the train carve labelled "train"
  (`run_a7.py:313-315`); calibration-method selection uses validation Brier only
  (`run_a7.py:343-346`); test scoring is cache-guarded, once-only
  (`run_a7.py:359-365`).

### 3. Leakage spot-checks (file:line)

- (a) test never fit/scored in A1/A4: `run_a1.py:265-266` (fit "train", predict
  "validation"); `run_a4.py:243-244` (same); enforced centrally by
  `model_framework.py:63-65` (fit raises unless partition == "train").
- (b) preprocessing/encoders fit on train only: `preprocessing.py:85-87` (fit raises
  unless partition == "train"), pipeline fitted once at `preprocessing.py:103`;
  `preprocessing.py:120-121` (`fit_transform_train`).
- (c) A4 prior history uses strictly prior b3 and excludes same-month twins:
  `a4_history.py:20` (strict b3 ordering; multiple births never prior);
  per-(caseid,b3) blocks `a4_history.py:176-184`; earlier counts = group cumsum
  MINUS own block `a4_history.py:185-193` (same-month siblings share a block and are
  therefore never counted as earlier); index join on own block
  `a4_history.py:194-204`.
- (d) swapped validation/test label invariance test exists and passes:
  `test_a7_modules.py:190-202` (preprocessing output identical before/after
  flipping validation labels); tuning never accepts evaluation labels
  `test_a7_modules.py:175-188`. A7 suite: 13 OK.
- (e) mother-exclusive partition hard error: `split_contracts.py:80`
  ("LEAKAGE: a mother appears in multiple partitions"); test
  `test_split_contracts.py:33-38` (`test_reject_mother_leakage`), plus
  `test_group_exclusive` (`test_split_contracts.py:17-22`). All pass.

### 4. Output manifests

- Added `MANIFEST.json` to each experiment's `outputs/` folder (A1 7 files, A4 9
  files, A7 12 files; `private/` excluded from A7 listing, noted in the manifest).
  Each manifest records per-file SHA-256 and size, the recorded run provenance
  (repo_commit, branch, timestamp, software versions), split and feature manifest
  hashes (all five A4 arms; both A7 tiers), the manifest-generation environment
  (global Python versions), and the base commit `dcb7754`. Existing table schemas
  unchanged.

### 5. Reproduced vs NOT REPRODUCED in the global environment (as of this entry)

- Reproduced by re-run: A1 full validation-only pipeline (byte-identical tables,
  summary identical except provenance fields); A4 full pipeline (identical values,
  figures pixel-identical); all 111 unit tests; config validation.
- NOT re-run / NOT REPRODUCED: A7 real-data outputs (candidate fitting, calibration,
  SHAP, robustness, F0 reconciliation) — deliberately not re-run at this stage;
  CatBoost-based recorded A7 results specifically remain NOT REPRODUCED (CatBoost is
  installed and importable in the global env; it has not been exercised on real data);
  A7 development-test metrics NOT rescored by design.
- Nothing was relabeled as verified without a re-run.

## Phase 2 — experiment work

### A1 re-record (completed 2026-10-19)

- Re-ran `run_a1.py` under the global environment into the committed
  `version2_ml/experiments/A1/outputs` (approved re-record; old outputs remain in git
  history at `a7ef6e3`). Command:
  `python version2_ml/experiments/A1/src/run_a1.py --nfhs5-path "<authorized-dir>\IABR7EFL.DTA"`.
- Result: exact reproduction vs the 2026-10-17 recording — cohort 201,311/158,429/
  44,742; split 140,920/30,227/30,164; interaction gate unchanged; test partition NOT
  RUN; `a1_run_summary.json` 0 non-dynamic diffs; `a1_matched_ablation.csv` 0
  non-provenance value diffs; `a1_support_table.csv` and `a1_interaction_gate.csv`
  byte-identical; figures pixel-identical (PNG bytes differ only via matplotlib
  container metadata).
- New recorded provenance: `repo_commit` `e6f9c79`, timestamp `2026-10-19T04:59:43Z`,
  software versions = global environment (Python 3.13.11, numpy 2.4.6, pandas 3.0.5,
  scikit-learn 1.9.0, xgboost 3.4.0, matplotlib 3.11.1).
- Updated `A1/TEST_COMMAND_LOG.md` (re-record section) and regenerated
  `A1/outputs/MANIFEST.json` with the new hashes and recorded-run provenance.
- Status: A1 complete; STOPPED for Gaurav's CONTINUE before starting A4.

### A4 re-record (completed 2026-10-19)

- Re-ran `run_a4.py` under the global environment into the committed
  `version2_ml/experiments/A4/outputs` (old outputs remain in git history at
  `2eda72c`'s parent). Command:
  `python version2_ml/experiments/A4/src/run_a4.py --nfhs5-path "<authorized-dir>\IABR7EFL.DTA"`.
- Result: exact reproduction — cohort 201,311/158,429/44,742; feasibility 82,607
  first births and 6,276 observed prior institutional CS; frozen split hash and
  membership digests match A1; 0 non-dynamic diffs in `a4_run_summary.json`; 0
  non-provenance value diffs in `a4_matched_ablations.csv` (10 arms) and
  `a4_sensitivity_m15_96.csv` (10 arms); `a4_history_support.csv` and
  `a4_facility_crosswalk.csv` byte-identical; all three figures pixel-identical.
- New recorded provenance: `repo_commit` `2eda72c`, timestamp `2026-10-19T09:43:24Z`,
  global-environment versions.
- Updated `A4/TEST_COMMAND_LOG.md` and regenerated `A4/outputs/MANIFEST.json`.
- Known limitation unchanged: NFHS-4 facility-detail harmonization pending local
  `IABR74FL.DTA` label verification.
- Status: A4 complete; STOPPED for Gaurav's CONTINUE before starting A7.

### A7 global-environment run (2026-10-19) — INCIDENT DISCLOSURE

- Intent: run A7 with caches reused (no retune, no test re-score) into a sandbox
  out-dir, then compare.
- Mistake: the tuning/test caches were copied to
  `<temp>\phase2_A7\private`, but the runner resolves its cache as
  `out_dir/private` (`run_a7.py:1255`). The sandbox therefore had no cache and
  **re-scored the development test partition once** at `2026-10-19T10:00:15Z`
  (`source=computed`) and re-ran tuning (`source=computed`).
- Scope of impact: sandbox only. The committed/original private files were NOT
  modified: `a7_test_scored.json` sha256 `adb4febc...ad2bceb8` (mtime unchanged),
  `a7_tuning.json` sha256 `f4718a15...709e732` (mtime unchanged). Nothing from
  this run was committed. The test partition has now been scored twice on this
  machine (2026-10-17 recorded, 2026-10-19 sandbox), which breaches the
  once-only discipline. Values were identical, but the breach is recorded here
  for transparency.
- What the run did establish (all comparisons sandbox vs recorded):
  - `aggregate_model_comparison.csv` (26 rows), `calibration_summary.csv` (18),
    `a7_tuning_summary.csv` (10), `a7_robustness.csv` (15),
    `shap_feature_family_summary.csv` (72), `a7_shap_stability.csv` (3),
    `a7_reconciliation.csv` (13): zero value differences.
  - `a7_run_summary.json`: zero non-dynamic differences.
  - Development-test metrics: zero differences across all candidates/methods.
  - Tuning output: byte-identical to the recorded `a7_tuning.json`.
  - All three figures: pixel-identical.
  - CatBoost 1.2.10 was exercised on real data for the first time in the global
    environment and reproduced its recorded validation metrics exactly.
- Status: PAUSED pending Gaurav's decision on how to proceed; no A7 outputs
  committed.

#### Gaurav decision (2026-10-19)

- Option (a): keep the original recorded A7 outputs exactly as committed; the
  sandbox run is verification only. Do not commit sandbox-derived outputs and do
  not re-record A7.
- The A7 private cache stays exactly as-is (`a7_test_scored.json`
  `adb4febc...ad2bceb8`, `a7_tuning.json` `f4718a15...709e732`). No further A7
  real-data run of any kind without Gaurav's explicit approval for that specific
  command.

#### Sandbox artifact hashes (recorded before deletion)

All files under `<temp>\phase2_A7` at 2026-10-19, sha256 and bytes:

```
compare.py|cfe60304ecca7e869a9622458bc1b3e1ca34cbeee9a28857292ef5ac26d63d34|5110
out\a7_methods_results.md|d53150df2dfb1c37fe08943d9f8409227125777f646886c392a39b8c4642a5d7|6486
out\a7_reconciliation.csv|165715e616a867b6da321d222e82d6ae66a0aa2bbe0f367d2677d71972e35815|558
out\a7_robustness.csv|230c6a2b6a1476699641d902fe4f216b0a9ace78956fd26b18311bcdc8a87fe8|2198
out\a7_run_summary.json|aa42687e201d059dd35715b4c945e2d9cd1b6fce922a5785da3e5a171d321767|11885
out\a7_shap_stability.csv|b891139fef3d2b98f145666b82e49c2afc02e5043904a5c31b6ef48c8bc7c54b|253
out\a7_tuning_summary.csv|3f9d53628d0c54785648907ef9c3be623c20ffb135b39bf4d01bc341261a2508|1942
out\aggregate_model_comparison.csv|a1e838d28acd1db91d0bf475c185a933aec4b6f520df351d1748ad8c5699db32|21457
out\calibration_summary.csv|31a010594e16a8c12454919b6985f9e20643b64b6612c0c8551f8fb538b68b74|3217
out\fig_a7_calibration.png|b06f92c2767b48b462a16d3ce335ff4649a1c68989122cd30d6b94a367138c66|103321
out\fig_a7_roc_pr.png|e7de169b98b7429349a2e475f57a43fa34d19cb90a77d10b59e97303f9df3d05|142700
out\fig_a7_shap_stability.png|1b63792cc482bc90d6e89069311303e60838c229ebad06142c13d3dabe945c79|92868
out\old_fig_a7_calibration.png|8bec9056a27e4aaa3d202454a9af57d07b17a24f04e05877275e640bc384c751|103321
out\old_fig_a7_roc_pr.png|c7691a2b84fd0b9ca5b49806ae538728adf5fbac6473cee9a43a7c287217b01a|142700
out\old_fig_a7_shap_stability.png|2e7a89a291263a333e25a44d5e84586c5a879df0295d9ad414be81d2b7d56182|92868
out\private\a7_test_scored.json|8799f5801f8ce80ba5cb491429eb5b7acc94fff69718fcfbced6351c70652086|10379
out\private\a7_tuning.json|f4718a15e0a25c0c435b3f49626aeebf900693b501c705c0f1c94efac709e732|5019
out\shap_feature_family_summary.csv|be8c31eda6a0fac5512a6f105456e04442ff17a621a9a42c5c636df6c45f2124|4133
private\a7_test_scored.json|adb4febc2f85e19df644f4d40bba968a4d37898be049dc7de7ab3d03ad2bceb8|10379
private\a7_tuning.json|f4718a15e0a25c0c435b3f49626aeebf900693b501c705c0f1c94efac709e732|5019
```

- Deleted `<temp>\phase2_A7` only (recursively); no repository file and no other
  temp directory was touched.

#### Post-scoring safety-only change (2026-10-19)

- `run_a7.py`: added `--cache-dir` (default `<out-dir>/private`) and
  `--allow-test-score`. `_test_stage` now raises `RuntimeError` when
  `a7_test_scored.json` is absent and the flag is not given. Cache-hit behavior
  is unchanged (`cache_no_rescore`; the private file is never rewritten). No
  candidates, grids, tuning, thresholds or scoring logic were changed.
- Added three synthetic tests (no real data) in
  `A7/tests/test_a7_modules.py`: refusal without cache/flag and no files
  written; cache reuse without rescoring and unchanged cache bytes; CLI flag
  defaults.
- Full suite: 114 tests OK (43 base, 10 common, 15 A1, 30 A4, 16 A7), exit 0
  per suite; `py_compile` clean.
- A7 `README.md` and `outputs/MANIFEST.json` now carry the verification
  disclosure: outputs are the original 2026-10-17 recording; the
  global-environment verification reproduced all tables, figures and CatBoost
  validation metrics with zero value differences; the test partition was scored
  twice on this machine (identical values, no decision depended on the second);
  results remain non-confirmatory because of V1 test exposure.
- Manifest hash caveat: `MANIFEST.json` file hashes were computed on worktree
  bytes (CRLF under `core.autocrlf=true`), so they differ from the LF blobs in
  git; content is identical apart from line endings. This applies to the A1 and
  A4 manifests too; recorded for the stop gate, not changed in this checkpoint.

### Manifest hash rule fix (2026-10-19)

- A1/A4/A7 `MANIFEST.json` regenerated from committed git blobs: text files
  hashed in repository-normalized LF form, PNG files hashed as raw bytes. Each
  manifest now documents the rule under `hash_rule`. No output table or figure
  was regenerated; PNG hashes are unchanged and text-file hashes now match the
  committed blobs. All 28 listed files verified against `HEAD`.
- The previous `files_note` was replaced by the same committed-blob rule.
- Added `ISSUE_LOG.md` (blockers, incidents, unresolved scientific questions
  and coordinator escalations; no data values and no local paths) and
  `REVIEW_BUNDLE.md` (handoff section 12, items 1-7; existing files only;
  NOT RUN and NON-IDENTIFIABLE items marked).

### Final privacy/path scan vs `version2-ml-development` (2026-10-19)

- Scope: all files changed by this branch (three-dot diff against
  `version2-ml-development`) plus a full tracked-tree scan for absolute,
  drive-letter, UNC, home-directory, cloud-sync and temporary paths.
- Changed files: zero absolute/local path matches after removing the
  user-profile paths that this log itself contained. The only name-like
  matches are the project owner's name in provenance text and branch names;
  these are not machine-specific paths.
- Changed file types: no DTA, parquet, SAV, RDS or spreadsheet files. The 14
  changed CSV files are aggregate tables (3-72 data rows); none has a
  caseid/bidx/v###/m15/m17/b3/b19 header. No new row-level file was added.
- Pre-existing tracked V1 row-level files: all three exist at `HEAD` and at the
  base branch and are unchanged by this branch.
- Full tracked tree: 17 pre-existing files contain absolute paths, all under
  `extensions_work/**` and `notebooks/**`; none is changed by this branch and
  none belongs to A1/A4/A7. This confirms the E-02 escalation.
- No push and no pull request.

### A7 review documentation edits (2026-10-19)

- Added `Deviations and caveats` to `A7/README.md` and
  `A7/outputs/a7_methods_results.md`: fixed-0.50 thresholded metrics versus the
  handoff's validation-selected threshold; no `calibration_method` column and
  the verified uncalibrated/Platt/isotonic row order; combined facility SHAP
  share (0.4148 overall, folds 0.4114-0.4188, residence 0.4011-0.4177); F0
  uncalibrated and not a V1 reproduction; corrected two-scoring disclosure;
  model-versus-split variation with no model declared superior;
  `history_quality_flag` constant in validation (all rows `ok`, ambiguous
  cohort rows outside validation) and zero SHAP share.
- Documented per-line robustness tier and calibration state: repeated splits
  on both tiers uncalibrated; m15=96 exclusion and most-recent-birth `F1`-only
  uncalibrated; subgroup and pooled-state rows `F1_detail_history`-only with
  the selected Platt calibrator; PSU overlap not applicable.
- Added ISSUE_LOG questions S-07 (validation-selected threshold), S-08
  (`calibration_method` column, coordinate with Rifa) and S-09
  (F1-replace-facility tier).
- Regenerated A1/A4/A7 `MANIFEST.json` from staged index blobs under the
  committed-blob rule; no output table or figure was recomputed or modified.
- One checkpoint commit; no push and no pull request.
