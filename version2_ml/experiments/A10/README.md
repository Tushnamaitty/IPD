# A10: V2 subgroup checks

**Question.** Is V2 performance and calibration acceptable in every major
subgroup?

> **Reading the results.** Within-subgroup ROC-AUC is usually lower than the
> pooled ROC-AUC, because conditioning on a subgroup removes the
> between-group differences in risk that the pooled AUC also rewards. **A
> flag is a reporting requirement, not a model failure.** It marks a
> subgroup whose performance or calibration must be reported explicitly.

## Source of predictions (no new modelling)

The model is A9's V2: XGBoost on `F1_detail_history_state`, Platt-calibrated,
with each seed's F1-optimal validation threshold (seeds 11, 22, 33, 44, 55,
A9 protocol `0ce1d48`).

A9 deliberately stored no per-birth predictions or model state, so A10
**re-derives** V2's test predictions deterministically:
- A9's own split, carve and fit functions are called with A9's seeds and
  parameters.
- A10 proceeds only if, for every seed, the following all equal the A9 guard
  exactly:
  - the split manifest hash and the train/validation/test membership digests;
  - the V2 validation thresholds;
  - all 11 V2 test metrics stored in the guard (55 values in total).
- Any mismatch stops the run with nothing written.
- Each verified re-derivation is appended to the A9 guard file
  (`rederivations`). A9's status and results are untouched.

This choice was approved by the user because the A8 → A9 → A10 approval
sequence made a same-process A9/A10 run impossible.

## Design

- **Rows:** the 5 seeds' test partitions concatenated: about 150k birth-test
  observations. A birth can sit in more than one seed's test partition, so
  unique births are reported too.
- **Subgroups:**
  - residence (urban, rural);
  - facility sector (public m15 20-27, private 30-33, other 96);
  - birth order (first, second or later);
  - wealth quintile 1-5;
  - maternal age at delivery (<20, 20-34, 35+);
  - previous-CS history (the four `history_four_level` levels, plus
    `unknown_history` so that it is visibly reported as under-supported);
  - state/UT.
- **Support screen** (every subgroup, pooled test rows): at least 200 births
  and at least 30 per class. Otherwise the subgroup is UNDER_SUPPORTED and
  reported as "not estimable".
- **Metrics:** n, unique births, events, prevalence, ROC-AUC, AP, Brier,
  calibration intercept and slope (A7 joint fit), observed vs mean predicted
  rate, and recall and precision at each row's own seed threshold.
- **Uncertainty:** PSU bootstrap within each seed's test partition, B =
  1,000, seed 2028, percentile 95% CIs. The per-seed range of point estimates
  is also given.
- **Predeclared flags** (pooled point estimates):
  - LOW_DISCRIMINATION: ROC-AUC < 0.70;
  - MISCALIBRATED: |intercept| > 0.10, or slope outside 0.8-1.2;
  - UNDER_SUPPORTED: fails the support screen.
- **Gap:** within each dimension, the largest minus smallest supported-group
  ROC-AUC, with a CI from the replicate-wise gap.

## Interpreting the MISCALIBRATED flag (added after the run; flags unchanged)

The calibration intercept is A7's **joint-fit intercept**, from the fit
logit(observed) = a + b * logit(predicted). It is the calibration line's
value at a predicted probability of 0.5.

For subgroups whose risks lie far from 0.5, a slope only slightly different
from 1 moves that intercept a long way. That happens even when observed and
predicted rates agree, i.e. when calibration-in-the-large is good. For
example, `higher_order_no_observed_previous_institutional_cs` has observed
5.2% vs mean predicted 4.9% and slope 1.09, but intercept +0.30.

The flags are reported exactly as predeclared. The observed and
mean-predicted rate columns in `a10_subgroup_metrics.csv` show the absolute
calibration gap, which should be read alongside each flag.

## Run

```powershell
$env:NFHS5_PATH = "C:\path\to\IABR7EFL.DTA"
& .venv\Scripts\python.exe version2_ml\experiments\A10\src\run_a10.py
& .venv\Scripts\python.exe -m unittest discover -s version2_ml\experiments\A10\tests -p "test_*.py" -v
```

## Outputs (`outputs/`, aggregate only, counts < 10 suppressed)

- `a10_subgroup_metrics.csv`, `a10_flags.csv`, `a10_gaps.csv`.
- `a10_calibration_bins_by_sector.csv`.
- `fig_a10_subgroup_forest.png`: ROC-AUC with CI per subgroup, grouped by
  dimension, with the pooled-AUC and 0.70 reference lines.
- `fig_a10_calibration_by_sector.png`.
- `a10_methods_results.md`, `MANIFEST.json`.
