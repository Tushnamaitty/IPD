# V2 final results summary (A8 -> A9 -> A10)

Branch `v2-final-eval`. All numbers below are copied from committed aggregate
outputs:
- `A8/outputs/a8_decision.json`, `a8_methods_results.md` (commit `f03ba25`);
- `A9/outputs/a9_methods_results.md`, `a9_statements.json` (protocol `0ce1d48`,
  results `f2a90c7`);
- `A10/outputs/a10_flags.csv`, `a10_subgroup_metrics.csv`, `a10_gaps.csv`
  (`0ecfc5f`).

Cohort: frozen NFHS-5 C1, 201,311 institutional births to 158,429 mothers,
44,742 C-sections. All analyses are unweighted point estimates; the
survey-weighted results are descriptive sensitivities only.

## 1. Selected final tier: `F1_detail_history_state`

The selection comes from **A8**, a matched ablation on the A7 split (seed 42)
plus 3 repeated mother-grouped splits (seeds 7, 2026, 99), scored on
validation only.

- **Arms compared:**
  - H0 = `F1_detail_history`;
  - H1 = H0 + `state_ut`;
  - H2 = H0 + state x sector OOF target encoding;
  - H3 = H0 + both.
- **Gate:** H0 reproduced A7 exactly (XGBoost validation ROC-AUC 0.8298, AP
  0.6113).
- **Predeclared rule:** adopt an arm only if, for XGBoost, ΔROC-AUC ≥ +0.005
  **and** ΔAP ≥ +0.010 vs H0 on **all 4 seeds**. If several pass, take the
  fewest added features, with ties going to H1.
- **Outcome:** H1, H2 and H3 all passed on every seed. H1 and H2 each add one
  feature, and the tie goes to **H1**.

XGBoost H1 − H0 (validation):

| Seed | ΔROC-AUC | ΔAP |
| --- | --- | --- |
| 42 | +0.0187 | +0.0411 |
| 7 | +0.0203 | +0.0440 |
| 2026 | +0.0215 | +0.0474 |
| 99 | +0.0192 | +0.0448 |

H2 and H3 added only a further +0.001 to +0.003 ROC-AUC over H1.

**FINAL_TIER** = the 8 core features + `facility_detail` + 3 observed-history
columns + `state_ut` (13 model columns). In the H1 SHAP, `state_ut` ranks
second, with 19.7% of mean |SHAP|, behind `facility_type` (20.8%).

## 2. Final evaluation (A9): paper table and predeclared statements

**Design:**
- protocol committed before scoring;
- 5 new PSU (`v021`) grouped 70/15/15 splits, seeds 11/22/33/44/55, with zero
  PSU and mother overlap asserted;
- Platt calibration on a PSU-grouped carve;
- thresholds selected on validation;
- each test partition scored once, behind guard files;
- PSU cluster bootstrap, B = 1,000, seed 2027.

**Pooled** = mean across the 5 seeds; the 95% CI comes from the across-seed
mean of the bootstrap replicates.

| Model | Features | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier (95% CI) | Cal. slope | Cal. intercept |
| --- | --- | --- | --- | --- | --- | --- |
| V2 | F1_detail_history + state_ut (FINAL_TIER, 13) | 0.8487 (0.8457-0.8516) | 0.6477 (0.6396-0.6550) | 0.1194 (0.1179-0.1211) | 1.008 | -0.011 |
| V2_LR | F1_detail_history + state_ut (FINAL_TIER, 13) | 0.8464 (0.8434-0.8495) | 0.6425 (0.6346-0.6500) | 0.1204 (0.1187-0.1221) | 1.006 | -0.017 |
| F1_core | 8 core features (F1) | 0.7782 (0.7745-0.7820) | 0.4923 (0.4849-0.4997) | 0.1419 (0.1399-0.1439) | 1.024 | +0.000 |
| V1_spec | V1 F0_historical_36 (incl. BMI, ANC) | 0.8013 (0.7977-0.8048) | 0.5283 (0.5208-0.5360) | 0.1364 (0.1345-0.1384) | 1.015 | -0.008 |
| V1 (recorded) | original V1 report, different split, not paired | 0.8037 | 0.5334 | 0.1358 | not reported | not reported |

Paired pooled differences (95% CI):

| Comparison | ΔROC-AUC | ΔPR-AUC |
| --- | --- | --- |
| V2 − V1_spec | +0.047 (+0.045 to +0.050) | +0.119 (+0.114 to +0.125) |
| V2 − F1_core | +0.070 (+0.068 to +0.073) | +0.155 (+0.150 to +0.161) |
| V2 − V2_LR | +0.0023 (+0.0016 to +0.0029) | +0.0052 (+0.0039 to +0.0065) |

**Predeclared statements** (wording fixed in `A9/PROTOCOL.md`):

1. **"V2 discrimination is superior to the V1 specification."** The lower
   95% bound of ΔROC-AUC (V2 − V1_spec) is > 0 on 5 of 5 seeds (required
   ≥ 4). The per-seed lower bounds are +0.0448, +0.0420, +0.0411, +0.0428 and
   +0.0413.
2. **"V2 is well calibrated."** Pooled slope 1.008 (required 0.9–1.1);
   pooled intercept −0.011 (required |·| ≤ 0.05).
3. **"XGBoost and LR are equivalent."** Pooled ΔROC-AUC (V2 − V2_LR) is
   +0.0023, with 95% CI [+0.0016, +0.0029], inside ±0.01. The CI excludes
   zero, so XGBoost is statistically, but not materially, better.

## 3. Subgroup checks (A10): flags with numbers

**Source:** A9's V2 test predictions, re-derived deterministically and
verified against the A9 guards (55 metrics, thresholds, split and
membership hashes, all exactly equal).

**Pooled test rows:** 150,810 observations (111,573 unique births). Overall
ROC-AUC 0.8486, slope 1.007, intercept −0.011.

**Predeclared flags:**
- LOW_DISCRIMINATION: ROC-AUC < 0.70;
- MISCALIBRATED: |joint-fit intercept| > 0.10 or slope outside 0.8–1.2;
- UNDER_SUPPORTED: < 200 births or < 30 per class.

**A flag is a reporting requirement, not a model failure.** Within-subgroup
AUC is lower than pooled AUC by construction.

| Dimension | Group | Flag | n | Evidence |
| --- | --- | --- | --- | --- |
| Facility sector | other (m15 = 96) | UNDER_SUPPORTED | 406 | 0 events; not estimable |
| Wealth | Q1 (poorest) | MISCALIBRATED | 34,290 | intercept +0.122, slope 1.048 (observed 8.9% vs predicted 8.7%) |
| Wealth | Q2 | MISCALIBRATED | 34,584 | intercept +0.101, slope 1.066 (15.5% vs 15.4%) |
| Maternal age | 35+ | MISCALIBRATED | 5,775 | intercept +0.128, slope 1.036 (26.4% vs 25.1%) |
| Previous-CS history | higher order, no observed previous CS | MISCALIBRATED | 26,380 | intercept +0.301, slope 1.088 (5.2% vs 4.9%) |
| Previous-CS history | unknown_history | UNDER_SUPPORTED | <10 | not estimable |
| State | Arunachal Pradesh | LOW_DISCRIMINATION + MISCALIBRATED | 3,216 | ROC-AUC 0.652 (0.625–0.678); intercept −0.419, slope 0.628 |
| State | Kerala | LOW_DISCRIMINATION + MISCALIBRATED | 2,258 | ROC-AUC 0.687 (0.665–0.710); intercept −0.206, slope 0.857 |
| State | Ladakh | LOW_DISCRIMINATION + MISCALIBRATED | 379 | ROC-AUC 0.624 (0.553–0.691); intercept +0.032, slope 0.538 |
| State | Lakshadweep | LOW_DISCRIMINATION + MISCALIBRATED | 230 | ROC-AUC 0.624 (0.535–0.693); intercept −0.198, slope 0.496 |
| State | Chandigarh | UNDER_SUPPORTED | 91 | not estimable |
| State | 21 further states | MISCALIBRATED | 362–10,846 | see `A10/outputs/a10_flags.csv` |

The 21 further states are:
- Andaman & Nicobar, Andhra Pradesh, Assam, Dadra & Nagar Haveli and Daman &
  Diu, Goa, Gujarat, Haryana, Himachal Pradesh, Jharkhand, Madhya Pradesh,
  Maharashtra;
- Manipur, Meghalaya, Mizoram, Nagaland, NCT of Delhi, Odisha, Puducherry,
  Rajasthan, Sikkim, Tripura.

The largest |intercept| is Nagaland's (−0.76). The largest absolute gap
between observed and predicted rate is Sikkim's, +10.9 percentage points
(n = 404). Across the 35 supported states, the median |observed − predicted|
is 1.5 percentage points.

**Unflagged, but notable:**
- weaker discrimination in **private** facilities: ROC-AUC 0.750
  (0.744–0.755), vs public 0.825;
- **urban** births: 0.794 vs rural 0.857;
- **first births**: 0.793 vs later births 0.879.

**ROC-AUC gaps within dimensions** (largest − smallest supported group, 95%
CI):

| Dimension | Gap (95% CI) |
| --- | --- |
| Residence | 0.063 (0.057–0.069) |
| Facility sector | 0.075 (0.068–0.081) |
| Birth order | 0.086 (0.081–0.091) |
| Wealth | 0.076 (0.069–0.086) |
| Maternal age | 0.061 (0.048–0.074) |
| Previous-CS history | 0.086 (0.063–0.111) |
| **State** | **0.266 (0.237–0.362)**, Bihar to Ladakh |

**Reading MISCALIBRATED.** The joint-fit intercept is the calibration line's
value at a predicted probability of 0.5. In low-prevalence groups it is
large even when calibration-in-the-large is good (see the
observed-vs-predicted rates above and `A10/README.md`). Several non-state
flags reflect this rather than material miscalibration. Small-state flags
(Sikkim, Puducherry, Ladakh, Lakshadweep) do show real absolute deviations.

## 4. Limitations

- **Retrospective survey data.** Delivery mode, place of delivery and birth
  history are mothers' recall at interview, for births up to 5 years
  earlier. Misreporting is not corrected.
- **One wave.** Everything is NFHS-5 (2019–21). Practice patterns and
  calibration may differ in other waves.
- **History observed only within the 5-year recall window.** "No previous
  C-section" means none observed among earlier births in the window, not a
  lifetime negative. Births outside the window are invisible.
- **BMI excluded.** Maternal height and weight are measured at interview,
  after the birth, so BMI was not used in V2. It is used in V1_spec, which
  keeps V1's definitions.
- **Not causal.** Facility type, state and history are strong predictors
  because of where and to whom C-sections happen. Neither the scores nor the
  SHAP attributions estimate the effect of changing an input.
- **Not an external validation.** A9 is a predeclared *internal* validation
  on new cluster-grouped splits. Its test rows come from the same NFHS-5
  outcomes that were inspected during V1 and A1–A8 development, and every
  state appears on both sides of each split.
- **NFHS-4 external validation was not run.** Facility-detail and history
  harmonisation for NFHS-4 needs local label verification (see the A4 notes).
- **Uncertainty.** The CIs come from a PSU cluster bootstrap that ignores
  survey stratification, so they are not design-based standard errors.
  Survey-weighted metrics are descriptive only.
- **Subgroups.** State-level discrimination and calibration vary widely
  (state AUC gap 0.27). Some small states and UTs are under-supported or
  flagged.

## 5. Does the implementation (`version2_ml/serving`) need retraining?

**Yes.** `FINAL_TIER` (`F1_detail_history_state`) differs from the tier the
serving code was built on (`F1_detail_history`, A7). The serving code is on
branch `v2-implementation` (pushed; not merged into this branch). Its
exported model `v2.0-a7-f1dh-xgb-platt-46e1bff` is now superseded.

Exact changes required (no other modelling change: same XGBoost parameters,
Platt calibration and F1-optimal threshold rule):

1. **`train_export.py`**
   - Set `TIER = "F1_detail_history_state"`. `a7_candidates.TIERS` has no
     such entry, so build the frame with A8's `a8_arms.arm_frame(base,
     map_states(cohort["v024"]), None, "H1")` and fit with
     `arm_manifest("H1")` (13 columns: `state_ut` appended to the
     categoricals).
   - Change `MODEL_VERSION_PREFIX` to something like
     `v2.1-a8h1-f1dhs-xgb-platt`.
   - Replace the A7 acceptance targets with the **A8 H1 seed-42 validation
     values**: XGBoost ROC-AUC **0.8484**, AP **0.6524** (from
     `A8/outputs/a8_ablation.csv`). Keep the slope and threshold-range checks.
   - Add `state_ut` to the exported vocabulary and display labels: the A1
     verbatim labels of the states present in the fit pool.
   - Retrain with `train_export.py --nfhs5-path ...`.
2. **`schemas.py`:** add a required input field `state` (enum built from
   `config.json` vocabulary `state_ut`). The API then takes **11 user
   inputs**.
3. **`mapping.py`:** write the state into the raw row (`v024` code, or the
   label mapped via A1's `STATE_LABELS`) and set the `state_ut` column with
   `a1_state_context.map_states`. The frame then has 13 columns in the new
   `feature_order`.
4. **`explain.py`:** add `"state_ut": "state"` to `FIELD_BY_COLUMN` and a
   display name, for example "State/UT", so SHAP factors can name the state.
5. **`api.py` `/options` and `app_streamlit.py`:** add a State/UT dropdown.
   Mark Chandigarh as under-supported (A10), and add the subgroup caveats to
   the model card. Low discrimination: Arunachal Pradesh, Kerala, Ladakh,
   Lakshadweep.
6. **`model_card.md` and README:** replace the A7 validation and A7
   dev-test numbers with the A9 pooled results above, add the A10 subgroup
   flags, and update the input table (11 inputs).
7. **Tests:**
   - the fixture model gains a `state_ut` column and vocabulary;
   - the schema tests need an unknown-state 422 case;
   - the mapping tests need a state-label check;
   - the golden test targets become the new `metrics.json`.

Until this retraining is done, the serving demo must not be described as the
final V2 model.
