# Step 4C: survey-uncertainty method — restricted prototype

**Status: NOT VALIDATED for publication or design-based confidence intervals (2026-10-15).**

The held-out fold contains 40,263 births, 2,685 observed strata, and 19,078 stratum-PSU pairs; 78 strata have one PSU. Float64 Kish effective sample size = 20,482.431256, reconciled with weighted-metrics pilot.

Prototype code (`version2_ml/variance_prototype.py`) provides:
- Integrity/Kish checks
- Within-observed-stratum PSU resampling *for exploration only*
- A default hard-stop on singleton PSU strata and an explicitly marked certainty-PSU sensitivity option
- Synthetic deterministic tests.

**No NFHS confidence intervals should be generated or published from this prototype.** The original multistage DHS sampling design, PSU and stratum scope, survey weights, independent mother-group holdout construction, and variance estimation under the fitted prediction model require methodological review. In particular, the test fold is a model-validation subsample of the survey, not a new probability sample; standard within-stratum PSU resampling from that fold does not automatically reproduce the original survey sample design. A certainty assumption for 78 singleton strata may understate variance.

**Next evidence gate:** run synthetic unit tests; review official DHS design details and whether an analytic PSU-based linearization or properly specified replicate-weight procedure fits the precise estimand. Distinguish inference for a fitted model's fixed predictions from uncertainty including training and model selection. Do not assert significance until established.
