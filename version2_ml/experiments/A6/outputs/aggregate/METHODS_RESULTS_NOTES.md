# A6 — Geographic and public/private sector transport: Methods and Results note

**Status: EXPLORATORY development result.** Only shared non-test NFHS-5 rows were used; the test partition was not scored. No confidence intervals, no significance claims, not external clinical validation.

## Methods

- **Data:** NFHS-5 201,311 institutional births (institutional m15 in 20-27/30-33/96; b19 in 0..59); analysis rows 171,147 (train 140,920, validation 30,227) from the shared mother-grouped split.
- **Geographic (leave-one-state-out):** for each state passing the screen, models train on other states' train mothers, choose the threshold on other states' validation mothers (i), and score all of the held-out state's births (ii) and its validation mothers (ii-b). Reference (iii): the same arm trained on all states' train mothers (target state seen), scored on the same validation rows — context only. Arm G0 reference reproduces A2 B0: PASS.
- **Geographic arms:** G0 = eight core features, state-free; G_hist = G0 + NFHS-4 historical state × sector rate (A2 mapping; exists before any NFHS-5 outcome); G_te = G0 + A3 grouped OOF state × sector encoding fitted on source states only — the held-out state is unseen, so it receives the source sector fallback (asserted). No numeric state code is used as a feature.
- **Sector:** public (m15 20–27) and private (m15 30–33); m15 = 96 excluded (444 births). Prespecified policy: the 3,800 mothers with births in both sectors (8,117 births) are excluded from both directions. Sector-free feature set (seven core features, facility_type dropped). (i) source-sector validation; (ii) all target-sector births; (ii-b)/(iii) target validation mothers scored by the transferred model and by a target-sector-trained reference.
- **Common:** logistic regression and XGBoost with fixed defaults; train-only preprocessing; max-F1 threshold from source validation applied unchanged to the target.

## Sector matrix

| Direction | Model | Target n | Target prev. | Source AUC | Target AUC | ΔAUC | Source AP | Target AP | ΔAP | Target cal. slope / intercept | Paired ΔAUC vs target ref. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| public->private | logistic_regression | 38,906 | 0.487 | 0.7139 | 0.6016 | -0.1123 | 0.2771 | 0.5836 | +0.3065 | 0.483 / 1.381 | -0.0147 |
| public->private | xgboost | 38,906 | 0.487 | 0.7146 | 0.6064 | -0.1082 | 0.2791 | 0.5858 | +0.3067 | 0.585 / 1.425 | -0.0136 |
| private->public | logistic_regression | 123,680 | 0.137 | 0.6269 | 0.6815 | +0.0546 | 0.6085 | 0.2504 | -0.3581 | 1.558 / -1.611 | -0.0319 |
| private->public | xgboost | 123,680 | 0.137 | 0.6298 | 0.6820 | +0.0522 | 0.6087 | 0.2510 | -0.3577 | 1.653 / -1.631 | -0.0309 |

## Post-execution verification

### 1. Zero-mother-overlap and target-outcome isolation tests
- Synthetic suite: 10/10 passed.
- Zero mother overlap: geographic **PASS**, sector policy **PASS**, overlap detection **PASS**.
- Target-outcome isolation: geographic **PASS**, sector **PASS**, held-out-state TE **PASS**, source/target mask guard **PASS**.
- Runtime checks on the real data: geographic_G0_ft_state_free_target_state_labels_flipped_predictions_identical **PASS**, geographic_te_target_state_labels_flipped_encodings_identical **PASS**, geographic_G_te_source_fallback_target_state_labels_flipped_predictions_identical **PASS**, sector_public_to_private_target_labels_flipped_predictions_identical **PASS**, sector_private_to_public_target_labels_flipped_predictions_identical **PASS**.
- Every geographic fold and both sector directions asserted zero shared mothers; each mother has one state (**PASS**).

### 2. Sector transport degradation (target − within-source validation; sector-free 7 features)

- **public->private, logistic_regression:** ROC-AUC 0.7139 → 0.6016 (ΔAUC -0.1123); AP 0.2771 → 0.5836 (ΔAP +0.3065; prevalence 0.137 → 0.487; ΔAP lift -0.825). Paired vs a target-sector-trained reference on the same validation rows: ΔAUC -0.0147, ΔAP -0.0172.
- **public->private, xgboost:** ROC-AUC 0.7146 → 0.6064 (ΔAUC -0.1082); AP 0.2791 → 0.5858 (ΔAP +0.3067; prevalence 0.137 → 0.487; ΔAP lift -0.835). Paired vs a target-sector-trained reference on the same validation rows: ΔAUC -0.0136, ΔAP -0.0143.
- **private->public, logistic_regression:** ROC-AUC 0.6269 → 0.6815 (ΔAUC +0.0546); AP 0.6085 → 0.2504 (ΔAP -0.3581; prevalence 0.482 → 0.137; ΔAP lift +0.570). Paired vs a target-sector-trained reference on the same validation rows: ΔAUC -0.0319, ΔAP -0.0291.
- **private->public, xgboost:** ROC-AUC 0.6298 → 0.6820 (ΔAUC +0.0522); AP 0.6087 → 0.2510 (ΔAP -0.3577; prevalence 0.482 → 0.137; ΔAP lift +0.574). Paired vs a target-sector-trained reference on the same validation rows: ΔAUC -0.0309, ΔAP -0.0311.
- AP is prevalence-dependent: public and private sectors have very different cesarean prevalence, so raw ΔAP mostly reflects the target prevalence; ΔAUC and ΔAP lift are the fairer comparisons. Differences reflect case mix and reporting by sector, not a causal sector effect.

### 3. Geographic transport across held-out states

- 35/36 states passed the support screen (n ≥ 200, ≥ 30 positives, ≥ 30 negatives on non-test births). Failed screen (NOT ESTIMABLE, counts suppressed, not pooled): chandigarh (n<200).

| Arm | Model | States | Held-out AUC min / p25 / median / p75 / max | ΔAUC vs source median (min, max) | States below source | Paired unseen − seen median (min, max) |
|---|---|---|---|---|---|---|
| G0_ft_state_free | logistic_regression | 35 | 0.596 / 0.681 / 0.722 / 0.774 / 0.836 | -0.0515 (-0.178, +0.071) | 26 | -0.0004 (-0.003, +0.003) |
| G0_ft_state_free | xgboost | 35 | 0.597 / 0.684 / 0.724 / 0.774 / 0.846 | -0.0512 (-0.179, +0.080) | 26 | -0.0004 (-0.003, +0.006) |
| G_hist_nfhs4_context | logistic_regression | 35 | 0.598 / 0.681 / 0.725 / 0.775 / 0.844 | -0.0841 (-0.210, +0.042) | 30 | -0.0002 (-0.004, +0.001) |
| G_hist_nfhs4_context | xgboost | 35 | 0.599 / 0.686 / 0.720 / 0.776 / 0.851 | -0.0903 (-0.213, +0.047) | 31 | -0.0017 (-0.031, +0.005) |
| G_te_source_fallback | logistic_regression | 35 | 0.603 / 0.680 / 0.726 / 0.776 / 0.844 | -0.0860 (-0.209, +0.040) | 31 | -0.0003 (-0.061, +0.007) |
| G_te_source_fallback | xgboost | 35 | 0.599 / 0.685 / 0.720 / 0.776 / 0.850 | -0.0903 (-0.213, +0.046) | 31 | -0.0019 (-0.078, +0.009) |

- Held-out-state AUC differences mix genuine transport loss with differences in each state's case mix and prevalence; per-state values are in `geographic_matrix.csv`. Small states have unstable estimates; no intervals are attached.

### Interpretation: raw deltas vs paired deltas

- **Geographic, G0, logistic_regression:** held-out-state AUC is a median -0.0515 below other-state validation, but on identical held-out-state validation rows a model that DID see the state differs from the unseen-state model by a median -0.0004 (27 states). The raw gap therefore mainly reflects lower discrimination *within* a single state (a pooled multi-state validation set also ranks between-state differences), not a loss caused by the state being unseen.
- **Geographic, G0, xgboost:** held-out-state AUC is a median -0.0512 below other-state validation, but on identical held-out-state validation rows a model that DID see the state differs from the unseen-state model by a median -0.0004 (27 states). The raw gap therefore mainly reflects lower discrimination *within* a single state (a pooled multi-state validation set also ranks between-state differences), not a loss caused by the state being unseen.
- **Context arms (G_hist, G_te):** median held-out AUC 0.720–0.726 vs G0 0.722–0.724: state-level context mostly helps rank births *between* states, so it adds little within one unseen state; its larger raw gap vs source reflects a stronger source-validation AUC, not worse target performance.
- **Sector public->private, logistic_regression:** raw ΔAUC vs source validation -0.1123, but vs a target-sector-trained model on the same rows -0.0147. The paired value is the transport penalty attributable to training on the other sector; the remainder of the raw difference reflects how hard discrimination is within each sector.
- **Sector public->private, xgboost:** raw ΔAUC vs source validation -0.1082, but vs a target-sector-trained model on the same rows -0.0136. The paired value is the transport penalty attributable to training on the other sector; the remainder of the raw difference reflects how hard discrimination is within each sector.
- **Sector private->public, logistic_regression:** raw ΔAUC vs source validation +0.0546, but vs a target-sector-trained model on the same rows -0.0319. The paired value is the transport penalty attributable to training on the other sector; the remainder of the raw difference reflects how hard discrimination is within each sector.
- **Sector private->public, xgboost:** raw ΔAUC vs source validation +0.0522, but vs a target-sector-trained model on the same rows -0.0309. The paired value is the transport penalty attributable to training on the other sector; the remainder of the raw difference reflects how hard discrimination is within each sector.

## Limitations

- Exploratory, one provisional split, no intervals; V1 test previously examined.
- State is interview residence, not verified delivery-facility state; migration and cross-state delivery are unresolved. Small held-out states give unstable metrics.
- Sector transport is conditional on observed institutional delivery; differences reflect case mix, selection and reporting, not causal public/private effects. Excluding both-sector mothers changes the population slightly.
- G_hist uses NFHS-4 labels from the held-out state (prior wave, different respondents); it is external historical context, not target-wave outcome information.
- Retrospective survey-recorded features; not prenatal prediction.
