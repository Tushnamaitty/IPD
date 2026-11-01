# A1 - State/area context: methods and results (auto-generated)

Generated: 2026-10-19T04:59:43+00:00 | Commit: e6f9c79e841b691c72648e9fa8f59f6bce0b15a5 (gaurav-v2-A1-A4-A7-r3) | Seed: 42

## Methods (as executed)

- Frozen cohort C1: 201311 births / 158429 mothers (44742 events); m15 = 96 rows: 517; states observed: 36; missing v024: 0.
- Mother-grouped provisional split (70/15/15, seed 42), split manifest hash `6ff45fcb76af00a6de5a33a225c482b1c9b3153f35544d887d8e3b2c3467904b`; group-exclusive audit passed; evaluation on the validation partition only; test partition NOT RUN (deferred to the single frozen A7 evaluation).
- Features: F1 = eight-core conservative set (maternal_age_at_index_birth, education_years, wealth_index, birth_order, facility_type, residence, religion, twin_order); F2 = F1 + `state_ut` (36 verified NFHS-5 states/UTs plus explicit unknown_state token), one-hot fitted on training rows only with handle_unknown=ignore.
- No external area descriptor with validated year/geography/coverage provenance was available, so the handoff fallback applies: categorical state and residence (residence is already inside F1). No contextual enrichment is claimed.
- XGBoost params: `{"colsample_bytree": 0.8, "eval_metric": "logloss", "learning_rate": 0.05, "max_depth": 6, "n_estimators": 200, "n_jobs": 1, "random_state": 42, "scale_pos_weight": 3.497207595340673, "subsample": 0.8, "tree_method": "hist", "verbosity": 0}`.
- Logistic regression params: `{"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "penalty": "l2 (library default; deprecated explicit argument omitted)", "random_state": 42, "solver": "liblinear"}`.
- Matched arms: identical training/validation rows in identical order, proven by row-membership digests (train `b96807778dbdfddc17d1446db1a9b918ba49a8eac9b09a9f710799d0cfeec37f`, validation `800e7130876165e2f2e72ba40555b742295baa411d03f04197ee4920234e8499`); threshold fixed at 0.50 (predeclared_conventional_not_optimized).
- Point estimates only: no design-based intervals, no significance or confirmatory claims; small cells (<10) suppressed.

## Results (development validation, point estimates)

| model | tier | n_births | n_mothers | n_events | prevalence | ROC-AUC | average precision | Brier | precision@0.50 | recall@0.50 | F1@0.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | F1 (eight-core) | 30227 | 23765 | 6701 | 0.2217 | 0.7764 | 0.4939 | 0.1930 | 0.4122 | 0.7038 | 0.5199 |
| XGBoost | F2 (F1 + state) | 30227 | 23765 | 6701 | 0.2217 | 0.8127 | 0.5690 | 0.1761 | 0.4368 | 0.7550 | 0.5534 |
| Logistic regression | F1 (eight-core) | 30227 | 23765 | 6701 | 0.2217 | 0.7722 | 0.4853 | 0.1941 | 0.4304 | 0.6612 | 0.5214 |
| Logistic regression | F2 (F1 + state) | 30227 | 23765 | 6701 | 0.2217 | 0.8068 | 0.5495 | 0.1786 | 0.4484 | 0.7227 | 0.5535 |

- XGBoost F2 minus F1 (point estimates): ROC-AUC +0.0364, average precision +0.0751, Brier -0.0168.
- Logistic regression F2 minus F1 (point estimates): ROC-AUC +0.0346, average precision +0.0643, Brier -0.0155.
- State x residence interaction: NOT RUN - D9 support screen failed on training rows (insufficient_support: andaman & nicobar islands | urban; chandigarh | rural; chandigarh | urban; goa | rural; goa | urban; ...); the reported fallback is F1 vs F1+state as above.

## Interpretation and limitations (as executed)

- The comparison is predictive, not ecological-causal; state identity may encode local response prevalence.
- Within-state generalization only; unseen-state validity is outside A1 (A6 scope).
- Split fractions remain provisional until the A7 freeze; the cohort rule C1 is frozen; m15 = 96 births are included per the recorded D2 decision.
- V1's held-out test was already examined in the original project; even a future single test scoring is a development-test, not independent validation.
- Statewise metrics are not reported; the support table shows training support flags only, with counts below 10 suppressed.

