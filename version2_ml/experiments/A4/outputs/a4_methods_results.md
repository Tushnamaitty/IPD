# A4 - Observed delivery history and facility detail: methods and results (auto-generated)

Generated: 2026-10-19T09:43:24+00:00 | Commit: 2eda72cb0f83cf8a116d05328f209e47d141cac9 (gaurav-v2-A1-A4-A7-r3) | Seed: 42

## Methods (as executed)

- Frozen cohort C1: 201311 births / 158429 mothers (44742 events); m15 = 96 rows: 517 (included per D2).
- Provisional mother-grouped split (70/15/15, seed 42); split hash `6ff45fcb76af00a6de5a33a225c482b1c9b3153f35544d887d8e3b2c3467904b` equals the frozen A1 split, and train/validation membership digests equal the A1 digests (train `b96807778dbdfddc17d1446db1a9b918ba49a8eac9b09a9f710799d0cfeec37f`, validation `800e7130876165e2f2e72ba40555b742295baa411d03f04197ee4920234e8499`).
- Observed-history timeline: all survey-window roster births (b19 0-59) of each mother, sorted by (caseid, b3); home deliveries stay in the timeline as observed noninstitutional records; births outside the window are never used. Only strictly earlier calendar months contribute (same-month twins never count as previous deliveries); the outcome of the index birth or later births is never used.
- Four-category history (index births only): first_recorded_live_birth (bord==1); observed_previous_institutional_cs (strictly earlier institutional birth with m17==1); higher_order_no_observed_previous_institutional_cs (strictly earlier institutional birth(s), none with m17==1); higher_order_earlier_institutional_history_unobserved (no strictly earlier institutional record; includes earlier noninstitutional and out-of-recall births). Missing or ambiguous records yield unknown_history plus history_quality_flag tokens; they are never treated as negatives.
- Facility detail labels are verbatim NFHS-5 metadata: NFHS-5 IABR7EFL.DTA m15 value metadata (local read 2026-10-17); crosswalk `a4_facility_crosswalk.csv` (NFHS-4 pending local verification).
- Ablation arms (identical rows and order across arms): F1; F1 + detailed m15 facility; F1 with detailed facility replacing broad facility_type; F1 + observed history; F1 + both. Models: XGBoost and logistic regression (same configurations as A1).
- m15 != 96 sensitivity: 96 rows removed consistently from the index cohort and the history timeline, with the mother partition inherited from the main split; threshold fixed at 0.50.
- Point estimates only; no design-based intervals or significance claims; cells below 10 suppressed.

## Feasibility cross-checks (Phase-0 audit reproduction)

- first_recorded_live_birth: observed 82607, audit expected 82607, match=True.
- observed_previous_institutional_cs: observed 6276, audit expected 6276, match=True.

## Results (development validation, point estimates)

| model | tier | n_births | n_events | ROC-AUC | average precision | Brier | precision@0.50 | recall@0.50 | F1@0.50 |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | F1 | 30227 | 6701 | 0.7764 | 0.4939 | 0.1930 | 0.4122 | 0.7038 | 0.5199 |
| XGBoost | F1_detail | 30227 | 6701 | 0.7914 | 0.5034 | 0.1888 | 0.4075 | 0.7568 | 0.5297 |
| XGBoost | F1_detail_replace | 30227 | 6701 | 0.7914 | 0.5046 | 0.1887 | 0.4075 | 0.7578 | 0.5300 |
| XGBoost | F1_history | 30227 | 6701 | 0.8189 | 0.6046 | 0.1719 | 0.4584 | 0.7277 | 0.5625 |
| XGBoost | F1_detail_history | 30227 | 6701 | 0.8302 | 0.6119 | 0.1685 | 0.4529 | 0.7654 | 0.5690 |
| Logistic regression | F1 | 30227 | 6701 | 0.7722 | 0.4853 | 0.1941 | 0.4304 | 0.6612 | 0.5214 |
| Logistic regression | F1_detail | 30227 | 6701 | 0.7890 | 0.4973 | 0.1898 | 0.4148 | 0.7366 | 0.5307 |
| Logistic regression | F1_detail_replace | 30227 | 6701 | 0.7890 | 0.4973 | 0.1898 | 0.4148 | 0.7368 | 0.5307 |
| Logistic regression | F1_history | 30227 | 6701 | 0.8148 | 0.5967 | 0.1733 | 0.4785 | 0.6945 | 0.5666 |
| Logistic regression | F1_detail_history | 30227 | 6701 | 0.8273 | 0.6070 | 0.1697 | 0.4577 | 0.7548 | 0.5699 |

- XGBoost F1_detail minus F1 (point estimates): ROC-AUC +0.0150, average precision +0.0095, Brier -0.0042.
- XGBoost F1_detail_replace minus F1 (point estimates): ROC-AUC +0.0151, average precision +0.0107, Brier -0.0043.
- XGBoost F1_history minus F1 (point estimates): ROC-AUC +0.0426, average precision +0.1107, Brier -0.0210.
- XGBoost F1_detail_history minus F1 (point estimates): ROC-AUC +0.0538, average precision +0.1180, Brier -0.0245.
- Logistic regression F1_detail minus F1 (point estimates): ROC-AUC +0.0168, average precision +0.0121, Brier -0.0043.
- Logistic regression F1_detail_replace minus F1 (point estimates): ROC-AUC +0.0168, average precision +0.0121, Brier -0.0043.
- Logistic regression F1_history minus F1 (point estimates): ROC-AUC +0.0427, average precision +0.1114, Brier -0.0208.
- Logistic regression F1_detail_history minus F1 (point estimates): ROC-AUC +0.0552, average precision +0.1217, Brier -0.0243.

### m15 != 96 sensitivity

| model | tier | n_births | n_events | ROC-AUC | average precision | Brier |
|---|---|---|---|---|---|---|
| XGBoost | F1 | 30139 | 6701 | 0.7755 | 0.4945 | 0.1931 |
| XGBoost | F1_detail | 30139 | 6701 | 0.7907 | 0.5046 | 0.1888 |
| XGBoost | F1_detail_replace | 30139 | 6701 | 0.7905 | 0.5042 | 0.1889 |
| XGBoost | F1_history | 30139 | 6701 | 0.8180 | 0.6040 | 0.1722 |
| XGBoost | F1_detail_history | 30139 | 6701 | 0.8295 | 0.6124 | 0.1687 |
| Logistic regression | F1 | 30139 | 6701 | 0.7713 | 0.4853 | 0.1944 |
| Logistic regression | F1_detail | 30139 | 6701 | 0.7882 | 0.4974 | 0.1901 |
| Logistic regression | F1_detail_replace | 30139 | 6701 | 0.7882 | 0.4973 | 0.1901 |
| Logistic regression | F1_history | 30139 | 6701 | 0.8142 | 0.5967 | 0.1735 |
| Logistic regression | F1_detail_history | 30139 | 6701 | 0.8267 | 0.6070 | 0.1700 |

## Disclosure-safe history availability

- See `a4_history_support.csv` for support by level, has-any flag and quality flag (<10 suppressed).
- Quality flags overall: {"ambiguous_prior_institutional_outcome": 14, "ok": 201297}.

## Interpretation and limitations (as executed)

- Only observed history within survey recall is captured; a missing earlier institutional record is not evidence of no lifetime prior cesarean.
- Facility type is observed at delivery; historical selection effects are possible and no claim about necessity of cesareans is made.
- The survey-window definition follows the frozen C1 cohort (b19 0-59); birth dates (b3) are used only for ordering.
- NFHS-4 facility-detail harmonization is not implemented because the file is not available in the authorized environment.
- Point estimates only; no intervals or significance; V1 exposure caveat applies to any later A7 test scoring.

