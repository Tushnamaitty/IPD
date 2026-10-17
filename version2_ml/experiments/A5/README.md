# A5 — NFHS-4 → NFHS-5 source-only temporal transfer

Status: **EXPLORATORY development** (feature tier FT). Not confirmatory; no confidence intervals.

Question: how much classification performance does a model trained only on NFHS-4 retain or lose when
scored zero-shot on NFHS-5, using the eight harmonized common-core features?

## Layout

| File | Purpose |
|---|---|
| `A5_temporal_transfer.py` | `SourceOnlyTransferModel` (NFHS-4-only fit, NFHS-4 validation threshold, frozen + hashed), `fit_source_model`, `zero_shot_pipeline`, `harmonize_core`, `harmonization_report`, `AdaptationRecalibrator` |
| `config.yaml` | Splits, models, sensitivities, optional adaptation arm, artifact folder |
| `run.py` | Synthetic tests → NFHS-4 fit/freeze → NFHS-5 zero-shot (B) → within-NFHS-5 reference (C) → recalibration (D) → diagnostics |
| `tests/test_a5_transfer.py` | Source-only fit, target-label mutation invariance, FT schema, unseen categories, frozen-model drift, adaptation disjointness |
| `outputs/aggregate/` | Aggregate outputs only |

## Run (from repo root)

```
python -m unittest discover -s version2_ml/experiments/A5/tests -p "test_*.py" -v
python version2_ml/experiments/A5/run.py                  # NFHS-5 test partition untouched
python version2_ml/experiments/A5/run.py --evaluate-test  # also scores the provisional NFHS-5 test partition
```

Requires both authorized local DTA files (gitignored). Frozen source models are written to the gitignored
`artifacts/A5/`; only their SHA-256 hashes enter the manifest.

## Design

- **Source split:** NFHS-4 mother-grouped 70/15/15 (seed 42): train fits preprocessing and model; validation
  chooses the max-F1 threshold and the primary model (highest ROC-AUC; tie → logistic regression); test is a
  clean source holdout.
- **Comparisons:** (A) NFHS-4 validation / holdout; (B) frozen model zero-shot on the shared NFHS-5 validation
  partition and on all non-test NFHS-5 rows; (C) NFHS-5 within-wave model on the same validation rows (context
  only; must reproduce A2 B0 exactly); (D) logistic recalibration fitted on NFHS-5 train mothers and evaluated on
  disjoint NFHS-5 validation mothers — target-supervised, reported separately.
- **Sensitivities for B:** NFHS-5 CMC 0..59 (drops the 1,634 records with v008 − b3 = 60), excluding m15 = 96,
  excluding the NFHS-4/NFHS-5 calendar overlap; v005-weighted descriptive point metrics.
- **Harmonization:** fixed codebook vocabularies per categorical feature (out-of-vocabulary values fail closed);
  in-vocabulary levels unseen in NFHS-4 training encode as all-zero one-hot and are counted.
- **Leakage guards:** `fit` and `select_threshold` refuse non-NFHS-4 rows and wrong partitions; models are frozen
  and hashed before NFHS-5 is loaded; `predict` re-checks the hash; the runner re-runs the full source pipeline with
  permuted and removed NFHS-5 labels and requires bitwise-identical predictions.
