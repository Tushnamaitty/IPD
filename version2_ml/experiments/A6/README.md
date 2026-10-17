# A6 — Geographic and public/private sector transport

Status: **EXPLORATORY development**. Not external clinical validation; no confidence intervals.

Questions: does performance degrade on completely unseen states, and when transporting public → private or
private → public?

## Layout

| File | Purpose |
|---|---|
| `A6_transport.py` | `SupportScreen`, `assert_mother_disjoint`, `mother_state_check`, `sector_overlap_policy`, `assert_sector_free`, `loso_masks`, `sector_masks`, `fit_predict_domain` |
| `config.yaml` | Support screen, arms, sector directions and overlap policy, models |
| `run.py` | Synthetic tests → NFHS-5 non-test rows → leave-one-state-out → sector directions → matrices and notes |
| `tests/test_a6_transport.py` | Zero mother overlap, target-outcome isolation (geographic, sector, TE), sector policy, constant/unseen features, support screen |
| `outputs/aggregate/` | Aggregate outputs only |

## Run (from repo root)

```
python -m unittest discover -s version2_ml/experiments/A6/tests -p "test_*.py" -v
python version2_ml/experiments/A6/run.py                  # shared non-test rows only
python version2_ml/experiments/A6/run.py --evaluate-test  # adds test rows as target-only evaluation rows
```

## Design

- **Rows:** the shared NFHS-5 mother-grouped split; only train + validation rows are used, so the provisional test
  partition stays untouched.
- **Geographic (leave-one-state-out):** each state passing the predeclared screen (n ≥ 200, ≥ 30 positives, ≥ 30
  negatives) is held out completely. Source = other states' train mothers (fit) and validation mothers (threshold,
  reference i); target = all held-out-state births (ii) and its validation mothers (ii-b, paired with the
  all-state reference iii). States failing the screen are NOT ESTIMABLE with counts suppressed; no regional pooling.
- **Geographic arms:** G0 eight core features (state-free); G_hist + A2 NFHS-4 historical state × sector rate;
  G_te + A3 OOF encoding fitted on source states only (held-out state → source sector fallback, asserted).
- **Sector:** public (m15 20–27) ↔ private (m15 30–33); m15 = 96 excluded; mothers with births in both sectors are
  excluded from both directions (prespecified); sector-free seven-feature set (facility_type rejected in code).
  Reference (iii) is a target-sector-trained model on the same target validation rows.
- **Guards:** `fit_predict_domain` fits only on source-train rows and thresholds on source-validation rows; target
  masks may not overlap source rows; mother disjointness asserted in every fold and direction; runtime checks
  flip target-domain labels and require bitwise-identical predictions.
