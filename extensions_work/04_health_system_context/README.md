# Extension 4 - rural state health-system context

**Status: frozen-V2 run completed; RHS source-table audit passed; interpretation
review remains.** The current outputs were generated on the user's frozen
`df_model_v2.parquet` (input SHA-256 and package versions in
`outputs/context_metadata.json`). The notebook file supplied for review does
not retain execution counts or cell outputs; the generated CSV, JSON and PNG
files document the run.

The retained rural cohort has 156,583 birth records across 34 states. The
adjusted private-minus-public risk difference is 35.17 percentage points in
the lower-availability group and 28.17 in the higher-availability group. The
high-minus-low contrast is -7.00 pp, with a 500-draw conditional paired-PSU
bootstrap interval of -8.63 to -5.27 pp. These are associations in the
sampled context groups, not an effect of changing staffing or facility.

**State influence:** removing West Bengal changes the fixed-prediction
contrast to -2.10 pp; the full 34-state range is -9.34 to -2.10 pp. The sign
does not reverse, but the magnitude depends materially on that state. The
reported PSU interval conditions on the observed states and fitted nuisance
models; it does not capture uncertainty from resampling states or refitting
models. Discuss this sensitivity before using the contrast as a headline
manuscript result.

**Source audit (28 September 2026):** All 36 state/UT rows in
`local_source/rhs20-21_table23_transcribed.csv` were compared with Table 23,
printed page 138 (PDF page 161), of the Ministry of Health and Family Welfare's
*Rural Health Statistics 2020-21*. Required, sanctioned, and in-position
counts match, including `NA` and `N App`; all derived availability percentages
match the source counts. No numerical results need rerunning. The ministry's
previous PDF URL currently returns an error; a copy of the report is available
at https://wagtail.ruralindiaonline.org/en/library/resource/rural-health-statistics-2020-21/.

## Question

Among NFHS-5 rural-residence institutional deliveries, does the adjusted
private-minus-public Cesarean risk difference differ between states with lower
versus higher rural public CHC specialist availability? This is a contextual
association. The indicator does not measure the staffing of a woman's actual
delivery facility, and it may not precede every NFHS-5 delivery.

## Files

- `notebook.ipynb`: runs the analysis and checks generated output files.
- `analysis.py`: input audit, linkage, cross-fitting, AIPW, diagnostics,
  paired PSU bootstrap, leave-one-state-out analysis and output writing.
- `local_source/rhs20-21_table23_transcribed.csv`: aggregate transcription of
  MoHFW RHS 2020-21 Table 23, independently checked against the source PDF.
- `outputs/`: aggregate results regenerated on the frozen V2 file.

No changes to the frozen V2 notebooks, data, or output directories are needed.
`shared_config_change.patch` from the earlier package is unnecessary for this
self-contained extension.

## Run on the user's Windows laptop

1. Rename the existing `extensions_work/04_health_system_context` folder to
   `04_health_system_context_before_fix`. Keep it as a backup.
2. Put this new `04_health_system_context` folder inside `extensions_work`.
3. In the Python environment used for the other IPD extensions, ensure
   `pandas`, `numpy`, `scikit-learn`, `xgboost`, `pyarrow`, and `matplotlib`
   are installed. Open `extensions_work/04_health_system_context/notebook.ipynb`.
4. Run cells from top to bottom. The default input is
   `data/processed/df_model_v2.parquet` relative to the repository root.
   If needed, set `IPD_EXT04_V2_PATH` to the **actual frozen Parquet** path
   before running the first notebook cell. Do not use a reconstructed `.pkl`.
5. The modeling cell fits ten nuisance-model pairs (five folds in each of
   two groups); allow it to finish. The final cell checks four descriptive
   rows, 500 paired bootstrap rows and 34 leave-one-state-out rows.
6. Zip the new `04_health_system_context` folder with its generated outputs
   and send it back for independent review before publishing or pushing it.

The analysis refuses to run if previously generated result CSV/JSON/PNG files
are already present in `outputs/`. Move them to a backup folder before a
fresh rerun; do not copy the old outputs into this folder.

## What is computed

The frozen V2 confounders and weighted nuisance model specifications are
reproduced: respondent-grouped 5-fold cross-fitting within the two pre-fixed
context groups; logistic propensity, XGBoost S-learner, AIPW risk scores,
survey-weighted group means, and a joint 500-draw PSU bootstrap within survey
strata. The same PSU draw contributes to both context groups. Nuisance models
are held fixed in this conditional bootstrap. The external state-level median
is based on the 34 non-missing state indicators, not on birth frequencies.

The code reports the high-minus-low difference in the two adjusted sector
risk differences, overlap and measured-covariate balance, a fixed-propensity
trimming check, and a fixed-prediction leave-one-state-out check. The latter
is an influence diagnostic, not an independent confirmatory analysis.

## Review criteria

- Counts should be checked before fitting: V2 total 201,311;
  public/private 200,794; rural public/private 156,717; missing context 134;
  retained 156,583; 34 states; 17 lower and 17 higher states.
- `context_descriptives.csv` must contain all four group-by-sector rows. The
  weighted public and private shares must sum to 100% **within each context
  group**.
- `context_modifier_results.csv`, the plot and JSON must agree. Every bootstrap
  row must satisfy `Delta = RD_high - RD_low`; its percentile interval must
  come from the paired Delta column.
- Inspect extreme propensity counts, ESS and balance, the trimming result,
  and all 34 leave-one-state-out Deltas. Flag if one state strongly changes the
  sign or magnitude of the finding.
- The earlier reconstructed-V2 run reported low 35.17 pp, high 28.17 pp,
  Delta -7.00 pp, CI -8.88 to -5.17 pp. The current frozen-V2 point values
  agree to rounding, while its jointly regenerated bootstrap interval is
  -8.63 to -5.27 pp. Do not force a match to either set of numbers.
- Do not make claims about causation, actual facility staffing, all Indian
  states beyond those observed, or a uniformly pre-delivery exposure.

All generated outputs are aggregate. Do not share the restricted Parquet,
row-level predictions, respondent IDs, or PSU-linked records.
