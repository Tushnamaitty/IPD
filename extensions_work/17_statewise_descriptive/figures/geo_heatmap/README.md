# NFHS-5 state/UT geographic heat map

This folder contains a **candidate** descriptive choropleth and an independent geographic-comparison audit for the Professor Murthy manuscript.

## Inputs

- Survey aggregates (committed): `../../outputs/statewise_descriptive/statewise_primary_results.csv`
- NWIC boundary reference (local, **not committed**): `C:\Users\Tushna\Downloads\NWIC_Boundaries\state_NWIC.GeoJSON`
- Historical comparison boundary (local): `india_states.geojson`

The government-reference dataset was downloaded separately and is attributed in its metadata to the Survey of India. Confirm provenance, license, boundary vintage, and legal suitability before formal publication.

## Run locally

From the IPD repository root:

```powershell
.\.venv\Scripts\python.exe extensions_work\17_statewise_descriptive\figures\geo_heatmap\verify_nfhs5_boundaries_all36.py
.\.venv\Scripts\python.exe extensions_work\17_statewise_descriptive\figures\geo_heatmap\plot_geo_heatmap_government.py
```

Both scripts use existing aggregates; they do not process or disclose restricted individual NFHS data.

## Included audit outputs

- `boundary_verification_summary.json`
- `state_boundary_comparison.csv`

Comparison matched all 36 NFHS-5 state/UT names after documented aliases. It quantifies polygon similarity, **not** definitive certification of historical or legal boundaries. Important mismatches with the older third-party shapes exist, including Puducherry and the merged Dadra and Nagar Haveli/Daman and Diu UT.

## Figure use

The figure maps *unadjusted, survey-weighted descriptive proportions* for NFHS-5 narrow facility codes `m15=21` and `m15=31`. Values are not causal effects and are not adjusted national associations. The private Ladakh cell has one birth and is intentionally greyed/hatched. Small-sample flags remain essential.

The generated PNG/PDF are local review artifacts; upload them separately after final visual confirmation. The government reference GeoJSON itself is intentionally omitted from GitHub.
