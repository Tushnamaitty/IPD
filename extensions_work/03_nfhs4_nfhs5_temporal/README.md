# Extension 3: NFHS-4 versus NFHS-5

This corrected notebook compares the adjusted private versus public Cesarean association separately in NFHS-4 and NFHS-5. It subtracts the wave-specific risk differences. Each wave uses its own weights, clusters, strata, cross-fitted models, and covariate distribution. The change describes adjusted associations, not a sector-switch or policy effect.

## Files and folder layout

Copy this package's extensions_work/03_nfhs4_nfhs5_temporal/notebook.ipynb and state_crosswalk.py into C:\Users\Tushna\IPD\extensions_work\03_nfhs4_nfhs5_temporal\. Replace the old notebook after retaining a backup if needed. Keep the existing extensions_work/shared/config.py and utils.py. This package does not contain shared files or outputs.

If your current directory is extensions_work/extensions_work/, move that inner directory's contents up one level so that shared and 03_nfhs4_nfhs5_temporal are direct children of the repository's extensions_work folder. Run only the new notebook, not a duplicate in the inner folder.

The project should contain:

| Purpose | Relative path |
| --- | --- |
| NFHS-4 raw BR | data/raw/nfhs4/IABR74FL.DTA |
| NFHS-5 frozen V2 cohort | data/processed/df_model_v2.parquet |
| Shared code | extensions_work/shared/config.py and utils.py |
| New Extension 3 | extensions_work/03_nfhs4_nfhs5_temporal/notebook.ipynb and state_crosswalk.py |

## Run from PowerShell at the IPD root

    Test-Path .\extensions_work\shared\config.py
    Test-Path .\extensions_work\03_nfhs4_nfhs5_temporal\state_crosswalk.py
    Test-Path .\data\raw\nfhs4\IABR74FL.DTA
    Test-Path .\data\processed\df_model_v2.parquet
    python -m pip install -r requirements.txt
    python -m pip install pyreadstat jupyterlab
    jupyter lab .\extensions_work\03_nfhs4_nfhs5_temporal\notebook.ipynb

All four Test-Path results must be True. In Jupyter choose the project Python kernel, select Run All, then save the executed notebook. If jupyter is not on PATH, use python -m jupyter lab with the same notebook path. The notebook reads only 14 NFHS-4 DTA columns, saves data/processed/df_model_nfhs4.parquet locally, and saves aggregate results in extensions_work/03_nfhs4_nfhs5_temporal/outputs/. Do not commit or share raw or respondent-level data.

## Expected cohort counts

| Round | Institutional | Public | Private group | Other | Public/private analysis |
| --- | ---: | ---: | ---: | ---: | ---: |
| NFHS-4 | 195,997 | 141,028 | 54,338 | 631 | 195,366 |
| NFHS-5 | 201,311 | 150,299 | 50,495 | 517 | 200,794 |

The notebook asserts these counts and checks NFHS-4 institutional m17 codes. Numeric v024 codes differ across waves. The state map creates 35 common geographies, combining Jammu and Kashmir with Ladakh and the Dadra/Nagar Haveli and Daman/Diu territories. Original codes are retained.

The primary model adjusts for birth_order, wealth_index, education_years, residence, religion, social_group, twin_order, state. The separate six-variable sensitivity omits social_group and state in both waves and refits the models. Never present this sensitivity as the primary estimate.

## Checks before publication

- m15, m17, s116, and v024 labels were checked against local metadata. Compare the remaining definitions, special codes, and survey design to both round codebooks, especially v133, v130, v190, v025, bord, b0, v001, v005, and v022.
- Inspect weighted rates, propensity overlap, extreme weights, and balance diagnostics in each wave. Running successfully does not establish overlap or remove unmeasured sector differences.
- Percentile intervals resample PSUs within recorded strata with nuisance predictions held fixed. These are provisional conditional intervals. A survey methods specialist must review singleton strata, design variance, and full uncertainty before publication claims.
- The two adjusted gaps are standardized to different wave-specific populations. Their difference can reflect composition shifts and is not a policy effect.
- Only use figures and tables from a successful authorized real-data run of this corrected notebook. This package contains no saved results.
