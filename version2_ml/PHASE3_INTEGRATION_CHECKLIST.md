# IPD Version 2 — Phase 3 integration checklist (provisional)

## Scope

Synthetic integration combines the previously implemented **data contracts**,
**mother-grouped partitioning**, **cross-wave mapping/preprocessing**, **train-only
model fitting**, and **aggregate evaluation**. It does not read the raw NFHS files.

## Run locally

From repository root:

```powershell
git pull origin main
python -m unittest discover -s version2_ml/tests -p "test_*.py" -v
python version2_ml/validate_configs.py
```

Use the console test total and failures as evidence. Do not claim a pass until
the user runs these tests and shares the output.

## Limitations that block a full real-data infrastructure freeze

1. **Cohorts**: confirm the NFHS-5 60-month boundary rule and official eligibility; retain original full V1 cohort for historical reproduction.
2. **Features**: finalize all 36 raw-to-feature definitions, NFHS-4/5 questionnaire/skip universes, pregnancy-time availability, verified age/parity derivations, and previous-CS observed-history limitations.
3. **Splits**: define and freeze actual proportions and IDs before fitting; inspect class/balance/geographic support on real NFHS. V1 final-test metrics were previously examined, so no automatic claim of independent confirmatory final test.
4. **Leakage controls**: partition string guards alone cannot prove provenance; protect original records and fitted transformations from accidental reuse.
5. **Models**: reproduce F0 baseline on matched evaluation rows; establish validation-only selection, model/calibration procedures and fixed final evaluation design.
6. **Inference**: paired stratified PSU bootstrap/replicate construction, stratification and singleton handling remain **NOT STATISTICALLY VALIDATED**; no inferential CI claims based on synthetic unit tests.
7. **Privacy**: keep licensed microdata, per-row predictions, identifiers and small cells out of public GitHub.

## Completion labels

- **Phase 3 synthetic scaffold:** ready for completion review **only if all local tests pass**.
- **Phase 3 real-data infrastructure freeze:** **BLOCKED** until the above items are resolved.
- **Phase 4:** real-data pilot, baseline reproduction, and design-based statistical validation are next, before teammate experiment release.
