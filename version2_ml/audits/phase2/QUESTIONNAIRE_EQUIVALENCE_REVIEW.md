# Phase 2: corrected seven mappings — documentary review

**Reviewed:** 2026-10-15. **Status:** Labels strongly supported; skip-universe equivalence **NOT** signed off.

Sources:
- NFHS-4 REC94 official World Bank microdata dictionary: https://microdata.worldbank.org/index.php/catalog/2949/data-dictionary/F31
- NFHS-5 REC94 official World Bank microdata dictionary: https://microdata.worldbank.org/index.php/catalog/4482/data-dictionary/F34
- NFHS-4 counselling questionnaire item 419: https://microdata.worldbank.org/catalog/3110/variable/W_ANCARECOM_ALL ; documented **most recent birth only**.

Matching public dictionary descriptions:
- NFHS-4 s419a-e ↔ NFHS-5 s420a-e: informed about vaginal bleeding, convulsions, prolonged labour, severe abdominal pain and high blood pressure.
- NFHS-4 s432 ↔ NFHS-5 s434: pregnancy convulsions not caused by fever.
- NFHS-4 s433 ↔ NFHS-5 s435: swelling of legs/body/face.

**Critical exposure timing:** Some variables are retrospective reports about the index pregnancy, while age/BMI are interview-time, and counselling information may be **conditionally collected for most recent birth only**. Differential missingness found in Step 9 suggests missingness by eligibility/recall is plausible. An absent value is NOT no. For whole-institutional-cohort modeling, do not use the seven as fully supported FT features until routing constraints have been examined. Some may need most-recent-birth subgroup analyses instead.

Next local check: `questionnaire_skip_audit.py` reports raw item missingness by `bidx=1` vs `bidx>1` separately in both waves. This is **descriptive evidence**; full questionnaire routing and remaining 29 mapped features still require independent review.

No licensed microdata or row-level predictions may be published.
