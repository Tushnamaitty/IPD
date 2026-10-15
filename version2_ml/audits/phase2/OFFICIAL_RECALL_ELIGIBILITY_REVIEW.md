# Phase 2 official recall-window evidence review

**Date:** 2026-10-15. **Status:** Official five-year definition documented, local cohort sign-off pending b19 audit.

## Documentary evidence

The DHS-7 Guide to DHS Statistics explicitly uses `b19 < 60` to select births in the five years before the interview. Older surveys lacking b19 use `v008-b3 < 60`. This excludes completed month 60. Source: https://preview.dhsprogram.com/pubs/pdf/DHSG1/Guide_to_DHS_Statistics_DHS-7_v2.pdf

DHS older recode documentation defines five-year maternity history as months **0–59**, explicitly excluding month 60. Source: https://dhsprogram.com/pubs/pdf/dhsg4/recode2dhs.pdf

NFHS-5 BR dictionary explicitly lists `b19` (current age of child in months / months since birth for deceased children): https://microdata.worldbank.org/catalog/4482/data-dictionary/F14

## Our verified local aggregates

Broad institutional NFHS-4 = 195,997, NFHS-5 = 201,311. In prior CMC audit NFHS-5 `v008-b3=60`: 1,634; NFHS-4: 0. NFHS-5 CMC `0..59`: 199,677. Do **not** assert that 199,677 must equal the final DHS-7 b19-defined count until checking the b19 values; day-based calculations can diverge from CMC.

## Policy

- Preserve historical F0/V1 full NFHS-5 cohort 201,311 for faithful reproduction.
- For harmonized 5-year transfer, select `b19<60` **if b19 is available and valid**; otherwise documented `0<=v008-b3<60` fallback, with wave-specific provenance.
- Before freezing exact cohort IDs: run `official_five_year_eligibility.py`, compare b19 and CMC, verify questionnaire/denominators, and choose matched comparison subsets. Boundary inclusion sensitivity is planned.
- All F0-vs-F1-vs-FT comparisons of model performance require matched evaluation rows.
- The **remaining 29-feature semantic/time harmonization is not yet complete**; separate from the five-year boundary sign-off.
- Survey design/paired bootstrap and independent final-test claims remain Phase 4 work.
