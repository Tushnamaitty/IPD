# IPD source ↔ output ↔ PDF consistency audit — 11 October 2026

**Audit level:** connected GitHub repository file/aggregate metadata + user-provided PDFs. **Not** execution of licensed NFHS raw microdata, package/environment replay, or automated full 237-file code semantics validation. Base repo head reviewed `8082ede`; README updated afterward. Distinguish verified matches from unresolved final sign-off.

## Verified comparisons
| Check | Finding | Basis |
|---|---|---|
| Facility codes | PASS: primary narrow m15=21 vs 31 throughout October 3/7/9 updates and primary national/state JSON | national/state summary; PDFs |
| Outcome/recall | PASS on recorded definitions: m17, birth CMC within 60 months | national/state summary + October 3 PDF |
| NFHS-4 narrow all births | PASS: 114263, observed 22.417→22.42 pp, adjusted 19.16554→19.17 | national JSON vs October 3 p.1 |
| NFHS-5 narrow all births | PASS: 115197, observed 25.155→25.16 pp, adjusted 23.75678→23.76 | same |
| NFHS-4 first delivery | PASS: 52851, observed 23.616→23.62 pp, adjusted 21.20783→21.21 | same |
| NFHS-5 first delivery | PASS: 51854, observed 25.61697→25.62 pp, adjusted 25.237→25.24 | same |
| National CIs | PASS: all eight observed/adjusted gap lower/upper limits match October 3 p.1 at two decimals | national JSON vs October 3 PDF |
| State national reproduction | PASS as internally reported by statewise summary: national proportions/gaps differences ~0 to 4e-14 pp; SE equal within precision | statewise summary |
| Imputation point changes | PASS at documented report level: NFHS-4 -0.007/-0.018; NFHS-5 -0.004/-0.004 pp; no independent microdata rerun | October 7 p.1 |
| Temporal shared reference | DISTINCT ESTIMAND: +4.66 and +4.00 pp; not own-wave national differences | October 3 p.2 and shared-reference output provenance |
| State plots | FILES PRESENT: all-birth and first-delivery PNG/PDF, plot source, tables; plotted-content pixel-by-pixel confirmation not performed | GitHub tree and Oct 9 PDF |
| Federated | MATCHING documented numerical full-refit gap: 23.7567805 and pooled gap match to < 1e-6 pp; computational prototype only | federated extension summary + October 3 p.4 |

## Identified mismatches / qualifications
1. **Outdated root README** portrayed the original 201311-row prediction model as present main study. Updated to lead with current narrow-facility aim and archive original text.
2. **Different cohorts are intentionally different**: original broad 200794, earlier ML 201311; narrow NFHS-5 115197; narrow NFHS-4 114263. Do not mix headline risk differences or denominators.
3. **Different references**: own-wave national 19.17 vs 23.76 does not estimate the exact same target as shared-reference temporal +4.66. Exclude boundary-changing geographies only for harmonized cross-wave work; statewise descriptive retains wave-specific geography.
4. **Conditionally approximated intervals**, missing exact complete multistage design, and unresolved 28-PSU household-roster discrepancy. Older period variance key explicitly non-equivalent.
5. **Review-status flags remain** in saved aggregate JSON (`completed_*_require_review`). Statuses should not be relabeled as independently verified.
6. **First delivery is a subgroup**, not universally a woman-level singleton record; twins are retained.
7. **Period, policy, historical and heterogeneity analyses** are exploratory; no causal/time-policy language or generalized state ranking warranted.
8. **Professor's last presentation request**: state figure main text, detailed tables supplement; optional WHO-style geographic heat map. Manuscript placement itself not verifiable from aggregate repo.

## Checks requiring local microdata or additional packaging
- Re-execute exact cohort selection from the two original DHS Birth Recode files and independently recompute each reported result/SE.
- Freeze source input checksums, deterministic environment/versions, raw-data access instructions and complete table-producing commands.
- Match corrected state/stratum/PSU frame and primary interval versions, especially policy-period results.
- Inspect graphic exported content at publication resolution and ensure complete source pointers/captions and sparse flags.
- Reconcile all manuscript/supplement numbers after draft is prepared. No complete manuscript was part of this audit.

**Conclusion:** No contradiction found among the checked primary national headline estimates and the October 3 update. Strong documented methodological alignment with Murthy. Full numerical certification and final publication uncertainty sign-off remain open.
