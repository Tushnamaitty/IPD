# Remaining 29 F0 predictors: harmonization review gate

**Status: candidate only. Not fully harmonized.** Local aggregate check:
`remaining29_feature_audit.py`. Reviews completeness and birth-index
missingness in both waves, not text equivalence.

Prior evidence: F0 36 predictors, seven counselling/complication fields have
wave-dependent birth-index skip patterns. Primary full-birth FT excludes those
seven until questionnaire universe matches.

Remaining 29 feature list is embedded explicitly in the script. Review gates:
- **Interview-time measurements:** maternal_age `v012`, total_children_ever_born
  `v201`, BMI `v445`, current wealth index `v190`, current residence/
  education and other sociodemographics. Age-at-index may be constructed
  from `b3-v011`; `v201` excluded from timing-restricted F1 and FT.
- **Birth-recode history:** `bord`, `b11`, `b0`, `v212`. Birth order is
  individual live-birth order, not previous pregnancies; verify twin codes.
- **ANC/facility questions:** `m13,m14,m2*,m42*,m43,m15` eligibility and
  specific reference birth may vary. Age-at-delivery and facility variables
  must be reviewed for whether available at target prediction time.
- **Wave-relative constructs:** wealth quintile and state/social-group labels
  are not necessarily comparable absolute measurements.
- **Observed-code compatibility is insufficient:** compare variable labels,
  value labels, skip universes, questionnaire wording, and transformations,
  separately for all 29.

Do not freeze FT based solely on a 29/29 column-presence audit. Do not publish
licensed DHS microdata. Final feature sign-off requires a feature-by-feature
evidence ledger.
