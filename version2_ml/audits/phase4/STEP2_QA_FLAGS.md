# Phase 4 Step 2 — QA flags carried forward (2026-10-15)

**Flag S1: facility code m15=96 ("other")** — In aggregate split report 517 births (356 train, 88 validation, 73 test) and **zero observed cesareans**. Before modeling, verify raw m17 coding/eligibility for these births, and report an explicit sensitivity excluding this ambiguous category. Do not assume this is a valid negative category based only on zero positives. All models with primary broad cohort need this flagged.

**Flag S2: small state / UT category** — One category suppressed due to at least one partition with <30 births. Before per-state scoring or A6 subgroup claims, predefine disclosure thresholds, minimum positive and negative events, and eligible state holdout rules; do not fabricate state-specific results.

**Design caveat:** Counts of stratum / PSU pairs indicate representation, NOT PSU disjointness or valid survey variance. Design-based interval construction requires separate validated methods.

**Follow-up requirement:** Add aggregate facility96 outcome cross-tab with explicit raw m17 levels and missingness; small-area suppression audit. All retained checks must avoid public microdata or identifying group-level details.
