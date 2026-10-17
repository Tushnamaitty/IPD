# A6 metric schema

`metrics.csv`: one row per (design, comparison, domain, arm, model, eval_set).

| Column | Meaning |
|---|---|
| design | geographic (leave-one-state-out) or sector (public <-> private) |
| comparison | i_within_source; ii_cross_domain (all target non-test births); ii_b_cross_domain_paired_rows (target validation mothers); iii_* target-domain reference on the same rows as ii_b (context only) |
| source_domain, target_domain | domains whose labels trained the model / that were scored |
| arm | G0_ft_state_free (8 core), G_hist_nfhs4_context (+ NFHS-4 historical state x sector rate), G_te_source_fallback (+ source-only OOF TE; unseen state -> sector fallback), S_sector_free_seven (7 core, no facility_type) |
| support_status | ESTIMABLE or NOT_ESTIMABLE (n >= 200, >= 30 positives, >= 30 negatives) |
| n_births ... f1 | shared schema; threshold from source validation for ii/ii_b |
| confidence_interval_status | NOT_APPROVED |

`geographic_matrix.csv`: per held-out state x arm x model; delta vs source = held-out state minus within-source
validation; paired delta = state unseen (ii_b) minus state seen (iii) on identical validation rows.
`geographic_summary.csv`: distribution across estimable states. `sector_matrix.csv`: per direction x model.
`support_screen.csv`: screen result per state (counts suppressed when NOT_ESTIMABLE).
