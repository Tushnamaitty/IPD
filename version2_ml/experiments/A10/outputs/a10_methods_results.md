# A10 methods and results: V2 subgroup checks

- source: A9 V2 (protocol `0ce1d48`, A9 commit `f2a90c7`), seeds [11, 22, 33, 44, 55]
- predictions: verified deterministic re-derivation. Per seed, the split hash, membership digests, V2 thresholds and all 11 V2 test metrics stored in the A9 guard were reproduced exactly (55 values); A9 results are unchanged.
- repo_commit at run: `f2a90c732b90e3942d0746e634739954af54e768`; generated_at_utc: 2026-11-02T12:25:57+00:00
- pooled test rows: 150,810 birth-test observations (111,573 unique births; a birth can be in more than one seed's test partition)
- all pooled test rows: ROC-AUC 0.8486, AP 0.6475, slope 1.007, intercept -0.011
- PSU bootstrap within each seed's test partition, B = 1,000, seed 2028; percentile 95% CIs.

Within-subgroup ROC-AUC is usually lower than the all-rows ROC-AUC because between-group differences in risk are removed. **A flag is a reporting requirement, not a model failure.**

## Predeclared flags

- LOW_DISCRIMINATION: ROC-AUC < 0.70
- MISCALIBRATED: |calibration intercept| > 0.10 or slope outside 0.8-1.2
- UNDER_SUPPORTED: < 200 births or < 30 per class in the pooled test rows (not estimable)

## Flag table

| dimension | group | flag | n | events | evidence |
| --- | --- | --- | --- | --- | --- |
| facility_sector | other | UNDER_SUPPORTED | 406 | 0 | support screen requires >= 200 births and >= 30 per class; not estimable |
| wealth | 1 | MISCALIBRATED | 34290 | 3062 | intercept +0.122 (limit |.| <= 0.10); slope 1.048 (limit 0.8-1.2) |
| wealth | 2 | MISCALIBRATED | 34584 | 5361 | intercept +0.101 (limit |.| <= 0.10); slope 1.066 (limit 0.8-1.2) |
| maternal_age | 35+ | MISCALIBRATED | 5775 | 1522 | intercept +0.128 (limit |.| <= 0.10); slope 1.036 (limit 0.8-1.2) |
| previous_cs_history | higher_order_no_observed_previous_institutional_cs | MISCALIBRATED | 26380 | 1375 | intercept +0.301 (limit |.| <= 0.10); slope 1.088 (limit 0.8-1.2) |
| previous_cs_history | unknown_history | UNDER_SUPPORTED | <10 (suppressed) | <10 (suppressed) | support screen requires >= 200 births and >= 30 per class; not estimable |
| state | andaman & nicobar islands | MISCALIBRATED | 401 | 89 | intercept +0.668 (limit |.| <= 0.10); slope 1.501 (limit 0.8-1.2) |
| state | andhra pradesh | MISCALIBRATED | 2094 | 904 | intercept +0.105 (limit |.| <= 0.10); slope 0.967 (limit 0.8-1.2) |
| state | arunachal pradesh | LOW_DISCRIMINATION | 3216 | 613 | ROC-AUC 0.6517 (95% CI 0.6251-0.6775) < 0.70 |
| state | arunachal pradesh | MISCALIBRATED | 3216 | 613 | intercept -0.419 (limit |.| <= 0.10); slope 0.628 (limit 0.8-1.2) |
| state | assam | MISCALIBRATED | 6673 | 1382 | intercept +0.214 (limit |.| <= 0.10); slope 1.031 (limit 0.8-1.2) |
| state | chandigarh | UNDER_SUPPORTED | 91 | 16 | support screen requires >= 200 births and >= 30 per class; not estimable |
| state | dadra & nagar haveli and daman & diu | MISCALIBRATED | 694 | 187 | intercept +0.192 (limit |.| <= 0.10); slope 1.123 (limit 0.8-1.2) |
| state | goa | MISCALIBRATED | 360 | 148 | intercept -0.088 (limit |.| <= 0.10); slope 0.739 (limit 0.8-1.2) |
| state | gujarat | MISCALIBRATED | 6539 | 1384 | intercept -0.121 (limit |.| <= 0.10); slope 0.976 (limit 0.8-1.2) |
| state | haryana | MISCALIBRATED | 4866 | 987 | intercept -0.237 (limit |.| <= 0.10); slope 0.908 (limit 0.8-1.2) |
| state | himachal pradesh | MISCALIBRATED | 1768 | 413 | intercept -0.127 (limit |.| <= 0.10); slope 1.080 (limit 0.8-1.2) |
| state | jharkhand | MISCALIBRATED | 5478 | 901 | intercept +0.194 (limit |.| <= 0.10); slope 1.145 (limit 0.8-1.2) |
| state | kerala | LOW_DISCRIMINATION | 2258 | 872 | ROC-AUC 0.6870 (95% CI 0.6646-0.7100) < 0.70 |
| state | kerala | MISCALIBRATED | 2258 | 872 | intercept -0.206 (limit |.| <= 0.10); slope 0.857 (limit 0.8-1.2) |
| state | ladakh | LOW_DISCRIMINATION | 379 | 149 | ROC-AUC 0.6237 (95% CI 0.5527-0.6914) < 0.70 |
| state | ladakh | MISCALIBRATED | 379 | 149 | intercept +0.032 (limit |.| <= 0.10); slope 0.538 (limit 0.8-1.2) |
| state | lakshadweep | LOW_DISCRIMINATION | 230 | 80 | ROC-AUC 0.6243 (95% CI 0.5352-0.6926) < 0.70 |
| state | lakshadweep | MISCALIBRATED | 230 | 80 | intercept -0.198 (limit |.| <= 0.10); slope 0.496 (limit 0.8-1.2) |
| state | madhya pradesh | MISCALIBRATED | 10688 | 1238 | intercept +0.127 (limit |.| <= 0.10); slope 1.193 (limit 0.8-1.2) |
| state | maharashtra | MISCALIBRATED | 6808 | 1671 | intercept -0.230 (limit |.| <= 0.10); slope 0.860 (limit 0.8-1.2) |
| state | manipur | MISCALIBRATED | 1483 | 392 | intercept -0.103 (limit |.| <= 0.10); slope 0.901 (limit 0.8-1.2) |
| state | meghalaya | MISCALIBRATED | 2792 | 310 | intercept -0.477 (limit |.| <= 0.10); slope 0.819 (limit 0.8-1.2) |
| state | mizoram | MISCALIBRATED | 1591 | 161 | intercept -0.431 (limit |.| <= 0.10); slope 0.955 (limit 0.8-1.2) |
| state | nagaland | MISCALIBRATED | 962 | 83 | intercept -0.760 (limit |.| <= 0.10); slope 0.952 (limit 0.8-1.2) |
| state | nct of delhi | MISCALIBRATED | 2279 | 591 | intercept -0.359 (limit |.| <= 0.10); slope 0.877 (limit 0.8-1.2) |
| state | odisha | MISCALIBRATED | 5936 | 1211 | intercept +0.420 (limit |.| <= 0.10); slope 1.162 (limit 0.8-1.2) |
| state | puducherry | MISCALIBRATED | 503 | 246 | intercept +0.342 (limit |.| <= 0.10); slope 0.819 (limit 0.8-1.2) |
| state | rajasthan | MISCALIBRATED | 10846 | 1057 | intercept -0.154 (limit |.| <= 0.10); slope 1.005 (limit 0.8-1.2) |
| state | sikkim | MISCALIBRATED | 404 | 144 | intercept +0.530 (limit |.| <= 0.10); slope 0.915 (limit 0.8-1.2) |
| state | tripura | MISCALIBRATED | 1270 | 314 | intercept +0.642 (limit |.| <= 0.10); slope 1.254 (limit 0.8-1.2) |

## ROC-AUC gaps within dimensions (largest minus smallest supported group)

| dimension | groups | max (group) | min (group) | gap (95% CI) |
| --- | --- | --- | --- | --- |
| residence | 2 | 0.8572 (rural) | 0.7944 (urban) | 0.0629 (0.0568-0.0693) |
| facility_sector | 2 | 0.8246 (public) | 0.7499 (private) | 0.0748 (0.0682-0.0813) |
| birth_order | 2 | 0.8793 (second_or_later) | 0.7934 (first_birth) | 0.0859 (0.0813-0.0906) |
| wealth | 5 | 0.8431 (2) | 0.7675 (5) | 0.0756 (0.0690-0.0855) |
| maternal_age | 3 | 0.8678 (35+) | 0.8066 (<20) | 0.0612 (0.0476-0.0735) |
| previous_cs_history | 4 | 0.8299 (higher_order_earlier_institutional_history_unobserved) | 0.7440 (observed_previous_institutional_cs) | 0.0859 (0.0627-0.1108) |
| state | 35 | 0.8899 (bihar) | 0.6237 (ladakh) | 0.2663 (0.2370-0.3621) |

## Subgroup metrics (pooled; 95% CI; per-seed ROC-AUC range)

| dimension | group | n | events | ROC-AUC (95% CI) | seeds | AP | Brier | intercept | slope | obs / pred | recall | precision | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residence | urban | 33170 | 10836 | 0.794 (0.789-0.800) | 0.787-0.800 | 0.670 | 0.166 | -0.067 | 0.927 | 0.327 / 0.331 | 0.761 | 0.529 | none |
| residence | rural | 117640 | 22624 | 0.857 (0.854-0.860) | 0.854-0.861 | 0.635 | 0.106 | +0.022 | 1.035 | 0.192 / 0.194 | 0.658 | 0.531 | none |
| facility_sector | public | 112605 | 15615 | 0.825 (0.821-0.828) | 0.819-0.828 | 0.499 | 0.092 | +0.081 | 1.045 | 0.139 / 0.137 | 0.427 | 0.497 | none |
| facility_sector | private | 37799 | 17845 | 0.750 (0.744-0.755) | 0.742-0.760 | 0.742 | 0.201 | -0.059 | 1.010 | 0.472 / 0.484 | 0.922 | 0.545 | none |
| facility_sector | other | 406 | 0 | not estimable | | | | | | | | | UNDER_SUPPORTED |
| birth_order | first_birth | 61843 | 16516 | 0.793 (0.789-0.798) | 0.787-0.797 | 0.580 | 0.153 | -0.039 | 0.993 | 0.267 / 0.272 | 0.709 | 0.490 | none |
| birth_order | second_or_later | 88967 | 16944 | 0.879 (0.876-0.882) | 0.876-0.884 | 0.694 | 0.096 | +0.021 | 1.019 | 0.190 / 0.190 | 0.673 | 0.579 | none |
| wealth | 1 | 34290 | 3062 | 0.842 (0.834-0.850) | 0.828-0.855 | 0.458 | 0.063 | +0.122 | 1.048 | 0.089 / 0.087 | 0.389 | 0.501 | MISCALIBRATED |
| wealth | 2 | 34584 | 5361 | 0.843 (0.837-0.849) | 0.827-0.851 | 0.572 | 0.095 | +0.101 | 1.066 | 0.155 / 0.154 | 0.531 | 0.524 | MISCALIBRATED |
| wealth | 3 | 30917 | 7464 | 0.834 (0.828-0.840) | 0.826-0.840 | 0.658 | 0.128 | -0.008 | 1.046 | 0.241 / 0.248 | 0.684 | 0.529 | none |
| wealth | 4 | 28267 | 8756 | 0.815 (0.809-0.821) | 0.808-0.821 | 0.688 | 0.154 | -0.005 | 1.003 | 0.310 / 0.311 | 0.758 | 0.541 | none |
| wealth | 5 | 22752 | 8817 | 0.767 (0.761-0.774) | 0.760-0.779 | 0.685 | 0.187 | -0.074 | 0.920 | 0.388 / 0.397 | 0.833 | 0.529 | none |
| maternal_age | <20 | 17178 | 2833 | 0.807 (0.798-0.815) | 0.800-0.812 | 0.500 | 0.109 | -0.093 | 1.006 | 0.165 / 0.176 | 0.555 | 0.444 | none |
| maternal_age | 20-34 | 127857 | 29105 | 0.851 (0.848-0.854) | 0.846-0.855 | 0.656 | 0.121 | -0.010 | 1.003 | 0.228 / 0.229 | 0.701 | 0.535 | none |
| maternal_age | 35+ | 5775 | 1522 | 0.868 (0.857-0.878) | 0.863-0.873 | 0.714 | 0.124 | +0.128 | 1.036 | 0.264 / 0.251 | 0.751 | 0.596 | MISCALIBRATED |
| previous_cs_history | first_recorded_live_birth | 61843 | 16516 | 0.793 (0.789-0.798) | 0.787-0.797 | 0.580 | 0.153 | -0.039 | 0.993 | 0.267 / 0.272 | 0.709 | 0.490 | none |
| previous_cs_history | observed_previous_institutional_cs | 4616 | 4095 | 0.744 (0.720-0.767) | 0.732-0.761 | 0.952 | 0.088 | -0.034 | 1.061 | 0.887 / 0.881 | 1.000 | 0.890 | none |
| previous_cs_history | higher_order_no_observed_previous_institutional_cs | 26380 | 1375 | 0.791 (0.779-0.803) | 0.783-0.808 | 0.198 | 0.046 | +0.301 | 1.088 | 0.052 / 0.049 | 0.073 | 0.424 | MISCALIBRATED |
| previous_cs_history | higher_order_earlier_institutional_history_unobserved | 57966 | 11473 | 0.830 (0.826-0.834) | 0.825-0.836 | 0.551 | 0.120 | +0.011 | 1.028 | 0.198 / 0.200 | 0.629 | 0.486 | none |
| previous_cs_history | unknown_history | <10 (suppressed) | <10 (suppressed) | not estimable | | | | | | | | | UNDER_SUPPORTED |
| state | andaman & nicobar islands | 401 | 89 | 0.850 (0.792-0.895) | not estimable-not estimable | 0.663 | 0.124 | +0.668 | 1.501 | 0.222 / 0.207 | 0.517 | 0.561 | MISCALIBRATED |
| state | andhra pradesh | 2094 | 904 | 0.809 (0.790-0.828) | 0.776-0.844 | 0.785 | 0.172 | +0.105 | 0.967 | 0.432 / 0.412 | 0.813 | 0.635 | MISCALIBRATED |
| state | arunachal pradesh | 3216 | 613 | 0.652 (0.625-0.678) | 0.616-0.669 | 0.389 | 0.142 | -0.419 | 0.628 | 0.191 / 0.176 | 0.313 | 0.428 | LOW_DISCRIMINATION;MISCALIBRATED |
| state | assam | 6673 | 1382 | 0.823 (0.807-0.837) | 0.819-0.832 | 0.604 | 0.120 | +0.214 | 1.031 | 0.207 / 0.187 | 0.513 | 0.589 | MISCALIBRATED |
| state | bihar | 12346 | 1544 | 0.890 (0.880-0.899) | 0.876-0.903 | 0.592 | 0.073 | +0.050 | 1.094 | 0.125 / 0.130 | 0.718 | 0.485 | none |
| state | chandigarh | 91 | 16 | not estimable | | | | | | | | | UNDER_SUPPORTED |
| state | chhattisgarh | 5218 | 724 | 0.849 (0.829-0.868) | 0.842-0.879 | 0.558 | 0.086 | -0.055 | 1.077 | 0.139 / 0.153 | 0.590 | 0.512 | none |
| state | dadra & nagar haveli and daman & diu | 694 | 187 | 0.819 (0.779-0.855) | not estimable-not estimable | 0.666 | 0.142 | +0.192 | 1.123 | 0.269 / 0.256 | 0.684 | 0.557 | MISCALIBRATED |
| state | goa | 360 | 148 | 0.718 (0.662-0.771) | not estimable-not estimable | 0.663 | 0.211 | -0.088 | 0.739 | 0.411 / 0.415 | 0.851 | 0.519 | MISCALIBRATED |
| state | gujarat | 6539 | 1384 | 0.787 (0.774-0.801) | 0.773-0.802 | 0.547 | 0.133 | -0.121 | 0.976 | 0.212 / 0.225 | 0.612 | 0.438 | MISCALIBRATED |
| state | haryana | 4866 | 987 | 0.782 (0.765-0.798) | 0.760-0.810 | 0.499 | 0.132 | -0.237 | 0.908 | 0.203 / 0.222 | 0.620 | 0.417 | MISCALIBRATED |
| state | himachal pradesh | 1768 | 413 | 0.791 (0.767-0.813) | 0.773-0.807 | 0.579 | 0.140 | -0.127 | 1.080 | 0.234 / 0.262 | 0.637 | 0.466 | MISCALIBRATED |
| state | jammu & kashmir | 4065 | 1747 | 0.703 (0.684-0.720) | 0.686-0.724 | 0.662 | 0.213 | +0.042 | 0.825 | 0.430 / 0.407 | 0.823 | 0.516 | none |
| state | jharkhand | 5478 | 901 | 0.874 (0.861-0.888) | 0.852-0.888 | 0.590 | 0.094 | +0.194 | 1.145 | 0.164 / 0.162 | 0.684 | 0.544 | MISCALIBRATED |
| state | karnataka | 5958 | 1890 | 0.784 (0.770-0.798) | 0.771-0.803 | 0.644 | 0.168 | +0.008 | 0.884 | 0.317 / 0.303 | 0.647 | 0.555 | none |
| state | kerala | 2258 | 872 | 0.687 (0.665-0.710) | 0.676-0.699 | 0.633 | 0.211 | -0.206 | 0.857 | 0.386 / 0.421 | 0.813 | 0.450 | LOW_DISCRIMINATION;MISCALIBRATED |
| state | ladakh | 379 | 149 | 0.624 (0.553-0.691) | not estimable-not estimable | 0.534 | 0.236 | +0.032 | 0.538 | 0.393 / 0.311 | 0.584 | 0.489 | LOW_DISCRIMINATION;MISCALIBRATED |
| state | lakshadweep | 230 | 80 | 0.624 (0.535-0.693) | not estimable-not estimable | 0.508 | 0.224 | -0.198 | 0.496 | 0.348 / 0.314 | 0.550 | 0.431 | LOW_DISCRIMINATION;MISCALIBRATED |
| state | madhya pradesh | 10688 | 1238 | 0.866 (0.853-0.879) | 0.850-0.875 | 0.584 | 0.070 | +0.127 | 1.193 | 0.116 / 0.127 | 0.569 | 0.575 | MISCALIBRATED |
| state | maharashtra | 6808 | 1671 | 0.773 (0.759-0.788) | 0.762-0.784 | 0.551 | 0.151 | -0.230 | 0.860 | 0.245 / 0.263 | 0.642 | 0.443 | MISCALIBRATED |
| state | manipur | 1483 | 392 | 0.768 (0.739-0.794) | 0.748-0.782 | 0.585 | 0.157 | -0.103 | 0.901 | 0.264 / 0.267 | 0.622 | 0.469 | MISCALIBRATED |
| state | meghalaya | 2792 | 310 | 0.748 (0.711-0.785) | 0.692-0.792 | 0.379 | 0.086 | -0.477 | 0.819 | 0.111 / 0.128 | 0.400 | 0.389 | MISCALIBRATED |
| state | mizoram | 1591 | 161 | 0.788 (0.745-0.828) | 0.775-0.790 | 0.365 | 0.079 | -0.431 | 0.955 | 0.101 / 0.132 | 0.391 | 0.406 | MISCALIBRATED |
| state | nagaland | 962 | 83 | 0.777 (0.715-0.837) | not estimable-not estimable | 0.337 | 0.074 | -0.760 | 0.952 | 0.086 / 0.146 | 0.410 | 0.340 | MISCALIBRATED |
| state | nct of delhi | 2279 | 591 | 0.742 (0.720-0.762) | 0.722-0.762 | 0.509 | 0.167 | -0.359 | 0.877 | 0.259 / 0.306 | 0.689 | 0.445 | MISCALIBRATED |
| state | odisha | 5936 | 1211 | 0.859 (0.847-0.871) | 0.827-0.873 | 0.651 | 0.110 | +0.420 | 1.162 | 0.204 / 0.180 | 0.583 | 0.631 | MISCALIBRATED |
| state | puducherry | 503 | 246 | 0.730 (0.682-0.768) | not estimable-not estimable | 0.724 | 0.217 | +0.342 | 0.819 | 0.489 / 0.405 | 0.760 | 0.611 | MISCALIBRATED |
| state | punjab | 4093 | 1611 | 0.762 (0.745-0.779) | 0.744-0.779 | 0.672 | 0.191 | +0.017 | 0.947 | 0.394 / 0.387 | 0.788 | 0.551 | none |
| state | rajasthan | 10846 | 1057 | 0.827 (0.812-0.840) | 0.806-0.841 | 0.438 | 0.070 | -0.154 | 1.005 | 0.097 / 0.109 | 0.384 | 0.401 | MISCALIBRATED |
| state | sikkim | 404 | 144 | 0.733 (0.676-0.785) | not estimable-not estimable | 0.605 | 0.207 | +0.530 | 0.915 | 0.356 / 0.248 | 0.542 | 0.619 | MISCALIBRATED |
| state | tamil nadu | 4723 | 2168 | 0.769 (0.755-0.783) | 0.736-0.787 | 0.744 | 0.193 | +0.035 | 0.998 | 0.459 / 0.452 | 0.899 | 0.565 | none |
| state | telangana | 5097 | 3154 | 0.824 (0.811-0.836) | 0.800-0.843 | 0.878 | 0.163 | +0.051 | 0.991 | 0.619 / 0.611 | 0.960 | 0.705 | none |
| state | tripura | 1270 | 314 | 0.808 (0.773-0.842) | 0.742-0.881 | 0.610 | 0.145 | +0.642 | 1.254 | 0.247 / 0.200 | 0.516 | 0.628 | MISCALIBRATED |
| state | uttar pradesh | 22391 | 3297 | 0.859 (0.851-0.866) | 0.854-0.867 | 0.554 | 0.090 | -0.063 | 0.993 | 0.147 / 0.152 | 0.662 | 0.470 | none |
| state | uttarakhand | 2365 | 520 | 0.823 (0.801-0.843) | 0.786-0.871 | 0.567 | 0.129 | -0.009 | 1.107 | 0.220 / 0.234 | 0.654 | 0.528 | none |
| state | west bengal | 3945 | 1262 | 0.822 (0.808-0.839) | 0.805-0.849 | 0.716 | 0.147 | +0.092 | 0.983 | 0.320 / 0.305 | 0.672 | 0.637 | none |

## Re-derivation log

- seed 11: 11 metrics, thresholds, split hash and 3 membership digests matched A9
- seed 22: 11 metrics, thresholds, split hash and 3 membership digests matched A9
- seed 33: 11 metrics, thresholds, split hash and 3 membership digests matched A9
- seed 44: 11 metrics, thresholds, split hash and 3 membership digests matched A9
- seed 55: 11 metrics, thresholds, split hash and 3 membership digests matched A9

## Disclosure

- Aggregate outputs only; counts below 10 suppressed (births, events, non-events); no keys or per-birth predictions written.
- Unweighted; CIs ignore survey strata and are not design-based.
