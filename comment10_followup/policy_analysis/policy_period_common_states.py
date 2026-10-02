"""Comment 10 follow-up: NFHS-5 common-state, common-target period associations.

Place beside policy_period_analysis.py and state_crosswalk.py. Reads the local
NFHS-5 raw file; writes new aggregate outputs. No bootstrap or causal effects.
Uses the same survey-weighted pooled covariate distribution for every period.
This target is a selected 11-state analytic sample, not all India.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import policy_period_analysis as base

PERIODS = ['A_before_2016', 'B_2017_Jan_Nov', 'C_2018_Jan_2019_Sep',
           'D_2019_Nov_2020_Feb', 'E_2020_Mar_onward']
EXPECTED_STATES = {'Chhattisgarh', 'Delhi', 'Haryana', 'Jharkhand',
                   'Madhya Pradesh', 'Odisha', 'Punjab', 'Rajasthan',
                   'Tamil Nadu', 'Uttar Pradesh', 'Uttarakhand'}

def common_cohort(df):
    eligible = df.loc[df.period.isin(PERIODS)].copy()
    sets = []
    for period in PERIODS:
        part = eligible.loc[eligible.period.eq(period)]
        if part.empty:
            raise ValueError(f'No births in required period: {period}')
        counts = part.groupby(['state', 'sector']).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
        sets.append(set(counts.index[(counts[0] >= 20) & (counts[1] >= 20)]))
    common = set.intersection(*sets)
    if common != EXPECTED_STATES:
        raise ValueError(f'Common states differ from the audited 11 states: {sorted(common)}. Investigate data/version changes.')
    return eligible.loc[eligible.state.isin(common)].reset_index(drop=True), sorted(common)

def features(df):
    x = df[base.NUM + base.CAT + ['sector']].copy()
    for col in base.CAT:
        x[col] = x[col].astype('string').fillna('Missing').astype(str)
    return x

def support_diagnostics(train, target):
    rows = []
    x, ref = features(train), features(target)
    for col in base.CAT:
        for sector in [0, 1]:
            observed = set(x.loc[x.sector.eq(sector), col])
            absent = ~ref[col].isin(observed)
            rows.append({'variable': col, 'sector': sector,
                         'check': 'target category absent in training sector',
                         'target_weight_pct': float(np.average(absent, weights=target.weight) * 100),
                         'levels': '|'.join(sorted(set(ref.loc[absent, col])))})
    for col in base.NUM:
        for sector in [0, 1]:
            values = x.loc[x.sector.eq(sector), col].dropna()
            if values.empty:
                raise ValueError(f'{col}: all training values missing in sector {sector}')
            outside = ref[col].notna() & ~ref[col].between(values.min(), values.max())
            rows.append({'variable': col, 'sector': sector,
                         'check': 'target numeric value outside training-sector range',
                         'target_weight_pct': float(np.average(outside, weights=target.weight) * 100),
                         'levels': f'training range {values.min()} to {values.max()}'})
    return rows

def fit_to_target(train, target):
    x = features(train)
    if train.outcome.nunique() != 2 or set(train.sector) != {0, 1}:
        raise ValueError('Need both outcome classes and both sectors')
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                          ('scale', StandardScaler())]), base.NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore'), base.CAT + ['sector'])])
    # C=infinity yields the unpenalized specification without the deprecated
    # penalty=None argument in the user's current scikit-learn version.
    model = Pipeline([('pre', pre), ('fit', LogisticRegression(C=np.inf, max_iter=3000))])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, train.outcome, fit__sample_weight=train.weight.to_numpy())
    result = {}
    for label, ref in [('own_period', train), ('common_target', target)]:
        predictions = []
        for sector in [0, 1]:
            new = features(ref)
            new['sector'] = sector
            predictions.append(float(np.average(model.predict_proba(new)[:, 1], weights=ref.weight) * 100))
        result[label + '_public_pct'] = predictions[0]
        result[label + '_private_pct'] = predictions[1]
        result[label + '_gap_pp'] = predictions[1] - predictions[0]
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--nfhs5', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {args.output}. Choose a new directory.')
    path = args.nfhs5 or args.repo_root / 'data/raw/IABR7EFL.DTA'
    df, design, audit = base.load(path, 'NFHS-5', 60, False)
    target, states = common_cohort(df)
    expected_counts = {'A_before_2016': (2451, 1469), 'B_2017_Jan_Nov': (5592, 4053),
                       'C_2018_Jan_2019_Sep': (10819, 8272), 'D_2019_Nov_2020_Feb': (2023, 1605),
                       'E_2020_Mar_onward': (3240, 2846)}
    results, diagnostics = [], []
    for period in PERIODS:
        part = target.loc[target.period.eq(period)].reset_index(drop=True)
        counts = tuple(int(part.sector.eq(s).sum()) for s in [0, 1])
        if counts != expected_counts[period]:
            raise ValueError(f'{period}: counts {counts} differ from audited {expected_counts[period]}')
        diagnostics.extend(dict(period=period, **r) for r in support_diagnostics(part, target))
        result = {'wave': 'NFHS-5', 'period': period, 'public_n': counts[0], 'private_n': counts[1],
                  'states_retained': len(states), 'reference_n': len(target),
                  'status': 'point_estimate_pending_support_and_uncertainty_review'}
        result.update(fit_to_target(part, target))
        results.append(result)
        print(f'{period}: common-target gap {result["common_target_gap_pp"]:.3f} pp; n={len(part):,}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output / 'common_state_standardized_gaps.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(args.output / 'common_target_support_diagnostics.csv', index=False)
    base.describe(target, ['period', 'sector']).to_csv(args.output / 'common_state_descriptives.csv', index=False)
    state_weights = target.groupby('state').weight.sum()
    (state_weights / state_weights.sum() * 100).rename('pooled_target_weight_pct').to_csv(args.output / 'reference_state_weights.csv')
    metadata = {'status': 'point_estimates_only', 'wave': 'NFHS-5', 'common_states': states,
                'periods': PERIODS, 'reference_n': len(target), 'raw_date_audit': audit,
                'facility_codes': [21, 31], 'recall_rule': 'v008-b3 < 60 months',
                'reference': 'Survey-weighted pooled births in these five periods and common states; identical target for every period',
                'model': 'Separate survey-weighted unpenalized logistic outcome models per period; same eight measured covariates',
                'support': 'At least 20 births in each state-sector-period cell; additional marginal covariate support diagnostics supplied; joint overlap not established',
                'interpretation': 'Selected common-state period associations, not national estimates or individual programme receipt or causal policy effects',
                'uncertainty': 'Not yet estimated; do not claim statistical change or absence of change',
                'remaining': ['Review support/extrapolation diagnostics', 'Validate birth-date completeness flags',
                              'Joint full-design PSU resampling and model refitting for period contrasts, with singleton-stratum treatment reviewed',
                              'Check recall/date sensitivity and alternative model specification',
                              'Historical programme rollout/coverage and budgets still unmeasured']}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    print(f'Aggregate outputs saved: {args.output}', flush=True)
    print('No confidence intervals or policy effects have been estimated.', flush=True)

if __name__ == '__main__':
    main()
