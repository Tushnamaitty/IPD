"""Comment 10: supplementary birth-period associations, not policy effects.

Read authorized local NFHS Birth Recode files. Write aggregate outputs only.
See README.md for interpretation, date exclusions and uncertainty limits.
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
from state_crosswalk import harmonize_state

REQUIRED = ['m15', 'm17', 'caseid', 'bidx', 'b3', 'v008', 'v005',
            'v001', 'v022', 'v024', 'v025', 'v130', 'v133', 'v190', 'bord', 'b0', 's116']
RENAME = dict(v024='state', v025='residence', v130='religion',
              v133='education_years', v190='wealth_index', bord='birth_order',
              b0='twin_order', s116='social_group')
NUM = ['birth_order', 'wealth_index', 'education_years']
CAT = ['residence', 'religion', 'social_group', 'twin_order', 'state']
EXPECTED = {'NFHS-4': 195366, 'NFHS-5': 200794}

def cmc(year, month):
    return (year - 1900) * 12 + month

def periods(birth_cmc):
    # Entire 2016 is a PMSMA transition year; do not resolve June/November
    # announcements by assigning individual receipt from a launch date.
    b = np.asarray(birth_cmc)
    return np.select([
        b < cmc(2016, 1), b < cmc(2017, 1), b < cmc(2017, 12),
        b == cmc(2017, 12), b < cmc(2019, 10), b == cmc(2019, 10),
        b < cmc(2020, 3)], [
        'A_before_2016', 'transition_2016', 'B_2017_Jan_Nov',
        'launch_month_2017_Dec', 'C_2018_Jan_2019_Sep',
        'launch_month_2019_Oct', 'D_2019_Nov_2020_Feb'],
        default='E_2020_Mar_onward')

def load(path, wave, max_recall, allow_count_change):
    if not path.is_file():
        raise FileNotFoundError(f'{wave}: local Birth Recode file missing: {path}')
    with pd.io.stata.StataReader(path, convert_categoricals=False) as reader:
        labels = reader.variable_labels()
    missing = sorted(set(REQUIRED) - set(labels))
    if missing:
        raise ValueError(f'{wave}: required raw variables missing: {missing}')
    columns = REQUIRED + (['b10'] if 'b10' in labels else [])
    raw = pd.read_stata(path, columns=columns, convert_categoricals=False)
    design = raw[['v022', 'v001']].drop_duplicates()
    if design.isna().any().any():
        raise ValueError(f'{wave}: missing full-file design identifiers')
    broad = raw.m15.between(20, 27) | raw.m15.between(30, 33)
    broad_n = int(broad.sum())
    if broad_n != EXPECTED[wave] and not allow_count_change:
        raise ValueError(f'{wave}: broad cohort {broad_n} differs from audited {EXPECTED[wave]}; investigate before proceeding')
    df = raw.loc[raw.m15.isin([21, 31])].copy()
    if df.duplicated(['caseid', 'bidx']).any():
        raise ValueError(f'{wave}: duplicate mother/birth-index identifiers')
    if not df.m17.isin([0, 1]).all():
        raise ValueError(f'{wave}: invalid/missing cesarean outcome in narrow cohort')
    for col in ['b3', 'v008']:
        values = pd.to_numeric(df[col], errors='coerce')
        valid = np.isfinite(values) & (values % 1 == 0) & values.between(cmc(1900, 1), cmc(2100, 12))
        df[col] = values.where(valid)
    recall = df.v008 - df.b3
    date_ok = df.b3.notna() & df.v008.notna() & (recall >= 0)
    audit = {'broad_cohort_n': broad_n, 'narrow_cohort_n': len(df),
             'invalid_or_future_date_n': int((~date_ok).sum()),
             'recall_max_months_before_restriction': float(recall[date_ok].max()),
             'b10_present': 'b10' in columns,
             'date_variable_labels': {c: labels.get(c) for c in ['b3', 'v008', 'b10']},
             'recall_limit_months': max_recall}
    if 'b10' in df:
        audit['b10_counts_before_date_exclusions'] = df.b10.value_counts(dropna=False).astype(int).to_dict()
    df = df.loc[date_ok & (recall < max_recall)].copy()
    audit['retained_n'] = len(df)
    audit['date_or_recall_excluded_n'] = audit['narrow_cohort_n'] - len(df)
    if df.empty:
        raise ValueError(f'{wave}: no eligible dated births')
    df = df.rename(columns=RENAME)
    df.education_years = df.education_years.replace(97, np.nan)
    df.social_group = df.social_group.replace(8, np.nan)
    df = harmonize_state(df, wave)
    df['weight'] = df.v005 / 1e6
    if not np.isfinite(df.weight).all() or (df.weight <= 0).any():
        raise ValueError(f'{wave}: invalid survey weights')
    df['sector'] = (df.m15 == 31).astype(int)
    df['outcome'] = df.m17.astype(int)
    df['period'] = periods(df.b3)
    df['birth_year'] = ((df.b3 - 1) // 12 + 1900).astype(int)
    df['wave'] = wave
    audit['birth_cmc_range'] = [int(df.b3.min()), int(df.b3.max())]
    audit['singleton_full_file_strata'] = int((design.groupby('v022').v001.nunique() == 1).sum())
    return df, design, audit

def effective_n(w):
    return float(w.sum() ** 2 / np.square(w).sum())

def describe(df, group_cols):
    rows = []
    for keys, part in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(dict(zip(group_cols, keys), n=len(part),
                         weighted_csection_pct=float(np.average(part.outcome, weights=part.weight) * 100),
                         survey_weight_sum=float(part.weight.sum()), effective_n=effective_n(part.weight.to_numpy())))
    return pd.DataFrame(rows)

def fit_gap(df, weights=None):
    w = df.weight.to_numpy() if weights is None else np.asarray(weights)
    active = w > 0
    if set(df.loc[active, 'sector']) != {0, 1} or df.loc[active, 'outcome'].nunique() != 2:
        raise ValueError('Resample lacks both sectors or both outcome classes')
    x = df[NUM + CAT + ['sector']].copy()
    for col in CAT:
        x[col] = x[col].astype('string').fillna('Missing').astype(str)
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                          ('scale', StandardScaler())]), NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT + ['sector'])])
    # Deliberately a supplementary logistic standardization; not a causal
    # estimator, and not interchangeable with the existing AIPW headline.
    model = Pipeline([('pre', pre), ('fit', LogisticRegression(penalty=None, max_iter=3000))])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x.loc[active], df.loc[active, 'outcome'], fit__sample_weight=w[active])
    pred = []
    for sector in [0, 1]:
        new = x.copy()
        new['sector'] = sector
        pred.append(float(np.average(model.predict_proba(new)[:, 1], weights=w)))
    return {'public_standardized_pct': pred[0] * 100,
            'private_standardized_pct': pred[1] * 100,
            'standardized_gap_pp': (pred[1] - pred[0]) * 100}

def psu_multipliers(design, df, rng):
    counts = []
    for _, part in design.groupby('v022', sort=True):
        # Ordinary within-stratum PSU bootstrap, with singleton PSUs fixed.
        # Full-file design is used even where a domain contributes zero rows.
        draws = rng.choice(part.v001.to_numpy(), size=len(part), replace=True)
        count = pd.Series(draws).value_counts().rename('mult').reset_index()
        count.columns = ['v001', 'mult']
        count['v022'] = part.v022.iloc[0]
        counts.append(count)
    lookup = pd.concat(counts).set_index(['v022', 'v001']).mult
    return lookup.reindex(pd.MultiIndex.from_frame(df[['v022', 'v001']])).fillna(0).to_numpy()

def analyze(df, designs, bootstrap, min_per_sector):
    results, replicates = [], []
    rng = np.random.default_rng(42)
    for (wave, period), domain in df.groupby(['wave', 'period'], sort=True):
        if period.startswith(('transition', 'launch_month')):
            continue
        # Restrict each domain to states with observations in both sectors.
        counts = domain.groupby(['state', 'sector']).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
        supported = counts.index[(counts[0] >= min_per_sector) & (counts[1] >= min_per_sector)]
        part = domain.loc[domain.state.isin(supported)].reset_index(drop=True)
        record = {'wave': wave, 'period': period, 'eligible_n': len(domain),
                  'model_n': len(part), 'states_retained': len(supported),
                  'states_excluded': '|'.join(sorted(set(domain.state) - set(supported))),
                  'target': 'own wave-period distribution within states meeting sector-count threshold',
                  'method': 'survey-weighted unpenalized logistic standardization',
                  'causal_interpretation': False}
        if len(part) < 100 or part.sector.nunique() < 2 or part.outcome.nunique() < 2:
            record['status'] = 'insufficient_data'
            results.append(record)
            continue
        record.update(fit_gap(part))
        record['status'] = 'point_estimate_only' if bootstrap == 0 else 'bootstrap_attempted'
        reps = []
        for b in range(bootstrap):
            mult = psu_multipliers(designs[wave], part, rng)
            try:
                gap = fit_gap(part, part.weight.to_numpy() * mult)['standardized_gap_pp']
            except (ValueError, ConvergenceWarning) as exc:
                replicates.append({'wave': wave, 'period': period, 'replicate': b, 'status': type(exc).__name__})
                continue
            reps.append(gap)
            replicates.append({'wave': wave, 'period': period, 'replicate': b, 'gap_pp': gap, 'status': 'ok'})
        if bootstrap:
            record['bootstrap_success_n'] = len(reps)
            if len(reps) >= max(200, int(np.ceil(bootstrap * .95))):
                record['provisional_interval_lower_pp'], record['provisional_interval_upper_pp'] = np.percentile(reps, [2.5, 97.5])
                record['status'] = 'provisional_refit_PSU_bootstrap_singletons_fixed'
            else:
                record['status'] = 'bootstrap_insufficient_success_no_interval'
        results.append(record)
        print(f'{wave} {period}: {record["status"]}; model n={len(part):,}', flush=True)
    return pd.DataFrame(results), pd.DataFrame(replicates)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--nfhs4', type=Path)
    p.add_argument('--nfhs5', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-recall-months', type=int, default=60)
    p.add_argument('--bootstrap', type=int, default=0)
    p.add_argument('--min-state-sector-n', type=int, default=20)
    p.add_argument('--fit', action='store_true')
    p.add_argument('--allow-count-change', action='store_true')
    args = p.parse_args()
    if args.bootstrap < 0 or args.max_recall_months <= 0 or args.min_state_sector_n < 1:
        p.error('Invalid bootstrap, recall limit or count threshold')
    if args.bootstrap and not args.fit:
        p.error('--bootstrap requires --fit')
    paths = {'NFHS-4': args.nfhs4 or args.repo_root / 'data/raw/nfhs4/IABR74FL.DTA',
             'NFHS-5': args.nfhs5 or args.repo_root / 'data/raw/IABR7EFL.DTA'}
    frames, designs, audits = [], {}, {}
    for wave, path in paths.items():
        frame, design, audit = load(path, wave, args.max_recall_months, args.allow_count_change)
        frames.append(frame); designs[wave] = design; audits[wave] = audit
    df = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    describe(df, ['wave', 'period', 'sector']).to_csv(args.output / 'period_sector_descriptives.csv', index=False)
    describe(df, ['wave', 'birth_year', 'sector']).to_csv(args.output / 'year_sector_descriptives.csv', index=False)
    # State counts expose geographic coverage; do not interpret sparse cells.
    df.groupby(['wave', 'period', 'state', 'sector']).size().rename('n').to_csv(args.output / 'state_sector_support_counts.csv')
    if args.fit:
        results, reps = analyze(df, designs, args.bootstrap, args.min_state_sector_n)
        results.to_csv(args.output / 'period_standardized_gaps.csv', index=False)
        if not reps.empty:
            reps.to_csv(args.output / 'bootstrap_replicates.csv', index=False)
    metadata = {'status': 'executed_on_local_data', 'audits': audits,
                'narrow_facility_codes': [21, 31], 'max_recall_months': args.max_recall_months,
                'fit_requested': args.fit, 'bootstrap_requested': args.bootstrap,
                'min_state_sector_n': args.min_state_sector_n,
                'date_completeness_note': 'b10 audited only; consult wave-specific labels before excluding imputed dates',
                'interpretation': 'Period-specific sector associations; no policy receipt or causal policy effect measured',
                'uncertainty': 'Refit ordinary PSU bootstrap within full-file strata; singleton strata fixed; provisional pending survey review',
                'remaining': ['Verify birth-date metadata/imputation', 'Inspect state support and date exclusions',
                              'Verify historical state rollout/coverage and budgets before quantitative policy attribution',
                              'Use one consistent estimator and comparable target if formally contrasting periods or waves']}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    print(f'Aggregate outputs saved: {args.output}', flush=True)

if __name__ == '__main__':
    main()
