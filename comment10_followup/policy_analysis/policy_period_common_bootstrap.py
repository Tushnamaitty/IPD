"""NFHS-5 common-state period uncertainty; exploratory, provisional survey inference.

Place beside the existing three policy scripts. First run --replicates 10;
inspect the pilot, then extend to 500 with --resume in the same output folder.
Resamples n_h-1 PSUs within each provided birth-file stratum and multiplies
weights by n_h/(n_h-1) times selection multiplicity. The same draw is used
for every period and the pooled reference. All five outcome models are refit.
Singleton strata remain fixed. No first-stage FPC or household-stage variance
is available; design-methods review is needed before final inference.

Method source (equation 2.12 and its n_h-1 special case):
https://www150.statcan.gc.ca/n1/pub/12-001-x/2019003/article/00009/02-eng.htm

The state set and time bins were chosen after examining the data: comparisons
are exploratory. Marginal support checks do not establish joint positivity.
Calendar periods do not measure individual programme receipt or policy effects.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import policy_period_common_states as common

PERIODS = common.PERIODS
PAIRS = [(PERIODS[i], PERIODS[i-1]) for i in range(1, 5)] + [(PERIODS[4], PERIODS[0])]

def fit_model(train):
    if set(train.sector) != {0, 1} or train.outcome.nunique() != 2:
        raise ValueError('Training sample lacks a sector or outcome class')
    pre = common.ColumnTransformer([
        ('num', common.Pipeline([
            ('impute', common.SimpleImputer(strategy='median', add_indicator=True)),
            ('scale', common.StandardScaler())]), common.base.NUM),
        ('cat', common.OneHotEncoder(handle_unknown='ignore'), common.base.CAT + ['sector'])])
    model = common.Pipeline([('pre', pre), ('fit', common.LogisticRegression(C=np.inf, max_iter=3000))])
    with warnings.catch_warnings():
        warnings.simplefilter('error', common.ConvergenceWarning)
        warnings.filterwarnings('ignore', message='Setting penalty=None will ignore the C and l1_ratio parameters', category=UserWarning)
        model.fit(common.features(train), train.outcome, fit__sample_weight=train.weight.to_numpy())
    return model

def gap(model, reference):
    if reference.empty or not np.isfinite(reference.weight).all() or (reference.weight <= 0).any():
        raise ValueError('Reference weights must be finite and positive')
    pred = []
    for sector in [0, 1]:
        x = common.features(reference)
        x['sector'] = sector
        pred.append(float(np.average(model.predict_proba(x)[:, 1], weights=reference.weight) * 100))
    return pred[1] - pred[0]

def marginal_reference_mask(target):
    # One identical restricted reference for ALL periods. Training samples
    # stay unchanged: this sensitivity changes the target, not fitted models.
    ref = common.features(target)
    mask = np.ones(len(target), dtype=bool)
    for period in PERIODS:
        x = common.features(target.loc[target.period.eq(period)])
        for sector in [0, 1]:
            part = x.loc[x.sector.eq(sector)]
            for col in common.base.CAT:
                mask &= ref[col].isin(set(part[col])).to_numpy()
            for col in common.base.NUM:
                observed = part[col].dropna()
                if observed.empty:
                    raise ValueError(f'{period}, sector {sector}: no observed {col}')
                mask &= (ref[col].isna() | ref[col].between(observed.min(), observed.max())).to_numpy()
    if not mask.any():
        raise ValueError('Marginally supported reference is empty')
    return mask

def prepare_design(design, target):
    active_strata = set(target.v022)
    pairs = design.loc[design.v022.isin(active_strata), ['v022', 'v001']].drop_duplicates().sort_values(['v022', 'v001']).reset_index(drop=True)
    ids = pd.MultiIndex.from_frame(pairs)
    row_ids = ids.get_indexer(pd.MultiIndex.from_frame(target[['v022', 'v001']]))
    if (row_ids < 0).any():
        raise ValueError('Analytic PSU missing from full provided birth-file design frame')
    groups = [np.asarray(idx, dtype=int) for idx in pairs.groupby('v022', sort=True).indices.values()]
    singleton_ids = {int(idx[0]) for idx in groups if len(idx) == 1}
    singleton_rows = np.isin(row_ids, list(singleton_ids))
    audit = {'strata_with_target_births': len(groups), 'psus_in_those_full_file_strata': len(pairs),
             'singleton_strata_with_target_births': len(singleton_ids),
             'singleton_target_birth_n': int(singleton_rows.sum()),
             'singleton_target_weight_pct': float(np.average(singleton_rows, weights=target.weight) * 100)}
    return groups, row_ids, len(pairs), audit

def draw_factors(groups, n_psus, rng):
    factors = np.zeros(n_psus)
    for idx in groups:
        n = len(idx)
        if n == 1:
            factors[idx] = 1.0
        else:
            multiplicity = rng.multinomial(n-1, np.full(n, 1/n))
            factors[idx] = multiplicity * n / (n-1)
    return factors

def atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)

def atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)

def intervals(reps, points, attempted):
    good = reps.loc[reps.status.eq('ok')]
    if len(good) < max(200, int(np.ceil(attempted * .95))):
        return pd.DataFrame(), pd.DataFrame()
    period_rows, contrast_rows = [], []
    for period in PERIODS:
        values = good[period].to_numpy(dtype=float)
        lo, hi = np.percentile(values, [2.5, 97.5])
        period_rows.append({'period': period, 'gap_pp': points[period],
                            'provisional_ci_lower_pp': lo, 'provisional_ci_upper_pp': hi,
                            'successful_joint_replicates': len(good),
                            'interval_type': '95% marginal percentile; no multiplicity adjustment'})
    for later, earlier in PAIRS:
        values = good[later].to_numpy(dtype=float) - good[earlier].to_numpy(dtype=float)
        lo, hi = np.percentile(values, [2.5, 97.5])
        contrast_rows.append({'later_period': later, 'earlier_period': earlier,
                              'change_in_gap_pp': points[later] - points[earlier],
                              'provisional_ci_lower_pp': lo, 'provisional_ci_upper_pp': hi,
                              'successful_joint_replicates': len(good),
                              'interpretation': 'exploratory descriptive period contrast; not a policy effect',
                              'interval_type': '95% marginal percentile; five exploratory contrasts, no multiplicity adjustment'})
    return pd.DataFrame(period_rows), pd.DataFrame(contrast_rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--nfhs5', type=Path)
    parser.add_argument('--point-results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error('--replicates must be positive')
    raw_path = args.nfhs5 or args.repo_root / 'data/raw/IABR7EFL.DTA'
    point_path = args.point_results or args.repo_root / 'extensions_work/03_nfhs4_nfhs5_temporal/outputs/policy_period_common_states/common_state_standardized_gaps.csv'
    if not point_path.is_file():
        raise FileNotFoundError(f'Prior common-target results are required: {point_path}')
    prior = pd.read_csv(point_path).set_index('period')
    if prior.index.duplicated().any() or set(prior.index) != set(PERIODS):
        raise ValueError('Previous results must contain exactly the five common-target periods')
    df, design, raw_audit = common.base.load(raw_path, 'NFHS-5', 60, False)
    target, states = common.common_cohort(df)
    if len(target) != 42370:
        raise ValueError(f'Common-state sample changed: {len(target)} instead of 42,370')
    groups, row_ids, n_psus, design_audit = prepare_design(design, target)
    mask = marginal_reference_mask(target)
    restricted = target.loc[mask].copy()
    dropped_weight_pct = float(np.average(~mask, weights=target.weight) * 100)
    if set(restricted.state) != set(states):
        raise ValueError('Support restriction removed an entire state')
    signature = {'raw_name': raw_path.name, 'raw_bytes': raw_path.stat().st_size,
                 'raw_mtime_ns': raw_path.stat().st_mtime_ns,
                 'point_results_sha256': hashlib.sha256(point_path.read_bytes()).hexdigest(),
                 'states': states, 'target_n': len(target), 'seed': 42,
                 'method': 'n_h-1 PSU resampling with n_h/(n_h-1) multiplicities; singleton factors fixed at 1',
                 'code_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__), Path(common.__file__), Path(common.base.__file__)]}}
    manifest_path = args.output / 'metadata.json'
    reps_path = args.output / 'joint_bootstrap_replicates.csv'
    records = []
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume:
            raise FileExistsError('Output is not empty. Use --resume for this same checkpoint, or choose a new folder.')
        if not manifest_path.is_file() or not reps_path.is_file():
            raise ValueError('Checkpoint files are incomplete; choose a new output directory')
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['signature'] != signature:
            raise ValueError('Data, code or previous point results changed; cannot resume this checkpoint')
        saved = pd.read_csv(reps_path)
        if saved.replicate.duplicated().any() or saved.replicate.tolist() != list(range(len(saved))):
            raise ValueError('Checkpoint replicate IDs are not consecutive and unique')
        if len(saved) > args.replicates:
            raise ValueError('Requested total is below the existing checkpoint length')
        records = saved.to_dict('records')
    elif args.resume:
        raise FileNotFoundError('No checkpoint exists to resume')
    args.output.mkdir(parents=True, exist_ok=True)
    points, sensitivity = {}, []
    for period in PERIODS:
        part = target.loc[target.period.eq(period)]
        model = fit_model(part)
        points[period] = gap(model, target)
        old_gap = float(prior.loc[period, 'common_target_gap_pp'])
        if abs(points[period] - old_gap) > .01:
            raise ValueError(f'{period}: baseline gap {points[period]} differs from prior {old_gap}')
        restricted_gap = gap(model, restricted)
        sensitivity.append({'period': period, 'original_target_gap_pp': points[period],
                            'marginal_support_restricted_gap_pp': restricted_gap,
                            'difference_pp': restricted_gap - points[period],
                            'original_reference_n': len(target), 'restricted_reference_n': len(restricted),
                            'excluded_reference_weight_pct': dropped_weight_pct,
                            'interpretation': 'same models, identical restricted reference across all periods; not a joint-overlap check'})
    atomic_csv(pd.DataFrame(sensitivity), args.output / 'support_reference_sensitivity.csv')
    print(f'Baseline estimates reproduced. Support-restricted reference removes {dropped_weight_pct:.4f}% of target weight.', flush=True)
    print(f'Design: {design_audit}', flush=True)
    metadata = {'signature': signature, 'raw_audit': raw_audit, 'design_audit': design_audit,
                'status': 'running', 'requested_total': args.replicates,
                'interpretation': 'Selected 11-state calendar-period associations; no measured policy receipt or causal policy effects',
                'reference': 'Pooled common-state empirical reference is reweighted with the SAME PSU draw as all five model samples',
                'uncertainty': 'Provisional rescaled ultimate-PSU approximation; singleton strata fixed, FPC and lower stages unavailable; survey review required',
                'multiplicity': 'Five exploratory contrasts; marginal intervals unadjusted; no confirmatory significance claims',
                'remaining': ['Joint covariate overlap not established', 'Birth-date completeness/recall sensitivity and model sensitivity remain',
                              'Historical state policy rollout, coverage and budgets not measured'],
                'method_source': 'https://www150.statcan.gc.ca/n1/pub/12-001-x/2019003/article/00009/02-eng.htm'}
    atomic_json(metadata, manifest_path)
    start = time.perf_counter()
    for b in range(len(records), args.replicates):
        t = time.perf_counter()
        rng = np.random.default_rng(np.random.SeedSequence([42, b]))
        factors = draw_factors(groups, n_psus, rng)[row_ids]
        boot = target.copy()
        boot['weight'] = target.weight.to_numpy() * factors
        boot = boot.loc[boot.weight.gt(0)].copy()
        record = {'replicate': b, 'status': 'ok', 'error': ''}
        try:
            for period in PERIODS:
                part = boot.loc[boot.period.eq(period)]
                if not set(boot.state).issubset(set(part.state)):
                    raise ValueError(f'{period}: target state absent from resampled training domain')
                record[period] = gap(fit_model(part), boot)
        except (ValueError, common.ConvergenceWarning, FloatingPointError) as exc:
            record['status'] = 'failed'
            record['error'] = f'{type(exc).__name__}: {str(exc)[:250]}'
            # A joint draw is usable for contrasts only if ALL five models succeed.
            for period in PERIODS:
                record[period] = np.nan
        record['seconds'] = time.perf_counter() - t
        records.append(record)
        # Every completed draw is saved atomically, enabling interruption/resume.
        atomic_csv(pd.DataFrame(records), reps_path)
        metadata.update(completed_total=len(records), successful_joint_replicates=sum(r['status']=='ok' for r in records))
        atomic_json(metadata, manifest_path)
        if b == 0 or (b+1) % 5 == 0 or b+1 == args.replicates:
            print(f'{b+1}/{args.replicates} attempted; {metadata["successful_joint_replicates"]} jointly successful; this draw {record["seconds"]:.1f}s', flush=True)
    reps = pd.DataFrame(records)
    period_ci, contrast_ci = intervals(reps, points, len(reps))
    if not period_ci.empty:
        atomic_csv(period_ci, args.output / 'period_gap_intervals.csv')
        atomic_csv(contrast_ci, args.output / 'exploratory_period_contrasts.csv')
        metadata['status'] = 'completed_provisional_intervals_require_review'
    else:
        # Never leave stale interval files after an insufficient checkpoint.
        for name in ['period_gap_intervals.csv', 'exploratory_period_contrasts.csv']:
            (args.output / name).unlink(missing_ok=True)
        metadata['status'] = 'pilot_or_insufficient_success_no_intervals'
    metadata['median_seconds_per_joint_draw'] = float(reps.seconds.median())
    metadata['estimated_minutes_for_remaining_to_500'] = max(0, 500-len(reps)) * metadata['median_seconds_per_joint_draw'] / 60
    metadata['current_invocation_bootstrap_seconds'] = time.perf_counter() - start
    atomic_json(metadata, manifest_path)
    print(f'Finished: {metadata["status"]}', flush=True)
    print(f'Median time per joint draw: {metadata["median_seconds_per_joint_draw"]:.1f}s', flush=True)
    print(f'Estimated remaining time to 500: {metadata["estimated_minutes_for_remaining_to_500"]:.1f} minutes (rough estimate)', flush=True)
    print(f'Aggregate outputs saved: {args.output}', flush=True)

if __name__ == '__main__':
    main()
