"""Aggregate-only diagnostic: how much variation in each historical context term survives
state + calendar adjustment, and how much of it is related to facility sector?

Place beside IPD_historical_state_context.py, IPD_narrow_facility_analysis.py,
IPD_narrow_support_sensitivity.py and historical_state_context_data/. Run:
    python IPD_historical_context_residual_diagnostic.py --repo-root .

What it does: rebuilds exactly the linked cohorts and the unweighted state/year/seasonality
projection used by IPD_historical_state_context.matrices(), then reports descriptive aggregates
per wave x analysis x case x term. It fits NO logistic or other outcome model (only the same
linear least-squares projection), exports NO births, households, PSUs or identifiers, and
writes only to a new folder. Existing outputs are untouched.

Self-checks (abort before writing anything): linked_n and the stored
within_state_calendar_residual_norm for EVERY completed case must reproduce the existing
historical_context_summary.json.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

CELL_TOLERANCE = 1e-8


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(value):
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def wmean(v, w):
    return float(np.sum(w * v) / np.sum(w))


def wsd(v, w):
    m = wmean(v, w)
    return float(np.sqrt(np.sum(w * (v - m) ** 2) / np.sum(w)))


def wcorr(a, b, w):
    ma, mb = wmean(a, w), wmean(b, w)
    va, vb = np.sum(w * (a - ma) ** 2), np.sum(w * (b - mb) ** 2)
    if va <= 1e-24 or vb <= 1e-24:
        return float('nan')
    return float(np.sum(w * (a - ma) * (b - mb)) / np.sqrt(va * vb))


def project_out(z, v, w):
    """Weighted least-squares residual of each column of v on z (linear algebra only)."""
    s = np.sqrt(w)[:, None]
    coefficients = np.linalg.lstsq(s * z, s * v, rcond=None)[0]
    return v - z @ coefficients


def diagnose(base, hist, eligible, columns, first, stored, label):
    x, schema = base.encode(eligible, first_delivery=first)
    years = sorted(eligible.birth_year.unique())
    month = eligible.birth_month.to_numpy(float)
    calendar = np.column_stack([eligible.birth_year.eq(y).to_numpy(float) for y in years[1:]]
                               + [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)])
    indices = [0] + [i for i, s in enumerate(schema['labels']) if s.startswith('state=')]
    nuisance = np.column_stack([x[:, indices], calendar])
    scales = np.array([hist.SCALES[c] for c in columns])
    values = eligible[columns].to_numpy(float) / scales
    residual = values - nuisance @ np.linalg.lstsq(nuisance, values, rcond=None)[0]
    # Self-check against the stored historical output (same expressions as matrices()).
    for j, detail in enumerate(stored['schema']['context_terms']):
        got = float(np.linalg.norm(residual[:, j]))
        want = float(detail['within_state_calendar_residual_norm'])
        if abs(got - want) > 1e-6 * max(1., abs(want)) + 1e-8:
            raise RuntimeError(f'{label} {detail["term"]}: residual norm {got:.8g} does not reproduce stored {want:.8g}.')
    w = eligible.weight.to_numpy(float)
    private = eligible.sector.to_numpy(float)
    # Partial out the other design columns (everything except the private column) and calendar terms.
    z = np.column_stack([np.delete(x, 1, axis=1), calendar])
    private_partial = project_out(z, private[:, None], w)[:, 0]
    partial = project_out(z, residual, w)
    rows = []
    included = set(stored['schema']['included_context_terms'])
    for j, name in enumerate(columns):
        raw = values[:, j] * scales[j]
        res = residual[:, j] * scales[j]
        ss_total = float(np.sum((raw - raw.mean()) ** 2))
        ss_resid = float(np.sum(res ** 2))
        cells = pd.DataFrame({'state': eligible.state_name.to_numpy(), 'year': eligible.birth_year.to_numpy(),
                              'raw': raw, 'res': res})
        cell = cells.groupby(['state', 'year']).agg(raw_value_n=('raw', 'nunique'), mean_residual=('res', 'mean'))
        retained = cell.mean_residual.abs() > CELL_TOLERANCE
        state_retained = cell.loc[retained].reset_index().state.nunique()
        row = {'case_label': label, 'term': name, 'stored_included': bool(name in included),
               'linked_n': int(len(eligible)), 'linked_states': int(eligible.state_name.nunique()),
               'linked_state_year_cells': int(len(cell)),
               'state_year_cells_with_retained_variation': int(retained.sum()),
               'states_with_retained_variation': int(state_retained),
               'distinct_raw_values': int(pd.Series(raw).nunique()),
               'raw_sd_unweighted': float(raw.std()), 'raw_sd_weighted': wsd(raw, w),
               'residual_sd_unweighted': float(np.sqrt(np.mean(res ** 2))), 'residual_sd_weighted': float(np.sqrt(wmean(res ** 2, w))),
               'share_of_unweighted_raw_variance_absorbed': (1. - ss_resid / ss_total) if ss_total > 0 else float('nan'),
               'between_cell_residual_rms': float(np.sqrt(np.mean(cell.mean_residual.to_numpy() ** 2))),
               'weighted_corr_residual_with_private_raw': wcorr(residual[:, j], private, w),
               'weighted_corr_residual_with_private_partial': wcorr(partial[:, j], private_partial, w),
               'sd_units': 'original term units (percentage points of expenditure share, specialists per CHC, or 0/1 indicator)',
               'residual_orthogonality_note': 'unweighted projection on state, calendar year and month harmonics, as in the fitted models'}
        rows.append(row)
    if len(columns) == 2:
        c = wcorr(partial[:, 0], partial[:, 1], w)
        for row in rows:
            row['weighted_partial_corr_between_the_two_residual_terms'] = c
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--existing-summary', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--data-dir', type=Path)
    for name in ('nfhs4', 'nfhs5', 'nfhs4-household', 'nfhs5-household'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    beside = Path(__file__).resolve().parent
    root = args.repo_root.resolve()
    hist = load(beside / 'IPD_historical_state_context.py', 'ipd_hist_context')
    helper = load(beside / 'IPD_narrow_support_sensitivity.py', 'context_support')
    base = helper.load_base(beside / 'IPD_narrow_facility_analysis.py')
    output = args.output or root / 'extensions_work/04_health_system_context/outputs/historical_context_residual_diagnostic'
    if output.exists():
        raise FileExistsError('Existing output preserved; choose another --output.')
    existing_path = args.existing_summary or root / 'extensions_work/04_health_system_context/outputs/historical_state_context/historical_context_summary.json'
    existing = json.loads(existing_path.read_text())
    baseline = args.baseline or root / 'extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json'
    original = json.loads(baseline.read_text())
    finance, facility, manifest = hist.load_sources(args.data_dir or beside / 'historical_state_context_data', base)
    start = time.monotonic()
    rows, checked = [], []
    for wave in ('NFHS-4', 'NFHS-5'):
        br = (args.nfhs4 if wave == 'NFHS-4' else args.nfhs5) or base.find_raw(
            root, 'IABR74' if wave == 'NFHS-4' else 'IABR7', exclude_prefix=None if wave == 'NFHS-4' else 'IABR74')
        explicit = args.nfhs4_household if wave == 'NFHS-4' else args.nfhs5_household
        hr, _ = helper.hr_candidates(root, wave, explicit)
        old = original['cohort_audit'][wave]
        if old['household_roster_available'] and hr is None:
            raise FileNotFoundError('Reviewed household file required.')
        frame, roster, audit = base.load_wave(br, hr, wave)
        if audit['birth_sha256'] != old['birth_sha256']:
            raise ValueError('Raw birth input changed.')
        names = hist.state_names(frame, audit, wave)
        linked12 = hist.link(frame, names, finance, facility, 12)
        linked24 = hist.link(frame, names, finance, facility, 24)
        for analysis in ('all_narrow', 'first_delivery'):
            select = np.ones(len(frame), bool) if analysis == 'all_narrow' else frame.delivery_order.eq(1).to_numpy()
            full12 = linked12.loc[select].reset_index(drop=True)
            full24 = linked24.loc[select].reset_index(drop=True)
            for case, columns in hist.CASES.items():
                q = full24 if case.endswith('24m') else full12
                key = f'{wave}_{analysis}_{case}'
                stored = existing['cases'][key]
                keep, _ = hist.coverage(q, columns, case, wave, analysis)
                eligible = q.loc[keep].reset_index(drop=True)
                if stored['linked_n'] != len(eligible):
                    raise RuntimeError(f'{key}: linked_n {len(eligible)} differs from stored {stored["linked_n"]}.')
                if 'schema' not in stored:
                    checked.append({'case': key, 'status': stored['status'], 'linked_n': len(eligible)})
                    continue
                print(f'{key}: {len(eligible):,} linked births...', flush=True)
                for r in diagnose(base, hist, eligible, columns, analysis == 'first_delivery', stored, key):
                    r.update(wave=wave, analysis=analysis, case=case)
                    rows.append(r)
                checked.append({'case': key, 'status': 'residual_norms_reproduced', 'linked_n': len(eligible)})
    table = pd.DataFrame(rows)
    lead = ['wave', 'analysis', 'case', 'term', 'stored_included']
    table = table[lead + [c for c in table.columns if c not in lead]]
    summary = {'status': 'aggregate_residual_diagnostic_completed_requires_review',
               'purpose': 'Residual variation in context terms after state/calendar projection; no outcome model refit.',
               'existing_summary_sha256': base.sha256(existing_path), 'script_sha256': base.sha256(Path(__file__)),
               'historical_script_sha256': base.sha256(beside / 'IPD_historical_state_context.py'),
               'self_checks': checked, 'rows': len(table), 'record_level_data_exported': False,
               'existing_outputs_changed': False,
               'note': 'Descriptive; correlations and variances are unweighted/weighted summaries of residualized macro terms, not effect estimates.',
               'elapsed_seconds': round(time.monotonic() - start, 1)}
    output.mkdir(parents=True)
    table.to_csv(output / 'context_residual_variation.csv', index=False)
    (output / 'context_residual_variation_summary.json').write_text(
        json.dumps({k: clean(v) if not isinstance(v, (list, dict)) else v for k, v in summary.items()}, indent=2, allow_nan=False), encoding='utf-8')
    print('Aggregate outputs saved: ' + str(output))
    print('Share context_residual_variation.csv and context_residual_variation_summary.json only.')


if __name__ == '__main__':
    main()
