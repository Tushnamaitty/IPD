"""State/UT descriptive cesarean comparisons for the reviewed narrow cohorts.

NOT run or tested here. Run locally beside IPD_narrow_facility_analysis.py:
python IPD_statewise_descriptive.py --repo-root .
Requires numpy, pandas, scipy and the existing reviewed national outputs.
Exports aggregate CSV/JSON only; preserves existing outputs. No adjusted models,
imputation, causal estimates, state rankings or cross-wave change tests.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import t


def load_base(root):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('statewise_base', root/'IPD_narrow_facility_analysis.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_labels(path):
    # Read actual variable-bound categoricals; never guess a label-table binding
    # or treat NFHS-4 raw codes as NFHS-5 codes.
    raw = pd.read_stata(path, columns=['v024'], convert_categoricals=False)
    labelled = pd.read_stata(path, columns=['v024'], convert_categoricals=True)
    if len(raw) != len(labelled) or not isinstance(labelled.v024.dtype, pd.CategoricalDtype):
        raise ValueError('Cannot establish variable-bound state labels from '+str(path))
    pairs = pd.DataFrame({'code': raw.v024, 'name': labelled.v024.astype('string')}).drop_duplicates()
    if pairs.isna().any().any() or pairs.code.duplicated().any():
        raise ValueError('Missing or ambiguous state label mapping.')
    return {int(r.code): str(r['name']) for _, r in pairs.iterrows()}


def observed(frame, domain):
    w = frame.weight.to_numpy(float)
    y = frame.outcome.to_numpy(float)
    sector = frame.sector.to_numpy(int)
    points = np.full(3, np.nan)
    influence = np.zeros((len(frame), 6))
    counts = []
    for a in (0, 1):
        mask = domain & (sector == a)
        counts.append(int(mask.sum()))
        denominator = w[mask].sum()
        if denominator > 0:
            points[a] = np.dot(w[mask], y[mask])/denominator
            influence[:, a] = w*mask*(y-points[a])/denominator
    if all(counts):
        points[2] = points[1]-points[0]
        influence[:, 2] = influence[:, 1]-influence[:, 0]
    if np.max(np.abs(influence.sum(axis=0))) > 1e-7:
        raise RuntimeError('Descriptive influence centering failed.')
    return points, influence, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=Path('.'))
    p.add_argument('--nfhs4', type=Path)
    p.add_argument('--nfhs5', type=Path)
    p.add_argument('--nfhs4-household', type=Path)
    p.add_argument('--nfhs5-household', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.repo_root.resolve()
    base = load_base(root)
    baseline_path = root/'extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    current_hash = base.sha256(root/'IPD_narrow_facility_analysis.py')
    reviewed_current_hash = '6ee895abac082b9c6a8173764606f64167bf7700f69f578a0d91c5058b71e836'
    if current_hash not in (baseline['script_sha256'], reviewed_current_hash):
        raise ValueError('Unreviewed base script version; stop for source review.')
    compatibility_checks = []
    if current_hash != baseline['script_sha256']:
        print('Reviewed current source differs from original recorded hash. Reproducing all national estimates and SEs before state results.', flush=True)
    out = (args.output or root/'extensions_work/17_statewise_descriptive/outputs/statewise_descriptive').resolve()
    if out.exists():
        raise FileExistsError('Existing output preserved. Choose a new --output directory.')
    results, diagnostic_rows, mappings, checks = [], [], [], []
    audits = {}
    for wave in ('NFHS-4', 'NFHS-5'):
        four = wave == 'NFHS-4'
        br = (args.nfhs4 if four else args.nfhs5) or base.find_raw(
            root, 'IABR74' if four else 'IABR7', exclude_prefix=None if four else 'IABR74')
        hr = (args.nfhs4_household if four else args.nfhs5_household) or base.find_raw(
            root, 'IAHR74' if four else 'IAHR7', exclude_prefix=None if four else 'IAHR74', required=False)
        frame, roster, audit = base.load_wave(br, hr, wave)
        previous = baseline['cohort_audit'][wave]
        for key in ('birth_sha256', 'analytic_n', 'public_n', 'private_n', 'household_roster_available'):
            if audit[key] != previous[key]:
                raise ValueError(wave+': reviewed input/cohort mismatch: '+key)
        if audit.get('household_sha256') != previous.get('household_sha256'):
            raise ValueError(wave+': household input differs from baseline.')
        labels = state_labels(br)
        audits[wave] = audit
        for code, name in labels.items():
            mappings.append({'wave': wave, 'raw_state_code': code, 'state_name': name,
                             'in_narrow_cohort': bool(frame.state.eq(code).any()),
                             'cross_wave_mapping_status': 'Wave-specific labels only; geographic harmonisation requires review'})
        for analysis in ('all_narrow', 'first_delivery'):
            domain = np.ones(len(frame), dtype=bool) if analysis == 'all_narrow' else frame.delivery_order.eq(1).to_numpy()
            if current_hash != baseline['script_sha256']:
                sub = frame.loc[domain].copy().reset_index(drop=True)
                reproduced, _, _, _, _, _ = base.run_analysis(sub, roster, wave, analysis)
                for old in baseline['primary_intervals']:
                    if old['wave'] != wave or old['analysis'] != analysis:
                        continue
                    row = next(r for r in reproduced if r['metric'] == old['metric'] and r['singleton_treatment'] == old['singleton_treatment'])
                    differences = {key: abs(row[key]-old[key]) for key in ('estimate','se','ci_lower','ci_upper')}
                    if max(differences.values()) > 1e-5:
                        raise ValueError('Current source fails baseline reproduction: '+wave+'/'+analysis+'/'+old['metric']+' '+str(differences))
                    compatibility_checks.append({'wave':wave,'analysis':analysis,'metric':old['metric'],'differences_pp':differences})
            point, influence, _ = observed(frame, domain)
            covs, _ = base.covariance_from_roster(frame, influence, roster)
            for j, metric in enumerate(base.METRICS[:3]):
                old = next(r for r in baseline['primary_intervals'] if r['wave'] == wave and r['analysis'] == analysis and r['metric'] == metric)
                difference = abs(100*point[j]-old['estimate'])
                if difference > 1e-6:
                    raise ValueError('National observed estimate not reproduced: '+wave+'/'+analysis+'/'+metric)
                checks.append({'wave': wave, 'analysis': analysis, 'metric': metric,
                               'estimate_difference_pp': float(difference),
                               'baseline_se_pp': old['se'],
                               'full_domain_roster_se_pp': float(100*np.sqrt(max(0, covs['adjust'][j,j]))),
                               'note': 'Point reproduced. First-delivery variance retains all narrow-cohort active strata, so SE need not equal older subgroup-only variance.'})
        # Include labelled geographies even when no eligible narrow births exist.
        for code, name in sorted(labels.items(), key=lambda item: item[1].lower()):
            f = frame.loc[frame.state.eq(code)].copy().reset_index(drop=True)
            for analysis in ('all_narrow', 'first_delivery'):
                domain = np.ones(len(f), dtype=bool) if analysis == 'all_narrow' else f.delivery_order.eq(1).to_numpy()
                point, influence, counts = observed(f, domain)
                design, covariance_error, covs = {}, '', {}
                if len(f):
                    try:
                        # Keep all state narrow rows as zero-contribution rows outside
                        # first-delivery domain. Roster supplies zero-domain PSUs.
                        covs, design = base.covariance_from_roster(f, influence, roster)
                    except ValueError as exc:
                        if 'Insufficient nonsingleton strata' not in str(exc):
                            raise
                        covariance_error = str(exc)
                else:
                    covariance_error = 'No eligible narrow births in this geography.'
                diagnostics = {'wave': wave, 'analysis': analysis, 'raw_state_code': code,
                               'state_name': name, 'n_public': counts[0], 'n_private': counts[1],
                               'both_sectors_observed': bool(all(counts)),
                               'below_30_either_sector': bool(min(counts)<30),
                               'uncertainty_error': covariance_error, **design}
                diagnostic_rows.append(diagnostics)
                for j, metric in enumerate(base.METRICS[:3]):
                    relevant = [0] if j == 0 else [1] if j == 1 else [0,1]
                    masks = [domain & f.sector.eq(a).to_numpy() for a in relevant]
                    flags = []
                    if any(counts[a] == 0 for a in relevant): flags.append('absent_sector_nonestimable')
                    if any(0 < counts[a] < 30 for a in relevant): flags.append('sparse_n_below_30')
                    boundary = any(mask.any() and f.loc[mask,'outcome'].nunique()<2 for mask in masks)
                    if boundary: flags.append('boundary_outcome_no_reliable_wald_interval')
                    # Kish effective n is a weight-only diagnostic, not survey df.
                    effective = []
                    for mask in masks:
                        ww = f.loc[mask,'weight'].to_numpy(float)
                        effective.append(float(ww.sum()**2/np.dot(ww,ww)) if len(ww) else 0.)
                    if any(0 < value < 30 for value in effective): flags.append('weight_only_effective_n_below_30')
                    for treatment in ('adjust', 'average', 'zero_sensitivity_only'):
                        se = low = high = None
                        status = 'point_only'
                        value = float(100*point[j]) if np.isfinite(point[j]) else None
                        if value is None: status = 'nonestimable'
                        elif covariance_error: status = 'point_only_insufficient_design'
                        elif boundary: status = 'point_only_boundary_outcome'
                        else:
                            variance = float(covs[treatment][j,j])
                            if not np.isfinite(variance) or variance < -1e-12:
                                raise RuntimeError('Invalid state-domain variance.')
                            se = float(100*np.sqrt(max(0., variance)))
                            if se == 0:
                                status = 'point_only_zero_variance';se = None
                            else:
                                critical = float(t.ppf(.975, design['design_df']))
                                low, high = value-critical*se, value+critical*se
                                status = 'approximate_interval_sparse' if flags else 'approximate_interval'
                                if j < 2 and (low < 0 or high > 100):
                                    status = 'approximate_interval_outside_bounds'; flags_here = flags+['wald_interval_outside_0_100']
                                else: flags_here = flags
                        if status != 'approximate_interval_outside_bounds': flags_here = flags
                        results.append({'wave': wave, 'analysis': analysis, 'raw_state_code': code,
                                        'state_name': name, 'metric': metric, 'estimate': value,
                                        'se': se, 'ci_lower': low, 'ci_upper': high,
                                        'units': 'percentage points' if j == 2 else 'percent',
                                        'n_public': counts[0], 'n_private': counts[1],
                                        'minimum_weight_only_effective_n': min(effective),
                                        'singleton_treatment': treatment, 'design_df': design.get('design_df'),
                                        'status': status, 'flags': ';'.join(flags_here)})
            print(wave+': '+name+' completed', flush=True)
    summary = {'status': 'completed_descriptive_outputs_require_review',
               'facility_codes': {'public':21, 'private':31},
               'cohorts': ['all_narrow','first_delivery'],
               'method': 'Survey-weighted domain ratio proportions and joint private-minus-public influence; existing state/stratum/PSU roster Taylor approximation.',
               'primary_singleton_treatment':'adjust', 'sparse_flag_threshold':30,
               'threshold_note':'Diagnostic only, not a guarantee of precision; sparse estimates retained and flagged.',
               'state_mapping':'Read from actual v024 categorical binding independently per wave; no code equivalence or geographic harmonisation inferred.',
               'national_reproduction': checks, 'cohort_audit':audits,
               'original_baseline_script_sha256':baseline['script_sha256'],
               'source_compatibility_reproduction':compatibility_checks,
               'source_compatibility_note':'Known reviewed current source allowed only with full national estimate/SE/CI reproduction when original recorded hash differs.',
               'script_sha256':base.sha256(__file__),
               'base_script_sha256':base.sha256(root/'IPD_narrow_facility_analysis.py'),
               'limitations':['Descriptive associations only; no clinical appropriateness or causal conclusions.',
                              'With-replacement ultimate-PSU approximation; observed roster, singleton and no-FPC/lower-stage limitations remain.',
                              'Marginal approximate t/Wald intervals; no simultaneous coverage, state rankings or multiplicity-adjusted significance claims.',
                              'Intervals at boundary outcomes withheld; zero-variance intervals withheld; sparse and weight-dominated samples flagged.',
                              'State/UT labels and boundary changes require manual review before cross-wave presentation; no temporal difference tests calculated.',
                              'Describes eligible narrow facility births, not all births or general hospital utilisation.',
                              'No social-group imputation needed for unadjusted descriptive ratios.'],
               'remaining_work':'Review mapping, sparse flags and interval precision; then prepare publication table/figure.',
               'execution_note':'Author did not execute or test this script before delivery.',
               'existing_outputs_changed':False}
    out.mkdir(parents=True)
    table = pd.DataFrame(results)
    table.to_csv(out/'statewise_all_intervals.csv', index=False)
    table.loc[table.singleton_treatment.eq('adjust')].to_csv(out/'statewise_primary_results.csv', index=False)
    pd.DataFrame(diagnostic_rows).to_csv(out/'statewise_design_support.csv', index=False)
    pd.DataFrame(mappings).to_csv(out/'statewise_label_mapping.csv', index=False)
    base.save_json(out/'statewise_summary.json', summary)
    print('Aggregate outputs saved: '+str(out), flush=True)
    print('Share the five CSV/JSON files for review; keep raw records local.', flush=True)


if __name__ == '__main__':
    main()
