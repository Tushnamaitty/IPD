"""Audit local NFHS-5 PSU identifiers and birth-file versus household-file coverage.

Place alongside the existing policy scripts. This reads data and saved bootstrap
results; it does not refit models or change previous outputs. Only share the
small design_review.json report, not household_psu_frame.csv (local design IDs).

Sources reviewed:
https://microdata.worldbank.org/catalog/4482 (NFHS-5 design and 30,198 fieldwork PSUs)
https://cran.r-universe.dev/survey/doc/manual.html (subbootstrap and lonely PSUs)
https://www150.statcan.gc.ca/n1/pub/12-001-x/2019003/article/00009/02-eng.htm
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import policy_period_common_states as common
import policy_period_common_bootstrap as bootstrap


def read_columns(path, required, optional=()):
    with pd.io.stata.StataReader(path, convert_categoricals=False) as reader:
        labels = reader.variable_labels()
    missing = sorted(set(required) - set(labels))
    if missing:
        raise ValueError(f'{path.name}: missing required variables {missing}')
    cols = list(required) + [c for c in optional if c in labels]
    df = pd.read_stata(path, columns=cols, convert_categoricals=False)
    return df, {c: labels[c] for c in cols}


def design_pairs(df, stratum, psu):
    pairs = df[[stratum, psu]].drop_duplicates().copy()
    for c in [stratum, psu]:
        a = pd.to_numeric(pairs[c], errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(a).all() or (a <= 0).any() or (a != np.floor(a)).any():
            raise ValueError(f'Invalid design identifiers: {c}')
        pairs[c] = a.astype(np.int64)
    return pairs.rename(columns={stratum: 'stratum', psu: 'psu'}).sort_values(['stratum', 'psu']).reset_index(drop=True)


def compare_frames(birth_pairs, household_pairs, active_strata):
    br = birth_pairs.loc[birth_pairs.stratum.isin(active_strata)]
    hr = household_pairs.loc[household_pairs.stratum.isin(active_strata)]
    bset, hset = set(map(tuple, br.to_numpy())), set(map(tuple, hr.to_numpy()))
    bc, hc = br.groupby('stratum').size(), hr.groupby('stratum').size()
    old_single = set(bc.index[bc.eq(1)])
    new_single = set(hc.index[hc.eq(1)])
    return {'birth_file_psus_in_active_strata': len(br),
            'household_file_psus_in_active_strata': len(hr),
            'additional_household_psus_in_active_strata': len(hset-bset),
            'birth_psu_pairs_absent_from_household_frame': len(bset-hset),
            'birth_file_singleton_active_strata': len(old_single),
            'household_file_singleton_active_strata': len(new_single),
            'birth_file_singletons_resolved_by_household_frame': len(old_single-new_single)}, hr, new_single


def check_saved_results(folder):
    reps = pd.read_csv(folder / 'joint_bootstrap_replicates.csv')
    if reps.replicate.duplicated().any() or reps.replicate.tolist() != list(range(len(reps))):
        raise ValueError('Saved replicate IDs are not consecutive and unique')
    good = reps.loc[reps.status.eq('ok')]
    if len(reps) != 500 or len(good) != 500:
        raise ValueError(f'Expected 500 successful joint draws; found {len(good)}/{len(reps)}')
    if not np.isfinite(good[common.PERIODS].to_numpy()).all():
        raise ValueError('Saved successful replicates contain nonfinite estimates')
    period_ci = pd.read_csv(folder / 'period_gap_intervals.csv').set_index('period')
    contrast_ci = pd.read_csv(folder / 'exploratory_period_contrasts.csv').set_index(['later_period', 'earlier_period'])
    if period_ci.index.duplicated().any() or set(period_ci.index) != set(common.PERIODS):
        raise ValueError('Saved period rows do not match the five periods')
    if contrast_ci.index.duplicated().any() or set(contrast_ci.index) != set(bootstrap.PAIRS):
        raise ValueError('Saved contrast rows do not match the five comparisons')
    errors = []
    for period in common.PERIODS:
        q = np.percentile(good[period], [2.5, 97.5])
        stored = period_ci.loc[period, ['provisional_ci_lower_pp', 'provisional_ci_upper_pp']].to_numpy(dtype=float)
        errors.append(float(np.max(np.abs(q-stored))))
    for later, earlier in bootstrap.PAIRS:
        q = np.percentile(good[later]-good[earlier], [2.5, 97.5])
        stored = contrast_ci.loc[(later, earlier), ['provisional_ci_lower_pp', 'provisional_ci_upper_pp']].to_numpy(dtype=float)
        errors.append(float(np.max(np.abs(q-stored))))
    if max(errors) > 1e-8:
        raise ValueError('Saved percentile intervals do not reproduce from saved paired draws')
    return {'joint_draws': len(reps), 'successful_joint_draws': len(good),
            'max_interval_reproduction_error_pp': max(errors),
            'same_draw_period_contrasts_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--nfhs5', type=Path)
    parser.add_argument('--household-frame', type=Path)
    parser.add_argument('--bootstrap-results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose a new output folder; the audit never overwrites prior results')
    raw_path = args.nfhs5 or args.repo_root / 'data/raw/IABR7EFL.DTA'
    results = args.bootstrap_results or args.repo_root / 'extensions_work/03_nfhs4_nfhs5_temporal/outputs/policy_period_common_bootstrap'
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'audit_started', 'existing_intervals_finalized': False,
              'method_review': 'n_h-1 scaling is correct for a with-replacement ultimate-PSU approximation; exact PPS/systematic multistage variance is not established',
              'singleton_review': 'Fixed singleton factors give zero between-PSU variance; small weighted share alone does not establish negligible variance impact',
              'point_estimates_changed': False}
    try:
        report['saved_results'] = check_saved_results(results)
        raw, labels = read_columns(raw_path, ['caseid', 'bidx', 'v001', 'v021', 'v022'], ['v024', 'v025'])
        report['birth_design_labels'] = {c: labels[c] for c in ['v001', 'v021', 'v022']}
        mapping = raw[['v022', 'v001', 'v021']].drop_duplicates()
        equivalent = (mapping.groupby(['v022', 'v001']).v021.nunique().le(1).all()
                      and mapping.groupby(['v022', 'v021']).v001.nunique().le(1).all())
        # A bijection is sufficient even when the numeric identifiers differ.
        report['v001_v021_same_cluster_partition_within_strata'] = bool(equivalent)
        birth_pairs = design_pairs(raw, 'v022', 'v021')
        df, _, _ = common.base.load(raw_path, 'NFHS-5', 60, False)
        target, states = common.common_cohort(df)
        if len(target) != 42370:
            raise ValueError('The audited common cohort changed')
        target = target.merge(raw[['caseid', 'bidx', 'v021']], on=['caseid', 'bidx'], validate='one_to_one')
        active = set(target.v022)
        report.update(target_n=len(target), common_states=states,
                      birth_file_national_psus=len(birth_pairs), active_strata=len(active))
        if not equivalent:
            report['status'] = 'correction_required_v001_and_v021_cluster_partitions_differ'
        else:
            expected_name = 'IAHR' + raw_path.stem[4:] + '.DTA'
            local_files = [p for p in (args.repo_root / 'data/raw').rglob('*') if p.is_file() and p.suffix.lower() == '.dta']
            candidates = [p for p in local_files if p.name.upper() == expected_name.upper()]
            report['household_recode_files_found'] = [str(p) for p in local_files if p.name.upper().startswith('IAHR')]
            hr_path = args.household_frame or (candidates[0] if len(candidates) == 1 else None)
            if hr_path is None:
                report['status'] = 'household_frame_needed_to_finish_coverage_and_singleton_review'
                report['expected_household_filename'] = expected_name
            else:
                hr, hr_labels = read_columns(hr_path, ['hv021', 'hv022'], ['hv001', 'hv024', 'hv025', 'hv015'])
                household_pairs = design_pairs(hr, 'hv022', 'hv021')
                report.update(household_filename=hr_path.name,
                              household_design_labels=hr_labels,
                              household_national_psus=len(household_pairs),
                              matches_reported_30198_fieldwork_psus=len(household_pairs) == 30198)
                coverage, frame, lonely = compare_frames(birth_pairs, household_pairs, active)
                report['frame_comparison'] = coverage
                report['household_singleton_target_birth_n'] = int(target.v022.isin(lonely).sum())
                report['household_singleton_target_weight_pct'] = float(np.average(target.v022.isin(lonely), weights=target.weight)*100)
                if coverage['birth_psu_pairs_absent_from_household_frame']:
                    report['status'] = 'household_and_birth_design_identifiers_do_not_match_investigate'
                else:
                    # Local design IDs only; no individual/household records written.
                    frame.to_csv(args.output / 'household_psu_frame.csv', index=False)
                    if coverage['additional_household_psus_in_active_strata']:
                        report['status'] = 'rerun_uncertainty_with_household_psu_frame_required'
                    elif lonely:
                        report['status'] = 'singleton_variance_adjustment_or_documented_certainty_needed'
                    elif len(household_pairs) != 30198:
                        report['status'] = 'frame_matches_birth_domains_but_national_coverage_requires_explanation'
                    else:
                        report['status'] = 'frame_and_identifiers_match_existing_resampling_approximation'
                        report['existing_intervals_finalized'] = True
                        report['scope_of_finalization'] = 'Exploratory intervals under the stated with-replacement ultimate-PSU approximation only; not exact full-design coverage'
    except (ValueError, FileNotFoundError, KeyError) as exc:
        report['status'] = 'audit_blocked'
        report['error'] = f'{type(exc).__name__}: {exc}'
    (args.output / 'design_review.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(f'Audit saved: {args.output / "design_review.json"}', flush=True)
    print('No model refits or bootstrap repetitions were run; previous results were not changed.', flush=True)


if __name__ == '__main__':
    main()
