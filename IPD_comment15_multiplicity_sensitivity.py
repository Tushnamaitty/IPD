"""Added EXPLORATORY multiplicity sensitivity for the Comment 15 heterogeneity-change analysis.

Run from the repo folder with explicit inputs:
  python IPD_comment15_multiplicity_sensitivity.py --intervals <intervals.csv> --summary <summary.json>

Reads ONLY the two aggregate files already produced by IPD_comment15_heterogeneity_change.py.
Uses the EXISTING two-sided p-values (p_value_exploratory_unadjusted); no p-value is recomputed,
no model is refit, no raw data are read, and no existing file is modified.

Selection: singleton_treatment == 'adjust', wave_or_contrast == 'NFHS-5_minus_NFHS-4',
quantity == 'change_in_heterogeneity_gap'.
Families (Holm step-down, alpha 0.05, applied separately to each):
  primary_threshold_1        10 tests at support threshold 1  (2 analyses x 5 contrasts)
  supplementary_threshold_30 the corresponding 10 tests at support threshold 30
Contrasts: rural vs urban, and wealth quintiles 2-5 vs quintile 1.

Status: an added exploratory sensitivity, NOT a prospectively specified confirmatory analysis.
Estimates, SEs, original p-values and CIs are preserved unchanged. The CIs remain the original
marginal, UNADJUSTED 95% intervals; they are NOT simultaneous and NOT Holm-adjusted.
Writes new aggregate CSV/JSON only and refuses to overwrite an existing output folder.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

CHANGE = 'NFHS-5_minus_NFHS-4'
QUANTITY = 'change_in_heterogeneity_gap'
RULE = 'adjust'
ALPHA = 0.05
ANALYSES = ('all_narrow', 'first_delivery')
CONTRAST_GROUP = {'rural_minus_urban': 'residence_2_minus_1',
                  'quintile_2_minus_quintile_1': 'wealth_index_2_minus_1',
                  'quintile_3_minus_quintile_1': 'wealth_index_3_minus_1',
                  'quintile_4_minus_quintile_1': 'wealth_index_4_minus_1',
                  'quintile_5_minus_quintile_1': 'wealth_index_5_minus_1'}
CONTRASTS = tuple(CONTRAST_GROUP)
FAMILIES = {1: ('primary_threshold_1', 'primary'), 30: ('supplementary_threshold_30', 'supplementary_sensitivity')}
FAMILY_SIZE = len(ANALYSES) * len(CONTRASTS)
P = 'p_value_exploratory_unadjusted'
THRESHOLD = 'reference_minimum_each_sector_each_wave'
REQUIRED = ['analysis', THRESHOLD, 'contrast', 'group', 'wave_or_contrast', 'quantity', 'estimate_pp', 'se_pp',
            'ci_lower_pp', 'ci_upper_pp', 't_statistic', P, 'singleton_treatment', 'conservative_design_df']
KEY = ['analysis', THRESHOLD, 'contrast', 'group', 'wave_or_contrast', 'singleton_treatment']
NUMERIC = ['estimate_pp', 'se_pp', 'ci_lower_pp', 'ci_upper_pp', 't_statistic', P]
CI_LABEL = 'UNADJUSTED marginal 95% CI from the original analysis; NOT simultaneous and NOT Holm-adjusted'
STATUS_LABEL = ('Added EXPLORATORY multiplicity sensitivity (Holm step-down, alpha 0.05, within the declared family); '
                'not prospectively specified; not a confirmatory analysis; association only, no causal interpretation')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def holm(p):
    """Holm step-down adjusted p-values (same definition as R p.adjust(method='holm'))."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(m)
    ranks = np.empty(m, int)
    running = 0.
    for rank, index in enumerate(order, start=1):
        running = max(running, (m - rank + 1) * p[index])
        adjusted[index] = min(1., running)
        ranks[index] = rank
    return adjusted, ranks


def self_check_holm():
    # Hand-verified: sorted 0.005,0.01,0.03,0.04 -> x4,x3,x2,x1 = 0.02,0.03,0.06,0.04 -> running max 0.02,0.03,0.06,0.06.
    adjusted, ranks = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06, 0.02], rtol=0, atol=1e-15) and ranks.tolist() == [2, 4, 3, 1]
    assert np.allclose(holm([0.9, 0.95])[0], [1., 1.], rtol=0, atol=1e-15)      # capped at 1
    assert np.allclose(holm([0.2])[0], [0.2], rtol=0, atol=1e-15)               # family of one is unchanged
    tied, _ = holm([0.01, 0.01, 0.5])
    assert np.allclose(tied, [0.03, 0.03, 0.5], rtol=0, atol=1e-15)             # ties get equal adjusted values
    shuffled = np.array([0.3, 0.001, 0.2, 0.04])
    a1, _ = holm(shuffled)
    a2, _ = holm(shuffled[::-1])
    assert np.allclose(a1, a2[::-1], rtol=0, atol=1e-15)                        # order of input does not matter


def load_inputs(intervals_path, summary_path):
    frame = pd.read_csv(intervals_path)
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError('Intervals file lacks required columns: ' + str(missing))
    if frame.duplicated(KEY).any():
        raise ValueError('Ambiguous input: duplicated analysis/threshold/contrast/wave/rule rows.')
    summary = json.loads(Path(summary_path).read_text())
    if not str(summary.get('status', '')).startswith('completed'):
        raise ValueError('Summary status is not a completed run: ' + repr(summary.get('status')))
    if summary.get('primary_singleton_treatment') != RULE:
        raise ValueError('Summary primary singleton treatment is not ' + RULE)
    if sorted(summary.get('support_minima_each_sector_each_wave', [])) != sorted(FAMILIES):
        raise ValueError('Summary support thresholds differ from the expected 1 and 30.')
    return frame, summary


def select_rows(frame):
    wide = frame.loc[frame.singleton_treatment.eq(RULE) & frame.wave_or_contrast.eq(CHANGE)]
    chosen = wide.loc[wide.quantity.eq(QUANTITY)]
    if len(chosen) != len(wide):
        raise ValueError('Ambiguous input: NFHS-5 minus NFHS-4 rows with an unexpected quantity label.')
    if set(chosen[THRESHOLD].unique()) != set(FAMILIES):
        raise ValueError('Unexpected support thresholds among selected rows: ' + str(sorted(chosen[THRESHOLD].unique())))
    expected = set(itertools.product(ANALYSES, CONTRASTS))
    pieces = []
    for threshold, (family, role) in FAMILIES.items():
        fam = chosen.loc[chosen[THRESHOLD].eq(threshold)].copy()
        observed = list(zip(fam.analysis, fam.contrast))
        if len(fam) != FAMILY_SIZE or len(set(observed)) != FAMILY_SIZE or set(observed) != expected:
            raise ValueError(f'{family}: need exactly {FAMILY_SIZE} unique expected analysis/contrast rows; '
                             f'found {len(fam)} rows, {len(set(observed))} unique.')
        for row in fam.itertuples():
            if CONTRAST_GROUP[row.contrast] != row.group:
                raise ValueError(f'{family}: contrast {row.contrast} has unexpected group {row.group}.')
        values = fam[NUMERIC].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f'{family}: non-finite estimate, SE, CI, t statistic or p-value.')
        p = fam[P].to_numpy(float)
        if not ((p >= 0) & (p <= 1)).all():
            raise ValueError(f'{family}: p-value outside [0, 1].')
        est, se, t = (fam[c].to_numpy(float) for c in ('estimate_pp', 'se_pp', 't_statistic'))
        if not (se > 0).all() or not np.allclose(est / se, t, rtol=1e-6, atol=1e-9):
            raise ValueError(f'{family}: estimate/SE does not match the stored t statistic.')
        if not ((fam.ci_lower_pp < fam.estimate_pp) & (fam.estimate_pp < fam.ci_upper_pp)).all():
            raise ValueError(f'{family}: estimate lies outside its stored confidence interval.')
        fam['family'] = family
        fam['family_role'] = role
        pieces.append(fam)
    return pd.concat(pieces, ignore_index=True)


def cross_check_summary(selected, summary):
    """The supplied summary repeats these rows; both files must agree exactly."""
    rows = summary.get('primary_change_results')
    if not isinstance(rows, list) or len(rows) != 2 * FAMILY_SIZE:
        raise ValueError('Summary primary_change_results does not hold the expected 20 rows.')
    index = selected.set_index(['analysis', THRESHOLD, 'contrast'])
    for r in rows:
        key = (r['analysis'], int(r[THRESHOLD]), r['contrast'])
        if key not in index.index:
            raise ValueError('Summary row absent from intervals CSV: ' + str(key))
        row = index.loc[key]
        for col in ('estimate_pp', 'se_pp', 'ci_lower_pp', 'ci_upper_pp', P):
            if not np.isclose(float(row[col]), float(r[col]), rtol=1e-12, atol=1e-15):
                raise ValueError(f'Summary and CSV disagree for {key} {col}.')


def adjust_families(selected):
    out = []
    for threshold, (family, role) in FAMILIES.items():
        fam = selected.loc[selected.family.eq(family)].copy()
        adjusted, ranks = holm(fam[P].to_numpy(float))
        if not (adjusted >= fam[P].to_numpy(float) - 1e-15).all() or not (adjusted <= 1).all():
            raise RuntimeError(family + ': Holm adjustment violated basic bounds.')
        fam['family_size'] = len(fam)
        fam['holm_rank_by_p'] = ranks
        fam['holm_adjusted_p_value'] = adjusted
        fam['unadjusted_p_at_most_0_05'] = fam[P].to_numpy(float) <= ALPHA
        fam['reject_holm_at_0_05'] = adjusted <= ALPHA
        fam['holm_overturns_unadjusted_result'] = fam.unadjusted_p_at_most_0_05 & ~fam.reject_holm_at_0_05
        fam['ci_label'] = CI_LABEL
        fam['multiplicity_label'] = STATUS_LABEL
        out.append(fam.sort_values(['analysis', 'holm_rank_by_p'], kind='stable'))
    result = pd.concat(out, ignore_index=True)
    if (result.reject_holm_at_0_05 & ~result.unadjusted_p_at_most_0_05).any():
        raise RuntimeError('Holm rejected a test that the unadjusted p-value did not reject.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--intervals', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or args.repo_root.resolve() / 'extensions_work/12_narrow_scope/outputs/comment15_multiplicity_sensitivity'
    if out.exists():
        raise FileExistsError('Existing output preserved; choose a new --output.')
    for path in (args.intervals, args.summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    self_check_holm()
    frame, summary = load_inputs(args.intervals, args.summary)
    selected = select_rows(frame)
    cross_check_summary(selected, summary)
    result = adjust_families(selected)
    columns = ['family', 'family_role', 'family_size', 'analysis', THRESHOLD, 'contrast', 'group', 'wave_or_contrast',
               'quantity', 'estimate_pp', 'se_pp', 'ci_lower_pp', 'ci_upper_pp', 'ci_label', P, 'holm_rank_by_p',
               'holm_adjusted_p_value', 'unadjusted_p_at_most_0_05', 'reject_holm_at_0_05',
               'holm_overturns_unadjusted_result', 'singleton_treatment', 'conservative_design_df', 'multiplicity_label']
    result = result[columns]
    families = {}
    for threshold, (family, role) in FAMILIES.items():
        fam = result.loc[result.family.eq(family)]
        label = lambda q: [f'{a}:{c}' for a, c in zip(q.analysis, q.contrast)]
        families[family] = {
            'role': role, 'support_threshold_each_sector_each_wave': threshold, 'size': int(len(fam)),
            'members': label(fam.sort_values(['analysis', 'contrast'])),
            'definition': f'{len(ANALYSES)} analyses x {len(CONTRASTS)} contrasts, NFHS-5 minus NFHS-4 change in heterogeneity gap, singleton rule {RULE}',
            'method': 'Holm step-down on the existing unadjusted two-sided p-values; reject if Holm-adjusted p <= 0.05',
            'n_unadjusted_p_at_most_0_05': int(fam.unadjusted_p_at_most_0_05.sum()),
            'n_reject_holm_at_0_05': int(fam.reject_holm_at_0_05.sum()),
            'rejected_by_holm': label(fam.loc[fam.reject_holm_at_0_05]),
            'unadjusted_significant_but_not_holm': label(fam.loc[fam.holm_overturns_unadjusted_result])}
    report = {
        'status': 'completed_exploratory_multiplicity_sensitivity',
        'description': STATUS_LABEL,
        'ci_note': CI_LABEL,
        'family_note': ('Holm controls the familywise error rate only within each declared 10-test family, under arbitrary '
                        'dependence but conservatively. The threshold-30 family re-tests the same estimand on a restricted reference, '
                        'so it is a sensitivity analysis, not independent evidence; the two analyses share births. No adjustment '
                        'is made across families, singleton rules, or earlier Comment 15 analyses.'),
        'selection': {'singleton_treatment': RULE, 'wave_or_contrast': CHANGE, 'quantity': QUANTITY,
                      'contrasts': list(CONTRASTS), 'analyses': list(ANALYSES), 'alpha': ALPHA},
        'families': families,
        'inputs': {'intervals_csv': str(args.intervals.resolve()), 'intervals_csv_sha256': sha256(args.intervals),
                   'summary_json': str(args.summary.resolve()), 'summary_json_sha256': sha256(args.summary),
                   'input_summary_script_sha256': summary.get('script_sha256'),
                   'input_summary_existing_intervals_sha256': summary.get('existing_intervals_sha256')},
        'self_checks': {'holm_reference_cases': 'passed', 'exact_family_membership': 'passed',
                        'finite_p_values_in_0_1': 'passed', 't_statistic_equals_estimate_over_se': 'passed',
                        'summary_json_matches_csv_rows': 'passed', 'holm_bounds_and_monotonicity': 'passed'},
        'script_sha256': sha256(Path(__file__)),
        'p_values_recomputed': False, 'models_refit': False, 'raw_data_accessed': False, 'existing_outputs_changed': False}
    out.mkdir(parents=True)
    with (out / 'comment15_multiplicity_sensitivity.csv').open('x', newline='', encoding='utf-8') as handle:
        result.to_csv(handle, index=False)
    with (out / 'comment15_multiplicity_sensitivity_summary.json').open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False))
    print(result[['family', 'analysis', 'contrast', P, 'holm_adjusted_p_value', 'reject_holm_at_0_05']].to_string(index=False))
    print('Aggregate outputs saved: ' + str(out))
    print('Share comment15_multiplicity_sensitivity.csv and comment15_multiplicity_sensitivity_summary.json only.')


if __name__ == '__main__':
    main()
