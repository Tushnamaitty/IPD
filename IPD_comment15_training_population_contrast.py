"""Comment 15 training-population SENSITIVITY on a first-delivery reference.

Place beside IPD_comment15_analysis.py, IPD_narrow_facility_analysis.py and the corrected
IPD_comment15_heterogeneity_change.py. Run --self-test first (synthetic data only), then:
    python IPD_comment15_training_population_contrast.py --repo-root .

Question: within each wave, how much does the training population change the private-minus-public
gap when everything else is held fixed?
  Model A: trained on ALL narrow births of the wave (codes 21/31, same cohort rules).
  Model B: trained on FIRST-DELIVERY narrow births of the wave.
Both models use the identical design matrix: same predictors, vocabulary, scales, missing coding,
private x residence and private x wealth interactions, and the fixed 1e-5 ridge. Birth order is
omitted from BOTH models, so only the training population differs.
Both are evaluated on the identical first-delivery reference (same rows, same weights): the
equal-wave (50/50 NFHS-4/NFHS-5) survey-weighted pooled first-delivery births in supported
state x residence x wealth cells (>=1 or >=30 births per sector per wave, classified on the
first-delivery cohort, exactly as in the existing first-delivery Comment 15 analysis).

Reported (percentage points, association only):
  model_A gap, model_B gap, and B - A, per wave, with SEs and 95% intervals.

Uncertainty: joint fitted-coefficient + empirical-reference influence for all four estimates
(A and B, both waves) is stacked over the SAME rows (B training rows are a subset of A training
rows; the reference rows are shared). Survey covariance of the stacked influence is computed once
per wave on the all-narrow design (state/stratum/PSU keys, household-roster zero-domain PSUs,
singleton adjust/average/zero rules) and waves are summed as independent. The B - A SE uses
L V L-transpose, so overlapping training samples and shared references are accounted for.
The fits are NOT treated as independent.

The existing first-delivery Comment 15 rows are reproduced before anything is written (point
estimates for every rule; SEs under the zero rule, which is design-domain invariant).
Existing outputs are never overwritten; only aggregate CSV/JSON files are written.
This is a training-population sensitivity, exploratory and associational, not a causal estimate.
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
from scipy.special import expit
from scipy.stats import t

HERE = Path(__file__).resolve().parent
WAVES = ('NFHS-4', 'NFHS-5')
RULES = ('adjust', 'average', 'zero_sensitivity_only')
PRIMARY_RULE = 'adjust'
SUPPORT_MINIMA = (1, 30)
ANALYSIS_LABEL = 'training_population_sensitivity_first_delivery_reference'
REFERENCE_LABEL = 'first_delivery_births_equal_wave_pooled_supported_cells'
MODEL_A = 'A_all_narrow_trained'
MODEL_B = 'B_first_delivery_trained'
QA = 'model_A_all_narrow_trained_gap'
QB = 'model_B_first_delivery_trained_gap'
QD = 'B_minus_A_gap_difference'
COLS = [(MODEL_A, 0), (MODEL_A, 1), (MODEL_B, 0), (MODEL_B, 1)]
REPRODUCTION_TOLERANCE_PP = 1e-6
INFERENCE_LABEL = ('EXPLORATORY training-population sensitivity on a first-delivery reference: approximate 95% Taylor '
                   't interval and unadjusted two-sided p-value; no multiplicity adjustment; association only; '
                   'conditional on fixed ridge, model, support rule and geography mapping')


# ----------------------------------------------------------------------------- module loading

def load_het():
    path = HERE / 'IPD_comment15_heterogeneity_change.py'
    if not path.exists():
        raise FileNotFoundError('Place the corrected IPD_comment15_heterogeneity_change.py beside this script.')
    spec = importlib.util.spec_from_file_location('ipd_comment15_het', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('load_modules', 'load_inputs', 'verify_toggles', 'reference_means', 'corrected_limitations'):
        if not hasattr(module, name):
            raise ValueError('IPD_comment15_heterogeneity_change.py lacks %s; copy the latest corrected version.' % name)
    return module


def design():
    """Rows of L map the four estimates (A4, A5, B4, B5) to every reported quantity."""
    pos = {c: i for i, c in enumerate(COLS)}
    keys, rows = [], []
    for w in (0, 1):
        for quantity, coefficients in [(QA, [((MODEL_A, w), 1.)]), (QB, [((MODEL_B, w), 1.)]),
                                       (QD, [((MODEL_B, w), 1.), ((MODEL_A, w), -1.)])]:
            v = np.zeros(len(COLS))
            for column, c in coefficients:
                v[pos[column]] += c
            keys.append((quantity, WAVES[w]))
            rows.append(v)
    return keys, np.array(rows)


# ----------------------------------------------------------------------------- data preparation

def prepare(base, c15, het, frames, minimum):
    pieces = []
    for wave, frame in enumerate(frames):
        q = frame.copy()
        q['wave'] = wave
        pieces.append(q)
    f = pd.concat(pieces, ignore_index=True)
    is_fd = f.delivery_order.eq(1).to_numpy()
    if not is_fd.any():
        raise ValueError('No first-delivery births.')
    for wave in (0, 1):
        if not (is_fd & f.wave.eq(wave).to_numpy()).any():
            raise ValueError('A wave lacks first-delivery births.')
    fd_index = np.flatnonzero(is_fd)
    f_fd = f.loc[is_fd].reset_index(drop=True)
    # One design for both models: birth order omitted from BOTH (first_delivery=True encoding),
    # vocabulary/scales taken from the pooled all-narrow cohort and verified against the
    # first-delivery cohort encoded on its own, so the design equals the existing first-delivery design.
    x, schema = base.encode(f, first_delivery=True)
    labels = list(schema['labels'])
    x_check, schema_check = base.encode(f_fd, first_delivery=True)
    if (schema_check['labels'] != labels or schema_check['categories'] != schema['categories']
            or not np.array_equal(x_check, x[is_fd])):
        raise ValueError('Vocabulary or design differs between all-narrow and first-delivery encodings; '
                         'models would not be matched. Review categories before analysis.')
    del x_check
    c15.INTERACTIONS = []
    c15.INTERACTION_VALUES = {}
    extra = []
    for variable, levels in [('residence', [2]), ('wealth_index', [2, 3, 4, 5])]:
        for level in levels:  # same construction and order as IPD_comment15_analysis.analysis
            value = f[variable].eq(level).to_numpy(float)
            index = x.shape[1] + len(extra)
            extra.append(f.sector.to_numpy(float) * value)
            c15.INTERACTIONS.append(index)
            c15.INTERACTION_VALUES[index] = value
            labels.append(f'private:{variable}={level}')
    x = np.column_stack([x] + extra)
    if len(labels) != x.shape[1]:
        raise RuntimeError('Label/design width mismatch.')
    interactions = list(c15.INTERACTIONS)
    full_values = dict(c15.INTERACTION_VALUES)
    x0, x1 = het.verify_toggles(c15, f, x, labels)
    x_fd = x[is_fd]
    fd_values = {i: v[is_fd] for i, v in full_values.items()}
    c15.INTERACTION_VALUES = fd_values
    x0_fd, x1_fd = het.verify_toggles(c15, f_fd, x_fd, labels)
    c15.INTERACTION_VALUES = full_values
    # Reference: first-delivery rows only; support classified on the first-delivery cohort.
    m_fd = c15.support(f_fd, minimum)
    if not m_fd.any():
        raise ValueError('No common supported first-delivery reference cells.')
    mask_full = np.zeros(len(f), bool)
    mask_full[fd_index] = m_fd
    wr_full = c15.reference_weights(f, mask_full)
    wr_fd = c15.reference_weights(f_fd, m_fd)
    if wr_full[~is_fd].any() or not np.allclose(wr_full[is_fd], wr_fd, rtol=0, atol=1e-15):
        raise RuntimeError('Models A and B do not share an identical reference (rows and weights).')
    return {'f': f, 'f_fd': f_fd, 'is_fd': is_fd, 'fd_index': fd_index, 'x': x, 'x_fd': x_fd,
            'x0': x0, 'x1': x1, 'x0_fd': x0_fd, 'x1_fd': x1_fd, 'mask_full': mask_full, 'm_fd': m_fd,
            'labels': labels, 'interactions': interactions, 'full_values': full_values, 'fd_values': fd_values}


def fit_models(base, ctx, tag='', quiet=False):
    f, x, is_fd = ctx['f'], ctx['x'], ctx['is_fd']
    y = f.outcome.to_numpy(float)
    w = f.weight.to_numpy(float)
    betas_a, betas_b, diagnostics = [], [], {'A': [], 'B': []}
    for wave in (0, 1):
        train = f.wave.eq(wave).to_numpy()
        for name, selected, store in (('A all narrow', train, betas_a), ('B first delivery', train & is_fd, betas_b)):
            if not quiet:
                print(f'{tag} wave {wave + 4} model {name}: fitting {selected.sum():,} births', flush=True)
            beta, diagnostic = base.fit_model(x[selected], y[selected], w[selected])
            store.append(beta)
            diagnostics[name[0]].append(diagnostic)
    return betas_a, betas_b, diagnostics


def direct_points(het, ctx, betas_a, betas_b):
    """Independent sector-gap means on the identical reference, no influence functions."""
    a = het.reference_means(ctx['f'], ctx['x0'], ctx['x1'], betas_a, ctx['mask_full'])
    b = het.reference_means(ctx['f_fd'], ctx['x0_fd'], ctx['x1_fd'], betas_b, ctx['m_fd'])
    return np.concatenate([a, b])


def influences(base, c15, ctx, betas_a, betas_b):
    """Stacked joint model+reference influence over ALL rows: columns A4, A5, B4, B5."""
    c15.INTERACTIONS = list(ctx['interactions'])
    c15.INTERACTION_VALUES = ctx['full_values']
    p_a, u_a = c15.targets(base, ctx['f'], ctx['x'], betas_a, ctx['mask_full'])
    c15.INTERACTION_VALUES = ctx['fd_values']
    p_b, u_b_fd = c15.targets(base, ctx['f_fd'], ctx['x_fd'], betas_b, ctx['m_fd'])
    c15.INTERACTION_VALUES = ctx['full_values']
    u_b = np.zeros((len(ctx['f']), 2))
    u_b[ctx['fd_index']] = u_b_fd  # first-delivery model contributes only on first-delivery rows
    return np.concatenate([p_a, p_b]), np.column_stack([u_a, u_b])


def covariance_for(c15, ctx, u, rosters):
    covariance, design_info = {}, {}
    f = ctx['f']
    for wave in (0, 1):
        train = f.wave.eq(wave).to_numpy()
        covariance[wave], design_info[wave] = c15.covariance(f.loc[train], u[train], rosters[wave])
    return {rule: covariance[0][rule] + covariance[1][rule] for rule in covariance[0]}, design_info


def compute(base, c15, het, frames, rosters, minimum, tag='', quiet=False):
    ctx = prepare(base, c15, het, frames, minimum)
    betas_a, betas_b, fit_diagnostics = fit_models(base, ctx, tag, quiet)
    theta, u = influences(base, c15, ctx, betas_a, betas_b)
    if np.max(np.abs(theta - direct_points(het, ctx, betas_a, betas_b))) > 1e-12:
        raise RuntimeError('Influence-function targets disagree with independently toggled counterfactuals.')
    if np.max(np.abs(u.sum(axis=0))) > 1e-7:
        raise RuntimeError('Influence centering failed.')
    cov, design_info = covariance_for(c15, ctx, u, rosters)
    if set(cov) != set(RULES):
        raise RuntimeError('Unexpected singleton rules.')
    for rule, matrix in cov.items():
        if np.linalg.eigvalsh(matrix).min() < -1e-10:
            raise RuntimeError('Covariance not positive semidefinite: ' + rule)
    f, is_fd, mask = ctx['f'], ctx['is_fd'], ctx['mask_full']
    w = f.weight.to_numpy(float)
    audit = {'reference_n': int(mask.sum()), 'model_terms': len(ctx['labels']), 'design': design_info,
             'fit_diagnostics': fit_diagnostics, 'design_domain': 'all narrow births (state/stratum/PSU, roster zero-domain PSUs)',
             'waves': {}}
    for wave in (0, 1):
        train = f.wave.eq(wave).to_numpy()
        fd_wave = train & is_fd
        audit['waves'][WAVES[wave]] = {
            'training_n_model_A_all_narrow': int(train.sum()), 'training_n_model_B_first_delivery': int(fd_wave.sum()),
            'reference_n': int((mask & train).sum()),
            'excluded_first_delivery_reference_weight_pct': float(100 * w[fd_wave & ~mask].sum() / w[fd_wave].sum())}
    return {'ctx': ctx, 'betas_a': betas_a, 'betas_b': betas_b, 'theta': theta, 'influence': u, 'cov': cov, 'audit': audit}


# ----------------------------------------------------------------------------- inference

def contrast_rows(result, minimum, keys, L):
    df = min(d['df'] for d in result['audit']['design'].values())
    crit = float(t.ppf(.975, df))
    est = L @ result['theta']
    rows, covariance_rows, ratios = [], [], []
    for rule in RULES:
        V = result['cov'][rule]
        LV = L @ V @ L.T
        for k, (quantity, wave) in enumerate(keys):
            se = float(100 * np.sqrt(max(0., LV[k, k])))
            if not np.isfinite(se) or se <= 0:
                raise RuntimeError('Non-positive contrast SE: ' + quantity + ' ' + wave)
            point = float(100 * est[k])
            independent = float(100 * np.sqrt(max(0., np.sum(L[k] ** 2 * np.diag(V)))))
            stat = point / se
            rows.append({
                'analysis_label': ANALYSIS_LABEL, 'reference_population': REFERENCE_LABEL,
                'reference_minimum_each_sector_each_wave': minimum, 'wave': wave, 'quantity': quantity,
                'estimate_pp': point, 'se_pp': se, 'ci_lower_pp': point - crit * se, 'ci_upper_pp': point + crit * se,
                't_statistic': float(stat), 'p_value_exploratory_unadjusted': float(2 * t.sf(abs(stat), df)),
                'singleton_treatment': rule, 'primary_rule': rule == PRIMARY_RULE, 'conservative_design_df': df,
                'se_if_fits_treated_independent_pp': independent, 'inference_status': INFERENCE_LABEL})
            if quantity == QD and rule == PRIMARY_RULE:
                ratios.append(abs(independent - se) / se)
        for a, (qa, wa) in enumerate(keys):
            for b, (qb, wb) in enumerate(keys):
                covariance_rows.append({
                    'analysis_label': ANALYSIS_LABEL, 'reference_minimum_each_sector_each_wave': minimum,
                    'singleton_treatment': rule, 'row_quantity': qa, 'row_wave': wa, 'column_quantity': qb,
                    'column_wave': wb, 'covariance_pp2': float(1e4 * LV[a, b])})
    return rows, covariance_rows, float(max(ratios))


def reproduce_existing(rows, existing, minima, strict, tolerance=REPRODUCTION_TOLERANCE_PP):
    """Model B is the existing first-delivery model (birth order is absent there too)."""
    new = pd.DataFrame(rows)
    new = new.loc[new.quantity.eq(QB) & new.reference_minimum_each_sector_each_wave.isin(minima)]
    key = ['reference_minimum_each_sector_each_wave', 'wave', 'singleton_treatment']
    old = existing.loc[existing.analysis.eq('first_delivery') & existing.group.eq('overall')
                       & existing.wave_or_contrast.isin(WAVES)
                       & existing.reference_minimum_each_sector_each_wave.isin(minima),
                       ['reference_minimum_each_sector_each_wave', 'wave_or_contrast', 'singleton_treatment',
                        'estimate_pp', 'se_pp']].rename(columns={'wave_or_contrast': 'wave', 'estimate_pp': 'estimate_existing',
                                                                'se_pp': 'se_existing'})
    merged = new.merge(old, on=key, how='left', validate='one_to_one', indicator=True)
    if len(merged) != len(minima) * 2 * len(RULES) or not merged['_merge'].eq('both').all():
        raise RuntimeError('Existing first-delivery rows missing for reproduction check.')
    estimate_difference = float((merged.estimate_pp - merged.estimate_existing).abs().max())
    se_difference = {rule: float((q.se_pp - q.se_existing).abs().max()) for rule, q in merged.groupby('singleton_treatment')}
    if estimate_difference > tolerance:
        raise RuntimeError('Model B point estimates do not reproduce existing first-delivery results: %g' % estimate_difference)
    enforced = list(RULES) if strict else ['zero_sensitivity_only']
    for rule in enforced:
        if se_difference[rule] > tolerance:
            raise RuntimeError('Model B SE does not reproduce existing results (%s): %g' % (rule, se_difference[rule]))
    return {'rows_compared': int(len(merged)), 'max_abs_estimate_difference_pp': estimate_difference,
            'max_abs_se_difference_pp_by_rule': se_difference, 'se_rules_enforced': enforced,
            'tolerance_pp': tolerance,
            'note': ('adjust/average SEs can differ slightly in real data because the joint design domain is all narrow '
                     'births (more state-strata/zero-domain PSUs than the first-delivery-only design); the zero rule is invariant.')}


# ----------------------------------------------------------------------------- self-test

def synthetic(base, mixed=True, seed=79):
    rng = np.random.default_rng(seed)
    frames, rosters = [], []
    for wave in (0, 1):
        n = 1200
        f = pd.DataFrame({
            'state': np.repeat([1, 2], n // 2), 'stratum': np.tile(np.repeat([1, 2], n // 4), 2),
            'psu': np.tile(np.arange(n // 2) % 30 + 1, 2), 'birth_order': rng.integers(1, 4, n),
            'delivery_order': rng.choice([1, 2, 3], n, p=[.4, .35, .25]) if mixed else np.ones(n),
            'education_years': rng.integers(0, 20, n), 'wealth_index': rng.integers(1, 6, n),
            'residence': rng.integers(1, 3, n), 'religion': rng.integers(1, 3, n),
            'social_group': rng.integers(1, 4, n), 'twin_order': np.zeros(n), 'sector': rng.integers(0, 2, n),
            'weight': rng.uniform(.2, 2, n)})
        later = (f.delivery_order > 1).astype(float)
        logit = (-1 + .5 * f.sector + .4 * later + .3 * f.sector * later
                 + wave * f.sector * (.5 * f.residence.eq(2) - .1 * f.wealth_index))
        f['outcome'] = rng.binomial(1, expit(logit))
        frames.append(f)
        rosters.append(base.design_pairs(f))
    return frames, rosters


def perturbed_points(base, het, ctx, index, sign, eps):
    f = ctx['f'].copy()
    f.loc[index, 'weight'] *= 1 + sign * eps
    other = dict(ctx)
    other['f'] = f
    other['f_fd'] = f.loc[ctx['is_fd']].reset_index(drop=True)
    betas_a, betas_b, _ = fit_models(base, other, quiet=True)
    return direct_points(het, other, betas_a, betas_b)


def self_test(base, c15, het):
    keys, L = design()
    assert L.shape == (6, 4) and len(keys) == 6
    # Existing first-delivery analysis on the same synthetic data (run FIRST: it resets c15 globals).
    frames, rosters = synthetic(base)
    existing_list, audit_existing = c15.analysis(base, frames, rosters, True, 1)
    existing_rows = pd.DataFrame(existing_list)
    result = compute(base, c15, het, frames, rosters, 1, 'self-test', quiet=True)
    ctx = result['ctx']
    f, is_fd = ctx['f'], ctx['is_fd']
    assert 0 < is_fd.sum() < len(f) and (ctx['mask_full'] <= is_fd).all() and ctx['mask_full'].any()
    assert result['audit']['reference_n'] == audit_existing['reference_n']
    # 1. Rows, arithmetic, labels.
    rows, cov_rows, overlap_effect = contrast_rows(result, 1, keys, L)
    assert len(rows) == 18 and len(cov_rows) == 3 * 36
    est = {(r['quantity'], r['wave']): r['estimate_pp'] for r in rows if r['singleton_treatment'] == PRIMARY_RULE}
    for wave in WAVES:
        assert abs(est[(QD, wave)] - (est[(QB, wave)] - est[(QA, wave)])) < 1e-10
    for r in rows:
        assert r['ci_lower_pp'] < r['estimate_pp'] < r['ci_upper_pp'] and r['se_pp'] > 0
        assert r['analysis_label'] == ANALYSIS_LABEL and r['inference_status'].startswith('EXPLORATORY')
    # 2. Model B equals the existing first-delivery Comment 15 analysis (all rules in this fixture).
    assert reproduce_existing(rows, existing_rows, (1,), strict=True)['rows_compared'] == 6
    altered = existing_rows.copy()
    altered['se_pp'] *= 1.001
    try:
        reproduce_existing(rows, altered, (1,), strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Altered existing SEs were accepted.')
    # 3. Joint covariance equals the survey covariance of directly differenced influences.
    direct, _ = covariance_for(c15, ctx, result['influence'] @ L.T, rosters)
    for rule in RULES:
        assert np.allclose(direct[rule], L @ result['cov'][rule] @ L.T, rtol=1e-9, atol=1e-12)
    # 4. Overlap matters: the independent-fits SE shortcut gives a different B - A SE.
    assert overlap_effect > 1e-4, 'Overlapping training samples had no effect on the B - A SE.'
    # 5. Joint perturbation: re-fit all four models and the shared reference; compare with influence rows.
    wave0 = f.wave.eq(0).to_numpy()
    wave1 = f.wave.eq(1).to_numpy()
    picks = [int(np.flatnonzero(wave0 & is_fd & ctx['mask_full'])[0]), int(np.flatnonzero(wave1 & ~is_fd)[0]),
             int(np.flatnonzero(wave1 & is_fd)[2])]
    for i in picks:
        eps = 1e-3
        plus = perturbed_points(base, het, ctx, i, 1, eps)
        minus = perturbed_points(base, het, ctx, i, -1, eps)
        assert np.max(np.abs((plus - minus) / (2 * eps) - result['influence'][i])) < 1e-6, 'weight perturbation failed'
    assert np.all(result['influence'][picks[1], 2:] == 0.), 'Non-first-delivery row leaked into model B influence.'
    # 6. Identical training populations: B - A is exactly zero for estimates and influences.
    same_frames, same_rosters = synthetic(base, mixed=False)
    same = compute(base, c15, het, same_frames, same_rosters, 1, 'self-test same', quiet=True)
    assert np.max(np.abs(same['theta'][2:] - same['theta'][:2])) < 1e-12
    assert np.max(np.abs(same['influence'][:, 2:] - same['influence'][:, :2])) < 1e-12
    # 7. Negative control: a wrongly toggled interaction indicator must be detected.
    saved = ctx['full_values']
    bad = dict(saved)
    first = ctx['interactions'][0]
    bad[first] = 1. - np.asarray(saved[first])
    c15.INTERACTIONS = list(ctx['interactions'])
    c15.INTERACTION_VALUES = bad
    try:
        het.verify_toggles(c15, f, ctx['x'], ctx['labels'])
    except RuntimeError:
        pass
    else:
        raise AssertionError('Wrong interaction toggle was not detected.')
    finally:
        c15.INTERACTION_VALUES = saved
    # 8. Matched-design negative control: a category present only in all-narrow births must be rejected.
    tweak = [frames[0].copy(), frames[1].copy()]
    tweak[0].loc[tweak[0].delivery_order.gt(1).to_numpy().nonzero()[0][:3], 'religion'] = 9
    try:
        prepare(base, c15, het, tweak, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched vocabulary was accepted.')
    print(json.dumps({'status': 'synthetic_checks_passed', 'model_B_reproduces_existing_first_delivery': 'PASS',
                      'contrast_arithmetic_and_labels': 'PASS', 'joint_covariance_equals_direct_difference': 'PASS',
                      'overlap_changes_B_minus_A_se': 'PASS', 'four_model_weight_perturbation_with_shared_reference': 'PASS',
                      'identical_training_populations_give_zero_contrast': 'PASS',
                      'interaction_toggle_negative_control': 'PASS', 'matched_vocabulary_negative_control': 'PASS',
                      'shared_reference_rows_and_weights': 'PASS'}, indent=2))


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--existing', type=Path, help='Existing comment15_shared_reference output folder.')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    het = load_het()
    base, c15 = het.load_modules()
    if args.self_test:
        self_test(base, c15, het)
        return
    root = args.repo_root.resolve()
    out = args.output or root / 'extensions_work/12_narrow_scope/outputs/comment15_training_population_contrast'
    existing_dir = args.existing or root / 'extensions_work/12_narrow_scope/outputs/comment15_shared_reference'
    baseline = root / 'extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json'
    if out.exists():
        raise FileExistsError('Existing output preserved; choose a new --output.')
    existing_summary = json.loads((existing_dir / 'comment15_summary.json').read_text())
    existing_intervals = pd.read_csv(existing_dir / 'comment15_intervals.csv')
    for label, path, key in [('comment15 script', Path(c15.__file__), 'script_sha256'),
                             ('base script', Path(base.__file__), 'base_script_sha256'),
                             ('baseline summary', baseline, 'baseline_sha256')]:
        if base.sha256(path) != existing_summary[key]:
            raise ValueError(label + ' differs from the version that produced the existing comment15 outputs '
                             '(check edits or CRLF/LF changes); methods may not be identical.')
    start = time.monotonic()
    frames, rosters, audits, _ = het.load_inputs(base, c15, root, baseline)
    keys, L = design()
    all_rows, covariance_rows, diagnostics, overlap = [], [], {}, []
    for minimum in SUPPORT_MINIMA:
        result = compute(base, c15, het, frames, rosters, minimum, f'min {minimum}')
        result.pop('ctx')
        rows, cov_rows, effect = contrast_rows(result, minimum, keys, L)
        if existing_summary['diagnostics'][f'True_{minimum}']['reference_n'] != result['audit']['reference_n']:
            raise RuntimeError('First-delivery reference size differs from existing comment15 output.')
        all_rows += rows
        covariance_rows += cov_rows
        overlap.append(effect)
        diagnostics[f'first_delivery_reference_min_{minimum}'] = result['audit']
        print(f'min {minimum}: training-population contrasts computed', flush=True)
        del result
    reproduction = reproduce_existing(all_rows, existing_intervals, SUPPORT_MINIMA, strict=False)
    primary = [r for r in all_rows if r['singleton_treatment'] == PRIMARY_RULE]
    summary = {
        'status': 'completed_exploratory_sensitivity_outputs_require_review',
        'analysis_label': ANALYSIS_LABEL, 'inference_label': INFERENCE_LABEL,
        'question': ('Does the training population (all narrow births vs first-delivery births only) change the '
                     'private-minus-public gap evaluated on the same first-delivery reference?'),
        'models': {'A': 'trained on all narrow births in the wave', 'B': 'trained on first-delivery narrow births in the wave',
                   'matching': ('identical predictors, interactions (private x residence, private x wealth 2-5), vocabulary, '
                                'scales, missing coding and fixed ridge; birth order omitted from BOTH; B design verified '
                                'identical to the existing first-delivery design')},
        'reference_rule': ('First-delivery births only, identical rows and weights for A and B; 50 percent weighted NFHS-4 plus '
                           '50 percent weighted NFHS-5 in supported state x residence x wealth cells classified on the '
                           'first-delivery cohort (>=1 or >=30 births per sector per wave)'),
        'facility_codes': {'public': 21, 'private': 31}, 'fixed_ridge': base.RIDGE,
        'support_minima_each_sector_each_wave': list(SUPPORT_MINIMA),
        'primary_singleton_treatment': PRIMARY_RULE, 'singleton_sensitivity': ['average', 'zero_sensitivity_only'],
        'uncertainty': ('Stacked joint fitted-coefficient and shared-empirical-reference influence for A and B in both waves '
                        'over the same rows; one survey covariance per wave on the all-narrow design (state/stratum/PSU, '
                        'roster zero-domain PSUs, singleton rules); waves independent; minimum wave design df; B - A uses '
                        'L V L-transpose so overlapping training samples are not treated as independent'),
        'max_relative_se_change_if_fits_treated_independent_primary_B_minus_A': float(max(overlap)),
        'reproduction_of_existing_first_delivery_rows': reproduction,
        'primary_results': primary, 'diagnostics': diagnostics, 'cohort_audit': audits,
        'script_sha256': base.sha256(Path(__file__)), 'heterogeneity_change_script_sha256': base.sha256(Path(het.__file__)),
        'comment15_script_sha256': base.sha256(Path(c15.__file__)), 'base_script_sha256': base.sha256(Path(base.__file__)),
        'baseline_sha256': base.sha256(baseline), 'existing_intervals_sha256': base.sha256(existing_dir / 'comment15_intervals.csv'),
        'limitations': het.corrected_limitations(base) + [
            'Training-population sensitivity only: model A is evaluated on a first-delivery reference that differs from its all-narrow training population, so A predictions on this reference may extrapolate; B - A is not a causal or policy effect.',
            'Birth order is omitted from both models to isolate the training population; model A therefore differs from the earlier all-narrow specification that included birth order.',
            'Joint variance uses the all-narrow design domain; B marginal SEs under adjust/average singleton rules can differ slightly from the earlier first-delivery-only design, while the zero rule reproduces exactly.',
            'Exploratory intervals and p-values without multiplicity adjustment; waves treated as independent with no cross-wave PSU dependence; minimum design df used.',
            'Reference support classification and geography mapping are fixed; coarse cells do not prove joint overlap.'],
        'elapsed_seconds': round(time.monotonic() - start, 1), 'existing_outputs_changed': False}
    out.mkdir(parents=True)
    pd.DataFrame(all_rows).to_csv(out / 'comment15_training_population_contrast_intervals.csv', index=False)
    pd.DataFrame(covariance_rows).to_csv(out / 'comment15_training_population_contrast_covariance.csv', index=False)
    base.save_json(out / 'comment15_training_population_contrast_summary.json', summary)
    print('Aggregate outputs saved: ' + str(out))
    print('Share comment15_training_population_contrast_summary.json, comment15_training_population_contrast_intervals.csv '
          'and comment15_training_population_contrast_covariance.csv only.')


if __name__ == '__main__':
    main()
