"""Comment 15 follow-up: change in residence and wealth heterogeneity, NFHS-4 -> NFHS-5.

Place beside IPD_comment15_analysis.py and IPD_narrow_facility_analysis.py.
Run --self-test first (synthetic data only), then:  python IPD_comment15_heterogeneity_change.py --repo-root .

Estimand (percentage points, all quantities from the declared comment15 models):
  Delta[g, w]  = mean over reference subset g of
                 Pr_w(C-section | private, X) - Pr_w(C-section | public, X),
                 from the wave-w fixed-ridge fit with private x residence and
                 private x wealth interactions toggled; reference = the shared
                 50/50 NFHS-4/NFHS-5 pooled supported cells restricted to g.
  Heterogeneity gap in wave w  = Delta[rural, w] - Delta[urban, w]
                                 (wealth: Delta[quintile q, w] - Delta[quintile 1, w], q = 2..5)
  Change in heterogeneity      = gap[NFHS-5] - gap[NFHS-4]

Uncertainty: every contrast is a linear combination of the 14 group-by-wave
estimates. Each estimate carries joint fitted-coefficient + empirical-reference
influence (from IPD_comment15_analysis.targets). Survey covariance of the full
14-column influence matrix is computed once per wave (IPD_comment15_analysis.covariance,
ultimate-PSU Taylor, independent waves) and contrasts use L V L'. Standard errors
are NEVER derived from marginal CSV standard errors.

The script reproduces the existing comment15_intervals.csv residence/wealth rows
(estimates, SEs, CIs) before exporting anything and aborts on any mismatch.
Existing outputs are never overwritten. Only aggregate CSV/JSON files are written.
Wording correction: the inherited 'Wave-specific references...' limitation is replaced locally
(see corrected_limitations); calculations and the base module are unchanged.
Inference is EXPLORATORY: unadjusted for multiplicity, association-only, not causal.
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
CHANGE = 'NFHS-5_minus_NFHS-4'
RULES = ('adjust', 'average', 'zero_sensitivity_only')
PRIMARY_RULE = 'adjust'
SUPPORT_MINIMA = (1, 30)
# DHS v025: 1 urban, 2 rural. DHS v190: 1 poorest ... 5 richest (verified from audit labels at run time).
GROUPS = [('residence', 1), ('residence', 2)] + [('wealth_index', k) for k in range(1, 6)]
CONTRASTS = [('residence', 2, 'rural_minus_urban')] + [
    ('wealth_index', k, f'quintile_{k}_minus_quintile_1') for k in range(2, 6)]
EXPECTED_INTERACTIONS = ['private:residence=2'] + [f'private:wealth_index={k}' for k in range(2, 6)]
REPRODUCTION_TOLERANCE_PP = 1e-6
OBSOLETE_LIMITATION_PREFIX = 'Wave-specific references'
CORRECTED_REFERENCE_LIMITATION = (
    'Both wave models use the same equal-wave pooled reference within each declared subgroup and support '
    'threshold. References differ across subgroups and thresholds; wealth quintiles represent within-wave '
    'relative ranks. Comparisons remain associational, not causal temporal or policy effects.')


def corrected_limitations(base):
    """Local copy of base.LIMITATIONS with the inapplicable wave-specific-reference entry replaced.

    The base module's list is copied, never mutated. Exactly one entry must match, so a changed
    base script fails loudly instead of silently keeping or dropping text.
    """
    items = list(base.LIMITATIONS)
    hits = [i for i, text in enumerate(items) if text.startswith(OBSOLETE_LIMITATION_PREFIX)]
    if len(hits) != 1:
        raise ValueError('Expected exactly one inherited wave-specific-reference limitation; found %d.' % len(hits))
    items[hits[0]] = CORRECTED_REFERENCE_LIMITATION
    return items


INFERENCE_LABEL = ('EXPLORATORY: approximate 95% Taylor t interval and unadjusted two-sided p-value; '
                   'no multiplicity adjustment; association only; conditional on fixed ridge, model, '
                   'support rule and geography mapping')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_modules():
    base = load_module('ipd_narrow_base', HERE / 'IPD_narrow_facility_analysis.py')
    c15 = load_module('ipd_comment15', HERE / 'IPD_comment15_analysis.py')
    if base.RIDGE != 1e-5 or not hasattr(base, 'objective_change'):
        raise ValueError('Use the corrected national analysis script with the frozen 1e-5 ridge.')
    for name in ('targets', 'covariance', 'support', 'state_mapping', 'STATES'):
        if not hasattr(c15, name):
            raise ValueError('IPD_comment15_analysis.py lacks ' + name)
    return base, c15


# ----------------------------------------------------------------------------- contrasts

def group_name(variable, level):
    return f'{variable}_{level}'


def contrast_design():
    """Rows of L map the 14 group-by-wave estimates to every reported quantity."""
    columns = [(group_name(c, l), w) for c, l in GROUPS for w in WAVES]
    position = {key: i for i, key in enumerate(columns)}
    keys, matrix = [], []

    def add(key, coefficients):
        v = np.zeros(len(columns))
        for column, coefficient in coefficients:
            v[position[column]] += coefficient
        keys.append(key)
        matrix.append(v)

    for c, l in GROUPS:  # existing group rows, reproduced for verification
        g = group_name(c, l)
        for w in WAVES:
            add((g, w, 'reproduced'), [((g, w), 1.)])
        add((g, CHANGE, 'reproduced'), [((g, WAVES[1]), 1.), ((g, WAVES[0]), -1.)])
    for c, l, _ in CONTRASTS:
        hi, lo = group_name(c, l), group_name(c, 1)
        g = hi + '_minus_1'
        for w in WAVES:  # existing within-wave heterogeneity gaps, reproduced
            add((g, w, 'reproduced'), [((hi, w), 1.), ((lo, w), -1.)])
        # NEW: change in heterogeneity gap = (hi-lo)[NFHS-5] - (hi-lo)[NFHS-4]
        add((g, CHANGE, 'new'), [((hi, WAVES[1]), 1.), ((lo, WAVES[1]), -1.),
                                 ((hi, WAVES[0]), -1.), ((lo, WAVES[0]), 1.)])
    return columns, keys, np.array(matrix)


# ----------------------------------------------------------------------------- model pieces

def verify_toggles(c15, f, x, labels):
    """Independently rebuild sector counterfactuals from column LABELS and check them.

    private=0 must zero the main private column and every private-x-modifier column;
    private=1 must set them to the modifier indicator. Observed rows must reproduce
    their own design row under their own sector, and nothing else may change.
    """
    sector = f.sector.to_numpy(float)
    if labels[1] != 'private':
        raise RuntimeError('Column 1 is not the private main effect.')
    idx = [i for i, label in enumerate(labels) if label.startswith('private:')]
    if [labels[i] for i in idx] != EXPECTED_INTERACTIONS:
        raise RuntimeError('Unexpected interaction labels: ' + str([labels[i] for i in idx]))
    if sorted(idx) != sorted(c15.INTERACTIONS):
        raise RuntimeError('Interaction columns differ from comment15 module state.')
    x0 = x.copy()
    x1 = x.copy()
    x0[:, 1] = 0.
    x1[:, 1] = 1.
    for i in idx:
        variable, level = labels[i][len('private:'):].split('=')
        indicator = f[variable].eq(int(level)).to_numpy(float)
        if not np.array_equal(x[:, i], sector * indicator):
            raise RuntimeError('Interaction column is not sector x indicator: ' + labels[i])
        if not np.array_equal(np.asarray(c15.INTERACTION_VALUES[i], float), indicator):
            raise RuntimeError('Counterfactual indicator mismatch: ' + labels[i])
        x0[:, i] = 0.
        x1[:, i] = indicator
    private = sector == 1
    if not (np.array_equal(x1[private], x[private]) and np.array_equal(x0[~private], x[~private])):
        raise RuntimeError('Counterfactual rows do not reproduce observed design rows.')
    changed = set(np.flatnonzero(np.any(x1 != x0, axis=0)).tolist())
    if not changed <= set([1] + idx):
        raise RuntimeError('Counterfactual toggles changed unrelated columns.')
    return x0, x1


def prepare(base, c15, frames, first, minimum):
    pieces = []
    for wave, frame in enumerate(frames):
        q = frame.loc[frame.delivery_order.eq(1)].copy() if first else frame.copy()
        q['wave'] = wave
        pieces.append(q)
    f = pd.concat(pieces, ignore_index=True)
    mask = c15.support(f, minimum)
    if not mask.any():
        raise ValueError('No common supported reference cells.')
    x, schema = base.encode(f, first_delivery=first)
    labels = list(schema['labels'])
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
    x0, x1 = verify_toggles(c15, f, x, labels)
    return f, x, x0, x1, mask, labels


def fit_betas(base, f, x, tag='', quiet=False):
    betas, diagnostics = [], []
    y = f.outcome.to_numpy(float)
    w = f.weight.to_numpy(float)
    for wave in (0, 1):
        train = f.wave.eq(wave).to_numpy()
        if not quiet:
            print(f'{tag} wave {wave + 4}: fitting {train.sum():,} births', flush=True)
        beta, diagnostic = base.fit_model(x[train], y[train], w[train])
        betas.append(beta)
        diagnostics.append(diagnostic)
    return betas, diagnostics


def reference_means(f, x0, x1, betas, mask):
    """Independent restatement of the shared-reference sector contrast (no influence)."""
    w = f.weight.to_numpy(float)
    wr = np.zeros(len(f))
    for wave in (0, 1):
        selected = mask & f.wave.eq(wave).to_numpy()
        wr[selected] = .5 * w[selected] / w[selected].sum()
    return np.array([wr @ (expit(x1 @ b) - expit(x0 @ b)) for b in betas])


def estimate(base, c15, f, x, x0, x1, betas, mask):
    """14 group-by-wave sector gaps and their joint model+reference influences."""
    theta, columns = [], []
    for variable, level in GROUPS:
        group_mask = mask & f[variable].eq(level).to_numpy()
        p, u = c15.targets(base, f, x, betas, group_mask)
        if np.max(np.abs(p - reference_means(f, x0, x1, betas, group_mask))) > 1e-12:
            raise RuntimeError('Target disagrees with independently toggled counterfactuals.')
        theta.extend(p)
        columns.extend([u[:, 0], u[:, 1]])
    return np.array(theta), np.column_stack(columns)


def compute(base, c15, frames, rosters, first, minimum, tag='', quiet=False):
    f, x, x0, x1, mask, labels = prepare(base, c15, frames, first, minimum)
    betas, fit_diagnostics = fit_betas(base, f, x, tag, quiet)
    theta, influence = estimate(base, c15, f, x, x0, x1, betas, mask)
    if np.max(np.abs(influence.sum(axis=0))) > 1e-7:
        raise RuntimeError('Influence centering failed.')
    covariance, design = {}, {}
    for wave in (0, 1):
        train = f.wave.eq(wave).to_numpy()
        covariance[wave], design[wave] = c15.covariance(f.loc[train], influence[train], rosters[wave])
    summed = {rule: covariance[0][rule] + covariance[1][rule] for rule in covariance[0]}
    if set(summed) != set(RULES):
        raise RuntimeError('Unexpected singleton rules.')
    for rule, cov in summed.items():
        if np.linalg.eigvalsh(cov).min() < -1e-10:
            raise RuntimeError('Covariance not positive semidefinite: ' + rule)
    w = f.weight.to_numpy(float)
    audit = {'reference_n': int(mask.sum()), 'design': design, 'fit_diagnostics': fit_diagnostics,
             'model_terms': len(labels), 'waves': {}}
    for wave in (0, 1):
        train = f.wave.eq(wave).to_numpy()
        audit['waves'][WAVES[wave]] = {
            'training_n': int(train.sum()), 'reference_n': int((mask & train).sum()),
            'excluded_reference_weight_pct': float(100 * w[train & ~mask].sum() / w[train].sum())}
    return {'f': f, 'x': x, 'x0': x0, 'x1': x1, 'mask': mask, 'betas': betas, 'theta': theta,
            'influence': influence, 'cov': summed, 'audit': audit}


# ----------------------------------------------------------------------------- inference

def contrast_rows(result, analysis, minimum, keys, L):
    df = min(d['df'] for d in result['audit']['design'].values())
    crit = float(t.ppf(.975, df))
    est = L @ result['theta']
    rows, covariance_rows, marginal_gap = [], [], 0.
    new_index = [k for k, key in enumerate(keys) if key[2] == 'new']
    family = {hi + '_minus_1': name for c, l, name in CONTRASTS for hi in [group_name(c, l)]}
    for rule in RULES:
        V = result['cov'][rule]
        LV = L @ V @ L.T
        for k, (group, wave, origin) in enumerate(keys):
            se = float(100 * np.sqrt(max(0., LV[k, k])))
            if not np.isfinite(se) or se <= 0:
                raise RuntimeError('Non-positive contrast SE: ' + group + ' ' + wave)
            point = float(100 * est[k])
            marginal_only = float(100 * np.sqrt(max(0., np.sum(L[k] ** 2 * np.diag(V)))))
            stat = point / se
            rows.append({
                'analysis': analysis, 'reference_minimum_each_sector_each_wave': minimum,
                'contrast': family.get(group, ''), 'group': group, 'wave_or_contrast': wave,
                'origin': origin,
                'quantity': ('change_in_heterogeneity_gap' if wave == CHANGE and origin == 'new'
                             else 'within_wave_heterogeneity_gap' if group.endswith('_minus_1')
                             and wave in WAVES else 'group_sector_gap_or_cross_wave_change'),
                'estimate_pp': point, 'se_pp': se,
                'ci_lower_pp': point - crit * se, 'ci_upper_pp': point + crit * se,
                't_statistic': stat, 'p_value_exploratory_unadjusted': float(2 * t.sf(abs(stat), df)),
                'singleton_treatment': rule, 'primary_rule': rule == PRIMARY_RULE,
                'conservative_design_df': df, 'inference_status': INFERENCE_LABEL,
                'se_if_covariance_ignored_pp': marginal_only})
            if origin == 'new' and rule == PRIMARY_RULE:
                marginal_gap = max(marginal_gap, abs(marginal_only - se) / se)
        for a in new_index:  # covariance among the five change contrasts, aggregate only
            for b in new_index:
                covariance_rows.append({
                    'analysis': analysis, 'reference_minimum_each_sector_each_wave': minimum,
                    'singleton_treatment': rule, 'row_contrast': keys[a][0], 'column_contrast': keys[b][0],
                    'covariance_pp2': float(1e4 * LV[a, b])})
    return rows, covariance_rows, marginal_gap


def reproduction_check(existing, all_rows):
    new = pd.DataFrame(all_rows)
    reproduced = new.loc[new.origin.eq('reproduced')]
    keys = ['analysis', 'reference_minimum_each_sector_each_wave', 'group', 'wave_or_contrast',
            'singleton_treatment']
    values = ['estimate_pp', 'se_pp', 'ci_lower_pp', 'ci_upper_pp', 'conservative_design_df']
    ex = existing[keys + values]
    if ex.duplicated(keys).any():
        raise ValueError('Existing intervals contain duplicate keys.')
    merged = reproduced.merge(ex, on=keys, how='left', suffixes=('', '_existing'),
                              validate='one_to_one', indicator=True)
    if not merged['_merge'].eq('both').all():
        raise RuntimeError('Existing comment15 rows missing for reproduction check.')
    differences = {c: float((merged[c] - merged[c + '_existing']).abs().max()) for c in values}
    if differences['conservative_design_df'] != 0:
        raise RuntimeError('Design df differs from existing comment15 output.')
    if max(differences[c] for c in values[:-1]) > REPRODUCTION_TOLERANCE_PP:
        raise RuntimeError('Existing comment15 estimates/SEs not reproduced: ' + json.dumps(differences))
    return {'rows_compared': int(len(merged)), 'max_abs_difference': differences,
            'tolerance_pp': REPRODUCTION_TOLERANCE_PP}


# ----------------------------------------------------------------------------- data loading (mirrors comment15 main)

def load_inputs(base, c15, root, baseline):
    old = json.loads(baseline.read_text())
    frames, rosters, audits = [], [], {}
    for wave in WAVES:
        nfhs4 = wave == 'NFHS-4'
        br = base.find_raw(root, 'IABR74' if nfhs4 else 'IABR7', exclude_prefix=None if nfhs4 else 'IABR74')
        hr = base.find_raw(root, 'IAHR74' if nfhs4 else 'IAHR7',
                           exclude_prefix=None if nfhs4 else 'IAHR74', required=False)
        f, roster, audit = base.load_wave(br, hr, wave)
        if audit['birth_sha256'] != old['cohort_audit'][wave]['birth_sha256']:
            raise ValueError('Birth input differs from reviewed baseline.')
        previous = old['cohort_audit'][wave]
        if previous['household_roster_available'] and audit.get('household_sha256') != previous.get('household_sha256'):
            raise ValueError('Required household file differs from reviewed baseline or is missing.')
        tables = audit.get('named_value_label_tables', {})
        if tables.get('V025') != {'1': 'urban', '2': 'rural'}:
            raise ValueError(wave + ': v025 labels are not 1=urban, 2=rural; contrast naming would be wrong.')
        v190 = tables.get('V190', {})
        if 'poorest' not in v190.get('1', '').lower() or 'richest' not in v190.get('5', '').lower():
            raise ValueError(wave + ': v190 labels do not run poorest (1) to richest (5).')
        mapping, _ = c15.state_mapping(br)
        names = f.state.map(mapping)
        keep = names.notna()
        if set(f.state) - set(mapping):
            raise ValueError('State mapping incomplete.')
        audit['geography_excluded_n'] = int((~keep).sum())
        audit['geography_excluded_weight_pct'] = float(100 * f.loc[~keep, 'weight'].sum() / f.weight.sum())
        index = {n: i + 1 for i, n in enumerate(c15.STATES)}
        f = f.loc[keep].copy()
        f['raw_state'] = f.state
        f['state'] = names.loc[keep].map(index).astype(int)
        roster = roster.copy()
        rn = roster.state.map(mapping)
        roster = roster.loc[rn.notna()].copy()
        roster.state = rn.loc[rn.notna()].map(index).astype(int)
        frames.append(f)
        rosters.append(roster)
        audits[wave] = audit
    return frames, rosters, audits, old


# ----------------------------------------------------------------------------- self-test

def synthetic(base, seed=79):
    rng = np.random.default_rng(seed)
    frames, rosters = [], []
    for wave in (0, 1):
        n = 1200
        f = pd.DataFrame({
            'state': np.repeat([1, 2], n // 2), 'stratum': np.tile(np.repeat([1, 2], n // 4), 2),
            'psu': np.tile(np.arange(n // 2) % 30 + 1, 2), 'birth_order': rng.integers(1, 4, n),
            'delivery_order': np.ones(n), 'education_years': rng.integers(0, 20, n),
            'wealth_index': rng.integers(1, 6, n), 'residence': rng.integers(1, 3, n),
            'religion': rng.integers(1, 3, n), 'social_group': rng.integers(1, 4, n),
            'twin_order': np.zeros(n), 'sector': rng.integers(0, 2, n), 'weight': rng.uniform(.2, 2, n)})
        logit = (-1 + .5 * f.sector + wave * f.sector * (.6 * f.residence.eq(2) - .12 * f.wealth_index)
                 + .3 * f.sector * f.residence.eq(2) + .1 * f.sector * f.wealth_index.ge(4))
        f['outcome'] = rng.binomial(1, expit(logit))
        frames.append(f)
        rosters.append(base.design_pairs(f))
    return frames, rosters


def self_test(base, c15):
    columns, keys, L = contrast_design()
    frames, rosters = synthetic(base)
    assert L.shape == (len(keys), 14) and len(keys) == 36 and sum(k[2] == 'new' for k in keys) == 5
    result = compute(base, c15, frames, rosters, False, 1, 'self-test', quiet=True)
    f, x, x0, x1, mask = result['f'], result['x'], result['x0'], result['x1'], result['mask']
    # 1. Structure and arithmetic of the change contrast.
    rows, cov_rows, marginal_gap = contrast_rows(result, 'all_narrow', 1, keys, L)
    assert len(rows) == 36 * 3 and len(cov_rows) == 25 * 3
    estimates = {(r['group'], r['wave_or_contrast']): r['estimate_pp'] for r in rows
                 if r['singleton_treatment'] == PRIMARY_RULE}
    for c, l, _ in CONTRASTS:
        g = group_name(c, l) + '_minus_1'
        assert abs(estimates[(g, CHANGE)] - (estimates[(g, WAVES[1])] - estimates[(g, WAVES[0])])) < 1e-10
    for r in rows:
        assert r['ci_lower_pp'] < r['estimate_pp'] < r['ci_upper_pp'] and r['se_pp'] > 0
        assert 0 <= r['p_value_exploratory_unadjusted'] <= 1 and r['inference_status'].startswith('EXPLORATORY')
    # 2. Covariance is retained: L V L' equals the survey covariance of directly differenced influences.
    direct = {}
    for wave in (0, 1):
        train = f.wave.eq(wave).to_numpy()
        direct[wave] = c15.covariance(f.loc[train], result['influence'][train] @ L.T, rosters[wave])[0]
    for rule in RULES:
        assert np.allclose(direct[0][rule] + direct[1][rule], L @ result['cov'][rule] @ L.T, rtol=1e-9, atol=1e-12)
    # 3. Marginal-only SEs would be wrong here: joint SEs must differ from them.
    assert marginal_gap > 1e-6, 'Covariance had no effect; shared-model covariance is not being retained.'
    # 4. Joint two-wave weight perturbation: re-fit BOTH models and shared reference, compare with influence.
    for i in (3, 1205):
        values = []
        eps = 1e-3
        for sign in (1, -1):
            ff = f.copy()
            ff.loc[i, 'weight'] *= 1 + sign * eps
            bs, _ = fit_betas(base, ff, x, quiet=True)
            values.append(L @ estimate(base, c15, ff, x, x0, x1, bs, mask)[0])
        numeric = (values[0] - values[1]) / (2 * eps)
        assert np.max(np.abs(numeric - L @ result['influence'][i])) < 1e-6, 'weight perturbation failed'
    # 5. Identical fitted models on an identical reference give exactly zero change in every gap.
    same = [frames[0], frames[0]]
    fs, xs, xs0, xs1, ms, _ = prepare(base, c15, same, False, 1)
    train0 = fs.wave.eq(0).to_numpy()
    b, _ = base.fit_model(xs[train0], fs.outcome.to_numpy(float)[train0], fs.weight.to_numpy(float)[train0])
    theta_same, _ = estimate(base, c15, fs, xs, xs0, xs1, [b, b], ms)
    for k, key in enumerate(keys):
        if key[1] == CHANGE:
            assert abs((L @ theta_same)[k]) < 1e-12
    # 6. Toggle verification really detects a wrongly toggled interaction column.
    label_set = ['intercept', 'private'] + ['z'] * (x.shape[1] - 7) + EXPECTED_INTERACTIONS
    c15.INTERACTION_VALUES[c15.INTERACTIONS[0]] = 1 - np.asarray(c15.INTERACTION_VALUES[c15.INTERACTIONS[0]])
    try:
        verify_toggles(c15, f, x, label_set)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Wrong interaction toggle was not detected.')
    # 7. Reproduction check accepts identical rows and rejects altered ones.
    existing = pd.DataFrame([r for r in rows if r['origin'] == 'reproduced'])
    assert reproduction_check(existing, rows)['rows_compared'] == 31 * 3
    altered = existing.copy()
    altered['se_pp'] *= 1.001
    try:
        reproduction_check(altered, rows)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Altered existing SEs were accepted.')
    # 8. First-delivery path (different design width) runs and passes the same internal checks.
    first = compute(base, c15, frames, rosters, True, 1, 'self-test first', quiet=True)
    assert first['audit']['model_terms'] == result['audit']['model_terms'] - 2  # birth_order scaled + missing dropped
    print(json.dumps({'status': 'synthetic_checks_passed', 'contrast_arithmetic': 'PASS',
                      'covariance_retained_vs_direct_influence': 'PASS',
                      'marginal_only_se_differs': 'PASS',
                      'joint_two_wave_weight_perturbation': 'PASS', 'same_model_zero_change': 'PASS',
                      'interaction_toggle_verification_and_negative_control': 'PASS',
                      'reproduction_check_accept_and_reject': 'PASS', 'first_delivery_path': 'PASS'}, indent=2))


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--existing', type=Path, help='Existing comment15_shared_reference output folder.')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    base, c15 = load_modules()
    if args.self_test:
        self_test(base, c15)
        return
    root = args.repo_root.resolve()
    out = args.output or root / 'extensions_work/12_narrow_scope/outputs/comment15_heterogeneity_change'
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
    frames, rosters, audits, _ = load_inputs(base, c15, root, baseline)
    _, keys, L = contrast_design()
    all_rows, covariance_rows, diagnostics, gaps = [], [], {}, []
    for first in (False, True):
        analysis = 'first_delivery' if first else 'all_narrow'
        for minimum in SUPPORT_MINIMA:
            result = compute(base, c15, frames, rosters, first, minimum, f'{analysis} min {minimum}')
            rows, cov_rows, gap = contrast_rows(result, analysis, minimum, keys, L)
            all_rows += rows
            covariance_rows += cov_rows
            gaps.append(gap)
            old = existing_summary['diagnostics'][f'{first}_{minimum}']
            if old['reference_n'] != result['audit']['reference_n']:
                raise RuntimeError('Reference size differs from existing comment15 output.')
            diagnostics[f'{analysis}_{minimum}'] = result['audit']
            print(f'{analysis} min {minimum}: change contrasts computed', flush=True)
    reproduction = reproduction_check(existing_intervals, all_rows)
    exported = [r for r in all_rows if r['group'].endswith('_minus_1')]
    for r in exported:
        r.pop('origin')
    primary = [r for r in exported if r['singleton_treatment'] == PRIMARY_RULE and r['wave_or_contrast'] == CHANGE]
    summary = {
        'status': 'completed_exploratory_outputs_require_review',
        'task': 'Change in rural-urban and wealth-quintile (2-5 vs 1) heterogeneity of the private-minus-public gap, NFHS-5 minus NFHS-4',
        'estimand': ('Delta[g,w]=mean over shared reference subset g of Pr_w(C-section|private,X)-Pr_w(C-section|public,X); '
                     'gap_w = Delta[rural,w]-Delta[urban,w] (wealth: Delta[q,w]-Delta[1,w]); change = gap_NFHS-5 - gap_NFHS-4'),
        'inference_label': INFERENCE_LABEL,
        'facility_codes': {'public': 21, 'private': 31}, 'fixed_ridge': base.RIDGE,
        'support_minima_each_sector_each_wave': list(SUPPORT_MINIMA), 'analyses': ['all_narrow', 'first_delivery'],
        'primary_singleton_treatment': PRIMARY_RULE, 'singleton_sensitivity': ['average', 'zero_sensitivity_only'],
        'uncertainty': ('Joint fitted-coefficient and empirical equal-wave pooled-reference influence for all 14 group-by-wave '
                        'estimates; contrasts use L V L-transpose from one survey covariance per wave; independent waves; '
                        'minimum wave design df; no marginal-SE shortcut'),
        'max_relative_se_change_if_covariance_ignored_primary_new_contrasts': float(max(gaps)),
        'reproduction_of_existing_comment15_rows': reproduction,
        'primary_change_results': primary, 'diagnostics': diagnostics, 'cohort_audit': audits,
        'script_sha256': base.sha256(Path(__file__)), 'comment15_script_sha256': base.sha256(Path(c15.__file__)),
        'base_script_sha256': base.sha256(Path(base.__file__)), 'baseline_sha256': base.sha256(baseline),
        'existing_intervals_sha256': base.sha256(existing_dir / 'comment15_intervals.csv'),
        'limitations': corrected_limitations(base) + [
            'Exploratory change-in-heterogeneity contrasts without multiplicity adjustment (5 contrasts x 2 analyses x 2 thresholds x 3 singleton rules).',
            'Contrasts are differences of model-based standardized sector gaps from interaction models; not a causal temporal, policy or effect-modification claim.',
            'Wave samples treated as independent; no cross-wave PSU dependence estimated; minimum design df used for t intervals.',
            'Wealth quintiles are within-wave relative ranks, not constant absolute purchasing power.',
            'Reference support classification and geography mapping are fixed; coarse cells do not prove joint overlap.',
            'First-delivery results use the separately fitted first-delivery models and their own support classification.'],
        'elapsed_seconds': round(time.monotonic() - start, 1), 'existing_outputs_changed': False}
    out.mkdir(parents=True)
    pd.DataFrame(exported).to_csv(out / 'comment15_heterogeneity_change_intervals.csv', index=False)
    pd.DataFrame(covariance_rows).to_csv(out / 'comment15_heterogeneity_change_covariance.csv', index=False)
    base.save_json(out / 'comment15_heterogeneity_change_summary.json', summary)
    print('Aggregate outputs saved: ' + str(out))
    print('Share comment15_heterogeneity_change_summary.json, comment15_heterogeneity_change_intervals.csv '
          'and comment15_heterogeneity_change_covariance.csv only.')


if __name__ == '__main__':
    main()
