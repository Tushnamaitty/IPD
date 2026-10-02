"""Household-roster Taylor uncertainty for the five common-target NFHS-5 gaps.

Place beside the existing policy scripts. Reads local data; writes aggregates
into a NEW output directory. Does not overwrite the previous 500 bootstrap draws.

This is a separate first-order, with-replacement ultimate-PSU approximation.
Both the fitted outcome model and the estimated pooled reference contribute
to the joint influence function. Preprocessing, common states and time bins
are held fixed. Primary singleton treatment: grand-mean centering (adjust).
Average-stratum and zero singleton treatments are sensitivity checks only.
The household recode is an observed roster, not a verified complete frame.
Version 3 profiles out separated categorical cells only when their level is
absent from the identical restricted reference. It preserves original fitted
preprocessing, rechecks separation and identification, and records every cell.
Intervals condition on this observed separation face and reference selection.
Actual local v2 audit, 1 October 2026: A_before_2016 n=3920; religion 96
has 3 births/0 C-sections, religion 9 has 2 births/0 C-sections. Both have
zero restricted-reference weight; the separating direction affects 5 rows.
Version 3 records the resulting training counts and all point-estimate shifts.
No new local NFHS results have yet been reviewed or finalized.
Version 4 allows an EXPLICIT additional reference religion-code restriction.
Local v3 output: E_2020_Mar_onward religion code 6 has 8 births/0 C-sections
and 0.147187877% of the supported pooled reference weight. Its separation
cannot be profiled while it remains in the reference. The supplied v4 command
excludes code 6 from the reference identically for ALL five periods, then
rechecks every fit. Estimates apply only to this further restricted target.

Method references:
https://cran.r-universe.dev/survey/doc/manual.html (svyCprod, svyglm, svyrecvar)
https://microdata.worldbank.org/catalog/4482 (NFHS-5 design metadata)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import qr, solve, solve_triangular
from scipy.optimize import linprog
from scipy.special import expit
from scipy.stats import t
import policy_period_common_states as common
import policy_period_common_bootstrap as bootstrap
import policy_period_design_audit as audit

PERIODS = common.PERIODS


def separation_check(x, y):
    """Find a nonzero improving likelihood direction without misclassified rows.

    Positive margins on any rows with all remaining margins zero indicate
    quasi-complete separation. A finite unpenalized coefficient fit does not
    then exist, even if an optimizer previously declared convergence.
    """
    scale = np.maximum(np.max(np.abs(x), axis=0), 1.)
    a = (2*y-1)[:, None]*(x/scale)
    result = linprog(-a.mean(axis=0), A_ub=-a, b_ub=np.zeros(len(y)),
                     bounds=[(-1., 1.)]*x.shape[1], method='highs',
                     options={'primal_feasibility_tolerance': 1e-9,
                              'dual_feasibility_tolerance': 1e-9})
    if not result.success:
        raise ValueError(f'Separation diagnostic inconclusive: {result.message}')
    margins = a@result.x
    if margins.min() < -1e-7:
        raise ValueError('Separation diagnostic feasibility check failed')
    return {'separation_detected': bool(margins.max() > 1e-6 and -result.fun > 1e-8),
            'mean_improving_margin': float(-result.fun),
            'minimum_signed_margin': float(margins.min()),
            'maximum_signed_margin': float(margins.max()),
            'rows_with_positive_margin': int(np.sum(margins > 1e-6))}


def pure_cells(train, target, reference_mask):
    """Descriptive clues only: pure cells alone do not prove separation."""
    x = common.features(train)
    ref = common.features(target.loc[reference_mask])
    for col in common.base.NUM:
        x[col+'_missing'] = x[col].isna().astype(str)
        ref[col+'_missing'] = ref[col].isna().astype(str)
    rows = []
    variables = common.base.CAT+['sector']+[c+'_missing' for c in common.base.NUM]
    rw = target.loc[reference_mask, 'weight'].to_numpy()
    for col in variables:
        for level in x[col].unique():
            which = x[col].eq(level).to_numpy()
            outcomes = train.outcome.to_numpy()[which]
            if len(outcomes) and (outcomes.min() == outcomes.max()):
                share = np.average(ref[col].eq(level).to_numpy(), weights=rw)*100
                rows.append({'variable': col, 'level': str(level), 'training_n': int(which.sum()),
                             'csection_n': int(outcomes.sum()), 'restricted_reference_weight_pct': float(share)})
    return rows


def dense_design(model, frame, sector=None):
    x = common.features(frame)
    if sector is not None:
        x['sector'] = sector
    z = model.named_steps['pre'].transform(x)
    if hasattr(z, 'toarray'):
        z = z.toarray()
    return np.column_stack([np.ones(len(frame)), np.asarray(z, dtype=float)])


def refine(x, y, w, start=None):
    """Weighted Newton fit on an independently selected full-rank design."""
    w = np.asarray(w, dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Invalid model weights')
    w = w / w.sum()
    beta = np.zeros(x.shape[1]) if start is None else start.copy()
    for iteration in range(100):
        eta = x @ beta
        p = expit(eta)
        score = x.T @ (w * (y-p))
        h = x.T @ ((w*p*(1-p))[:, None] * x)
        if np.linalg.cond(h) > 1e12:
            raise ValueError(f'Ill-conditioned information after weighted-QR scaling: condition={np.linalg.cond(h):.4g}, iteration={iteration}, score={np.max(np.abs(score)):.4g}')
        if np.max(np.abs(score)) < 1e-10:
            return beta, h, float(np.max(np.abs(score))), iteration
        step = solve(h, score, assume_a='pos')
        loss = np.dot(w, np.logaddexp(0, eta)-y*eta)
        alpha = 1.0
        for _ in range(40):
            candidate = beta + alpha*step
            new_eta = x @ candidate
            new_loss = np.dot(w, np.logaddexp(0, new_eta)-y*new_eta)
            if new_loss <= loss - 1e-4*alpha*np.dot(score, step) + 1e-15:
                beta = candidate
                break
            alpha *= 0.5
        else:
            raise ValueError('Logistic Newton line search failed')
    raise ValueError('Logistic score did not converge to the required tolerance')


def gap_influence(target, period, reference_mask=None, diagnostics=None):
    if reference_mask is None:
        reference_mask = np.ones(len(target), dtype=bool)
    # Own the array: pandas Copy-on-Write may expose a read-only NumPy view.
    training_mask = target.period.eq(period).to_numpy(copy=True)
    train = target.loc[training_mask]
    detail = {'period': period, 'stage': 'initial_fit',
              'training_n': len(train), 'pure_outcome_cells': pure_cells(train, target, reference_mask)}
    if diagnostics is not None:
        diagnostics.append(detail)
    print(f'{period}: checking identification and separation', flush=True)
    model = bootstrap.fit_model(train)
    x_full = dense_design(model, train)
    r_full = [dense_design(model, target, s) for s in (0, 1)]
    # Refuse predictions involving levels absent from the whole period's fit.
    # This checks identification, not joint covariate overlap or sector support.
    for col in common.base.CAT:
        if not set(common.features(target.loc[reference_mask])[col]).issubset(set(common.features(train)[col])):
            raise ValueError(f'{period}: pooled reference has unobserved training category in {col}')
    _, triangular, pivot = qr(x_full, mode='economic', pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    rank = int(np.sum(diagonal > diagonal.max()*1e-10))
    keep = np.sort(pivot[:rank])
    x = x_full[:, keep]
    detail['stage'] = 'separation_check'
    y = train.outcome.to_numpy(dtype=float)
    detail.update(separation_check(x, y))
    detail['original_separation_detected'] = detail['separation_detected']
    detail['profiled_separated_training_n'] = 0
    detail['profiled_cells'] = []
    if detail['separation_detected']:
        # A categorical level with only zeros (or only ones) has its own
        # likelihood-maximizing coefficient at -infinity (or +infinity).
        # Its limiting contribution is zero and its likelihood supplies no
        # information about coefficients for reference levels. Profile that
        # boundary cell only if it is COMPLETELY absent from the reference.
        tx = common.features(train)
        rx = common.features(target.loc[reference_mask])
        remove = np.zeros(len(train), dtype=bool)
        for col in common.base.CAT:
            for level in tx[col].unique():
                cell = tx[col].eq(level).to_numpy()
                if not rx[col].eq(level).any() and train.loc[cell, 'outcome'].nunique() == 1:
                    remove |= cell
                    detail['profiled_cells'].append({'variable': col, 'level': str(level),
                                                    'training_n': int(cell.sum()),
                                                    'csection_n': int(train.loc[cell, 'outcome'].sum()),
                                                    'restricted_reference_weight_pct': 0.0})
        if not remove.any():
            raise ValueError(f'{period}: separation affects supported cells; no automatic profiling is justified')
        positions = np.flatnonzero(training_mask)
        if reference_mask[positions[remove]].any():
            raise ValueError('A proposed profiled cell intersects the reference')
        training_mask[positions[remove]] = False
        detail['profiled_separated_training_n'] = int(remove.sum())
        train = train.loc[~remove]
        x_full = x_full[~remove]
        y = train.outcome.to_numpy(dtype=float)
        # Preserve the ORIGINAL transformer, medians and encoding. Only
        # remove now-zero/redundant design columns via a new rank check.
        _, triangular, pivot = qr(x_full, mode='economic', pivoting=True)
        diagonal = np.abs(np.diag(triangular))
        rank = int(np.sum(diagonal > diagonal.max()*1e-10))
        keep = np.sort(pivot[:rank])
        x = x_full[:, keep]
        detail['stage'] = 'separation_recheck_after_profiling'
        detail.update(separation_check(x, y))
        if detail['separation_detected']:
            raise ValueError(f'{period}: separation remains after profiling zero-reference cells; do not use intervals')
        print(f'{period}: profiled {remove.sum()} zero-reference separated rows; remaining model n={len(train)}', flush=True)
    detail['retained_training_n'] = len(train)
    relation = np.linalg.lstsq(x, x_full, rcond=1e-10)[0]
    span_error = max(float(np.max(np.abs(r[reference_mask][:, keep]@relation-r[reference_mask]))) for r in r_full)
    if span_error > 1e-7:
        raise ValueError(f'{period}: reference predictions are not identified by the training design (error {span_error})')
    r0, r1 = (r[:, keep] for r in r_full)
    original_beta = np.r_[model.named_steps['fit'].intercept_, model.named_steps['fit'].coef_.ravel()]
    start = np.linalg.lstsq(x, x_full@original_beta, rcond=1e-10)[0]
    w = train.weight.to_numpy(dtype=float)
    # Reparameterize the SAME model to make the weighted design orthonormal.
    # This adds no penalty and changes no covariates or predictions.
    _, weighted_r = qr(np.sqrt(w/w.sum())[:, None]*x, mode='economic')
    transform = solve_triangular(weighted_r, np.eye(x.shape[1]))
    x = x@transform
    r0, r1 = r0@transform, r1@transform
    start = weighted_r@start
    detail['stage'] = 'scaled_baseline_refinement'
    beta, h, score, iterations = refine(x, y, w, start)
    detail['baseline_information_condition'] = float(np.linalg.cond(h))
    reference_w = target.weight.to_numpy(dtype=float)
    reference_w = np.where(reference_mask, reference_w, 0.)
    reference_w = reference_w/reference_w.sum()
    p0, p1 = expit(r0@beta), expit(r1@beta)
    q = p1-p0
    gap = float(reference_w@q)
    gradient = r1.T@(reference_w*p1*(1-p1)) - r0.T@(reference_w*p0*(1-p0))
    # h uses training weights normalized to sum one; preserve that normalization.
    coefficient_direction = solve(h, gradient, assume_a='pos')
    influence = reference_w*(q-gap)
    influence[training_mask] += (w/w.sum())*(y-expit(x@beta))*(x@coefficient_direction)
    # Independently check the derivative of the FULL statistic: perturb both
    # training and reference weights, then refit with preprocessing fixed.
    rng = np.random.default_rng(904 + PERIODS.index(period))
    perturbation = rng.choice([-1.0, 1.0], size=len(target))
    epsilon = 1e-3
    perturbed_gaps = []
    for sign in (-1, 1):
        detail['stage'] = f'weight_perturbation_{sign:+d}'
        factors = 1+sign*epsilon*perturbation
        b, _, _, _ = refine(x, y, w*factors[training_mask], beta)
        rw = reference_w*factors
        perturbed_gaps.append(float(np.dot(rw, expit(r1@b)-expit(r0@b))/rw.sum()))
    numerical = (perturbed_gaps[1]-perturbed_gaps[0])/(2*epsilon)
    analytic = float(influence@perturbation)
    derivative_error_pp = abs(numerical-analytic)*100
    if derivative_error_pp > max(1e-5, abs(analytic)*100*2e-3):
        raise ValueError(f'{period}: full-statistic influence check failed ({derivative_error_pp:.6g} pp)')
    if abs(float(influence.sum())) > 1e-7:
        raise ValueError(f'{period}: influence does not sum to zero')
    detail['stage'] = 'verified'
    info = {'period': period, 'design_columns': x_full.shape[1], 'estimable_rank': rank,
            'original_training_n': detail['training_n'], 'retained_training_n': len(train),
            'profiled_separated_training_n': detail['profiled_separated_training_n'],
            'reference_span_error': span_error, 'maximum_normalized_score': score,
            'refinement_iterations': iterations, 'derivative_check_error_pp': derivative_error_pp,
            'public_pct': float(reference_w@p0*100), 'private_pct': float(reference_w@p1*100),
            'gap_pp': gap*100, 'preprocessing_gap_pp': bootstrap.gap(model, target.loc[reference_mask])}
    return gap*100, influence*100, info


def joint_covariance(psu_values, groups):
    """WR ultimate-PSU covariance, with explicit singleton sensitivities.

    groups includes every observed household PSU in active target strata,
    including zero-domain PSUs. No certainty status is assumed.
    """
    k = psu_values.shape[1]
    regular = np.zeros((k, k))
    adjusted = np.zeros((k, k))
    multi_count = 0
    lonely_count = 0
    grand_mean = psu_values.mean(axis=0)
    degrees = 0
    for ids in groups:
        n = len(ids)
        values = psu_values[ids]
        if n > 1:
            centered = values-values.mean(axis=0)
            regular += n/(n-1)*(centered.T@centered)
            multi_count += 1
            degrees += n-1
        else:
            delta = values[0]-grand_mean
            adjusted += np.outer(delta, delta)
            lonely_count += 1
    if not multi_count or degrees <= 0:
        raise ValueError('No multi-PSU strata available for variance estimation')
    return {'adjust': regular+adjusted,
            'average': regular*(1+lonely_count/multi_count),
            'zero_sensitivity_only': regular}, degrees, lonely_count


def interval_rows(points, covariance, degrees):
    quantile = float(t.ppf(.975, degrees))
    periods, contrasts = [], []
    for method, cov in covariance.items():
        if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() < -1e-8:
            raise ValueError('Nonfinite or non-positive-semidefinite covariance')
        for i, period in enumerate(PERIODS):
            se = float(np.sqrt(max(cov[i, i], 0)))
            periods.append({'period': period, 'singleton_treatment': method,
                            'gap_pp': points[i], 'se_pp': se,
                            'ci_lower_pp': points[i]-quantile*se,
                            'ci_upper_pp': points[i]+quantile*se,
                            'design_df': degrees, 'interval_type': '95% marginal Taylor t; no multiplicity adjustment'})
        for later, earlier in bootstrap.PAIRS:
            i, j = PERIODS.index(later), PERIODS.index(earlier)
            se = float(np.sqrt(max(cov[i, i]+cov[j, j]-2*cov[i, j], 0)))
            difference = points[i]-points[j]
            contrasts.append({'later_period': later, 'earlier_period': earlier,
                              'singleton_treatment': method, 'change_in_gap_pp': difference,
                              'se_pp': se, 'ci_lower_pp': difference-quantile*se,
                              'ci_upper_pp': difference+quantile*se, 'design_df': degrees,
                              'interval_type': '95% marginal Taylor t; five exploratory contrasts, no multiplicity adjustment',
                              'interpretation': 'Selected common-state period association; not a policy effect'})
    return pd.DataFrame(periods), pd.DataFrame(contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--supported-reference', action='store_true',
                        help='Use the previously checked identical marginal-support-restricted reference (training unchanged)')
    parser.add_argument('--reference-exclude-religion', type=int, nargs='+', default=[],
                        help='Explicitly omit specified religion codes from the same reference in ALL periods; report target-weight changes')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose a new output folder to preserve all previous results')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started', 'raw_data_run_review_required': True,
              'method': 'Joint Taylor linearization; with-replacement ultimate-PSU approximation',
              'primary_singleton_treatment': 'Grand-mean centering of PSU influence totals in active strata (adjust)',
              'limitations': ['Household recode roster completeness is not established; national count differs from published fieldwork count',
                             'Not exact full PPS/systematic multistage variance; no FPC or lower-stage identifiers used',
                             'Conditional on fitted preprocessing, median imputation, selected 11 states and time bins',
                             'Conditional on observed zero-reference separated-cell profiling; this is not unconditional uncertainty for the original full-coefficient model',
                             'Marginal support sensitivity does not establish joint overlap',
                             'Exploratory marginal intervals; no multiplicity adjustment; no causal policy effects']}
    start_time = time.monotonic()
    diagnostics = []
    try:
        raw_path = args.repo_root/'data/raw/IABR7EFL.DTA'
        hr_path = args.repo_root/'data/raw/IAHR7EFL.DTA'
        prior_path = args.repo_root/'extensions_work/03_nfhs4_nfhs5_temporal/outputs/policy_period_common_states/common_state_standardized_gaps.csv'
        print('Loading birth and household design variables...', flush=True)
        raw, _ = audit.read_columns(raw_path, ['caseid', 'bidx', 'v021', 'v022'])
        hr, labels = audit.read_columns(hr_path, ['hv021', 'hv022'], ['hv015'])
        df, _, _ = common.base.load(raw_path, 'NFHS-5', 60, False)
        target, states = common.common_cohort(df)
        target = target.merge(raw[['caseid', 'bidx', 'v021']], on=['caseid', 'bidx'], validate='one_to_one')
        if len(target) != 42370:
            raise ValueError('Common analytic sample differs from the audited 42,370 births')
        hr_pairs = audit.design_pairs(hr, 'hv022', 'hv021')
        br_pairs = audit.design_pairs(raw, 'v022', 'v021')
        coverage, frame, lonely = audit.compare_frames(br_pairs, hr_pairs, set(target.v022))
        if coverage['birth_psu_pairs_absent_from_household_frame']:
            raise ValueError('Birth design identifiers are absent from household roster')
        indices = pd.MultiIndex.from_frame(frame).get_indexer(pd.MultiIndex.from_arrays([target.v022, target.v021]))
        if (indices < 0).any():
            raise ValueError('Target PSU is absent from household roster')
        groups = [np.asarray(g, dtype=int) for g in frame.groupby('stratum', sort=True).indices.values()]
        prior = pd.read_csv(prior_path).set_index('period')
        if prior.index.duplicated().any() or set(prior.index) != set(PERIODS):
            raise ValueError('Saved point estimates do not contain exactly the five periods')
        points, contributions, fits = [], [], []
        mask = bootstrap.marginal_reference_mask(target) if args.supported_reference else np.ones(len(target), dtype=bool)
        supported_mask = mask.copy()
        if args.reference_exclude_religion:
            mask &= ~pd.to_numeric(target.religion, errors='coerce').isin(args.reference_exclude_religion).to_numpy()
        if not mask.any():
            raise ValueError('Additional reference restriction removes every reference row')
        report['reference_excluded_religion_codes'] = args.reference_exclude_religion
        report['supported_reference_n_before_additional_restriction'] = int(supported_mask.sum())
        report['additional_excluded_weight_pct_of_supported_reference'] = float(
            target.loc[supported_mask & ~mask, 'weight'].sum()/target.loc[supported_mask, 'weight'].sum()*100)
        report['reference_n'] = int(mask.sum())
        report['excluded_reference_weight_pct'] = float(np.average(~mask, weights=target.weight)*100)
        report['reference'] = 'Identical support-restricted pooled reference; training unchanged' if args.supported_reference else 'Original pooled reference'
        if args.reference_exclude_religion:
            report['reference'] += '; additionally excludes the declared religion codes identically across all periods'
        report['training_rule'] = 'Original period training data; pure separated categorical cells absent from reference are profiled and recorded'
        print(f'Common reference: {mask.sum():,} births; total excluded original-target weight {report["excluded_reference_weight_pct"]:.6f}%', flush=True)
        for period in PERIODS:
            value, influence, fit = gap_influence(target, period, mask, diagnostics)
            old = float(prior.loc[period, 'common_target_gap_pp'])
            fit['saved_gap_pp'] = old
            fit['change_from_saved_gap_pp'] = value-old
            # Report tighter numerical fits transparently; refuse a material change.
            # Account for the explicitly changed target in this comparison.
            # A gap lies in [-100,100] pp; removing reference fraction f can
            # shift its mean by at most 200*f pp, without changing any fit.
            tolerance = (.15 if args.supported_reference else .05) + 2*report['additional_excluded_weight_pct_of_supported_reference']
            report['point_difference_review_bound_pp'] = tolerance
            if abs(value-old) > tolerance:
                raise ValueError(f'{period}: gap differs from prior estimate by more than {tolerance} pp')
            points.append(value)
            contributions.append(influence)
            fits.append(fit)
            print(f'{period}: gap {value:.3f} pp; analytic derivative verified', flush=True)
        psu_values = np.zeros((len(frame), len(PERIODS)))
        np.add.at(psu_values, indices, np.column_stack(contributions))
        covariance, degrees, lonely_count = joint_covariance(psu_values, groups)
        periods, contrasts = interval_rows(points, covariance, degrees)
        periods.to_csv(args.output/'household_period_intervals.csv', index=False)
        contrasts.to_csv(args.output/'household_period_contrasts.csv', index=False)
        pd.DataFrame(fits).to_csv(args.output/'point_estimate_verification.csv', index=False)
        sensitivity = []
        for period in PERIODS:
            rows = periods.loc[periods.period.eq(period)].set_index('singleton_treatment')
            sensitivity.append({'period': period,
                                'adjust_se_pp': rows.loc['adjust', 'se_pp'],
                                'average_se_pp': rows.loc['average', 'se_pp'],
                                'zero_se_pp': rows.loc['zero_sensitivity_only', 'se_pp']})
        pd.DataFrame(sensitivity).to_csv(args.output/'singleton_sensitivity.csv', index=False)
        for name, cov in covariance.items():
            pd.DataFrame(cov, index=PERIODS, columns=PERIODS).to_csv(args.output/f'joint_covariance_{name}.csv')
        report.update(status='completed_approximate_intervals_require_output_review', target_n=len(target),
                      common_states=states, household_national_psus=len(hr_pairs),
                      difference_from_reported_30198=30198-len(hr_pairs), frame_comparison=coverage,
                      singleton_active_strata=lonely_count, design_df=degrees,
                      singleton_target_birth_n=int(target.v022.isin(lonely).sum()),
                      singleton_target_weight_pct=float(np.average(target.v022.isin(lonely), weights=target.weight)*100),
                      maximum_point_estimate_difference_pp=max(abs(f['change_from_saved_gap_pp']) for f in fits),
                      total_profiled_separated_training_n=sum(f['profiled_separated_training_n'] for f in fits),
                      maximum_derivative_check_error_pp=max(f['derivative_check_error_pp'] for f in fits),
                      input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), prior_path]},
                      previous_bootstrap_outputs_changed=False)
        if 'hv015' in hr:
            report['household_interview_result_label'] = labels['hv015']
            report['household_interview_result_counts'] = {str(k): int(v) for k, v in hr.hv015.value_counts(dropna=False).items()}
    except Exception as exc:
        report.update(status='blocked_do_not_use_new_intervals', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (args.output/'model_diagnostics.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
        if diagnostics:
            report['last_period'] = diagnostics[-1]['period']
            report['last_stage'] = diagnostics[-1]['stage']
            report['separation_detected'] = diagnostics[-1].get('separation_detected')
        report['elapsed_seconds'] = round(time.monotonic()-start_time, 1)
        (args.output/'household_uncertainty_review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
        print(f'Aggregate outputs saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
