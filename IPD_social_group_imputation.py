"""Exploratory social-group imputation sensitivity; NOT executed/tested by author.

Run: python IPD_social_group_imputation.py --repo-root . --imputations 30
Requires existing IPD_narrow_facility_analysis.py, numpy, pandas, scipy.
Uses a survey-weighted multinomial imputer, with stratified PSU Bayesian
bootstrap weight draws and stochastic category draws at mother level.
This is approximate multiple imputation, not an exact posterior sampler.
Rubin pooling is exploratory/conditional; see exported limitations.
No existing outputs are overwritten; raw/imputed records are never exported.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import t


def multinomial_fit(x, y, w, classes=4, ridge=1e-5):
    w = w / w.sum()
    nterms = x.shape[1]
    penalty = np.full((nterms, classes-1), ridge)
    penalty[0] = 0
    indicator = np.eye(classes)[y.astype(int)-1]
    def objective(flat):
        beta = flat.reshape(nterms, classes-1)
        logits = np.column_stack([np.zeros(len(x)), x @ beta])
        prob = softmax(logits, axis=1)
        logden = np.logaddexp.reduce(logits, axis=1)
        loss = w @ (logden - (logits * indicator).sum(axis=1))
        loss += .5 * np.sum(penalty * beta**2)
        grad = x.T @ (w[:, None] * (prob[:, 1:] - indicator[:, 1:])) + penalty * beta
        return float(loss), grad.ravel()
    fit = minimize(objective, np.zeros(nterms*(classes-1)), jac=True,
                   method='L-BFGS-B', options={'maxiter':3000, 'ftol':1e-13, 'gtol':1e-7, 'maxls':50})
    gradient = float(np.max(np.abs(objective(fit.x)[1])))
    if not fit.success or gradient > 1e-5:
        raise RuntimeError(f'Multinomial fit failed: {fit.message}; gradient={gradient}')
    return fit.x.reshape(nterms, classes-1), gradient


def mother_data(frame):
    # caseid identifies mothers within a wave. Social group is woman-level.
    if frame.caseid.isna().any():
        raise ValueError('Missing mother identifiers.')
    stable = ['social_group','state','stratum','psu','residence','wealth_index',
              'religion','education_years','weight']
    sizes = frame.groupby('caseid')[stable].nunique(dropna=False)
    if (sizes > 1).any().any():
        raise ValueError('Mother-level covariates/design/social group inconsistent across births; inspect raw records.')
    m = frame.groupby('caseid', sort=True)[stable].first()
    agg = frame.groupby('caseid').agg(
        birth_count=('sector','size'), private_share=('sector','mean'),
        outcome_share=('outcome','mean'), mean_birth_order=('birth_order','mean'),
        first_delivery_any=('delivery_order',lambda a:float(a.eq(1).any())),
        multiple_any=('twin_order',lambda a:float(a.gt(0).any())),
    )
    m = m.join(agg)
    if m.drop(columns='social_group').isna().any().any():
        raise ValueError('Imputation predictors contain missingness beyond social group.')
    observed = m.social_group.notna().to_numpy()
    if set(m.loc[observed,'social_group'].astype(int)) != {1,2,3,4}:
        raise ValueError('Social group must have the four reviewed observed categories.')
    parts = [np.ones((len(m),1))]
    # State and residence enter fixed imputation context. Design strata enter
    # the PSU multiplier draws; thousands of stratum/PSU dummies are avoided.
    cats = ['state','residence','wealth_index','religion']
    for col in cats:
        dummy = pd.get_dummies(m[col].astype(str), drop_first=True, dtype=float)
        parts.append(dummy.to_numpy())
    for col, scale in [('education_years',20),('birth_count',5),('private_share',1),
                       ('outcome_share',1),('mean_birth_order',10),
                       ('first_delivery_any',1),('multiple_any',1)]:
        parts.append(m[col].to_numpy(float)[:,None]/scale)
    # Weight is a predictor as well as the fitting weight.
    lw = np.log(m.weight.to_numpy(float)); sd = lw.std()
    parts.append(((lw-lw.mean())/max(sd,1e-12))[:,None])
    x = np.column_stack(parts)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite imputation design.')
    return m, x, observed


def cluster_weights(m, rng):
    keys = ['state','stratum','psu']
    units = m[keys].drop_duplicates().copy()
    units['factor'] = rng.exponential(1, len(units))
    # Normalize inside each observed stratum; singleton strata are fixed at 1.
    units['factor'] /= units.groupby(['state','stratum']).factor.transform('mean')
    factors = m.reset_index().merge(units,on=keys,how='left',validate='many_to_one')['factor'].to_numpy()
    return m.weight.to_numpy(float)*factors


def analyze(base, frame, roster, first):
    q = frame.loc[frame.delivery_order.eq(1)].copy().reset_index(drop=True) if first else frame.copy().reset_index(drop=True)
    x, _ = base.encode(q, first_delivery=first)
    beta, _ = base.fit_model(x,q.outcome.to_numpy(float),q.weight.to_numpy(float))
    point, influence, _ = base.targets_influence(q,x,beta)
    covs, design = base.covariance_from_roster(q,influence,roster)
    return 100*point, 10000*covs['adjust'], design['design_df']


def pool(estimates, covariances, design_df):
    q = np.stack(estimates); m = len(q)
    mean = q.mean(axis=0); within = np.mean(covariances,axis=0)
    between = np.cov(q,rowvar=False,ddof=1)
    total = within + (1+1/m)*between
    rows = []
    for j in range(q.shape[1]):
        u = max(0,float(within[j,j])); b = max(0,float(between[j,j]))
        variance = max(0,float(total[j,j])); se = np.sqrt(variance)
        if variance == 0:
            df = float(design_df); fraction = 0.
        else:
            fraction = (1+1/m)*b/variance
            old_df = np.inf if fraction == 0 else (m-1)/fraction**2
            observed_df = (design_df+1)/(design_df+3)*design_df*(1-fraction)
            df = observed_df if np.isinf(old_df) else 1/(1/old_df+1/max(observed_df,1e-12))
        critical = t.ppf(.975,df)
        rows.append({'estimate':float(mean[j]),'se':float(se),'ci_lower':float(mean[j]-critical*se),
                     'ci_upper':float(mean[j]+critical*se),'pooling_df':float(df),
                     'within_variance':u,'between_variance':b,
                     'between_imputation_variance_fraction':float(fraction),
                     'monte_carlo_se_of_mean':float(np.sqrt(b/m))})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path('.'))
    p.add_argument('--imputations',type=int,default=30)
    p.add_argument('--seed',type=int,default=20261007)
    p.add_argument('--output',type=Path)
    a = p.parse_args()
    if a.imputations < 20:
        raise ValueError('Use at least 20 imputations for this sensitivity.')
    root = a.repo_root.resolve(); sys.path.insert(0,str(root))
    source = root/'IPD_narrow_facility_analysis.py'
    spec = importlib.util.spec_from_file_location('ipd_base',source)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    out = a.output or root/'extensions_work/16_missing_data/outputs/social_group_imputation'
    if out.exists():
        raise FileExistsError('Existing output preserved; choose a new --output.')
    baseline_path = root/'extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json'
    baseline = json.loads(baseline_path.read_text())
    rng = np.random.default_rng(a.seed)
    pooled_rows, draw_rows, comparisons, diagnostics = [], [], [], {}
    for wave in ('NFHS-4','NFHS-5'):
        four = wave == 'NFHS-4'
        br = base.find_raw(root,'IABR74' if four else 'IABR7',exclude_prefix=None if four else 'IABR74')
        hr = base.find_raw(root,'IAHR74' if four else 'IAHR7',exclude_prefix=None if four else 'IAHR74',required=False)
        frame, roster, audit = base.load_wave(br,hr,wave)
        previous = baseline['cohort_audit'][wave]
        if audit['birth_sha256'] != previous['birth_sha256']:
            raise ValueError('Birth input differs from reviewed baseline.')
        if previous['household_roster_available'] and audit.get('household_sha256') != previous.get('household_sha256'):
            raise ValueError('Household roster differs from reviewed baseline.')
        if audit['named_value_label_tables'].get('S116') != previous['named_value_label_tables'].get('S116'):
            raise ValueError('Social-group labels changed.')
        m, x, observed = mother_data(frame)
        if observed.all():
            raise ValueError('No missing social group; audit differs from expected.')
        reference = {}
        for first in (False,True):
            label = 'first_delivery' if first else 'all_narrow'
            points, cov, df = analyze(base,frame,roster,first)
            old = {r['metric']:r for r in baseline['primary_intervals'] if r['wave']==wave and r['analysis']==label}
            for j, metric in enumerate(base.METRICS):
                if abs(points[j]-old[metric]['estimate']) > 1e-5 or abs(np.sqrt(cov[j,j])-old[metric]['se']) > 1e-5:
                    raise ValueError('Current baseline estimates/SEs not reproduced; stop for review.')
            reference[label] = (points,cov,df)
        estimates = {'all_narrow':[], 'first_delivery':[]}
        covariances = {'all_narrow':[], 'first_delivery':[]}
        fit_gradients = []
        for draw in range(a.imputations):
            print(f'{wave}: imputation {draw+1}/{a.imputations}',flush=True)
            ww = cluster_weights(m,rng)
            beta, gradient = multinomial_fit(x[observed],m.loc[observed,'social_group'].to_numpy(),ww[observed])
            fit_gradients.append(gradient)
            probabilities = softmax(np.column_stack([np.zeros((~observed).sum()),x[~observed]@beta]),axis=1)
            categories = 1+(rng.random((~observed).sum())[:,None] > probabilities.cumsum(axis=1)[:,:-1]).sum(axis=1)
            filled = m.social_group.copy(); filled.loc[~observed] = categories
            completed = frame.copy(); completed['social_group'] = completed.caseid.map(filled).astype(float)
            if completed.social_group.isna().any():
                raise RuntimeError('Imputation left missing values.')
            for first in (False,True):
                label = 'first_delivery' if first else 'all_narrow'
                point,cov,df = analyze(base,completed,roster,first)
                estimates[label].append(point); covariances[label].append(cov)
                for j,metric in enumerate(base.METRICS):
                    draw_rows.append({'wave':wave,'analysis':label,'imputation':draw+1,'metric':metric,
                                      'estimate':float(point[j]),'within_variance':float(cov[j,j])})
        for label in estimates:
            pooled = pool(estimates[label],covariances[label],reference[label][2])
            for j,metric in enumerate(base.METRICS):
                row = {'wave':wave,'analysis':label,'metric':metric,**pooled[j]}
                pooled_rows.append(row)
                comparisons.append({**row,'baseline_estimate':float(reference[label][0][j]),
                                    'change_from_baseline':float(row['estimate']-reference[label][0][j]),
                                    'units':'percentage points for gaps; percent for proportions',
                                    'change_ci':'Not calculated; baseline and imputation estimates are dependent.'})
        diagnostics[wave] = {'mothers':len(m),'missing_social_group_mothers':int((~observed).sum()),
                             'missing_social_group_births':int(frame.social_group.isna().sum()),
                             'imputation_predictor_terms':x.shape[1],
                             'maximum_imputer_gradient':max(fit_gradients),'baseline_reproduction':'passed',
                             'raw_birth_sha256':audit['birth_sha256']}
    summary = {'status':'completed_exploratory_imputation_requires_review',
               'imputations':a.imputations,'seed':a.seed,'diagnostics':diagnostics,
               'method':'Survey-weighted ridge multinomial imputation with within-stratum PSU exponential weight draws; stochastic social-group category draws at mother level; impute separately by wave; analyze nested first-delivery subgroup from the same completed wave.',
               'pooling':'Rubin within/between variance with Barnard-Rubin degrees of freedom using complete-data survey df; conditional approximate interpretation.',
               'scope':'National narrow cohorts only. Does not rerun shared-reference, rural/urban, policy-period or historical-context models.',
               'primary_gap_comparisons':[r for r in comparisons if r['metric']=='adjusted_gap_pp'],
               'limitations':[
                   'Conditional MAR working assumption; missingness mechanisms are not established.',
                   'PSU Bayesian-bootstrap parameter variation is approximate, not an exact proper posterior draw; Rubin coverage is not guaranteed.',
                   'Imputer includes state/residence fixed effects and stratified PSU multiplier uncertainty, but no stratum fixed effects, PSU random intercept or lower-stage model.',
                   'Fixed ridge regularization, imputation vocabulary and preprocessing; no hyperparameter uncertainty.',
                   'Outcome and sector enter mother-level imputation predictors; neither is imputed. Maternal summaries cover only eligible narrow births.',
                   'Survey Taylor variance retains original with-replacement, roster, singleton and no-FPC limitations.',
                   'Difference from baseline is descriptive; no paired CI or independent-estimate significance test.',
                   'This sensitivity cannot establish causal identification or robustness to missing-not-at-random mechanisms.',
               ],'script_sha256':base.sha256(__file__),'existing_outputs_changed':False}
    out.mkdir(parents=True)
    pd.DataFrame(pooled_rows).to_csv(out/'imputation_pooled_intervals.csv',index=False)
    pd.DataFrame(draw_rows).to_csv(out/'imputation_draw_estimates.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'imputation_baseline_comparison.csv',index=False)
    base.save_json(out/'imputation_summary.json',summary)
    print(f'Aggregate outputs saved: {out}',flush=True)
    print('Share these four CSV/JSON files; no row-level data are exported.',flush=True)


if __name__ == '__main__':
    main()
