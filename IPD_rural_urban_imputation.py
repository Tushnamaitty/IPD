"""Shared-reference residence/wealth imputation sensitivity. NOT run/tested here.

Place in IPD beside IPD_social_group_imputation.py and the existing comment15
scripts. Run: python IPD_rural_urban_imputation.py --repo-root .
Uses 30 imputations, both cohorts, primary and stricter support thresholds.
Reuses the approximate mother-level multinomial imputation method; interpret
Rubin intervals as exploratory. No row-level exports or existing-file changes.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import t


def load(name, path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def holm(values):
    values=np.asarray(values,float);order=np.argsort(values);n=len(values)
    adjusted=np.maximum.accumulate((n-np.arange(n))*values[order])
    result=np.empty(n);result[order]=np.minimum(1.,adjusted)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path('.'))
    p.add_argument('--imputations',type=int,default=30)
    p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--output',type=Path)
    a=p.parse_args();root=a.repo_root.resolve();sys.path.insert(0,str(root))
    if a.imputations<20:raise ValueError('Use at least 20 imputations.')
    out=a.output or root/'extensions_work/16_missing_data/outputs/rural_urban_imputation'
    if out.exists():raise FileExistsError('Existing output preserved; choose new --output.')
    base=load('rural_base',root/'IPD_narrow_facility_analysis.py')
    c15=load('rural_c15',root/'IPD_comment15_analysis.py')
    h=load('rural_h',root/'IPD_comment15_heterogeneity_change.py')
    mi=load('rural_mi',root/'IPD_social_group_imputation.py')
    baseline=root/'extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json'
    print('Reading raw waves and checking reviewed inputs...',flush=True)
    frames,rosters,audits,_=h.load_inputs(base,c15,root,baseline)
    olddir=root/'extensions_work/12_narrow_scope/outputs/comment15_shared_reference'
    existing=pd.read_csv(olddir/'comment15_intervals.csv')
    oldsummary=json.loads((olddir/'comment15_summary.json').read_text())
    for file,key in [(root/'IPD_comment15_analysis.py','script_sha256'),
                     (root/'IPD_narrow_facility_analysis.py','base_script_sha256'),
                     (baseline,'baseline_sha256')]:
        if base.sha256(file)!=oldsummary[key]:
            raise ValueError('Source differs from reviewed shared-reference output: '+str(file))
    _,keys,L=h.contrast_design()
    contexts={};baseline_rows=[];baseline_result={}
    for first in (False,True):
        label='first_delivery' if first else 'all_narrow'
        for minimum in (1,30):
            result=h.compute(base,c15,frames,rosters,first,minimum,f'Baseline {label}/{minimum}')
            rows,_,_=h.contrast_rows(result,label,minimum,keys,L)
            baseline_rows.extend(rows)
            df=min(d['df'] for d in result['audit']['design'].values())
            context=(label,minimum)
            baseline_result[context]=100*(L@result['theta'])
            contexts[context]={'estimates':[],'covariances':[],'df':df,
                               'reference_n':result['audit']['reference_n']}
            del result
    reproduction=h.reproduction_check(existing,baseline_rows)
    print('Baseline shared-reference estimates and standard errors reproduced.',flush=True)
    prepared=[mi.mother_data(f) for f in frames]
    rng=np.random.default_rng(a.seed);draw_rows=[];gradients=[[],[]]
    for draw in range(a.imputations):
        print(f'Imputation {draw+1}/{a.imputations}: completing both waves...',flush=True)
        completed=[]
        for wave,(frame,(m,x,observed)) in enumerate(zip(frames,prepared)):
            weights=mi.cluster_weights(m,rng)
            beta,g=mi.multinomial_fit(x[observed],m.loc[observed,'social_group'].to_numpy(),weights[observed])
            gradients[wave].append(g)
            probs=softmax(np.column_stack([np.zeros((~observed).sum()),x[~observed]@beta]),axis=1)
            categories=1+(rng.random((~observed).sum())[:,None]>probs.cumsum(axis=1)[:,:-1]).sum(axis=1)
            filled=m.social_group.copy();filled.loc[~observed]=categories
            f=frame.copy();f['social_group']=f.caseid.map(filled).astype(float)
            if f.social_group.isna().any():raise RuntimeError('Missing social group remains.')
            completed.append(f)
        for first in (False,True):
            label='first_delivery' if first else 'all_narrow'
            for minimum in (1,30):
                context=(label,minimum);stored=contexts[context]
                result=h.compute(base,c15,completed,rosters,first,minimum,f'Imputation {draw+1} {label}/{minimum}')
                if result['audit']['reference_n']!=stored['reference_n']:
                    raise RuntimeError('Reference population changed after social-group imputation.')
                estimate=100*(L@result['theta'])
                covariance=10000*(L@result['cov']['adjust']@L.T)
                stored['estimates'].append(estimate);stored['covariances'].append(covariance)
                for j,(group,wave,origin) in enumerate(keys):
                    draw_rows.append({'analysis':label,'support_minimum':minimum,'imputation':draw+1,
                                      'group':group,'wave_or_contrast':wave,
                                      'estimate_pp':float(estimate[j]),'within_variance_pp2':float(covariance[j,j])})
                del result
    pooled_rows=[]
    for (label,minimum),stored in contexts.items():
        pooled=mi.pool(stored['estimates'],stored['covariances'],stored['df'])
        for j,((group,wave,origin),row) in enumerate(zip(keys,pooled)):
            point=row['estimate'];se=row['se']
            pv=float(2*t.sf(abs(point/se),row['pooling_df'])) if se>0 else (1. if point==0 else 0.)
            pooled_rows.append({'analysis':label,'support_minimum':minimum,'group':group,
                               'wave_or_contrast':wave,**row,'baseline_estimate_pp':float(baseline_result[(label,minimum)][j]),
                               'change_from_baseline_pp':float(point-baseline_result[(label,minimum)][j]),
                               'p_value_exploratory_unadjusted':pv,'holm_adjusted_p_value':None,
                               'units':'percentage points','interval_status':'Unadjusted marginal approximate Rubin interval; not simultaneous',
                               'baseline_change_inference':'Descriptive difference only; no paired CI'})
    # Preserve the original family: five residence/wealth changes x two cohorts.
    # Wealth changes are included to avoid replacing Holm's ten-test family
    # with a smaller residence-only family.
    for minimum in (1,30):
        family=[r for r in pooled_rows if r['support_minimum']==minimum
                and r['group'].endswith('_minus_1') and r['wave_or_contrast']==h.CHANGE]
        if len(family)!=10:raise RuntimeError('Expected ten heterogeneity-change tests.')
        adjusted=holm([r['p_value_exploratory_unadjusted'] for r in family])
        for row,value in zip(family,adjusted):row['holm_adjusted_p_value']=float(value)
    rural=[r for r in pooled_rows if r['group'].startswith('residence_')]
    summary={'status':'completed_approximate_shared_reference_imputation_requires_review',
             'imputations':a.imputations,'seed':a.seed,'baseline_reproduction':reproduction,
             'scope':'33 comparable states/UTs; shared-reference interaction models; both cohorts; support minima 1 and 30. Imputer fitted separately in each retained-geography wave, unlike the earlier national imputer.',
             'imputation_method':'Reuses approximate survey-weighted mother-level multinomial imputation with stratified PSU weight draws; state/residence and maternal covariate/outcome/sector summaries plus log weight predict social group.',
             'covariance':'Full joint fitted-model/reference covariance transformed with L V L-transpose before Rubin pooling. No marginal-SE shortcut.',
             'multiplicity':'Holm correction within each support threshold across the original ten residence/wealth change tests; added exploratory sensitivity; marginal CIs are unadjusted.',
             'maximum_imputer_gradients':{h.WAVES[i]:max(g) for i,g in enumerate(gradients)},
             'rural_urban_primary_results':[r for r in rural if r['support_minimum']==1],
             'limitations':['Approximate imputation parameter draws are not an exact proper posterior; Rubin interval coverage is not guaranteed.',
                            'Conditional MAR assumption is unverified; this does not address MNAR mechanisms.',
                            'Existing Taylor, roster, singleton and no-FPC limitations remain.',
                            'References differ across residence groups and cohorts, but are shared across waves within each subgroup.',
                            'Only social group is imputed; original support and geography are fixed.',
                            'No causal interpretation or paired significance test of change from the baseline.',
                            'Policy-period and historical-context models are outside this sensitivity.'],
             'script_sha256':base.sha256(__file__),'imputer_script_sha256':base.sha256(root/'IPD_social_group_imputation.py'),
             'existing_outputs_changed':False}
    out.mkdir(parents=True)
    pd.DataFrame(pooled_rows).to_csv(out/'shared_reference_imputation_all_results.csv',index=False)
    pd.DataFrame(rural).to_csv(out/'rural_urban_imputation_results.csv',index=False)
    pd.DataFrame(draw_rows).to_csv(out/'shared_reference_imputation_draws.csv',index=False)
    base.save_json(out/'rural_urban_imputation_summary.json',summary)
    print('Aggregate results saved: '+str(out),flush=True)
    print('Share the four output CSV/JSON files; keep raw data local.',flush=True)


if __name__=='__main__':main()
