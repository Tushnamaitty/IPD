"""Support and survey-roster follow-up for the national narrow analysis.

Place beside IPD_narrow_facility_analysis.py and run --repo-root .
Reads local raw data, checks the previously reviewed aggregate baseline, and
writes only new aggregate CSV/JSON files. Existing results are preserved.
No cross-wave contrast, causal effect, or complete-overlap claim is made.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from scipy.stats import t


def load_base(path):
    spec=importlib.util.spec_from_file_location('ipd_narrow_base',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not hasattr(module,'objective_change') or module.RIDGE!=1e-5:
        raise ValueError('Use the corrected national analysis script with the frozen 1e-5 ridge.')
    return module


def support_mask(frame,columns,minimum):
    # Explicit missing categories are retained rather than silently dropped.
    keys=frame[columns].copy()
    for c in columns:
        keys[c]=keys[c].map(lambda a:'__MISSING__' if pd.isna(a) else str(a))
    counts=keys.copy();counts['private']=frame.sector.to_numpy()
    table=counts.groupby(columns,dropna=False)['private'].agg(['sum','count'])
    eligible=(table['sum']>=minimum)&((table['count']-table['sum'])>=minimum)
    index=pd.MultiIndex.from_frame(keys)
    accepted=pd.MultiIndex.from_frame(table.loc[eligible].reset_index()[columns])
    return index.isin(accepted)


def reference_targets(base,frame,x,beta,mask):
    """Joint fitted-model and empirical restricted-reference influence.

    The fit uses every training row. Reference contributions are zero outside
    the declared subset. Selection and the model schema/penalty are fixed.
    """
    mask=np.asarray(mask,dtype=bool)
    if len(mask)!=len(frame) or not mask.any(): raise ValueError('Empty or invalid reference.')
    w=frame.weight.to_numpy(float);wt=w/w.sum()
    wr=np.where(mask,w,0.);wr=wr/wr.sum()
    y=frame.outcome.to_numpy(float);a=frame.sector.to_numpy(int)
    x0=x.copy();x1=x.copy();x0[:,1]=0.;x1[:,1]=1.
    p=expit(x@beta);q0=expit(x0@beta);q1=expit(x1@beta)
    mu=np.array([wr@q0,wr@q1])
    derivatives=np.column_stack([x0.T@(wr*q0*(1-q0)),x1.T@(wr*q1*(1-q1))])
    directions=cho_solve(cho_factor(base.objective(beta,x,y,w,hessian=True)[2]),derivatives)
    penalty=np.full(len(beta),base.RIDGE);penalty[0]=0.
    model=wt[:,None]*((y-p)[:,None]*(x@directions)-(penalty*beta)@directions)
    influence_model=model+wr[:,None]*(np.column_stack([q0,q1])-mu)
    observed=[];observed_u=[]
    for sector in (0,1):
        selected=(a==sector);share=wr[selected].sum()
        if share<=0: raise ValueError('Reference lacks one sector.')
        value=float(wr[selected]@y[selected]/share)
        observed.append(value);observed_u.append(wr*selected*(y-value)/share)
    o0,o1=observed;u0,u1=observed_u
    points=np.array([o0,o1,o1-o0,mu[0],mu[1],mu[1]-mu[0]])
    influence=np.column_stack([u0,u1,u1-u0,influence_model,influence_model[:,1]-influence_model[:,0]])
    if np.max(np.abs(influence.sum(axis=0)))>1e-7:
        raise RuntimeError('Restricted-reference influence-centering check failed.')
    # Directional numeric derivative checks exercise all coefficients without
    # a separate full-cohort matrix evaluation for every coefficient.
    rng=np.random.default_rng(408);error=0.
    for _ in range(3):
        direction=rng.normal(size=len(beta));direction/=np.linalg.norm(direction)
        eps=1e-5
        def predict(b):
            z=x@b-x[:,1]*b[1]
            return np.array([wr@expit(z),wr@expit(z+b[1])])
        numeric=(predict(beta+eps*direction)-predict(beta-eps*direction))/(2*eps)
        error=max(error,float(np.max(np.abs(numeric-direction@derivatives))))
    if error>1e-7: raise RuntimeError('Restricted-reference derivative check failed.')
    return points,influence,error


def hr_candidates(root,wave,explicit=None):
    if explicit:
        path=Path(explicit).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        return path,1
    matches=[]
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower()!='.dta': continue
        if any(c.lower() in {'.git','.venv','venv','node_modules'} for c in p.relative_to(root).parts): continue
        name=p.name.upper()
        if wave=='NFHS-4': match=name.startswith('IAHR74')
        else: match=name.startswith('IAHR7') and not name.startswith('IAHR74')
        if match: matches.append(p)
    if len(matches)>1:
        raise ValueError(f'{wave}: multiple household files found; specify --nfhs4-household or --nfhs5-household.')
    return (matches[0] if matches else None),len(matches)


def compare_rosters(base,frame,birth_roster,household_roster):
    active=frame[['state','stratum']].drop_duplicates()
    birth=birth_roster.merge(active,on=['state','stratum'],how='inner')
    household=household_roster.merge(active,on=['state','stratum'],how='inner')
    b=pd.MultiIndex.from_frame(birth[['state','stratum','psu']])
    h=pd.MultiIndex.from_frame(household[['state','stratum','psu']])
    missing=int((~b.isin(h)).sum())
    if missing: raise ValueError('Birth PSUs absent from household roster; do not use revised variance.')
    def singletons(q):
        return q.groupby(['state','stratum']).size().loc[lambda a:a==1].index
    bs=singletons(birth);hs=singletons(household)
    return {'birth_active_psus':len(b),'household_active_psus':len(h),
            'additional_household_psus':int((~h.isin(b)).sum()),
            'birth_psus_absent_from_household':missing,
            'birth_singleton_strata':len(bs),'household_singleton_strata':len(hs),
            'birth_singletons_resolved':int((~bs.isin(hs)).sum())}


def intervals(base,frame,influence,points,roster,meta):
    covs,design=base.covariance_from_roster(frame,influence,roster)
    critical=float(t.ppf(.975,design['design_df']));rows=[]
    for treatment,cov in covs.items():
        if np.linalg.eigvalsh(cov).min() < -1e-12:
            raise RuntimeError('Covariance positive-semidefinite check failed.')
        for j in (3,4,5):
            se=float(100*np.sqrt(max(0.,cov[j,j])));point=float(100*points[j])
            rows.append(dict(meta,metric=base.METRICS[j],estimate=point,se=se,
                             ci_lower=point-critical*se,ci_upper=point+critical*se,
                             singleton_treatment=treatment,design_df=design['design_df']))
    return rows,design


def analyze(base,frame,roster,legacy_roster,wave,analysis,previous):
    frame=frame.reset_index(drop=True);x,schema=base.encode(frame,first_delivery=analysis=='first_delivery')
    beta,fit=base.fit_model(x,frame.outcome.to_numpy(float),frame.weight.to_numpy(float))
    full=np.ones(len(frame),dtype=bool)
    points,influence,derivative=reference_targets(base,frame,x,beta,full)
    old_points,old_influence,_=base.targets_influence(frame,x,beta)
    if not np.allclose(points,old_points,rtol=0,atol=1e-10) or not np.allclose(influence,old_influence,rtol=0,atol=1e-10):
        raise RuntimeError('General reference calculation disagrees with original implementation.')
    legacy_rows,_=intervals(base,frame,influence,points,legacy_roster,{'wave':wave,'analysis':analysis})
    for row in legacy_rows:
        if row['singleton_treatment']!='adjust': continue
        key=(wave,analysis,row['metric']);old=previous[key]
        if abs(row['estimate']-old['estimate'])>1e-5 or abs(row['se']-old['se'])>1e-5:
            raise RuntimeError(f'{key}: original point/SE reproduction failed; inspect changed inputs.')
    definitions=[('full_reference',full,False),
                 ('states_both_sectors',support_mask(frame,['state'],1),False),
                 ('states_30_each',support_mask(frame,['state'],30),False),
                 ('cells_both_sectors',support_mask(frame,['state','residence','wealth_index'],1),False),
                 ('cells_30_each',support_mask(frame,['state','residence','wealth_index'],30),False),
                 ('states_30_each_refit',support_mask(frame,['state'],30),True),
                 ('cells_30_each_refit',support_mask(frame,['state','residence','wealth_index'],30),True)]
    results=[];cases=[];w=frame.weight.to_numpy(float)
    for name,mask,refit in definitions:
        excluded=float(100*w[~mask].sum()/w.sum())
        if not mask.any(): raise ValueError(f'{wave} {analysis} {name}: empty reference.')
        f=frame.loc[mask].reset_index(drop=True) if refit else frame
        xx=x[mask] if refit else x
        bb,diagnostics=base.fit_model(xx,f.outcome.to_numpy(float),f.weight.to_numpy(float)) if refit else (beta,fit)
        reference=np.ones(len(f),dtype=bool) if refit else mask
        pp,uu,error=reference_targets(base,f,xx,bb,reference)
        if not refit:
            # A bounded probability difference changes by at most 2*q when
            # q of reference weight is removed. This checks reference handling.
            if abs(100*(pp[5]-points[5]))>2*excluded+1e-8:
                raise RuntimeError('Reference restriction exceeded its deterministic point bound.')
        meta={'wave':wave,'analysis':analysis,'scenario':name,
              'training_n':len(f),'reference_n':int(reference.sum()),
              'excluded_original_weight_pct':excluded,
              'training_rule':'Refit on supported rows; original coefficient schema and fixed ridge retained' if refit else 'Original full training fit unchanged',
              'gap_difference_from_full_reference_pp':float(100*(pp[5]-points[5]))}
        rows,design=intervals(base,f,uu,pp,roster,meta);results.extend(rows)
        cases.append(dict(meta,fit_diagnostics=diagnostics,design=design,
                          directional_derivative_max_error=error,
                          reference_gap_change_bound_pp=None if refit else 2*excluded))
        print(f'{wave} {analysis} {name}: gap {100*pp[5]:.4f} pp; change {meta["gap_difference_from_full_reference_pp"]:+.4f} pp; excluded weight {excluded:.4f}%',flush=True)
    return results,{'baseline_reproduced':True,'model_terms':len(beta),'scenarios':cases}


def self_test(base):
    rng=np.random.default_rng(66);n=720
    frame=pd.DataFrame({'state':np.repeat([1,2,3],n//3),'stratum':np.tile(np.repeat([1,2],n//6),3),
                        'psu':np.tile(np.arange(n//3)%12+1,3),'birth_order':rng.integers(1,5,n).astype(float),
                        'education_years':rng.integers(0,20,n).astype(float),
                        'wealth_index':rng.integers(1,6,n),'residence':rng.integers(1,3,n),
                        'religion':rng.integers(1,3,n),'social_group':rng.integers(1,5,n),
                        'twin_order':np.zeros(n),'sector':rng.integers(0,2,n),'weight':rng.uniform(.3,2,n)})
    frame.loc[frame.state.eq(3),'sector']=0
    frame['outcome']=rng.binomial(1,expit(-1+.8*frame.sector+.03*frame.education_years))
    x,_=base.encode(frame);b,_=base.fit_model(x,frame.outcome.to_numpy(float),frame.weight.to_numpy(float))
    full=np.ones(n,dtype=bool);mask=frame.state.ne(3).to_numpy()
    assert np.array_equal(support_mask(frame,['state'],1),mask)
    assert np.array_equal(support_mask(frame,['state'],30),mask)
    points,u,_=reference_targets(base,frame,x,b,full);original,original_u,_=base.targets_influence(frame,x,b)
    assert np.allclose(points,original,rtol=0,atol=1e-12) and np.allclose(u,original_u,rtol=0,atol=1e-12)
    _,restricted_u,_=reference_targets(base,frame,x,b,mask)
    # Rows outside the reference still affect fitted coefficients. Independently
    # perturb/refit one weight inside and one outside to test both contributions.
    for i in (2,600):
        values=[];eps=1e-3
        for sign in (1,-1):
            changed=frame.copy();changed.loc[i,'weight']*=1+sign*eps
            bb,_=base.fit_model(x,changed.outcome.to_numpy(float),changed.weight.to_numpy(float))
            ww=np.where(mask,changed.weight.to_numpy(float),0.);ww=ww/ww.sum()
            z=x@bb-x[:,1]*bb[1];a=ww@expit(z);c=ww@expit(z+bb[1])
            values.append(np.array([a,c,c-a]))
        assert np.max(np.abs((values[0]-values[1])/(2*eps)-restricted_u[i,3:]))<1e-7
    roster=base.design_pairs(frame)
    roster=pd.concat([roster,pd.DataFrame({'state':[1,2,3],'stratum':[1,1,1],'psu':[99,99,99]})],ignore_index=True)
    rows,design=intervals(base,frame,restricted_u,points,roster,{})
    assert len(rows)==9 and design['zero_domain_psus_added']==3
    # A missing key forms an explicit cell; it is not silently dropped.
    tiny=pd.DataFrame({'state':[1]*4,'wealth_index':[np.nan]*2+[1.,1.],
                       'sector':[0,1,0,0]})
    assert support_mask(tiny,['state','wealth_index'],1).tolist()==[True,True,False,False]
    try: compare_rosters(base,frame,base.design_pairs(frame),roster.loc[roster.state.ne(2)])
    except ValueError: pass
    else: raise AssertionError('Wrong-state household roster accepted.')
    print(json.dumps({'status':'synthetic_checks_passed','original_influence_equivalence':'PASS',
                      'restricted_reference_weight_perturbations_inside_and_outside':'PASS',
                      'missing_sector_and_missing_category_checks':'PASS',
                      'zero_domain_psus_and_wrong_roster_rejection':'PASS'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path('.'))
    parser.add_argument('--baseline',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--nfhs4',type=Path);parser.add_argument('--nfhs5',type=Path)
    parser.add_argument('--nfhs4-household',type=Path);parser.add_argument('--nfhs5-household',type=Path)
    parser.add_argument('--base-script',type=Path);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();root=args.repo_root.resolve()
    base_path=args.base_script or Path(__file__).resolve().with_name('IPD_narrow_facility_analysis.py')
    base=load_base(base_path)
    if args.self_test: self_test(base);return
    baseline=args.baseline or root/'extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json'
    output=args.output or root/'extensions_work/12_narrow_scope/outputs/narrow_support_sensitivity'
    if output.exists(): raise FileExistsError('Existing output preserved; choose another --output.')
    old=json.loads(baseline.read_text(encoding='utf-8'))
    if old['facility_codes']!={'public':21,'private':31} or old['fixed_ridge']!=base.RIDGE:
        raise ValueError('Baseline scope/penalty differs from this follow-up.')
    previous={(r['wave'],r['analysis'],r['metric']):r for r in old['primary_intervals']}
    start=time.monotonic();audits={};prepared={}
    for wave in ('NFHS-4','NFHS-5'):
        br=(args.nfhs4 if wave=='NFHS-4' else args.nfhs5) or base.find_raw(root,'IABR74' if wave=='NFHS-4' else 'IABR7',exclude_prefix=None if wave=='NFHS-4' else 'IABR74')
        explicit=args.nfhs4_household if wave=='NFHS-4' else args.nfhs5_household
        hr,count=hr_candidates(root,wave,explicit)
        original_audit=old['cohort_audit'][wave]
        if original_audit['household_roster_available'] and hr is None:
            raise FileNotFoundError(f'{wave}: original household file required for baseline verification.')
        frame,roster,audit=base.load_wave(br,hr,wave)
        if audit['birth_sha256']!=original_audit['birth_sha256']:
            raise ValueError(f'{wave}: birth file differs from reviewed baseline.')
        if original_audit['household_roster_available']:
            if audit.get('household_sha256')!=original_audit.get('household_sha256'):
                raise ValueError(f'{wave}: original household file differs; inspect before continuing.')
            legacy=roster
        else:
            raw_keys=pd.read_stata(br,columns=['v024','v022','v021'],convert_categoricals=False)
            legacy=base.design_pairs(raw_keys,('v024','v022','v021'))
        audit['standard_named_household_candidates_found']=count
        audit['household_filename_search_scope']='Entire repository, excluding dependency/git directories; custom names require an explicit argument.'
        audit['frame_status']='Observed household roster found; completeness remains unverified' if hr else 'Household file not found; full BR fallback remains incomplete'
        if not original_audit['household_roster_available'] and hr:
            audit['new_household_frame_comparison']=compare_rosters(base,frame,legacy,roster)
        audits[wave]=audit;prepared[wave]=(frame,roster,legacy)
    all_rows=[];diagnostics={}
    for wave,(frame,roster,legacy) in prepared.items():
        for analysis,sub in [('all_narrow',frame),('first_delivery',frame.loc[frame.delivery_order.eq(1)].copy())]:
            print(f'{wave} {analysis}: checking original fit and support restrictions...',flush=True)
            rows,diag=analyze(base,sub,roster,legacy,wave,analysis,previous)
            all_rows.extend(rows);diagnostics[wave+'_'+analysis]=diag
    summary={'status':'completed_support_sensitivity_requires_review',
             'script_sha256':base.sha256(Path(__file__)),'base_script_sha256':base.sha256(base_path),
             'baseline_summary_sha256':base.sha256(baseline),'fixed_ridge':base.RIDGE,
             'cohort_audit':audits,'diagnostics':diagnostics,
             'primary_intervals':[r for r in all_rows if r['singleton_treatment']=='adjust'],
             'original_baseline_reproduced':True,'existing_outputs_changed':False,
             'support_rule':'Within each wave/subgroup: state or state x residence x wealth counts of at least 1 or 30 in EACH sector. No outcome-based support selection.',
             'selection_uncertainty':'All intervals conditional on observed support classification, fixed schema and fixed penalty.',
             'refit_interpretation':'Refitting changes both training and reference populations; reference-only restrictions retain the original fit. Neither is the original full-population estimand.',
             'household_interpretation':'Finding a household file expands the observed roster if verified; does not establish full frame completeness or exact multistage variance.',
             'limitations':base.LIMITATIONS+['Coarse supported cells do not establish joint overlap across all adjustment variables.',
                 'Differences from the baseline are descriptive point sensitivities; no confidence interval or significance test of those differences is reported.'],
             'remaining_work':'Shared-reference cross-wave comparisons, heterogeneity and historically aligned state facility/policy/budget context require further work.',
             'elapsed_seconds':round(time.monotonic()-start,1)}
    output.mkdir(parents=True)
    pd.DataFrame(all_rows).to_csv(output/'support_sensitivity_intervals.csv',index=False)
    base.save_json(output/'support_sensitivity_summary.json',summary)
    base.save_json(output/'household_frame_followup.json',audits)
    print(f'Aggregate outputs saved: {output}',flush=True)
    print('Share ONLY support_sensitivity_summary.json, support_sensitivity_intervals.csv and household_frame_followup.json. Keep raw data local.',flush=True)


if __name__=='__main__': main()
