"""Comment 17: historical state context sensitivity for exact facilities 21/31.

Place beside the corrected national and support scripts, with the supplied
historical_state_context_data folder. Run --self-test then --repo-root .
Local raw files remain local; only aggregate JSON/CSV outputs are written.
Descriptive adjusted associations; no causal policy or budget effect.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse
import importlib.util
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t

METRICS=['original_full_reference_gap_pp','original_linked_reference_gap_pp',
         'linked_calendar_baseline_gap_pp','linked_context_gap_pp',
         'context_minus_calendar_baseline_pp','reference_composition_change_pp']
CASES={'actual_health_spending':['actual_share_pct'],
       'actual_and_planned_budget':['actual_share_pct','budget_be_share_pct'],
       'actual_and_facility_12m':['actual_share_pct','specialists_per_chc'],
       'actual_and_facility_24m':['actual_share_pct','specialists_per_chc'],
       'documented_state_policy_milestones':['wb_fpms_post','maharashtra_jssk_post']}
SCALES={'actual_share_pct':5.,'budget_be_share_pct':5.,'specialists_per_chc':1.,
        'wb_fpms_post':1.,'maharashtra_jssk_post':1.}
LIMITATIONS=[
    'Historical contextual associations only; no causal state policy, spending or facility-capacity effect. No avoidable-cesarean inference.',
    'State is interview residence, not verified birth-time residence or delivery-facility state; migration and cross-state delivery are not resolved.',
    'Actual expenditure is a retrospectively reported series, not information available prospectively at birth. Planned BE allocation is a separate archived estimate, not spending or programme uptake.',
    'Finance measures are rounded shares of aggregate state expenditure, including health revenue expenditure and capital outlay; they do not measure resources per birth or facility.',
    'Facility measure covers rural government CHCs only, not urban facilities, private capacity, individual staffing access or joint functional cesarean readiness.',
    'Two facility snapshots, March 2017 and March 2018, are available. Twelve-month lag is primary; twenty-four-month lag is an explicitly older-context sensitivity. No earlier or future snapshot is assigned.',
    'Ambiguous budget vintages and stale/unclear facility rows remain missing. Linked subsets can differ substantially from the national population; coverage is reported separately.',
    'Jammu and Kashmir/Ladakh and Dadra and Nagar Haveli/Daman and Diu are excluded from finance context because units changed. Andhra Pradesh/Telangana require preceding FY >=2015. Old totals are not split across new units.',
    'Policy register contains verified milestones, not a complete state implementation register. Zero on a state-specific interaction identifies the comparison group; it does not mean other states had no policies.',
    'Launch months are excluded for the two state policy terms because birth dates contain month/year. Post-launch is not delivery-specific participation or readiness.',
    'Calendar-year effects and seasonal terms are added to BOTH linked models. Context columns are residualized against state/calendar terms and dropped if absorbed; ridge is not used to claim identification of a constant state policy indicator.',
    'Ridge, observed preprocessing/category vocabulary, macro context, source selection and policy timing are held fixed. Survey intervals are conditional on these choices; they are not macro-policy coefficient inference or uncertainty in the historical source panel.',
    'PSU Taylor variance does not make births independent state policy units. No inferential policy/context coefficients are reported; coefficients are descriptive model diagnostics only.',
    'Missing NFHS-4 household roster and uncertified NFHS-5 frame completeness remain documented; ultimate-PSU with-replacement approximation, no FPC/lower-stage variance.',
    'Support is audited coarsely by state/birth year; this does not establish joint covariate overlap. Sparse cells can still require extrapolation.',
    'Exploratory marginal intervals without multiplicity adjustment. No shared-reference cross-wave trend or policy effect is estimated.',
]


def import_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise FileNotFoundError(path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def canonical(value):
    s=' '.join(re.sub(r'[#*]','',str(value)).strip().lower().replace('&',' and ').split())
    return {'chattisgarh':'chhattisgarh','orissa':'odisha','uttaranchal':'uttarakhand',
            'nct delhi':'delhi','nct of delhi':'delhi','a and n islands':'andaman and nicobar islands',
            'andaman and nicobar':'andaman and nicobar islands',
            'dadra and nagar haveli and daman and diu':'dadra and nagar haveli and daman and diu'}.get(s,s)


def load_sources(folder,base):
    manifest=json.loads((folder/'source_manifest.json').read_text())
    for name,digest in manifest['data_sha256'].items():
        if base.sha256(folder/name)!=digest:raise ValueError('Source data checksum mismatch: '+name)
    finance=pd.read_csv(folder/'state_finance_panel.csv')
    facility=pd.read_csv(folder/'state_facility_snapshots.csv')
    if finance.duplicated(['state_name','fy_start','measure']).any():raise ValueError('Duplicate state/financial-year/measure.')
    if facility.duplicated(['state_name','snapshot_year']).any():raise ValueError('Duplicate facility snapshot.')
    if not np.isfinite(finance.value).all() or not finance.value.between(0,100).all():raise ValueError('Invalid finance share.')
    if not ((facility.chcs>=0)&(facility.specialists>=0)).all():raise ValueError('Invalid facility count.')
    return finance,facility,manifest


def state_names(frame,audit,wave):
    tables=[v for k,v in audit['named_value_label_tables'].items() if k.lower()=='v024']
    if len(tables)!=1:raise ValueError('No unique literal v024 table; state names not guessed.')
    mapping={int(k):canonical(v) for k,v in tables[0].items()}
    expected=35 if wave=='NFHS-4' else 19
    if mapping.get(expected)!='west bengal':raise ValueError('Cross-wave state mapping failed.')
    names=frame.state.map(mapping)
    if names.isna().any():raise ValueError('Unmapped analytic state.')
    return names


def link(frame,names,finance,facility,lag=12):
    q=frame.reset_index(drop=True).copy();q['state_name']=np.asarray(names)
    cmc=q.b3.to_numpy(float)
    if not (np.isfinite(cmc)&(cmc%1==0)&(cmc>=1)).all():raise ValueError('Invalid birth CMC.')
    year=1900+(cmc.astype(int)-1)//12;month=(cmc.astype(int)-1)%12+1
    q['birth_year']=year;q['birth_month']=month
    q['preceding_fy_start']=year-(month<4)-1
    unstable=q.state_name.isin(['jammu and kashmir','ladakh','dadra and nagar haveli','daman and diu','dadra and nagar haveli and daman and diu'])
    ap=q.state_name.isin(['andhra pradesh','telangana']) & q.preceding_fy_start.lt(2015)
    q['finance_boundary_excluded']=unstable|ap
    for measure in ('actual_share_pct','budget_be_share_pct'):
        lookup={(r.state_name,int(r.fy_start)):r.value for r in finance.itertuples() if r.measure==measure}
        q[measure]=[lookup.get((s,int(fy)),np.nan) for s,fy in zip(q.state_name,q.preceding_fy_start)]
        q.loc[q.finance_boundary_excluded,measure]=np.nan
    q['specialists_per_chc']=np.nan;q['facility_snapshot_year']=np.nan;q['facility_age_months']=np.nan
    # Select newest preceding snapshot first; do not fall back around a flagged row.
    for state,records in facility.groupby('state_name'):
        loc=q.state_name.eq(state).to_numpy()
        for r in records.sort_values('snapshot_year').itertuples():
            snap=(int(r.snapshot_year)-1900)*12+3
            choose=loc&(cmc>snap)
            q.loc[choose,'facility_snapshot_year']=r.snapshot_year
            q.loc[choose,'facility_age_months']=cmc[choose]-snap
            q.loc[choose,'specialists_per_chc']=r.specialists_per_chc if bool(r.eligible) else np.nan
    q.loc[q.facility_age_months.gt(lag),'specialists_per_chc']=np.nan
    wb=q.state_name.eq('west bengal');ma=q.state_name.eq('maharashtra')
    wdate=(2012-1900)*12+12;mdate=(2011-1900)*12+10
    q['wb_fpms_post']=(wb & q.b3.gt(wdate)).astype(float)
    q['maharashtra_jssk_post']=(ma & q.b3.gt(mdate)).astype(float)
    q['policy_launch_month_ambiguous']=(wb & q.b3.eq(wdate))|(ma & q.b3.eq(mdate))
    return q


def matrices(base,frame,columns,first=False):
    x,schema=base.encode(frame,first_delivery=first)
    years=sorted(frame.birth_year.unique())
    calendar=np.column_stack([frame.birth_year.eq(y).to_numpy(float) for y in years[1:]]+
                             [np.sin(2*np.pi*frame.birth_month.to_numpy()/12),np.cos(2*np.pi*frame.birth_month.to_numpy()/12)])
    x0=np.column_stack([x,calendar])
    indices=[0]+[i for i,s in enumerate(schema['labels']) if s.startswith('state=')]
    nuisance=np.column_stack([x[:,indices],calendar])
    values=frame[columns].to_numpy(float)/np.array([SCALES[c] for c in columns])
    residual=values-nuisance@np.linalg.lstsq(nuisance,values,rcond=None)[0]
    # Sequential independent-column check also rejects mutually aliased context terms.
    accepted=[];details=[];span=nuisance
    for j,name in enumerate(columns):
        raw=residual[:,j];independent=raw-span@np.linalg.lstsq(span,raw,rcond=None)[0]
        norm=float(np.linalg.norm(independent));keep=norm>1e-8*max(1.,float(np.linalg.norm(values[:,j])))
        details.append({'term':name,'fixed_divisor':SCALES[name],
                        'within_state_calendar_residual_norm':float(np.linalg.norm(raw)),
                        'independent_residual_norm':norm,'included':bool(keep)})
        if keep:accepted.append(j);span=np.column_stack([span,raw])
    xc=np.column_stack([x0,residual[:,accepted]])
    return x0,xc,{'base_schema':schema,'calendar_years':list(map(int,years)),
                  'seasonality':'sin/cos birth month, one annual cycle',
                  'context_terms':details,'included_context_terms':[columns[j] for j in accepted],
                  'preprocessing':'Unweighted projection on observed state/year/seasonality terms, frozen for conditional inference; fixed divisors. Private column remains index 1; no sector interactions.'}


def six_points(base,helper,full,xfull,beta,linked,x0,xc,b0,bc):
    mask=np.zeros(len(full),bool);mask[linked.index.to_numpy()]=True
    a,ua,_=helper.reference_targets(base,full,xfull,beta,np.ones(len(full),bool))
    b,ub,_=helper.reference_targets(base,full,xfull,beta,mask)
    clean=linked.reset_index(drop=True)
    c,uc,e0=helper.reference_targets(base,clean,x0,b0,np.ones(len(clean),bool))
    d,ud,e1=helper.reference_targets(base,clean,xc,bc,np.ones(len(clean),bool))
    vc=np.zeros(len(full));vd=np.zeros(len(full));vc[mask]=uc[:,5];vd[mask]=ud[:,5]
    points=np.array([a[5],b[5],c[5],d[5],d[5]-c[5],b[5]-a[5]])
    influence=np.column_stack([ua[:,5],ub[:,5],vc,vd,vd-vc,ub[:,5]-ua[:,5]])
    if np.max(np.abs(influence.sum(axis=0)))>1e-7:raise RuntimeError('Paired influence centering failed.')
    return points,influence,max(e0,e1)


def interval_rows(base,full,roster,points,influence,meta):
    covs,design=base.covariance_from_roster(full,influence,roster)
    critical=float(t.ppf(.975,design['design_df']));rows=[]
    for treatment,cov in covs.items():
        if np.linalg.eigvalsh(cov).min() < -1e-12:raise RuntimeError('Joint covariance is not positive semidefinite.')
        for j,metric in enumerate(METRICS):
            value=float(100*points[j]);se=float(100*np.sqrt(max(0.,cov[j,j])))
            rows.append(dict(meta,metric=metric,estimate=value,se=se,ci_lower=value-critical*se,
                             ci_upper=value+critical*se,singleton_treatment=treatment,design_df=design['design_df']))
    return rows,design


def coverage(q,columns,case,wave,analysis):
    eligible=q[columns].notna().all(axis=1)
    if case=='documented_state_policy_milestones':eligible &= ~q.policy_launch_month_ambiguous
    c=q[['state_name','birth_year','preceding_fy_start','sector','weight']].copy()
    c['eligible']=eligible.to_numpy();c['private']=q.sector.to_numpy();c['eligible_weight']=np.where(eligible,q.weight,0.)
    c['linked_private']=np.where(eligible,q.sector,0);c['linked_public']=np.where(eligible,1-q.sector,0)
    out=c.groupby(['state_name','birth_year','preceding_fy_start'],dropna=False).agg(n=('sector','size'),private_n=('private','sum'),linked_n=('eligible','sum'),weight=('weight','sum'),linked_weight=('eligible_weight','sum'),linked_private_n=('linked_private','sum'),linked_public_n=('linked_public','sum')).reset_index()
    out['linked_weight_pct_of_cell']=100*out.linked_weight/out.weight
    out['cell_weight_pct_of_full_cohort']=100*out.weight/q.weight.sum()
    out['wave']=wave;out['analysis']=analysis;out['case']=case
    return eligible,out


def self_test(base,helper,finance,facility):
    # Exact date alignment, ambiguous launch month, stale and boundary keys.
    def cmc(y,m):return (y-1900)*12+m
    f=pd.DataFrame({'state':[1]*9,'b3':[cmc(2017,3),cmc(2017,4),cmc(2018,3),cmc(2018,4),cmc(2019,4),cmc(2012,12),cmc(2013,1),cmc(2018,5),cmc(2015,5)]})
    names=['west bengal']*7+['assam','telangana'];q=link(f,names,finance,facility)
    assert q.preceding_fy_start.tolist()==[2015,2016,2016,2017,2018,2011,2011,2017,2014]
    assert np.isnan(q.loc[0,'specialists_per_chc']) and q.loc[1,'facility_snapshot_year']==2017
    assert q.loc[2,'facility_snapshot_year']==2017 and q.loc[3,'facility_snapshot_year']==2018
    assert np.isnan(q.loc[4,'specialists_per_chc']) and np.isnan(q.loc[7,'specialists_per_chc'])
    assert q.loc[5,'policy_launch_month_ambiguous'] and q.loc[6,'wb_fpms_post']==1
    assert np.isnan(q.loc[8,'actual_share_pct']) and np.isnan(q.loc[4,'budget_be_share_pct'])
    rng=np.random.default_rng(1702);n=600
    f=pd.DataFrame({'state':np.repeat([1,2,3],n//3),'stratum':np.tile(np.repeat([1,2],n//6),3),
        'psu':np.tile(np.arange(n//3)%10+1,3),'sector':rng.integers(0,2,n),'weight':rng.uniform(.3,2,n),
        'education_years':rng.integers(0,20,n).astype(float),'birth_order':rng.integers(1,5,n).astype(float),
        'wealth_index':rng.integers(1,6,n),'residence':rng.integers(1,3,n),'religion':rng.integers(1,3,n),
        'social_group':rng.integers(1,5,n),'twin_order':np.zeros(n),'birth_year':rng.integers(2014,2019,n),
        'birth_month':rng.integers(1,13,n)})
    # Constant state context must be absorbed; time-varying state context retained.
    f['actual_share_pct']=.5*f.state+(.25*f.state*(f.birth_year-2014))
    f['wb_fpms_post']=f.state.eq(3).astype(float)
    f['outcome']=rng.binomial(1,expit(-1+.8*f.sector+.2*f.actual_share_pct))
    xfull,_=base.encode(f);beta,_=base.fit_model(xfull,f.outcome.to_numpy(float),f.weight.to_numpy(float))
    linked=f.loc[f.index%5!=0];x0,xc,s=matrices(base,linked,['actual_share_pct','wb_fpms_post'])
    assert s['included_context_terms']==['actual_share_pct']
    b0,_=base.fit_model(x0,linked.outcome.to_numpy(float),linked.weight.to_numpy(float))
    bc,_=base.fit_model(xc,linked.outcome.to_numpy(float),linked.weight.to_numpy(float))
    p,u,_=six_points(base,helper,f,xfull,beta,linked,x0,xc,b0,bc)
    def independent(full):
        bb,_=base.fit_model(xfull,full.outcome.to_numpy(float),full.weight.to_numpy(float))
        l=full.loc[linked.index];w=l.weight.to_numpy(float);w=w/w.sum()
        b,_=base.fit_model(x0,l.outcome.to_numpy(float),l.weight.to_numpy(float))
        d,_=base.fit_model(xc,l.outcome.to_numpy(float),l.weight.to_numpy(float))
        def gap(b,x,w):return float(w@(expit(x@b-x[:,1]*b[1]+b[1])-expit(x@b-x[:,1]*b[1])))
        a=gap(bb,xfull,full.weight.to_numpy(float)/full.weight.sum());v=gap(bb,xfull[l.index],w)
        c=gap(b,x0,w);d=gap(d,xc,w)
        return np.array([a,v,c,d,d-c,v-a])
    assert np.allclose(p,independent(f),atol=1e-10,rtol=0)
    errors=[]
    for i in (0,7):
        vals=[];eps=1e-3
        for sign in (1,-1):
            changed=f.copy();changed.loc[i,'weight']*=1+sign*eps;vals.append(independent(changed))
        errors.append(float(np.max(np.abs((vals[0]-vals[1])/(2*eps)-u[i]))))
    assert max(errors)<1e-7
    roster=base.design_pairs(f);roster=pd.concat([roster,pd.DataFrame({'state':[1,2,3],'stratum':[1,1,1],'psu':[99,99,99]})],ignore_index=True)
    rows,design=interval_rows(base,f,roster,p,u,{})
    assert len(rows)==18 and design['zero_domain_psus_added']==3
    null,nullu,_=six_points(base,helper,f,xfull,beta,linked,x0,x0,b0,b0)
    assert abs(null[4])<1e-14 and np.max(np.abs(nullu[:,4]))<1e-14
    for wave,code in [('NFHS-4',35),('NFHS-5',19)]:
        state_names(pd.DataFrame({'state':[code]}),{'named_value_label_tables':{'V024':{str(code):'west bengal'}}},wave)
    try:state_names(pd.DataFrame({'state':[35]}),{'named_value_label_tables':{'V024':{'35':'west bengal'}}},'NFHS-5')
    except ValueError:pass
    else:raise AssertionError('Cross-wave shortcut accepted.')
    print(json.dumps({'status':'synthetic_checks_passed','historical_alignment_and_no_future_snapshots':'PASS',
        'stale_rows_missing_vintages_boundary_and_launch_month_rejection':'PASS',
        'constant_state_policy_absorption':'PASS','independent_predictions':'PASS',
        'paired_weight_perturbations_inside_and_outside_linked_cohort':'PASS',
        'maximum_weight_perturbation_error':max(errors),'joint_covariance_zero_domain_psus':'PASS',
        'unchanged_model_null_contrast':'PASS','wave_state_mapping':'PASS'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path('.'));parser.add_argument('--output',type=Path)
    parser.add_argument('--baseline',type=Path);parser.add_argument('--data-dir',type=Path)
    for name in ('nfhs4','nfhs5','nfhs4-household','nfhs5-household','base-script','support-script'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    beside=Path(__file__).resolve().parent;root=args.repo_root.resolve()
    helper_path=args.support_script or beside/'IPD_narrow_support_sensitivity.py'
    base_path=args.base_script or beside/'IPD_narrow_facility_analysis.py'
    helper=import_module(helper_path,'context_support');base=helper.load_base(base_path)
    folder=args.data_dir or beside/'historical_state_context_data'
    finance,facility,manifest=load_sources(folder,base)
    if args.self_test:self_test(base,helper,finance,facility);return
    output=args.output or root/'extensions_work/04_health_system_context/outputs/historical_state_context'
    if output.exists():raise FileExistsError('Existing output preserved; choose another --output.')
    baseline=args.baseline or root/'extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json'
    original=json.loads(baseline.read_text())
    if original['facility_codes']!={'public':21,'private':31} or original['fixed_ridge']!=base.RIDGE:raise ValueError('Baseline scope/penalty differs.')
    previous={(r['wave'],r['analysis'],r['metric']):r for r in original['primary_intervals']}
    rows=[];tables=[];details={};audits={};terms=[];start=time.monotonic()
    for wave in ('NFHS-4','NFHS-5'):
        br=(args.nfhs4 if wave=='NFHS-4' else args.nfhs5) or base.find_raw(root,'IABR74' if wave=='NFHS-4' else 'IABR7',exclude_prefix=None if wave=='NFHS-4' else 'IABR74')
        explicit=args.nfhs4_household if wave=='NFHS-4' else args.nfhs5_household
        hr,count=helper.hr_candidates(root,wave,explicit);old=original['cohort_audit'][wave]
        if old['household_roster_available'] and hr is None:raise FileNotFoundError('Reviewed household file required.')
        frame,roster,audit=base.load_wave(br,hr,wave)
        if audit['birth_sha256']!=old['birth_sha256']:raise ValueError('Raw birth input changed.')
        if old['household_roster_available'] and audit.get('household_sha256')!=old.get('household_sha256'):raise ValueError('Household input changed.')
        names=state_names(frame,audit,wave);audits[wave]=audit
        linked12=link(frame,names,finance,facility,12);linked24=link(frame,names,finance,facility,24)
        for analysis in ('all_narrow','first_delivery'):
            select=np.ones(len(frame),bool) if analysis=='all_narrow' else frame.delivery_order.eq(1).to_numpy()
            full=linked12.loc[select].reset_index(drop=True);older=linked24.loc[select].reset_index(drop=True)
            print(f'{wave} {analysis}: reproducing original baseline...',flush=True)
            xfull,_=base.encode(full,first_delivery=analysis=='first_delivery')
            beta,fullfit=base.fit_model(xfull,full.outcome.to_numpy(float),full.weight.to_numpy(float))
            reproduced=float(100*np.diff(base.standardized(beta,xfull,full.weight.to_numpy(float)))[0])
            if abs(reproduced-previous[(wave,analysis,'adjusted_gap_pp')]['estimate'])>1e-5:raise RuntimeError('Baseline point failed reproduction.')
            for case,columns in CASES.items():
                q=older if case.endswith('24m') else full
                keep,table=coverage(q,columns,case,wave,analysis);tables.append(table)
                eligible=q.loc[keep];key=wave+'_'+analysis+'_'+case
                missing_weight=float(100*q.loc[~keep,'weight'].sum()/q.weight.sum())
                record={'status':'pending','full_n':len(q),'linked_n':len(eligible),
                    'excluded_weight_pct':missing_weight,'context_columns_requested':columns,
                    'coarse_linked_state_year_cells_lacking_a_sector':int(((table.linked_n>0)&((table.linked_private_n==0)|(table.linked_public_n==0))).sum())}
                details[key]=record
                if len(eligible)<100 or eligible.sector.nunique()<2 or eligible.outcome.nunique()<2:
                    record['status']='skipped_insufficient_linked_cohort';continue
                x0,xc,schema=matrices(base,eligible,columns,analysis=='first_delivery');record['schema']=schema
                if not schema['included_context_terms']:
                    record['status']='skipped_context_absorbed_by_state_calendar';continue
                print(f'{wave} {analysis} {case}: {len(eligible):,} linked births; excluded weight {missing_weight:.3f}%; fitting paired models...',flush=True)
                b0,fit0=base.fit_model(x0,eligible.outcome.to_numpy(float),eligible.weight.to_numpy(float))
                bc,fitc=base.fit_model(xc,eligible.outcome.to_numpy(float),eligible.weight.to_numpy(float))
                points,u,error=six_points(base,helper,full,xfull,beta,eligible,x0,xc,b0,bc)
                rr,design=interval_rows(base,full,roster,points,u,{'wave':wave,'analysis':analysis,'case':case,'full_n':len(full),'linked_n':len(eligible),'excluded_weight_pct':missing_weight})
                rows.extend(rr);record.update(status='completed_requires_review',calendar_baseline_fit=fit0,context_fit=fitc,original_fit=fullfit,design=design,maximum_derivative_error=error,
                    linked_context_gap_pp=float(100*points[3]),context_minus_calendar_baseline_pp=float(100*points[4]),reference_composition_change_pp=float(100*points[5]))
                for j,name in enumerate(schema['included_context_terms']):
                    terms.append({'wave':wave,'analysis':analysis,'case':case,'term':name,'coefficient':float(bc[x0.shape[1]+j]),'fixed_divisor':SCALES[name],'inference':'Descriptive coefficient only; macro-source uncertainty and state-level causal inference not estimated.'})
                print(f'{case}: context change {100*points[4]:+.4f} pp on identical linked reference.',flush=True)
    summary={'status':'historical_context_completed_requires_aggregate_output_review',
        'script_sha256':base.sha256(Path(__file__)),'base_script_sha256':base.sha256(base_path),'support_script_sha256':base.sha256(helper_path),
        'baseline_sha256':base.sha256(baseline),'source_manifest_sha256':base.sha256(folder/'source_manifest.json'),
        'facility_codes':{'public':21,'private':31},'fixed_ridge':base.RIDGE,'cohort_audit':audits,'cases':details,
        'primary_intervals':[r for r in rows if r['singleton_treatment']=='adjust'],
        'comparison':'Each context fit and its state/calendar baseline use identical linked births, reference, original covariates, calendar controls and fixed ridge. Original full-fit gaps on full and linked references separately describe reference composition.',
        'uncertainty':'Six joint empirical-reference/fitted-model influences; paired differences retain covariance; raw state/stratum/PSU keys, observed roster zero-domain PSUs and three singleton treatments. Full-cohort active-stratum union is used for all paired metrics.',
        'source_rule':'Exact preceding completed April-March FY; newest strictly preceding March31 facility snapshot, declared 12/24-month lag; no filling missing years. Policy terms use verified state launch milestones; ambiguous launch month excluded.',
        'source_extraction_validation':manifest['extraction_validation'],'limitations':LIMITATIONS,
        'comment_17_complete':False,'completion_note':'Do not close until local outputs are reviewed. Historical finance, limited facility context and documented state milestone sensitivities are implemented; comprehensive state implementation coverage and source gaps remain explicit limitations.',
        'existing_outputs_changed':False,'record_level_data_exported':False,'elapsed_seconds':round(time.monotonic()-start,1)}
    output.mkdir(parents=True)
    base.save_json(output/'historical_context_summary.json',summary)
    pd.DataFrame(rows).to_csv(output/'historical_context_intervals.csv',index=False)
    pd.concat(tables,ignore_index=True).to_csv(output/'historical_context_coverage.csv',index=False)
    pd.DataFrame(terms).to_csv(output/'historical_context_coefficients.csv',index=False)
    base.save_json(output/'historical_context_sources.json',manifest)
    print(f'Aggregate outputs saved: {output}',flush=True)
    print('Share historical_context_summary.json, historical_context_intervals.csv, historical_context_coverage.csv and historical_context_sources.json. Keep raw data local.',flush=True)


if __name__=='__main__':main()
