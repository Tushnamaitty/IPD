"""National narrow-facility observed and model-adjusted associations.

Run from IPD: python IPD_narrow_facility_analysis.py --repo-root .
No AIPW/TMLE or causal estimates. Inputs remain local. Exact m15 codes 21/31,
0 <= v008-b3 < 60 months, state adjustment and state/stratum/PSU variance keys.
Each wave and first-delivery subgroup has its own fitted model/reference.
Fixed ridge 1e-5 comes from the earlier NFHS-5 inner validation; no new tuning
or predictive validation is claimed. Schema uses the declared cohort.
Only output CSV/JSON files are for review; private cohort Parquets stay local.
Dependencies: numpy, pandas, scipy, pyarrow. Use --self-test without real data.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import t

RAW = ['caseid','bidx','m15','m17','b3','v008','v005','v021','v022','v024',
       'v025','v130','v133','v190','bord','b0','s116']
RENAME = {'v021':'psu','v022':'stratum','v024':'state','v025':'residence',
          'v130':'religion','v133':'education_years','v190':'wealth_index',
          'bord':'birth_order','b0':'twin_order'}
NUM = ['birth_order','education_years']
CAT = ['state','wealth_index','residence','religion','social_group','twin_order']
MISSING, UNKNOWN = '__MISSING__', '__UNKNOWN__'
RIDGE = 1e-5
METRICS = ['observed_public_pct','observed_private_pct','observed_gap_pp',
           'adjusted_public_pct','adjusted_private_pct','adjusted_gap_pp']
EXPECTED_BROAD = {'NFHS-4':195366,'NFHS-5':200794}
LIMITATIONS = [
    'Associations only; no causal facility effects, avoidable-cesarean estimate or individual facility-choice recommendation.',
    'With-replacement ultimate-PSU Taylor approximation, not exact multistage/PPS/systematic variance; no FPC/lower-stage identifiers.',
    'Household or birth-recode roster is observed rather than a verified complete sampling frame.',
    'Model uncertainty is conditional on fixed ridge, observed category vocabulary, scales, missing coding and subgroup definition; no tuning/schema-selection uncertainty.',
    'Full-reference model predictions can extrapolate in sparse covariate/sector combinations; coarse support checks do not prove joint overlap.',
    'Covariates such as wealth, schooling and residence are measured at interview; birth-time values are not certified.',
    'First delivery is a birth-record subgroup (including all first-delivery twins), not a sensitivity analysis or a woman-level estimand.',
    'Wave-specific references, within-wave wealth rankings, raw state codes and survey periods differ; side-by-side gaps are not a common-population temporal or policy effect.',
    'Exploratory marginal 95% intervals without multiplicity adjustment; original broad-sector predictive results have a separate scope.',
]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def design_pairs(frame, cols=('state','stratum','psu')):
    q=frame[list(cols)].copy()
    q.columns=['state','stratum','psu']
    for c in q:
        a=pd.to_numeric(q[c],errors='coerce').to_numpy(float)
        if not (np.isfinite(a)&(a%1==0)&(a>=0)).all():
            raise ValueError('Invalid design identifier in '+c+'; mapping not guessed.')
        q[c]=a.astype(np.int64)
    return q.drop_duplicates().sort_values(['state','stratum','psu']).reset_index(drop=True)


def prepare_births(raw, wave):
    missing=set(RAW)-set(raw)
    if missing: raise ValueError('Missing birth variables: '+str(sorted(missing)))
    broad=raw.m15.between(20,27)|raw.m15.between(30,33)
    broad_n=int(broad.sum())
    if broad_n!=EXPECTED_BROAD[wave]:
        raise ValueError(f'{wave}: broad count {broad_n} differs from reviewed {EXPECTED_BROAD[wave]}; inspect input first.')
    frame=raw.loc[raw.m15.isin([21,31])].copy()
    before=len(frame)
    if frame.duplicated(['caseid','bidx']).any(): raise ValueError('Duplicate birth identifiers.')
    if not frame.m17.isin([0,1]).all(): raise ValueError('Invalid/missing narrow-cohort outcome.')
    b=pd.to_numeric(frame.b3,errors='coerce'); interview=pd.to_numeric(frame.v008,errors='coerce')
    valid=(np.isfinite(b)&np.isfinite(interview)&b.mod(1).eq(0)&interview.mod(1).eq(0)
           &b.between(1,2400)&interview.between(1,2400))
    recall=interview-b
    keep=valid&recall.ge(0)&recall.lt(60)
    w=pd.to_numeric(frame.v005,errors='coerce')/1e6
    excluded_weight=float(100*w.loc[~keep].sum()/w.sum())
    audit={'wave':wave,'broad_sector_n':broad_n,'narrow_before_date_rule_n':before,
           'invalid_date_n':int((~valid).sum()),
           'valid_date_outside_recall_n':int((valid&~(recall.ge(0)&recall.lt(60))).sum()),
           'date_or_recall_excluded_n':int((~keep).sum()),
           'date_or_recall_excluded_weight_pct':excluded_weight,
           'narrow_before_date_public_n':int(frame.m15.eq(21).sum()),
           'narrow_before_date_private_n':int(frame.m15.eq(31).sum()),
           'recall_rule':'0 <= v008-b3 < 60 months; integer finite CMCs in [1,2400]',
           'b10_rule':'No additional b10 date-imputation exclusions.'}
    frame=frame.loc[keep].rename(columns=RENAME).reset_index(drop=True)
    frame['social_group']=frame.s116.replace(8,np.nan)
    frame.education_years=frame.education_years.replace(97,np.nan)
    frame['sector']=frame.m15.eq(31).astype(int)
    frame['outcome']=frame.m17.astype(int)
    frame['weight']=pd.to_numeric(frame.v005,errors='coerce')/1e6
    if frame.empty or not (np.isfinite(frame.weight)&frame.weight.gt(0)).all():
        raise ValueError('Empty cohort or invalid weights.')
    design_pairs(frame)
    for col,lo,hi in [('birth_order',1,30),('education_years',0,30),('wealth_index',1,5)]:
        a=pd.to_numeric(frame[col],errors='coerce').to_numpy(float)
        if not (np.isnan(a)|(np.isfinite(a)&(a>=lo)&(a<=hi))).all():
            raise ValueError('Unrecognized numeric codes/range: '+col)
    if not frame.residence.isin([1,2]).all(): raise ValueError('Unrecognized residence codes.')
    if frame[['birth_order','twin_order']].isna().any().any():
        raise ValueError('First-delivery rule requires birth and twin order.')
    if not (frame.twin_order.ge(0)&frame.twin_order.le(9)&frame.twin_order.mod(1).eq(0)).all():
        raise ValueError('Unrecognized twin-order codes.')
    frame['delivery_order']=np.where(frame.twin_order.eq(0),frame.birth_order,
                                    frame.birth_order-frame.twin_order+1)
    if not (frame.delivery_order.ge(1)&frame.delivery_order.mod(1).eq(0)).all():
        raise ValueError('Inconsistent birth order/plurality for first-delivery rule.')
    audit.update(analytic_n=len(frame),public_n=int(frame.sector.eq(0).sum()),
                 private_n=int(frame.sector.eq(1).sum()),n_states=int(frame.state.nunique()),
                 education_97_as_missing_n=int(raw.loc[raw.m15.isin([21,31]),'v133'].eq(97).sum()),
                 model_adjustment_variables=NUM+CAT)
    return frame,audit


def load_wave(birth_path, hr_path, wave):
    print(f'{wave}: reading local raw birth recode...',flush=True)
    with pd.io.stata.StataReader(birth_path,convert_categoricals=False) as r:
        labels=r.variable_labels(); value_labels=r.value_labels()
    missing=set(RAW)-set(labels)
    if missing: raise ValueError('Raw variables absent: '+str(sorted(missing)))
    raw=pd.read_stata(birth_path,columns=RAW,convert_categoricals=False)
    frame,audit=prepare_births(raw,wave)
    birth_roster=design_pairs(raw,('v024','v022','v021'))
    audit['birth_sha256']=sha256(birth_path)
    audit['raw_variable_labels']={c:labels[c] for c in RAW if c not in ('caseid','bidx')}
    requested=['m15','v133','v190','v025','v130','s116','b0','v024']
    # Public StataReader API returns label tables, not their column bindings.
    # Preserve only tables whose literal names match requested variable names;
    # do not infer an alternative table-to-variable mapping.
    audit['named_value_label_tables']={name:{str(k):str(v) for k,v in values.items()}
                                      for name,values in value_labels.items() if name.lower() in requested}
    audit['value_label_note']='Literal table-name matches only; variable-label metadata is complete for requested variables. No table binding was guessed.'
    if hr_path:
        hr=pd.read_stata(hr_path,columns=['hv024','hv022','hv021'],convert_categoricals=False)
        roster=design_pairs(hr,('hv024','hv022','hv021'))
        audit.update(roster_source='Observed Household Recode',household_sha256=sha256(hr_path))
    else:
        roster=birth_roster
        audit['roster_source']='Full Birth Recode roster; household recode not found; incomplete-frame limitation applies.'
    active=frame[['state','stratum']].drop_duplicates()
    selected=roster.merge(active,on=['state','stratum'],how='inner',validate='many_to_one')
    birth_active=birth_roster.merge(active,on=['state','stratum'],how='inner',validate='many_to_one')
    roster_index=pd.MultiIndex.from_frame(selected[['state','stratum','psu']])
    birth_index=pd.MultiIndex.from_frame(birth_active[['state','stratum','psu']])
    if not birth_index.isin(roster_index).all():
        raise ValueError(f'{wave}: birth PSUs absent from selected roster; raw state/stratum/PSU mapping requires review.')
    audit['national_roster_psus']=len(roster)
    audit['active_birth_roster_psus']=len(birth_active)
    audit['household_roster_available']=bool(hr_path)
    audit['state_design_keys_verified']=True
    if wave=='NFHS-5':
        audit['difference_from_previously_cited_30198_national_psus']=30198-len(roster) if hr_path else None
        if audit['birth_sha256']!='93267b17596929371dc3190d17af114271851f14d1f566678ffceda2db4cae6e':
            raise ValueError('NFHS-5 raw hash differs from the reviewed federated input.')
        if len(frame)!=115197:
            raise ValueError('NFHS-5 dated narrow count differs from reviewed 115197; inspect dates/cohort.')
    return frame,roster,audit


def cat_values(series):
    return np.array([MISSING if pd.isna(a) else str(int(a)) if isinstance(a,(int,float,np.integer,np.floating)) and float(a).is_integer() else str(a) for a in series],dtype=str)


def encode(frame, first_delivery=False):
    numeric=['education_years'] if first_delivery else NUM
    scales={'birth_order':10.,'education_years':20.}
    parts=[np.ones((len(frame),1)),frame.sector.to_numpy(float)[:,None]]
    labels=['intercept','private']
    num=frame[numeric].to_numpy(float)
    missing=np.isnan(num)
    parts.extend([np.where(missing,0.,num)/np.array([scales[c] for c in numeric]),missing.astype(float)])
    labels.extend(c+'_scaled' for c in numeric);labels.extend(c+'_missing' for c in numeric)
    vocab={}
    for col in CAT:
        a=cat_values(frame[col]); values=sorted(set(a)-{MISSING,UNKNOWN})+[MISSING,UNKNOWN]
        vocab[col]=values
        parts.append(np.column_stack([a==v for v in values[1:]]).astype(float))
        labels.extend(col+'='+v for v in values[1:])
    x=np.column_stack(parts)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite design matrix.')
    return x,{'labels':labels,'categories':vocab,'numeric_scales':{c:scales[c] for c in numeric},
              'numeric_missing_rule':'Value set to zero plus separate missing indicator.',
              'schema_source':'All rows of this declared wave/subgroup; no predictive validation claimed.',
              'birth_order_excluded_in_first_delivery':first_delivery}


def objective(beta,x,y,w,ridge=RIDGE,hessian=False):
    z=x@beta; p=expit(z); w=w/w.sum()
    penalty=np.full(len(beta),ridge); penalty[0]=0.
    loss=float(w@(np.logaddexp(0,z)-y*z)+.5*np.dot(penalty*beta,beta))
    gradient=x.T@(w*(p-y))+penalty*beta
    if hessian:
        h=x.T@((w*p*(1-p))[:,None]*x)+np.diag(penalty)
        return loss,gradient,h
    return loss,gradient


def objective_change(beta,delta,x,y,w,ridge=RIDGE):
    """Evaluate the same objective's change without subtracting two totals.

    For small changes in the logits, softplus(z+d)-softplus(z) equals
    log1p(expit(z)*expm1(d)). This avoids objective rounding at convergence.
    Larger changes use logaddexp to avoid overflow in expm1.
    """
    z=x@beta;dz=x@delta;p=expit(z)
    small=np.abs(dz)<=1.
    changes=np.empty_like(z)
    changes[small]=np.log1p(p[small]*np.expm1(dz[small]))-y[small]*dz[small]
    changes[~small]=np.logaddexp(0,z[~small]+dz[~small])-np.logaddexp(0,z[~small])-y[~small]*dz[~small]
    penalty=np.full(len(beta),ridge);penalty[0]=0.
    return float((w/w.sum())@changes+np.dot(penalty*beta,delta)+.5*np.dot(penalty*delta,delta))


def fit_model(x,y,w):
    if len(np.unique(y))<2: raise ValueError('Both outcome classes required for model fit.')
    beta=np.zeros(x.shape[1]);beta[0]=logit(np.average(y,weights=w))
    for iteration in range(100):
        loss,g,h=objective(beta,x,y,w,hessian=True)
        if np.max(np.abs(g))<1e-10: break
        step=cho_solve(cho_factor(h),g)
        multiplier=1.
        while multiplier>2**-25:
            delta=-multiplier*step
            candidate=beta+delta
            change=objective_change(beta,delta,x,y,w)
            if np.isfinite(change) and change<=-1e-4*multiplier*np.dot(g,step):
                beta=candidate;break
            multiplier/=2
        else: raise RuntimeError('Newton line search failed.')
    else: raise RuntimeError('Newton iteration limit.')
    if np.max(np.abs(objective(beta,x,y,w)[1]))>1e-8: raise RuntimeError('Gradient check failed.')
    check=minimize(lambda b:objective(b,x,y,w),np.zeros_like(beta),jac=True,method='L-BFGS-B',
                   options={'maxiter':3000,'ftol':1e-15,'gtol':1e-10,'maxls':50})
    diff=np.max(np.abs(standardized(check.x,x,w)-standardized(beta,x,w)))
    check_g=np.max(np.abs(objective(check.x,x,y,w)[1]))
    if not check.success or diff>1e-6 or check_g>1e-7:
        raise RuntimeError('Independent optimizer agreement failed.')
    return beta,{'newton_evaluations':iteration+1,'gradient_inf_norm':float(np.max(np.abs(objective(beta,x,y,w)[1]))),
                 'independent_optimizer_passed':True,'maximum_optimizer_target_difference_pp':float(100*diff),
                 'independent_optimizer_gradient_inf_norm':float(check_g)}


def standardized(beta,x,w):
    z=x@beta-x[:,1]*beta[1]
    return np.array([np.average(expit(z),weights=w),np.average(expit(z+beta[1]),weights=w)])


def targets_influence(frame,x,beta):
    w=frame.weight.to_numpy(float);w=w/w.sum()
    y=frame.outcome.to_numpy(float);a=frame.sector.to_numpy(int)
    x0=x.copy();x1=x.copy();x0[:,1]=0;x1[:,1]=1
    p=expit(x@beta);q0=expit(x0@beta);q1=expit(x1@beta)
    mu0,mu1=np.array([w@q0,w@q1])
    d=np.column_stack([x0.T@(w*q0*(1-q0)),x1.T@(w*q1*(1-q1))])
    eps=1e-5;numeric=np.zeros_like(d)
    for j in range(len(beta)):
        perturb=np.zeros_like(beta);perturb[j]=eps
        numeric[j]=(standardized(beta+perturb,x,w)-standardized(beta-perturb,x,w))/(2*eps)
    error=float(np.max(np.abs(numeric-d)))
    if error>1e-7: raise RuntimeError('Analytic target derivative check failed.')
    h=objective(beta,x,y,w,hessian=True)[2]
    directions=cho_solve(cho_factor(h),d)
    penalty=np.full(len(beta),RIDGE);penalty[0]=0
    # Normalized weight estimating equation includes penalty contribution.
    score=(y-p)[:,None]*(x@directions)-(penalty*beta)@directions
    model_u=w[:,None]*(np.column_stack([q0-mu0,q1-mu1])+score)
    observed=[];observed_u=[]
    for sector in (0,1):
        mask=a==sector;share=w[mask].sum()
        if share<=0: raise ValueError('Both sectors required.')
        mean=float(w[mask]@y[mask]/share)
        observed.append(mean);observed_u.append(w*mask*(y-mean)/share)
    obs0,obs1=observed;u0,u1=observed_u
    points=np.array([obs0,obs1,obs1-obs0,mu0,mu1,mu1-mu0])
    influence=np.column_stack([u0,u1,u1-u0,model_u,model_u[:,1]-model_u[:,0]])
    if np.max(np.abs(influence.sum(axis=0)))>1e-7: raise RuntimeError('Influence-centering check failed.')
    return points,influence,error


def covariance_from_roster(frame,influence,roster):
    active=frame[['state','stratum']].drop_duplicates()
    selected=roster.merge(active,on=['state','stratum'],how='inner',validate='many_to_one')
    index=pd.MultiIndex.from_frame(selected[['state','stratum','psu']])
    totals=pd.DataFrame(influence,columns=METRICS)
    for col in ('state','stratum','psu'): totals[col]=frame[col].to_numpy()
    totals=totals.groupby(['state','stratum','psu'])[METRICS].sum()
    if not totals.index.isin(index).all(): raise ValueError('Analytic PSUs absent from roster.')
    domain_psus=len(totals);totals=totals.reindex(index,fill_value=0.)
    groups=list(totals.groupby(level=[0,1],sort=True))
    regular=[];single=[];degrees=0
    for _,group in groups:
        a=group.to_numpy(float);n=len(a);degrees+=n-1
        if n==1: single.append(a[0])
        else:
            centered=a-a.mean(axis=0);regular.append(n/(n-1)*(centered.T@centered))
    if not regular or degrees<=0: raise ValueError('Insufficient nonsingleton strata for uncertainty.')
    base=sum(regular,np.zeros((len(METRICS),len(METRICS))))
    grand=totals.to_numpy(float).mean(axis=0)
    adjust=base+sum((np.outer(a-grand,a-grand) for a in single),np.zeros_like(base))
    average=base+len(single)*np.mean(regular,axis=0)
    return {'adjust':adjust,'average':average,'zero_sensitivity_only':base}, {
        'roster_active_psus':len(totals),'analytic_psus':domain_psus,
        'zero_domain_psus_added':len(totals)-domain_psus,'active_state_strata':len(groups),
        'singleton_active_state_strata':len(single),'design_df':degrees,
        'design_key':'raw state, stratum, PSU within each wave',
        'stratum_codes_shared_across_states':int((active.groupby('stratum').state.nunique()>1).sum())}


def support_rows(frame,wave,analysis):
    rows=[]
    for state,q in frame.groupby('state',sort=True):
        pub=int(q.sector.eq(0).sum());priv=int(q.sector.eq(1).sum())
        rows.append({'wave':wave,'analysis':analysis,'raw_state_code':int(state),'n':len(q),
                     'n_public':pub,'n_private':priv,'both_sectors_observed':bool(pub and priv),
                     'at_least_30_each_sector':bool(pub>=30 and priv>=30),
                     'cohort_weight_pct':float(100*q.weight.sum()/frame.weight.sum())})
    cells=frame.groupby(['state','residence','wealth_index'],dropna=False).sector.agg(['count','sum'])
    sparse=(cells['sum']<30)|((cells['count']-cells['sum'])<30)
    keys=pd.MultiIndex.from_frame(frame[['state','residence','wealth_index']])
    is_sparse=keys.isin(cells.index[sparse])
    summary={'state_support_threshold':30,'n_states_missing_one_sector':sum(not r['both_sectors_observed'] for r in rows),
             'n_states_below_30_each_sector':sum(not r['at_least_30_each_sector'] for r in rows),
             'weight_pct_in_states_below_30_each_sector':sum(r['cohort_weight_pct'] for r in rows if not r['at_least_30_each_sector']),
             'coarse_cell_definition':['state','residence','wealth_index'],
             'coarse_cells_below_30_each_sector_n':int(sparse.sum()),
             'weight_pct_in_coarse_sparse_cells':float(100*frame.weight.to_numpy()[is_sparse].sum()/frame.weight.sum()),
             'interpretation':'Descriptive support flags only; no automatic exclusions, no propensity balance or joint-overlap certification.'}
    return rows,summary


def run_analysis(frame,roster,wave,analysis):
    print(f'{wave} {analysis}: fitting fixed-ridge association model ({len(frame):,} births)...',flush=True)
    x,schema=encode(frame,first_delivery=analysis=='first_delivery')
    beta,diagnostics=fit_model(x,frame.outcome.to_numpy(float),frame.weight.to_numpy(float))
    points,influence,error=targets_influence(frame,x,beta)
    covs,design=covariance_from_roster(frame,influence,roster)
    support,support_summary=support_rows(frame,wave,analysis)
    diagnostics.update(target_derivative_max_error=error,model_terms=len(beta),n=len(frame),
                       support=support_summary,design=design)
    rows=[];critical=float(t.ppf(.975,design['design_df']))
    for treatment,cov in covs.items():
        if np.min(np.linalg.eigvalsh(cov)) < -1e-12: raise RuntimeError('Covariance positive-semidefinite check failed.')
        for j,metric in enumerate(METRICS):
            point=float(100*points[j]);se=float(100*np.sqrt(max(0.,cov[j,j])))
            rows.append({'wave':wave,'analysis':analysis,'metric':metric,'estimate':point,'se':se,
                         'ci_lower':point-critical*se,'ci_upper':point+critical*se,
                         'singleton_treatment':treatment,'design_df':design['design_df'],
                         'interval_type':'95% marginal approximate Taylor t; conditional; no multiplicity adjustment'})
    coefficient_rows=[{'wave':wave,'analysis':analysis,'term':term,'coefficient':float(b)}
                      for term,b in zip(schema['labels'],beta)]
    print(f'{wave} {analysis}: observed gap {100*points[2]:.4f} pp; adjusted gap {100*points[5]:.4f} pp; checks PASS',flush=True)
    return rows,support,coefficient_rows,diagnostics,schema,covs


def find_raw(root,prefix,exclude_prefix=None,required=True):
    candidates=sorted({p for p in (root/'data').rglob('*') if p.is_file() and p.suffix.lower()=='.dta'
                       and p.name.upper().startswith(prefix) and not (exclude_prefix and p.name.upper().startswith(exclude_prefix))})
    if len(candidates)==1: return candidates[0]
    if not candidates and not required: return None
    raise FileNotFoundError(f'Expected one local {prefix}*.DTA; found {len(candidates)}. Use the explicit input argument.')


def self_test():
    # This fixture previously exhausted the Armijo search near convergence:
    # gradient 1.52e-9, expected objective decrease 8.54e-17.
    regression_rng=np.random.default_rng(34);regression_n=1800
    rx=np.column_stack([np.ones(regression_n),regression_rng.integers(0,2,regression_n),
                        regression_rng.normal(size=(regression_n,4)),
                        np.eye(20)[regression_rng.integers(0,20,regression_n)][:,1:],np.zeros(regression_n)])
    rb=regression_rng.normal(size=rx.shape[1]);rb[0]=-1;rb[1]=.9
    ry=regression_rng.binomial(1,expit(rx@rb));rw=regression_rng.lognormal(0,1.5,regression_n)
    _,regression_diag=fit_model(rx,ry,rw)
    # Check the change calculation against ordinary objective subtraction when
    # the change is resolvable, including steps outside the expm1 branch.
    for scale in (1e-3,.3,3.):
        delta=scale*regression_rng.normal(size=len(rb))
        expected=objective(rb+delta,rx,ry,rw)[0]-objective(rb,rx,ry,rw)[0]
        assert np.isclose(objective_change(rb,delta,rx,ry,rw),expected,rtol=1e-11,atol=1e-14)
    rng=np.random.default_rng(719);n=480
    frame=pd.DataFrame({'state':np.repeat([1,2],n//2),'stratum':np.tile(np.repeat([1,2],n//4),2),
                        'psu':np.tile(np.arange(n//2)%12+1,2),
                        'birth_order':rng.integers(1,5,n).astype(float),'education_years':rng.integers(0,16,n).astype(float),
                        'wealth_index':rng.integers(1,6,n),'residence':rng.integers(1,3,n),
                        'religion':rng.integers(1,3,n),'social_group':rng.integers(1,5,n),'twin_order':np.zeros(n),
                        'sector':rng.integers(0,2,n),'weight':rng.uniform(.3,2,n)})
    frame['outcome']=rng.binomial(1,expit(-1+.8*frame.sector+.03*frame.education_years))
    frame.loc[0,'education_years']=np.nan
    x,_=encode(frame);beta,diag=fit_model(x,frame.outcome.to_numpy(float),frame.weight.to_numpy(float))
    points,u,error=targets_influence(frame,x,beta)
    # Independently refit after perturbing one sampling weight. This checks
    # both the estimating-equation normalization and empirical reference term.
    for i in (2,270):
        eps=1e-3;values=[]
        for sign in (1,-1):
            test=frame.copy();test.loc[i,'weight']*=1+sign*eps
            b,_=fit_model(x,test.outcome.to_numpy(float),test.weight.to_numpy(float))
            ww=test.weight.to_numpy(float);ww=ww/ww.sum();yy=test.outcome.to_numpy(float);aa=test.sector.to_numpy(int)
            o0=np.average(yy[aa==0],weights=ww[aa==0]);o1=np.average(yy[aa==1],weights=ww[aa==1])
            m0,m1=standardized(b,x,ww)
            values.append(np.array([o0,o1,o1-o0,m0,m1,m1-m0]))
        numeric=(values[0]-values[1])/(2*eps)
        if np.max(np.abs(numeric-u[i]))>1e-7: raise AssertionError('Weight-perturbation influence check failed.')
    roster=design_pairs(frame)
    roster=pd.concat([roster,pd.DataFrame({'state':[1,2,3],'stratum':[1,1,1],'psu':[99,99,99]})],ignore_index=True)
    covs,audit=covariance_from_roster(frame,u,roster)
    assert audit['zero_domain_psus_added']==2 and audit['stratum_codes_shared_across_states']==2
    # Independent grouped sample covariance includes two zero-domain PSUs.
    blocks=[]
    for st,hr in [(1,1),(1,2),(2,1),(2,2)]:
        indices=frame.index[(frame.state==st)&(frame.stratum==hr)]
        psus=sorted(frame.loc[indices,'psu'].unique())
        a=np.stack([u[frame.index[(frame.state==st)&(frame.stratum==hr)&(frame.psu==p)]].sum(axis=0) for p in psus])
        if hr==1:a=np.vstack([a,np.zeros(len(METRICS))])
        blocks.append(len(a)*np.cov(a,rowvar=False,ddof=1))
    assert np.allclose(covs['adjust'],sum(blocks),rtol=1e-10,atol=1e-12)
    scaled=frame.copy();scaled.weight*=17
    scaled_beta,_=fit_model(x,scaled.outcome.to_numpy(float),scaled.weight.to_numpy(float))
    assert np.max(np.abs(standardized(scaled_beta,x,scaled.weight)-points[3:5]))<1e-8
    wrong=roster.loc[roster.state!=2]
    try: covariance_from_roster(frame,u,wrong)
    except ValueError: pass
    else: raise AssertionError('Incorrect state coverage accepted.')
    print(json.dumps({'status':'synthetic_checks_passed','independent_optimizer':diag['independent_optimizer_passed'],
                      'target_derivative_max_error':error,'weight_perturbation_influence_checks':'PASS',
                      'independent_psu_covariance':'PASS','repeated_design_codes_across_states':'PASS',
                      'zero_domain_psus':'PASS','weight_rescaling':'PASS','wrong_state_roster_rejection':'PASS',
                      'near_convergence_line_search_regression':'PASS',
                      'regression_gradient_inf_norm':regression_diag['gradient_inf_norm'],
                      'stable_objective_change_check':'PASS'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path('.'))
    parser.add_argument('--nfhs4',type=Path);parser.add_argument('--nfhs5',type=Path)
    parser.add_argument('--nfhs4-household',type=Path);parser.add_argument('--nfhs5-household',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--private-cohorts',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: self_test();return
    root=args.repo_root.resolve();out=args.output or root/'extensions_work/12_narrow_scope/outputs/national_associations'
    private=args.private_cohorts or root/'data/processed/narrow_facility_study'
    if out.exists() or private.exists(): raise FileExistsError('Existing output/private folders preserved; choose new destinations.')
    inputs={'NFHS-4':(args.nfhs4 or find_raw(root,'IABR74'),args.nfhs4_household or find_raw(root,'IAHR74',required=False)),
            'NFHS-5':(args.nfhs5 or find_raw(root,'IABR7',exclude_prefix='IABR74'),args.nfhs5_household or find_raw(root,'IAHR7',exclude_prefix='IAHR74',required=False))}
    start=time.monotonic();prepared={};audits={}
    # Preflight both mappings before any row-level files or results are written.
    for wave,(br,hr) in inputs.items():
        frame,roster,audit=load_wave(br,hr,wave)
        prepared[wave]=(frame,roster);audits[wave]=audit
        print(f'{wave}: {len(frame):,} dated narrow births; date exclusions {audit["date_or_recall_excluded_n"]:,}; {audit["roster_source"]}',flush=True)
    intervals=[];support=[];coefficients=[];diagnostics={};schemas={};covariances={}
    for wave,(frame,roster) in prepared.items():
        for label,sub in [('all_narrow',frame),('first_delivery',frame.loc[frame.delivery_order.eq(1)].copy().reset_index(drop=True))]:
            rows,states,coefs,diag,schema,covs=run_analysis(sub,roster,wave,label)
            key=wave+'_'+label;intervals.extend(rows);support.extend(states);coefficients.extend(coefs)
            diagnostics[key]=diag;schemas[key]=schema;covariances[key]=covs
    summary={'status':'completed_approximate_association_outputs_require_review','script_sha256':sha256(Path(__file__)),
             'facility_codes':{'public':21,'private':31},'fixed_ridge':RIDGE,
             'ridge_provenance':'Earlier NFHS-5 inner validation selected 1e-5; frozen here for both waves and subgroups; no new tuning or untouched-test performance claim.',
             'reference_rule':'Own full dated narrow cohort within each wave; own first-delivery subgroup for subgroup models.',
             'adjustment_rule':'Birth order and schooling numeric; wealth, residence, religion, social group, plurality and state categorical. First delivery excludes birth order.',
             'state_rule':'Raw state code retained within each wave; no cross-wave state-code equivalence inferred.',
             'primary_singleton_treatment':'adjust (grand mean centering of PSU influence totals); average and zero sensitivity also saved.',
             'uncertainty_method':'Joint fitted-model/reference Taylor; with-replacement ultimate-PSU approximation.',
             'primary_intervals':[row for row in intervals if row['singleton_treatment']=='adjust'],
             'cohort_audit':audits,'model_diagnostics':diagnostics,'limitations':LIMITATIONS,
             'elapsed_seconds':round(time.monotonic()-start,1),
             'old_files_changed':False,'remaining_work':'Common-reference cross-wave comparison, predictive/group heterogeneity and historically aligned state-policy/budget context remain separate.'}
    out.mkdir(parents=True);private.mkdir(parents=True)
    for wave,(frame,_) in prepared.items(): frame.to_parquet(private/(wave.lower().replace('-','')+'_narrow.parquet'),index=False)
    pd.DataFrame(intervals).to_csv(out/'national_association_intervals.csv',index=False)
    pd.DataFrame(support).to_csv(out/'state_sector_support.csv',index=False)
    pd.DataFrame(coefficients).to_csv(out/'model_coefficients.csv',index=False)
    for key,covs in covariances.items():
        for name,cov in covs.items(): pd.DataFrame(cov,index=METRICS,columns=METRICS).to_csv(out/(key+'_'+name+'_covariance.csv'))
    save_json(out/'model_schema.json',schemas);save_json(out/'national_association_summary.json',summary)
    print(f'Aggregate results saved: {out}',flush=True)
    print('Share national_association_summary.json, national_association_intervals.csv and state_sector_support.csv. Keep private Parquet cohorts local.',flush=True)


if __name__=='__main__': main()
