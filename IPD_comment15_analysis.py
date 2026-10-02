"""Comment 15 shared-reference comparison and residence/wealth heterogeneity.
Place beside corrected IPD_narrow_facility_analysis.py. Raw data stay local.
Run --self-test first; then --repo-root . . Existing outputs never overwritten.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ.setdefault(key,'1')
import argparse, importlib.util, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import t

STATES='Andhra Pradesh|Arunachal Pradesh|Assam|Bihar|Chhattisgarh|Goa|Gujarat|Haryana|Himachal Pradesh|Jharkhand|Karnataka|Kerala|Madhya Pradesh|Maharashtra|Manipur|Meghalaya|Mizoram|Nagaland|Odisha|Punjab|Rajasthan|Sikkim|Tamil Nadu|Telangana|Tripura|Uttar Pradesh|Uttarakhand|West Bengal|Andaman and Nicobar Islands|Chandigarh|Delhi|Puducherry|Lakshadweep'.split('|')
EXCLUDED={'jammu and kashmir','ladakh','dadra and nagar haveli','daman and diu','dadra and nagar haveli and daman and diu'}

def canonical(label):
    if not isinstance(label,str): raise ValueError('State value labels are required; numeric equivalence is never assumed.')
    s=re.sub(r'[^a-z ]',' ',label.lower().replace('&',' and '));s=' '.join(s.split())
    aliases={'orissa':'odisha','uttaranchal':'uttarakhand','pondicherry':'puducherry','nct of delhi':'delhi','andaman and nicobar':'andaman and nicobar islands','a and n islands':'andaman and nicobar islands','jammu kashmir':'jammu and kashmir'}
    s=aliases.get(s,s)
    if s in EXCLUDED: return None
    names={n.lower():n for n in STATES}
    if s not in names: raise ValueError('Unrecognized state label '+repr(label)+'; review mapping before analysis.')
    return names[s]

def state_mapping(path):
    raw=pd.read_stata(path,columns=['v024'],convert_categoricals=False).v024
    labelled=pd.read_stata(path,columns=['v024'],convert_categoricals=True).v024
    if len(raw)!=len(labelled): raise ValueError('State label/raw alignment failed.')
    table=pd.DataFrame({'code':raw,'label':labelled.astype(object)}).drop_duplicates()
    if table.code.duplicated().any(): raise ValueError('Ambiguous state label binding.')
    return {int(r.code):canonical(r.label) for r in table.itertuples()},table

def support(f,minimum):
    q=f.groupby(['state','residence','wealth_index','wave','sector']).size().unstack(['wave','sector'],fill_value=0)
    q=q.reindex(columns=pd.MultiIndex.from_product([[0,1],[0,1]]),fill_value=0)
    accepted=q.index[(q>=minimum).all(axis=1)]
    return pd.MultiIndex.from_frame(f[['state','residence','wealth_index']]).isin(accepted)

def covariance(f,u,roster):
    cols=['state','stratum','psu']; active=f[cols[:2]].drop_duplicates()
    rr=roster.merge(active,on=cols[:2],how='inner',validate='many_to_one')
    idx=pd.MultiIndex.from_frame(rr[cols]);values=list(range(u.shape[1]))
    a=pd.DataFrame(u,columns=values)
    for c in cols:a[c]=f[c].to_numpy()
    a=a.groupby(cols)[values].sum()
    if not a.index.isin(idx).all():raise ValueError('PSUs absent from roster.')
    domain=len(a);a=a.reindex(idx,fill_value=0.);groups=list(a.groupby(level=[0,1]))
    regular=[];singles=[];df=0
    for _,g in groups:
        z=g.to_numpy();n=len(z);df+=n-1
        if n==1:singles.append(z[0])
        else: z=z-z.mean(axis=0);regular.append(n/(n-1)*(z.T@z))
    if not regular or df<=0:raise ValueError('Insufficient survey strata.')
    base=sum(regular,np.zeros((u.shape[1],u.shape[1])));grand=a.to_numpy().mean(axis=0)
    adjust=base+sum([np.outer(z-grand,z-grand) for z in singles],np.zeros_like(base))
    average=base+len(singles)*np.mean(regular,axis=0)
    return {'adjust':adjust,'average':average,'zero_sensitivity_only':base},{'df':df,'roster_psus':len(a),'zero_domain_psus':len(a)-domain,'singleton_strata':len(singles)}

def reference_weights(f,mask):
    wr=np.zeros(len(f))
    for wave in (0,1):
        selected=mask & f.wave.eq(wave).to_numpy();w=f.weight.to_numpy(float)
        if not selected.any():raise ValueError('Reference lacks one wave.')
        wr[selected]=.5*w[selected]/w[selected].sum()
    return wr

def targets(base,f,x,betas,mask):
    """Each target has both fitted coefficient and empirical reference influence."""
    wr=reference_weights(f,mask);w=f.weight.to_numpy(float);y=f.outcome.to_numpy(float)
    points=[];us=[]
    for target_wave,beta in enumerate(betas):
        x0=x.copy();x1=x.copy();x0[:,1]=0;x1[:,1]=1
        # All sector interactions are identified in labels passed by caller.
        for col in INTERACTIONS:
            x0[:,col]=0.
            x1[:,col]=INTERACTION_VALUES[col]
        q0=expit(x0@beta);q1=expit(x1@beta);delta=q1-q0;mu=wr@delta
        d=x1.T@(wr*q1*(1-q1))-x0.T@(wr*q0*(1-q0))
        train=f.wave.eq(target_wave).to_numpy();wt=w[train]/w[train].sum();xt=x[train]
        h=base.objective(beta,xt,y[train],w[train],hessian=True)[2]
        direction=cho_solve(cho_factor(h),d);penalty=np.full(len(beta),base.RIDGE);penalty[0]=0
        u=np.zeros(len(f));p=expit(xt@beta)
        u[train]+=wt*((y[train]-p)*(xt@direction)-(penalty*beta)@direction)
        for ref_wave in (0,1):
            selected=mask & f.wave.eq(ref_wave).to_numpy();mean=wr[selected]@delta[selected]/.5
            u[selected]+=wr[selected]*(delta[selected]-mean)
        if abs(u.sum())>1e-7:raise RuntimeError('Influence centering failed.')
        # Directional finite difference checks for every reference target.
        direction_test=np.random.default_rng(38).normal(size=len(beta));direction_test/=np.linalg.norm(direction_test);eps=1e-5
        derivative=wr@(expit(x1@(beta+eps*direction_test))-expit(x0@(beta+eps*direction_test))-expit(x1@(beta-eps*direction_test))+expit(x0@(beta-eps*direction_test)))/(2*eps)
        if abs(derivative-d@direction_test)>1e-7:raise RuntimeError('Reference derivative failed.')
        points.append(mu);us.append(u)
    return np.array(points),np.column_stack(us)

INTERACTIONS=[];INTERACTION_VALUES={}

def analysis(base,frames,rosters,first,minimum):
    global INTERACTIONS,INTERACTION_VALUES
    pieces=[]
    for wave,f in enumerate(frames):
        q=f.loc[f.delivery_order.eq(1)].copy() if first else f.copy();q['wave']=wave;pieces.append(q)
    f=pd.concat(pieces,ignore_index=True);mask=support(f,minimum)
    if not mask.any():raise ValueError('No common supported reference cells.')
    # Fit all retained geography rows, reference restricted identically across waves.
    x,schema=base.encode(f,first_delivery=first);labels=list(schema['labels']);INTERACTIONS=[];INTERACTION_VALUES={}
    extra=[]
    for c,levels in [('residence',[2]),('wealth_index',[2,3,4,5])]:
        for level in levels:
            value=f[c].eq(level).to_numpy(float);index=x.shape[1]+len(extra)
            extra.append(f.sector.to_numpy()*value);INTERACTIONS.append(index);INTERACTION_VALUES[index]=value;labels.append(f'private:{c}={level}')
    x=np.column_stack([x]+extra);betas=[];diagnostics=[]
    for wave in (0,1):
        tr=f.wave.eq(wave).to_numpy();print(f'{"first_delivery" if first else "all_narrow"} wave {wave+4}: fitting {tr.sum():,} births',flush=True)
        b,d=base.fit_model(x[tr],f.outcome.to_numpy(float)[tr],f.weight.to_numpy(float)[tr]);betas.append(b);diagnostics.append(d)
    specs=[('overall',mask)]
    for c,levels in [('residence',[1,2]),('wealth_index',[1,2,3,4,5])]:
        specs.extend((f'{c}_{level}',mask & f[c].eq(level).to_numpy()) for level in levels)
    estimates=[];influences=[];names=[]
    for group,m in specs:
        p,u=targets(base,f,x,betas,m)
        for wave in (0,1):names.append((group,f'NFHS-{wave+4}'));estimates.append(p[wave]);influences.append(u[:,wave])
        names.append((group,'NFHS-5_minus_NFHS-4'));estimates.append(p[1]-p[0]);influences.append(u[:,1]-u[:,0])
    # Within-wave contrasts preserve shared fitting and reference covariance.
    for c,levels in [('residence',[2]),('wealth_index',[2,3,4,5])]:
        for level in levels:
            for wave in ('NFHS-4','NFHS-5'):
                hi=names.index((f'{c}_{level}',wave));lo=names.index((f'{c}_1',wave))
                names.append((f'{c}_{level}_minus_1',wave));estimates.append(estimates[hi]-estimates[lo]);influences.append(influences[hi]-influences[lo])
    u=np.column_stack(influences);covs={};design={}
    for wave in (0,1):
        tr=f.wave.eq(wave).to_numpy();covs[wave],design[wave]=covariance(f.loc[tr],u[tr],rosters[wave])
    rows=[];df=min(d['df'] for d in design.values());crit=t.ppf(.975,df)
    for rule in covs[0]:
        cov=covs[0][rule]+covs[1][rule]
        if np.linalg.eigvalsh(cov).min() < -1e-10:raise RuntimeError('Covariance not positive semidefinite.')
        for j,(group,wave) in enumerate(names):
            se=100*np.sqrt(max(0,cov[j,j]));point=100*estimates[j]
            rows.append({'analysis':'first_delivery' if first else 'all_narrow','reference_minimum_each_sector_each_wave':minimum,'group':group,'wave_or_contrast':wave,'estimate_pp':point,'se_pp':se,'ci_lower_pp':point-crit*se,'ci_upper_pp':point+crit*se,'singleton_treatment':rule,'conservative_design_df':df})
    audit={'reference_n':int(mask.sum()),'reference_rule':'50 percent survey weighted NFHS-4 plus 50 percent survey weighted NFHS-5 in identical supported state residence wealth cells','waves':{},'fit_diagnostics':diagnostics,'design':design,'model_labels':labels}
    for wave in (0,1):
        tr=f.wave.eq(wave).to_numpy();w=f.weight.to_numpy(float)
        audit['waves'][f'NFHS-{wave+4}']={'training_n':int(tr.sum()),'reference_n':int((mask&tr).sum()),'excluded_reference_weight_pct':float(100*w[tr&~mask].sum()/w[tr].sum())}
    return rows,audit

def self_test(base):
    rng=np.random.default_rng(79);frames=[];rosters=[]
    for wave in (0,1):
        n=1200;f=pd.DataFrame({'state':np.repeat([1,2],n//2),'stratum':np.tile(np.repeat([1,2],n//4),2),'psu':np.tile(np.arange(n//2)%30+1,2),'birth_order':rng.integers(1,4,n),'delivery_order':np.ones(n),'education_years':rng.integers(0,20,n),'wealth_index':rng.integers(1,6,n),'residence':rng.integers(1,3,n),'religion':rng.integers(1,3,n),'social_group':rng.integers(1,4,n),'twin_order':np.zeros(n),'sector':rng.integers(0,2,n),'weight':rng.uniform(.2,2,n)})
        f['outcome']=rng.binomial(1,expit(-1+.6*f.sector+.2*wave*f.sector+.4*f.sector*f.residence.eq(2)));frames.append(f);rosters.append(base.design_pairs(f))
    rows,audit=analysis(base,frames,rosters,False,1)
    assert len(rows)==102 and audit['reference_n']==2400
    # Independent re-fit weight perturbations verify BOTH models and shared reference.
    f=pd.concat([q.assign(wave=w) for w,q in enumerate(frames)],ignore_index=True);x,_=base.encode(f);global INTERACTIONS,INTERACTION_VALUES
    INTERACTIONS=[];INTERACTION_VALUES={};mask=np.ones(len(f),bool);betas=[]
    value=f.residence.eq(2).to_numpy(float);INTERACTIONS=[x.shape[1]];INTERACTION_VALUES={x.shape[1]:value};x=np.column_stack([x,f.sector.to_numpy()*value])
    for wave in (0,1):
        tr=f.wave.eq(wave).to_numpy();betas.append(base.fit_model(x[tr],f.outcome.to_numpy()[tr],f.weight.to_numpy()[tr])[0])
    pp,u=targets(base,f,x,betas,mask)
    for i in (3,1300):
        vals=[];eps=1e-3
        for sign in (1,-1):
            ff=f.copy();ff.loc[i,'weight']*=1+sign*eps;bs=[]
            for wave in (0,1):
                tr=ff.wave.eq(wave).to_numpy();bs.append(base.fit_model(x[tr],ff.outcome.to_numpy()[tr],ff.weight.to_numpy()[tr])[0])
            vals.append(targets(base,ff,x,bs,mask)[0])
        assert np.max(np.abs((vals[0]-vals[1])/(2*eps)-u[i]))<1e-7
    assert canonical('Orissa')=='Odisha' and canonical('Ladakh') is None
    assert canonical('lakshadweep')=='Lakshadweep'
    assert len(STATES)==33 and len(set(STATES))==33
    for state_name in STATES: assert canonical(state_name.lower())==state_name
    for state_name in EXCLUDED: assert canonical(state_name) is None
    try:canonical(12)
    except ValueError:pass
    else:raise AssertionError('Numeric labels accepted')
    # Same fitted model gives exactly zero cross-wave contrast on the same reference.
    INTERACTIONS=[];INTERACTION_VALUES={}
    same=pd.concat([frames[0].assign(wave=0),frames[0].assign(wave=1)],ignore_index=True);xx,_=base.encode(same)
    bb=base.fit_model(xx[:1200],same.outcome.to_numpy()[:1200],same.weight.to_numpy()[:1200])[0]
    p,z=targets(base,same,xx,[bb,bb],np.ones(len(same),bool));assert abs(p[1]-p[0])<1e-12
    print(json.dumps({'status':'synthetic_checks_passed','joint_two_wave_weight_perturbation':'PASS','interaction_targets_and_covariance':'PASS','state_label_fail_closed':'PASS','same_reference_identity':'PASS'}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo-root',type=Path,default=Path('.'));p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true')
    a=p.parse_args();root=a.repo_root.resolve();spec=importlib.util.spec_from_file_location('base',Path(__file__).with_name('IPD_narrow_facility_analysis.py'));base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    if a.self_test:self_test(base);return
    out=a.output or root/'extensions_work/12_narrow_scope/outputs/comment15_shared_reference'
    if out.exists():raise FileExistsError('Existing output preserved; choose new --output.')
    baseline=root/'extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json';old=json.loads(baseline.read_text());frames=[];rosters=[];audits={};mapping_rows=[];start=time.monotonic()
    for wave in ('NFHS-4','NFHS-5'):
        br=base.find_raw(root,'IABR74' if wave=='NFHS-4' else 'IABR7',exclude_prefix=None if wave=='NFHS-4' else 'IABR74');hr=base.find_raw(root,'IAHR74' if wave=='NFHS-4' else 'IAHR7',exclude_prefix=None if wave=='NFHS-4' else 'IAHR74',required=False)
        f,roster,audit=base.load_wave(br,hr,wave)
        if audit['birth_sha256']!=old['cohort_audit'][wave]['birth_sha256']:raise ValueError('Birth input differs from reviewed baseline.')
        previous=old['cohort_audit'][wave]
        if previous['household_roster_available'] and audit.get('household_sha256')!=previous.get('household_sha256'):
            raise ValueError('Required household file differs from reviewed baseline or is missing.')
        mapping,table=state_mapping(br);names=f.state.map(mapping);keep=names.notna()
        if set(f.state)-set(mapping):raise ValueError('State mapping incomplete.')
        audit['geography_excluded_n']=int((~keep).sum());audit['geography_excluded_weight_pct']=float(100*f.loc[~keep,'weight'].sum()/f.weight.sum())
        for r in table.itertuples():mapping_rows.append({'wave':wave,'raw_state_code':int(r.code),'raw_label':r.label,'common_state':mapping[int(r.code)],'included':mapping[int(r.code)] is not None})
        f=f.loc[keep].copy();f['raw_state']=f.state;f['state']=names.loc[keep].map({n:i+1 for i,n in enumerate(STATES)}).astype(int)
        roster=roster.copy();rn=roster.state.map(mapping);roster=roster.loc[rn.notna()].copy();roster.state=rn.loc[rn.notna()].map({n:i+1 for i,n in enumerate(STATES)}).astype(int)
        frames.append(f);rosters.append(roster);audits[wave]=audit
    rows=[];diags={}
    for first in (False,True):
        for minimum in (1,30):
            rr,d=analysis(base,frames,rosters,first,minimum);rows+=rr;diags[f'{first}_{minimum}']=d
    summary={'status':'completed_requires_actual_output_review','facility_codes':{'public':21,'private':31},'fixed_ridge':base.RIDGE,'script_sha256':base.sha256(__file__),'base_script_sha256':base.sha256(Path(base.__file__)),'baseline_sha256':base.sha256(baseline),'cohort_audit':audits,'diagnostics':diags,'primary_rule':'At least 1 birth in each sector in EACH wave per common state residence wealth cell; >=30 sensitivity','geography_rule':'Verified labelled states with unchanged directly comparable units; JK Ladakh and Dadra Daman units excluded, no raw code equivalence assumed','model_rule':'Separate wave fits with shared coefficient vocabulary and private by residence and private by wealth interactions; first delivery fit separately','uncertainty':'Joint fitted coefficient and empirical equal-wave pooled reference influence; independent survey waves assumed; minimum wave design df; WR ultimate PSU Taylor; singleton adjust primary','limitations':base.LIMITATIONS+['Model changed by declared interactions; do not compare its gaps directly with earlier additive model as a temporal test.','Wave samples treated as independent; no cross-wave PSU dependence estimated.','Wealth represents relative survey rank, not constant absolute purchasing power.','Reference support classification and verified geographic mapping treated as fixed; coarse cells do not prove joint overlap.','Residence and wealth heterogeneity is model based; prior predictive score groups are not reproduced.','No causal temporal or policy effects and no multiplicity adjustment.'],'elapsed_seconds':round(time.monotonic()-start,1),'existing_outputs_changed':False}
    out.mkdir(parents=True);pd.DataFrame(rows).to_csv(out/'comment15_intervals.csv',index=False);pd.DataFrame(mapping_rows).to_csv(out/'comment15_state_mapping.csv',index=False);base.save_json(out/'comment15_summary.json',summary)
    print('Aggregate outputs saved: '+str(out));print('Share comment15_summary.json, comment15_intervals.csv, comment15_state_mapping.csv only.')
if __name__=='__main__':main()
