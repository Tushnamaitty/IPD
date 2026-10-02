"""Comment 17: West Bengal exclusion and historical-context source review.

Place beside IPD_narrow_facility_analysis.py and IPD_narrow_support_sensitivity.py.
Run --self-test, then --repo-root . after the support analysis finishes.
Exact codes 21/31; fixed-ridge adjusted associations, not policy effects.
Reads local raw files; writes only a NEW folder of aggregate JSON/CSV files.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t

METRICS = [
    'full_fit_full_reference_gap_pp',
    'full_fit_non_west_bengal_reference_gap_pp',
    'non_west_bengal_refit_same_reference_gap_pp',
    'reference_only_change_pp',
    'refit_change_on_same_reference_pp',
    'total_exclusion_change_pp',
]
REVIEWED_BIRTH_HASHES = {
    'NFHS-4': '6fcfebbc37879be3ddf7b8555df9be6df87a9de69480612235adf53eb59fc9b8',
    'NFHS-5': '93267b17596929371dc3190d17af114271851f14d1f566678ffceda2db4cae6e',
}

# Source review records historical facts, not respondent policy exposure.
# No numeric budget or facility panel is invented or silently extrapolated.
SOURCE_REVIEW = {
    'status': 'historical_source_review_started_full_linkage_not_completed',
    'review_date': '2026-10-02',
    'sources': [
        {
            'id': 'wb_fpms_history',
            'publisher': 'Government of West Bengal, Finance Department',
            'title': 'Budget Speech 2024-25',
            'url': 'https://finance.wb.gov.in/writereaddata/Budget_Speech/2024_English.pdf',
            'location': 'PDF page 55 (printed page 54), Health & Family Welfare',
            'historical_fact': 'The speech dates Fair Price Medicine Shops in PPP mode to December 2012.',
            'date': '2012-12', 'date_precision': 'month',
            'geographic_scope': 'West Bengal',
            'use': 'Rationale for an exploratory state exclusion; not evidence of individual coverage or an effect on cesareans.',
            'restriction': 'Current shop counts and diagnostic-centre counts in the 2024 speech cannot be assigned to NFHS birth years. No diagnostic-centre launch date is established here.',
        },
        {
            'id': 'national_jssk_launch', 'publisher': 'National Health Mission',
            'title': 'Janani Shishu Suraksha Karyakram (JSSK)',
            'url': 'https://www.nhm.gov.in/index4.php?lang=1&level=0&lid=171&linkid=150',
            'historical_fact': 'The national initiative was launched on 1 June 2011.',
            'date': '2011-06-01', 'date_precision': 'day',
            'geographic_scope': 'National launch',
            'restriction': 'A national announcement is not a verified date of implementation at each state or facility.',
        },
        {
            'id': 'maharashtra_jssk_launch', 'publisher': 'National Health Mission, Maharashtra',
            'title': 'RCH - Janani Shishu Suraksha Karyakram (JSSK)',
            'url': 'https://nhm.maharashtra.gov.in/en/scheme/rch-janani-shishu-suraksha-karyakram-jssk/',
            'historical_fact': 'The state page reports a Maharashtra launch on 7 October 2011, following a Government Resolution dated 26 September 2011.',
            'date': '2011-10-07', 'date_precision': 'day',
            'geographic_scope': 'Maharashtra',
            'restriction': 'A later retrospective page establishes a reported launch date, not delivery-specific uptake or facility readiness.',
        },
        {
            'id': 'national_laqshya_launch', 'publisher': 'National Health Mission',
            'title': 'LaQshya: Labour Room Quality Improvement Initiative',
            'url': 'https://nhm.gov.in/nhm/index1.php?lang=1&level=3&lid=690&sublinkid=1307',
            'historical_fact': 'The national initiative was launched on 11 December 2017.',
            'date': '2017-12-11', 'date_precision': 'day',
            'geographic_scope': 'National launch',
            'restriction': 'State or hospital implementation dates require additional evidence.',
        },
        {
            'id': 'nhp2020_actual_health_spending',
            'publisher': 'Central Bureau of Health Intelligence, MoHFW',
            'title': 'National Health Profile 2020, table 4.1.4',
            'url': 'https://cbhidghs.mohfw.gov.in/sites/default/files/NHP/National-health-2020.pdf',
            'location': 'PDF pages 272-273 (printed pages 246-247)',
            'period': '2015-16', 'estimate_type': 'Actuals',
            'unit': 'Rupees crore, nominal',
            'historical_fact': 'Table 4.1.4 provides state/UT health expenditure by components for 2015-16.',
            'restriction': 'This is actual expenditure, not a budget allocation. The neighbouring table 4.1.5 uses revised estimates for state health and total expenditure; do not mix the two.',
            'linkage_rule': 'Use an exact completed preceding financial year (April-March), with no carrying one year to every birth year. Obtain an annual series and a defensible denominator before comparing spending intensity.',
        },
        {
            'id': 'nhp2020_fru_infrastructure',
            'publisher': 'Central Bureau of Health Intelligence, MoHFW; underlying RHS 2018-19',
            'title': 'National Health Profile 2020, table 6.2.2(a)',
            'url': 'https://cbhidghs.mohfw.gov.in/sites/default/files/NHP/National-health-2020.pdf',
            'location': 'PDF page 448 (printed page 422)',
            'as_of': '2019-03-31',
            'historical_fact': 'The table reports state/UT counts of FRUs and those with operating theatres, labour rooms, and blood storage/linkage.',
            'restriction': 'A 2019 snapshot is unavailable as preceding context for earlier births. It is not a facility-level functional-capacity assessment; the marginal columns do not identify the number satisfying every criterion jointly.',
            'linkage_rule': 'Obtain repeated comparable snapshots; join only a preceding snapshot, report its age, and declare a maximum lag. Month-only birth dates leave the snapshot month ambiguous.',
        },
    ],
    'linkage_decisions': [
        'Use birth month/year from b3, not interview date, for historical alignment.',
        'Validate each wave state-label mapping. Numeric state codes cannot be matched directly between waves.',
        'Treat state-boundary changes explicitly: Jammu and Kashmir/Ladakh, Dadra and Nagar Haveli/Daman and Diu, and Andhra Pradesh/Telangana. Do not allocate old totals to new units without evidence.',
        'Reported state is interview residence, not certified birth-time residence or the delivery facility state; describe linkage as contextual.',
        'Keep budget estimates, revised estimates, actual expenditure, and NHA government expenditure definitions separate; retain units, source, and revision vintage.',
        'A time-invariant state indicator is absorbed by unpenalized state fixed effects. Ridge does not create substantive identification of a state-context main effect. Choose a justified time-varying or interaction analysis.',
        'National policy dates provide temporal context, not cross-state implementation variation. Month of launch is ambiguous when birth dates contain only month/year.',
        'Missing historical context remains missing. Report coverage and excluded weight; no retrospective assignment of current facilities to older births.',
        'Context associations require uncertainty reflecting their state-level variation. A large birth count does not create a large number of independent policy units.',
    ],
    'remaining_for_comment_17': [
        'Review the local West Bengal sensitivity outputs.',
        'Build and validate an annual state health-budget series, with estimate type and denominators preserved.',
        'Build comparable historical facility measures and a state policy implementation register.',
        'Audit state-by-birth-period linkage coverage before implementing the contextual model.',
        'Implement and review a justified state-context sensitivity model; do not infer a policy or spending effect.',
    ],
}


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def west_bengal_code(audit, wave):
    tables = audit['named_value_label_tables']
    found = [values for name, values in tables.items() if name.lower() == 'v024']
    if len(found) != 1:
        raise ValueError(f'{wave}: no unique literal v024 label table; mapping not guessed.')
    codes = [int(k) for k, value in found[0].items() if value.strip().lower() == 'west bengal']
    expected = {'NFHS-4': 35, 'NFHS-5': 19}[wave]
    if codes != [expected]:
        raise ValueError(f'{wave}: West Bengal label/code differs from the reviewed input.')
    return codes[0]


def calculate(base, helper, frame, keep, first_delivery=False):
    frame = frame.reset_index(drop=True)
    keep = np.asarray(keep, dtype=bool)
    if len(keep) != len(frame) or not keep.any():
        raise ValueError('Empty or invalid non-West Bengal reference.')
    if frame.loc[keep, 'sector'].nunique() != 2:
        raise ValueError('Non-West Bengal cohort lacks a sector.')
    x, schema = base.encode(frame, first_delivery=first_delivery)
    beta, full_fit = base.fit_model(x, frame.outcome.to_numpy(float), frame.weight.to_numpy(float))
    full_points, full_u, e0 = helper.reference_targets(base, frame, x, beta, np.ones(len(frame), bool))
    kept_points, kept_u, e1 = helper.reference_targets(base, frame, x, beta, keep)
    restricted = frame.loc[keep].reset_index(drop=True)
    # Retain the original schema/scales/penalty: only training data change.
    refit, refit_diagnostics = base.fit_model(x[keep], restricted.outcome.to_numpy(float), restricted.weight.to_numpy(float))
    refit_points, refit_u, e2 = helper.reference_targets(base, restricted, x[keep], refit, np.ones(len(restricted), bool))
    padded = np.zeros(len(frame)); padded[keep] = refit_u[:, 5]
    a, b, c = full_points[5], kept_points[5], refit_points[5]
    ua, ub, uc = full_u[:, 5], kept_u[:, 5], padded
    points = np.array([a, b, c, b-a, c-b, c-a])
    influence = np.column_stack([ua, ub, uc, ub-ua, uc-ub, uc-ua])
    if np.max(np.abs(influence.sum(axis=0))) > 1e-7:
        raise RuntimeError('Joint influence-centering check failed.')
    weight = frame.weight.to_numpy(float)
    excluded = float(100*weight[~keep].sum()/weight.sum())
    if abs(100*(b-a)) > 2*excluded+1e-8:
        raise RuntimeError('Reference-only change exceeds its deterministic bound.')
    return points, influence, {
        'full_fit': full_fit, 'restricted_refit': refit_diagnostics,
        'model_terms': len(beta), 'directional_derivative_max_error': max(e0,e1,e2),
        'excluded_weight_pct': excluded, 'excluded_n': int((~keep).sum()),
        'full_n': len(frame), 'restricted_n': len(restricted),
        'reference_only_bound_pp': 2*excluded,
        'schema': schema,
    }


def interval_rows(base, frame, roster, points, influence, metadata):
    # All estimates and paired differences use the same union of training
    # strata. Excluded-state rows have zero influence for the restricted fit.
    # This keeps covariance and singleton treatment coherent for contrasts.
    covariances, design = base.covariance_from_roster(frame, influence, roster)
    critical = float(t.ppf(.975, design['design_df']))
    rows = []
    for treatment, covariance in covariances.items():
        if np.linalg.eigvalsh(covariance).min() < -1e-12:
            raise RuntimeError('Joint covariance is not positive semidefinite.')
        for j, metric in enumerate(METRICS):
            se = float(100*np.sqrt(max(0., covariance[j,j])))
            value = float(100*points[j])
            rows.append(dict(metadata, metric=metric, estimate=value, se=se,
                             ci_lower=value-critical*se, ci_upper=value+critical*se,
                             singleton_treatment=treatment, design_df=design['design_df']))
    return rows, design


def independent_points(base, frame, keep):
    x, _ = base.encode(frame)
    beta, _ = base.fit_model(x, frame.outcome.to_numpy(float), frame.weight.to_numpy(float))
    refit, _ = base.fit_model(x[keep], frame.loc[keep,'outcome'].to_numpy(float), frame.loc[keep,'weight'].to_numpy(float))
    w = frame.weight.to_numpy(float); w = w/w.sum()
    wr = np.where(keep,w,0.); wr = wr/wr.sum()
    def gap(b, weights):
        z = x@b-x[:,1]*b[1]
        return float(weights@(expit(z+b[1])-expit(z)))
    a, b, c = gap(beta,w), gap(beta,wr), gap(refit,wr)
    return np.array([a,b,c,b-a,c-b,c-a])


def self_test(base, helper):
    rng = np.random.default_rng(173)
    n = 720
    frame = pd.DataFrame({
        'state': np.repeat([1,2,3],n//3),
        'stratum': np.tile(np.repeat([1,2],n//6),3),
        'psu': np.tile(np.arange(n//3)%12+1,3),
        'sector': rng.integers(0,2,n), 'weight': rng.uniform(.3,2,n),
        'education_years': rng.integers(0,20,n).astype(float),
        'birth_order': rng.integers(1,5,n).astype(float),
        'wealth_index': rng.integers(1,6,n), 'residence': rng.integers(1,3,n),
        'religion': rng.integers(1,3,n), 'social_group': rng.integers(1,5,n),
        'twin_order': np.zeros(n),
    })
    frame['outcome'] = rng.binomial(1,expit(-1+.8*frame.sector+.3*frame.state.eq(3)))
    keep = frame.state.ne(3).to_numpy()
    points, influence, _ = calculate(base,helper,frame,keep)
    assert np.allclose(points,independent_points(base,frame,keep),atol=1e-12,rtol=0)
    errors = []
    for i in (2,600):
        values = []; eps = 1e-3
        for sign in (1,-1):
            changed = frame.copy(); changed.loc[i,'weight'] *= 1+sign*eps
            values.append(independent_points(base,changed,keep))
        error = float(np.max(np.abs((values[0]-values[1])/(2*eps)-influence[i])))
        errors.append(error)
        assert error < 1e-7
    null, null_u, _ = calculate(base,helper,frame,np.ones(n,bool))
    assert np.max(np.abs(null[3:])) < 1e-12
    assert np.max(np.abs(null_u[:,3:])) < 1e-12
    roster = base.design_pairs(frame)
    roster = pd.concat([roster,pd.DataFrame({'state':[1,2,3],'stratum':[1,1,1],'psu':[99,99,99]})],ignore_index=True)
    rows, design = interval_rows(base,frame,roster,points,influence,{})
    assert len(rows) == 18 and design['zero_domain_psus_added'] == 3
    for wave, code in [('NFHS-4',35),('NFHS-5',19)]:
        assert west_bengal_code({'named_value_label_tables':{'V024':{str(code):'west bengal'}}},wave) == code
    try:
        west_bengal_code({'named_value_label_tables':{'V024':{'35':'west bengal'}}},'NFHS-5')
    except ValueError:
        pass
    else:
        raise AssertionError('Cross-wave numeric code shortcut accepted.')
    print(json.dumps({'status':'synthetic_checks_passed',
        'independent_predictions_and_refits':'PASS',
        'paired_influence_weight_perturbations_inside_and_outside':'PASS',
        'maximum_weight_perturbation_error':max(errors),
        'unchanged_population_null_contrast':'PASS',
        'zero_domain_psus_and_joint_covariance':'PASS',
        'wave_specific_state_mapping':'PASS'},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--nfhs4',type=Path); parser.add_argument('--nfhs5',type=Path)
    parser.add_argument('--nfhs4-household',type=Path); parser.add_argument('--nfhs5-household',type=Path)
    parser.add_argument('--base-script',type=Path); parser.add_argument('--support-script',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args(); root = args.repo_root.resolve()
    beside = Path(__file__).resolve().parent
    base_path = args.base_script or beside/'IPD_narrow_facility_analysis.py'
    helper_path = args.support_script or beside/'IPD_narrow_support_sensitivity.py'
    helper = import_module(helper_path,'ipd_support_helpers')
    base = helper.load_base(base_path)
    if args.self_test:
        self_test(base,helper); return
    output = args.output or root/'extensions_work/04_health_system_context/outputs/narrow_west_bengal_sensitivity'
    if output.exists():
        raise FileExistsError('Existing output preserved; choose another --output.')
    baseline = args.baseline or root/'extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json'
    original = json.loads(baseline.read_text(encoding='utf-8'))
    if original['facility_codes'] != {'public':21,'private':31} or original['fixed_ridge'] != base.RIDGE:
        raise ValueError('Baseline cohort scope or penalty differs.')
    previous = {(r['wave'],r['analysis'],r['metric']):r for r in original['primary_intervals']}
    start = time.monotonic(); rows = []; diagnostics = {}; audits = {}
    for wave in ('NFHS-4','NFHS-5'):
        br = (args.nfhs4 if wave=='NFHS-4' else args.nfhs5) or base.find_raw(root,'IABR74' if wave=='NFHS-4' else 'IABR7',exclude_prefix=None if wave=='NFHS-4' else 'IABR74')
        explicit = args.nfhs4_household if wave=='NFHS-4' else args.nfhs5_household
        hr, count = helper.hr_candidates(root,wave,explicit)
        old_audit = original['cohort_audit'][wave]
        if old_audit['household_roster_available'] and hr is None:
            raise FileNotFoundError(f'{wave}: original household file required.')
        frame, roster, audit = base.load_wave(br,hr,wave)
        if audit['birth_sha256'] != old_audit['birth_sha256'] or audit['birth_sha256'] != REVIEWED_BIRTH_HASHES[wave]:
            raise ValueError(f'{wave}: raw birth file differs from reviewed baseline.')
        if old_audit['household_roster_available'] and audit.get('household_sha256') != old_audit.get('household_sha256'):
            raise ValueError(f'{wave}: household input changed; inspect first.')
        code = west_bengal_code(audit,wave)
        audit['west_bengal_raw_code'] = code
        audit['household_candidates_found'] = count
        audit['household_frame_newly_found'] = bool(hr and not old_audit['household_roster_available'])
        audits[wave] = audit
        for analysis, sub in [('all_narrow',frame),('first_delivery',frame.loc[frame.delivery_order.eq(1)].copy())]:
            sub = sub.reset_index(drop=True); keep = sub.state.ne(code).to_numpy()
            if keep.all():
                raise ValueError(f'{wave} {analysis}: no West Bengal observations.')
            print(f'{wave} {analysis}: fitting full and non-West Bengal models...',flush=True)
            points, influence, checks = calculate(base,helper,sub,keep,analysis=='first_delivery')
            old = previous[(wave,analysis,'adjusted_gap_pp')]
            if abs(100*points[0]-old['estimate']) > 1e-5:
                raise RuntimeError('Original adjusted gap failed reproduction.')
            meta = {'wave':wave,'analysis':analysis,'full_n':len(sub),
                    'restricted_n':int(keep.sum()),'excluded_n':checks['excluded_n'],
                    'excluded_weight_pct':checks['excluded_weight_pct']}
            values, design = interval_rows(base,sub,roster,points,influence,meta)
            rows.extend(values)
            diagnostics[wave+'_'+analysis] = dict(checks,design=design,baseline_point_reproduced=True)
            print(f'{wave} {analysis}: same-reference refit change {100*points[4]:+.4f} pp; total exclusion change {100*points[5]:+.4f} pp.',flush=True)
    summary = {
        'status':'west_bengal_sensitivity_completed_requires_output_review',
        'script_sha256':base.sha256(Path(__file__)),
        'base_script_sha256':base.sha256(base_path),'support_script_sha256':base.sha256(helper_path),
        'baseline_sha256':base.sha256(baseline),'fixed_ridge':base.RIDGE,
        'facility_codes':{'public':21,'private':31},
        'cohort_audit':audits,'diagnostics':diagnostics,
        'primary_intervals':[r for r in rows if r['singleton_treatment']=='adjust'],
        'reference_rule':'For each wave/subgroup, the same survey-weighted non-West Bengal covariate reference is used for the original full model and the refitted model excluding West Bengal.',
        'contrast_rule':'Refit change on the identical reference isolates the model-fit sensitivity. Reference-only change captures removing West Bengal from standardization. Total exclusion changes both training and reference.',
        'uncertainty':'Joint fitted-model and empirical-reference Taylor influence; paired differences retain shared covariance. With-replacement ultimate-PSU approximation with raw state/stratum/PSU keys and observed roster zero-domain PSUs.',
        'design_rule':'For coherent paired covariance, all six quantities use the full training union of active strata and its design df. The restricted-fit influence is zero on excluded-state observations. Singleton sensitivities share this rule.',
        'state_adjustment':'Original state categorical terms retained; original schema/scales and ridge fixed for both fits.',
        'original_baseline_points_reproduced':True,'existing_outputs_changed':False,
        'comment_17_complete':False,
        'limitations':base.LIMITATIONS+[
            'Removing West Bengal is an exploratory geographic sensitivity, not identification of a medicine-shop, diagnostic-centre, budget or facility-capacity effect.',
            'Policy exposure, birth-time residence and delivery-facility state are not observed here.',
            'The non-West Bengal reference can still include sparse or one-sector cells; support audit findings remain relevant.',
            'Contrast intervals are exploratory and conditional on declared state exclusion, fixed model specification and observed roster. No multiplicity adjustment.',
        ],
        'remaining_work':SOURCE_REVIEW['remaining_for_comment_17'],
        'elapsed_seconds':round(time.monotonic()-start,1),
    }
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output/'west_bengal_sensitivity_intervals.csv',index=False)
    base.save_json(output/'west_bengal_sensitivity_summary.json',summary)
    base.save_json(output/'historical_state_context_source_review.json',SOURCE_REVIEW)
    print(f'Aggregate outputs saved: {output}',flush=True)
    print('Share ONLY west_bengal_sensitivity_summary.json, west_bengal_sensitivity_intervals.csv and historical_state_context_source_review.json. Keep raw data local.',flush=True)


if __name__ == '__main__':
    main()
