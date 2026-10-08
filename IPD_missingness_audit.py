"""Audit missingness without fitting models or changing existing outputs.
Run from IPD: python IPD_missingness_audit.py --repo-root .
Requires the existing IPD_narrow_facility_analysis.py, pandas and numpy.
This script has not been executed or tested by the assistant.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=Path('.'))
    p.add_argument('--nfhs4', type=Path)
    p.add_argument('--nfhs5', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.repo_root.resolve()
    source = root / 'IPD_narrow_facility_analysis.py'
    if not source.is_file():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('ipd_narrow_audit_source', source)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)  # Imports definitions only; does not call main/self_test.
    out = args.output or root / 'extensions_work/16_missing_data/outputs/missingness_audit'
    if out.exists():
        raise FileExistsError(f'Existing outputs preserved: {out}; choose --output with a new path.')
    inputs = {
        'NFHS-4': args.nfhs4 or base.find_raw(root, 'IABR74'),
        'NFHS-5': args.nfhs5 or base.find_raw(root, 'IABR7', exclude_prefix='IABR74'),
    }
    variables, cohorts, patterns, codes, audits = [], [], [], [], {}
    expected = {'NFHS-4': (114263, 52851), 'NFHS-5': (115197, 51854)}
    for wave, path in inputs.items():
        print(f'{wave}: reading raw birth recode; no model fitting.', flush=True)
        raw = pd.read_stata(path, columns=base.RAW, convert_categoricals=False)
        frame, audit = base.prepare_births(raw, wave)
        first = frame.loc[frame.delivery_order.eq(1)].copy()
        if (len(frame), len(first)) != expected[wave]:
            raise ValueError(f'{wave}: counts differ from the reviewed report; inspect before proceeding.')
        audit['birth_sha256'] = base.sha256(path)
        audits[wave] = audit
        for analysis, sub in [('all_narrow', frame), ('first_delivery', first)]:
            model_vars = (['education_years'] if analysis == 'first_delivery' else base.NUM) + base.CAT
            for sector, q in [('all', sub), ('public', sub.loc[sub.sector.eq(0)]), ('private', sub.loc[sub.sector.eq(1)])]:
                if q.empty:
                    raise ValueError(f'Empty cohort: {wave}/{analysis}/{sector}')
                weights = q.weight.to_numpy(float)
                any_missing = q[model_vars].isna().any(axis=1).to_numpy()
                cohorts.append({'wave': wave, 'analysis': analysis, 'sector': sector,
                                'n': len(q), 'any_covariate_missing_n': int(any_missing.sum()),
                                'any_covariate_missing_pct': float(100 * any_missing.mean()),
                                'any_covariate_missing_weight_pct': float(100 * weights[any_missing].sum() / weights.sum()),
                                'complete_covariates_n': int((~any_missing).sum())})
                for col in model_vars:
                    missing = q[col].isna().to_numpy()
                    variables.append({'wave': wave, 'analysis': analysis, 'sector': sector,
                                      'variable': col, 'n': len(q), 'missing_n': int(missing.sum()),
                                      'missing_pct': float(100 * missing.mean()),
                                      'missing_weight_pct': float(100 * weights[missing].sum() / weights.sum())})
                    for value, group in q.groupby(col, dropna=False):
                        codes.append({'wave': wave, 'analysis': analysis, 'sector': sector,
                                      'variable': col, 'code': 'MISSING' if pd.isna(value) else str(value),
                                      'n': len(group), 'weight_pct': float(100 * group.weight.sum() / q.weight.sum())})
                labels = q[model_vars].isna().apply(lambda row: '|'.join(row.index[row]) or 'COMPLETE', axis=1)
                tab = pd.DataFrame({'pattern': labels, 'weight': q.weight})
                for name, group in tab.groupby('pattern'):
                    patterns.append({'wave': wave, 'analysis': analysis, 'sector': sector,
                                     'pattern': name, 'n': len(group),
                                     'weight_pct': float(100 * group.weight.sum() / q.weight.sum())})
        # Record reviewed special-code handling separately from date exclusions.
        audit['special_code_rules'] = {'v133_97': 'missing schooling', 's116_8': 'missing social group'}
    affected = [r for r in cohorts if r['sector'] == 'all' and r['any_covariate_missing_n'] > 0]
    summary = {
        'status': 'audit_complete_requires_review',
        'imputation_decision': 'review_missingness_and_code_semantics_before_designing_imputation' if affected else 'no_covariate_missingness_under_current_recoding_review_code_semantics',
        'affected_cohorts': affected,
        'existing_handling': {'numeric': 'zero fill plus missing indicator', 'categorical': 'explicit missing category'},
        'cohort_audits': audits,
        'source_script_sha256': base.sha256(source),
        'limitations': [
            'Audit uses current reviewed recoding; code frequencies require review for additional unknown/refusal codes.',
            'Outcome, sector, survey design and cohort-defining variables are not imputed.',
            'Date/recall exclusions are separate from covariate missingness.',
            'Historical-context linkage gaps are separate coverage limitations; no state spending or facility values are imputed here.',
            'No missing-at-random assumption is established by this audit.',
            'No models, imputation, tests, or row-level exports are performed by this script.',
        ],
        'next_step': 'Review aggregate tables and raw code meanings; if warranted, design multiple imputation preserving survey structure, then compare adjusted associations with the current missing-indicator/category analysis.',
    }
    out.mkdir(parents=True)
    for name, rows in [('missingness_by_variable.csv', variables), ('missingness_by_cohort.csv', cohorts),
                       ('missingness_patterns.csv', patterns), ('covariate_code_frequencies.csv', codes)]:
        pd.DataFrame(rows).to_csv(out / name, index=False)
    (out / 'missingness_audit_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Aggregate audit saved to {out}', flush=True)
    print('Share the five CSV/JSON audit files. Keep raw NFHS data local.', flush=True)


if __name__ == '__main__':
    main()
