"""Read-only local scope inventory for IPD Comments 13/15/17.

No models, raw-data reads, row exports, source edits or network calls.
Exports aggregate parquet checks and source-file metadata; text-search hints
are not proof of a notebook's exposure definition or variance correctness.
Run: python IPD_narrow_scope_audit.py --repo-root .
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PARQUETS = (
    'data/processed/df_model_v2.parquet',
    'data/processed/df_model_nfhs4.parquet',
    'data/processed/df_model_nfhs5.parquet',
)
FIELDS = (
    'delivery_place_code', 'm15', 'facility_type', 'csection',
    'sample_weight_normalized', 'sample_weight',
    'state', 'v024', 'sample_stratum_v022', 'v022',
    'primary_sampling_unit', 'cluster_number', 'v021',
)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def first_column(columns, candidates):
    return next((x for x in candidates if x in columns), None)


def cohort_summary(df, code_col):
    code = pd.to_numeric(df[code_col], errors='coerce')
    out = {'n_code_21': int(code.eq(21).sum()),
           'n_code_31': int(code.eq(31).sum()),
           'n_other_or_missing_codes': int((~code.isin([21, 31])).sum())}
    narrow = df.loc[code.isin([21, 31])].copy()
    out['n_narrow_before_outcome_or_weight_filter'] = len(narrow)
    if 'csection' not in df:
        out['status'] = 'raw_facility_codes_present_outcome_missing'
        return out
    y = pd.to_numeric(narrow.csection, errors='coerce')
    w_col = first_column(df.columns, ('sample_weight_normalized', 'sample_weight'))
    valid_y = y.isin([0, 1])
    out['n_invalid_or_missing_outcome_in_narrow'] = int((~valid_y).sum())
    if not w_col:
        out['status'] = 'raw_facility_codes_present_weight_missing'
        return out
    w = pd.to_numeric(narrow[w_col], errors='coerce')
    valid_w = np.isfinite(w) & w.gt(0)
    out['weight_column'] = w_col
    out['n_invalid_or_missing_weight_in_narrow'] = int((~valid_w).sum())
    narrow = narrow.loc[valid_y & valid_w]
    y = y.loc[narrow.index]
    w = w.loc[narrow.index]
    code = code.loc[narrow.index]
    out['n_narrow_valid_outcome_and_weight'] = len(narrow)
    risks = {}
    for a in (21, 31):
        mask = code.eq(a)
        risks[str(a)] = {'n': int(mask.sum()), 'observed_cesarean_pct':
                         float(100 * np.average(y[mask], weights=w[mask])) if mask.any() else None}
    out['facility_observed_summary'] = risks
    out['observed_private_minus_public_gap_pp'] = (
        risks['31']['observed_cesarean_pct'] - risks['21']['observed_cesarean_pct']
        if all(risks[str(a)]['n'] for a in (21, 31)) else None)
    if 'facility_type' in narrow:
        labels = narrow.facility_type.astype(str).str.lower().str.strip()
        out['n_raw_code_sector_label_mismatches'] = int(
            ((code.eq(21) & labels.ne('public')) | (code.eq(31) & labels.ne('private'))).sum())
    state = first_column(narrow.columns, ('v024', 'state'))
    strata = first_column(narrow.columns, ('v022', 'sample_stratum_v022'))
    psu = first_column(narrow.columns, ('v021', 'primary_sampling_unit', 'cluster_number'))
    if all((state, strata, psu)):
        design = narrow[[state, strata, psu]]
        out['design_columns'] = {'state': state, 'stratum': strata, 'psu': psu}
        out['n_rows_missing_design_identifier'] = int(design.isna().any(axis=1).sum())
        pairs = design.dropna().drop_duplicates()
        state_strata = pairs[[state, strata]].drop_duplicates()
        out['n_active_states'] = int(pairs[state].nunique())
        out['n_active_state_strata'] = len(state_strata)
        out['n_active_state_stratum_psus'] = len(pairs)
        out['n_stratum_codes_shared_across_states'] = int((state_strata.groupby(strata)[state].nunique() > 1).sum())
        out['n_design_keys_lost_by_omitting_state'] = len(pairs) - len(pairs[[strata, psu]].drop_duplicates())
        out['n_analytic_singleton_state_strata'] = int((pairs.groupby([state, strata]).size() == 1).sum())
        out['design_note'] = 'Analytic-file counts only; no household frame or variance certification.'
    else:
        out['design_columns_missing'] = True
    out['status'] = 'aggregate_scope_check_completed_not_model_validation'
    return out


def inspect_parquet(path):
    if not path.is_file():
        return {'status': 'file_not_present'}
    try:
        meta = pq.ParquetFile(path)
        cols = meta.schema_arrow.names
        out = {'status': 'metadata_read', 'n_file_rows': meta.metadata.num_rows,
               'sha256': sha256(path), 'available_audit_columns': [x for x in FIELDS if x in cols]}
        code_col = first_column(cols, ('delivery_place_code', 'm15'))
        if not code_col:
            out.update(status='raw_facility_code_not_retained',
                       action='Reconstruct narrow cohort from verified raw m15; public/private labels cannot identify codes 21 and 31.')
            return out
        df = pd.read_parquet(path, columns=[x for x in FIELDS if x in cols])
        if {'delivery_place_code', 'm15'}.issubset(df.columns):
            a = pd.to_numeric(df.delivery_place_code, errors='coerce')
            b = pd.to_numeric(df.m15, errors='coerce')
            out['n_conflicting_raw_facility_code_columns'] = int((~(a.eq(b) | (a.isna() & b.isna()))).sum())
            if out['n_conflicting_raw_facility_code_columns']:
                out['status'] = 'conflicting_raw_facility_code_columns_requires_review'
                return out
        out['raw_facility_code_column'] = code_col
        out['cohort'] = cohort_summary(df, code_col)
        return out
    except Exception as e:
        # Report the exception class only; no data-bearing values/messages.
        return {'status': 'inspection_failed', 'error_type': type(e).__name__}


def code_inventory(root):
    files = set()
    for folder in ('notebooks', 'extensions_work', 'src', 'comment10_followup'):
        base = root / folder
        if base.is_dir():
            for pattern in ('*.py', '*.ipynb'):
                files.update(base.rglob(pattern))
    files.update(root.glob('*.py'))
    rows = []
    for path in sorted(files):
        if path.is_symlink() or any(x in {'outputs', '__pycache__', '.venv', 'venv', 'archive', 'archives', 'node_modules'} for x in path.relative_to(root).parts):
            continue
        row = {'path': path.relative_to(root).as_posix(), 'sha256': sha256(path), 'status': 'read'}
        try:
            content = path.read_text(encoding='utf-8-sig')
            if path.suffix == '.ipynb':
                cells = json.loads(content)['cells']
                content = '\n'.join(''.join(x.get('source', [])) for x in cells if x.get('cell_type') == 'code')
            row.update(
                mentions_raw_facility_code=bool(re.search(r'\b(m15|delivery_place_code)\b', content)),
                mentions_both_narrow_numbers=bool(re.search(r'\b21\b', content) and re.search(r'\b31\b', content)),
                uses_shared_broad_loader='load_public_private_analytic' in content,
                mentions_aipw_tmle=bool(re.search(r'\b(aipw|tmle|run_cross_fitted_aipw|aipw_point_estimate)\b', content, re.I)),
                mentions_state=bool(re.search(r'\b(state|v024|hv024|design_state)\b', content)),
                mentions_resampling=bool(re.search(r'bootstrap|resample|joint_covariance|Taylor', content, re.I)),
            )
        except Exception as e:
            row.update(status='read_failed', error_type=type(e).__name__)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path('extensions_work/12_narrow_scope/outputs/scope_audit'))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if not (root / 'data' / 'processed').is_dir():
        raise FileNotFoundError('Expected local repo containing data/processed.')
    out = args.output if args.output.is_absolute() else root / args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Audit output folder is nonempty; choose another --output.')
    start = time.monotonic()
    print('Inspecting local code and processed-cohort aggregates; no model fitting...', flush=True)
    inventory = code_inventory(root)
    report = {'status': 'scope_audit_completed_manual_source_review_required',
              'script_sha256': sha256(Path(__file__)),
              'model_refits_run': False, 'source_files_changed': False,
              'raw_data_read': False, 'record_level_data_exported': False,
              'inventory_note': 'Source flags are text-search hints, not verified cohort definitions or design-key correctness.',
              'cohorts': {p: inspect_parquet(root / p) for p in PARQUETS},
              'n_code_files_inventoried': len(inventory),
              'limitations': ['Processed cohorts may omit raw births or lack exact raw facility codes.',
                              'No confidence intervals, models, full survey frame, overlap or codebook validation in this audit.',
                              'Original predictive study and adjusted-association study must be assigned scopes explicitly.']}
    report['elapsed_seconds'] = round(time.monotonic() - start, 1)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'code_scope_inventory.csv').open('w', newline='', encoding='utf-8') as f:
        fields = list(dict.fromkeys(k for row in inventory for k in row)) or ['path', 'status']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)
    (out / 'narrow_scope_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f'Aggregate outputs saved: {out}')
    print('Share ONLY narrow_scope_audit.json and code_scope_inventory.csv for this review.')


if __name__ == '__main__':
    main()
