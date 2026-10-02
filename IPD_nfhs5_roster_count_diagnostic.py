"""Aggregate-only diagnostic: what do the NFHS-5 household-recode design identifiers count?

Run locally:  python IPD_nfhs5_roster_count_diagnostic.py --repo-root .
Reads only design columns (hv001, hv021, hv022, hv023, hv024, hv025, hv015) of the local household
recode. Writes ONLY aggregate counts (JSON + one state-level CSV). It writes no household records
and no list of identifiers, does not fill gaps, and does not change any existing output.

Purpose: the observed roster (distinct raw hv024/hv022/hv021 triples) is 30,170, versus the
published 30,198 PSUs with completed fieldwork (30,456 selected). This script reports how that
count moves under alternative definitions (PSU vs cluster number), whether hv001 and hv021 describe
the same units, and how many identifier gaps and very small clusters exist. It cannot say WHY
published and observed counts differ; it narrows which explanations remain possible.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

OFFICIAL = {'psus_selected': 30456, 'psus_fieldwork_completed': 30198, 'households_interviewed': 636699,
            'urban_selected': 7910, 'rural_selected': 22546}
DESIGN = ['hv024', 'hv022', 'hv021']          # existing roster key (state, stratum, PSU)
REQUIRED = ['hv021', 'hv022', 'hv024']
OPTIONAL = ['hv001', 'hv015', 'hv023', 'hv025']
BINS = [0, 2, 9, 14, 19, 21, 22, 10 ** 9]
BIN_LABELS = ['1-2', '3-9', '10-14', '15-19', '20-21', '22', '23+']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def distinct(df, cols):
    return int(len(df[cols].drop_duplicates()))


def find_household_file(root, explicit):
    if explicit:
        return Path(explicit)
    exact = root / 'data/raw/IAHR7EFL.DTA'
    if exact.exists():
        return exact
    found = sorted({p for p in (root / 'data').rglob('*') if p.is_file() and p.suffix.lower() == '.dta'
                    and p.name.upper().startswith('IAHR7') and not p.name.upper().startswith('IAHR74')})
    if len(found) != 1:
        raise FileNotFoundError('Expected exactly one NFHS-5 IAHR7*.DTA; found %d. Use --household.' % len(found))
    return found[0]


def read_design(path):
    with pd.io.stata.StataReader(path, convert_categoricals=False) as reader:
        labels = reader.variable_labels()
    missing = sorted(set(REQUIRED) - set(labels))
    if missing:
        raise ValueError('Missing required variables: ' + str(missing))
    columns = REQUIRED + [c for c in OPTIONAL if c in labels]
    df = pd.read_stata(path, columns=columns, convert_categoricals=False)
    for c in columns:
        if c == 'hv015':
            continue
        a = pd.to_numeric(df[c], errors='coerce').to_numpy(float)
        if not (np.isfinite(a) & (a % 1 == 0)).all():
            raise ValueError('Non-integer or missing design identifier in ' + c + '; nothing was dropped or guessed.')
        df[c] = a.astype(np.int64)
    return df, {c: labels[c] for c in columns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--household', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output or root / 'extensions_work/03_nfhs4_nfhs5_temporal/outputs/nfhs5_roster_count_diagnostic'
    if out.exists():
        raise FileExistsError('Existing output preserved; choose a new --output.')
    path = find_household_file(root, args.household)
    df, labels = read_design(path)
    has = set(df.columns)
    report = {'status': 'aggregate_diagnostic_completed_requires_review', 'household_file': path.name,
              'household_file_sha256': sha256(path), 'variable_labels': labels, 'household_rows': int(len(df)),
              'official_reference_values': OFFICIAL,
              'official_sources': 'IIPS & ICF NFHS-5 India Report Vol. I section 1.2 (30,456 selected; 30,198 completed); Vol. II sampling error appendix (30,456 clusters; urban 7,910, rural 22,546)'}
    if 'hv015' in has:
        report['hv015_counts'] = {str(k): int(v) for k, v in df.hv015.value_counts(dropna=False).items()}
    report['household_rows_match_official_636699'] = bool(len(df) == OFFICIAL['households_interviewed'])
    keys = {'state_stratum_psu_hv024_hv022_hv021_EXISTING_ROSTER_KEY': DESIGN,
            'psu_only_hv021': ['hv021'], 'stratum_psu_hv022_hv021': ['hv022', 'hv021'],
            'state_psu_hv024_hv021': ['hv024', 'hv021']}
    if 'hv001' in has:
        keys.update({'cluster_only_hv001': ['hv001'], 'state_cluster_hv024_hv001': ['hv024', 'hv001'],
                     'state_stratum_cluster_hv024_hv022_hv001': ['hv024', 'hv022', 'hv001'],
                     'state_stratum_cluster_psu_pairs': ['hv024', 'hv022', 'hv001', 'hv021']})
    if 'hv023' in has:
        keys['state_hv023_psu_hv024_hv023_hv021'] = ['hv024', 'hv023', 'hv021']
    counts = {name: distinct(df, cols) for name, cols in keys.items()}
    report['distinct_counts'] = counts
    report['difference_from_published'] = {name: {'minus_30198': n - OFFICIAL['psus_fieldwork_completed'],
                                                  'minus_30456': n - OFFICIAL['psus_selected']}
                                           for name, n in counts.items()}
    report['matches_existing_30170'] = bool(counts['state_stratum_psu_hv024_hv022_hv021_EXISTING_ROSTER_KEY'] == 30170)
    report['stratum_codes_shared_across_states'] = int((df[['hv024', 'hv022']].drop_duplicates()
                                                        .groupby('hv022').hv024.nunique() > 1).sum())
    if 'hv001' in has:
        g = df[['hv024', 'hv022', 'hv001', 'hv021']].drop_duplicates()
        ids = np.sort(df.hv001.unique())
        report['hv001_hv021_relationship_within_state_stratum'] = {
            'hv001_values_with_more_than_one_hv021': int((g.groupby(['hv024', 'hv022', 'hv001']).hv021.nunique() > 1).sum()),
            'hv021_values_with_more_than_one_hv001': int((g.groupby(['hv024', 'hv022', 'hv021']).hv001.nunique() > 1).sum()),
            'hv001_values_used_in_more_than_one_state': int((df[['hv024', 'hv001']].drop_duplicates()
                                                             .groupby('hv001').hv024.nunique() > 1).sum())}
        report['hv001_numbering'] = {'distinct': int(len(ids)), 'minimum': int(ids.min()), 'maximum': int(ids.max()),
                                     'unused_integers_between_min_and_max': int(ids.max() - ids.min() + 1 - len(ids)),
                                     'note': 'Counts only; gap identifiers are deliberately not listed and nothing is imputed.'}
    unit = DESIGN
    sizes = df.groupby(unit).size()
    report['households_per_existing_roster_unit'] = {
        'minimum': int(sizes.min()), 'median': float(sizes.median()), 'maximum': int(sizes.max()),
        'units_with_at_most_2_households': int((sizes <= 2).sum()),
        'histogram': {k: int(v) for k, v in pd.cut(sizes, BINS, labels=BIN_LABELS).value_counts().sort_index().items()}}
    if 'hv025' in has:
        r = df[DESIGN + ['hv025']].drop_duplicates()
        report['residence_of_existing_roster_units'] = {
            'units_with_more_than_one_residence_code': int((r.groupby(DESIGN).hv025.nunique() > 1).sum()),
            'by_hv025': {str(k): int(v) for k, v in r.drop_duplicates(DESIGN).hv025.value_counts().sort_index().items()},
            'note': 'Published 7,910 urban / 22,546 rural are for the 30,456 SELECTED PSUs, not the 30,198 completed.'}
    state_rows = []
    for state, q in df.groupby('hv024'):
        row = {'hv024': int(state), 'households': int(len(q)),
               'state_stratum_psu_units': distinct(q, ['hv022', 'hv021']), 'hv021_values': int(q.hv021.nunique())}
        if 'hv001' in has:
            u = np.sort(q.hv001.unique())
            row.update(hv001_distinct=int(len(u)), hv001_min=int(u.min()), hv001_max=int(u.max()),
                       hv001_unused_between_min_max=int(u.max() - u.min() + 1 - len(u)))
        state_rows.append(row)
    out.mkdir(parents=True)
    (out / 'roster_count_diagnostic.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(state_rows).to_csv(out / 'roster_count_by_state.csv', index=False)
    print(json.dumps({'distinct_counts': counts, 'matches_existing_30170': report['matches_existing_30170']}, indent=2))
    print('Aggregate outputs saved: ' + str(out))
    print('Share roster_count_diagnostic.json and roster_count_by_state.csv only.')


if __name__ == '__main__':
    main()
