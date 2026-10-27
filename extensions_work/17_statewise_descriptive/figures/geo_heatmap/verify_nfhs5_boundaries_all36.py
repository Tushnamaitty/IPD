"""Compare original NFHS-5 map polygons against downloaded NWIC/SOI boundary file.
Run locally from anywhere; no datasets are uploaded. Outputs small CSV/JSON summaries.
"""
from pathlib import Path
import json
import re
import unicodedata
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.validation import make_valid

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / 'india_states.geojson'
OFFICIAL = Path.home() / 'Downloads' / 'NWIC_Boundaries' / 'state_NWIC.GeoJSON'
OUT = ROOT / 'boundary_verification'
OUT.mkdir(exist_ok=True)


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).lower()
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.replace('&', ' and ')
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', s)).strip()

ALIASES = {
    'andaman and nicobar island': 'andaman and nicobar islands',
    'arunanchal pradesh': 'arunachal pradesh',
    'dadara and nagar havelli': 'dadra and nagar haveli and daman and diu',
    'dadra and nagar haveli': 'dadra and nagar haveli and daman and diu',
    'daman and diu': 'dadra and nagar haveli and daman and diu',
    'dadra and nagar haveli and daman and diu': 'dadra and nagar haveli and daman and diu',
    'dadra and nagar havelli and daman and diu': 'dadra and nagar haveli and daman and diu',
    'nct of delhi': 'delhi', 'delhi nct': 'delhi',
    'jammu and kashmir': 'jammu and kashmir',
    'orissa': 'odisha', 'pondicherry': 'puducherry',
    'uttaranchal': 'uttarakhand',
}


def canonical(value):
    key = norm(value)
    return ALIASES.get(key, key)


def fix_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.is_empty:
        return None
    return geom


def load(path, name_col):
    if not path.is_file():
        raise FileNotFoundError(f'Missing: {path}')
    gdf = gpd.read_file(path)
    if name_col not in gdf.columns:
        raise ValueError(f'{path.name}: expected {name_col}; got {list(gdf.columns)}')
    if gdf.crs is None:
        raise ValueError(f'{path.name}: CRS is absent; cannot compare coordinates safely')
    info = {'file': str(path), 'features': len(gdf), 'crs': str(gdf.crs),
            'empty_or_null_before': int(gdf.geometry.isna().sum() + (gdf.geometry.is_empty & ~gdf.geometry.isna()).sum()),
            'invalid_nonempty_before': int(((~gdf.geometry.is_valid) & gdf.geometry.notna() & (~gdf.geometry.is_empty)).sum())}
    gdf = gdf[[name_col, 'geometry']].copy()
    gdf['key'] = gdf[name_col].map(canonical)
    gdf['geometry'] = gdf.geometry.map(fix_geometry)
    gdf = gdf[gdf.geometry.notna()].copy()
    # Use an equal-area CRS so area comparisons have meaningful units.
    gdf = gdf.set_geometry('geometry').to_crs('EPSG:6933')
    grouped = gdf[['key', 'geometry']].dissolve(by='key')
    info['unique_names_after_aliasing'] = len(grouped)
    return grouped, info

original, original_info = load(ORIGINAL, 'ST_NM')
official, official_info = load(OFFICIAL, 'state_name')
original_keys = set(original.index)
official_keys = set(official.index)
only_original = sorted(original_keys - official_keys)
only_official = sorted(official_keys - original_keys)

records = []
for state in sorted(original_keys & official_keys):
    a = original.loc[state, 'geometry']
    b = official.loc[state, 'geometry']
    inter = a.intersection(b).area
    union = a.union(b).area
    old_area = a.area
    gov_area = b.area
    records.append({
        'state': state,
        'original_area_km2': old_area / 1e6,
        'reference_area_km2': gov_area / 1e6,
        'intersection_over_union_pct': 100 * inter / union if union else None,
        'reference_area_missing_from_original_pct': 100 * (gov_area - inter) / gov_area if gov_area else None,
        'original_area_outside_reference_pct': 100 * (old_area - inter) / old_area if old_area else None,
        'area_difference_vs_reference_pct': 100 * (old_area - gov_area) / gov_area if gov_area else None,
    })
result = pd.DataFrame(records).sort_values('intersection_over_union_pct')
result.to_csv(OUT / 'state_boundary_comparison.csv', index=False, float_format='%.5f')
summary = {
    'original': original_info,
    'official_reference': official_info,
    'original_only_names': only_original,
    'reference_only_names': only_official,
    'matched_states': len(result),
    'comparison_crs': 'EPSG:6933 (equal-area)',
    'worst_10': result.head(10).round(4).to_dict(orient='records'),
    'notes': [
        'This checks geometric similarity, not legal validity of disputed borders or reference vintage.',
        'A 2019-era input and a 2025 government dataset may legitimately differ in administrative vintage.',
        'Difference metrics include coastline/generalization effects; inspect notable mismatches before publication.',
        'No tolerance threshold automatically certifies boundaries as publication-ready.',
    ],
}
(OUT / 'boundary_verification_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('Original features:', original_info['features'], 'Reference features:', official_info['features'])
print('Matched names:', len(result))
if len(result) != 36 or only_original or only_official:
    raise ValueError('Expected 36 matched regions with no unmatched names; review report.')
print('Original unmatched:', only_original)
print('Reference unmatched:', only_official)
print('\nLowest geographic agreement:')
print(result[['state', 'intersection_over_union_pct', 'reference_area_missing_from_original_pct', 'original_area_outside_reference_pct']].head(10).to_string(index=False, float_format=lambda n: f'{n:.2f}'))
print('\nSaved:', OUT / 'state_boundary_comparison.csv')
print('Saved:', OUT / 'boundary_verification_summary.json')