"""NFHS-5 descriptive state/UT choropleths (government-boundary candidate; independent certification pending).

Run from the IPD repository; reads only existing aggregate CSV and local GeoJSON.
No NFHS microdata or model fitting involved.
"""
from pathlib import Path
import unicodedata

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
CSV = BASE / 'outputs' / 'statewise_descriptive' / 'statewise_primary_results.csv'
GEOJSON = Path.home() / 'Downloads' / 'NWIC_Boundaries' / 'state_NWIC.GeoJSON'
OUTPUT_PNG = HERE / 'nfhs5_cesarean_geo_heatmap_government_boundary.png'
OUTPUT_PDF = HERE / 'nfhs5_cesarean_geo_heatmap_government_boundary.pdf'
AUDIT_CSV = HERE / 'nfhs5_heatmap_government_boundary_audit.csv'

ALIASES = {
    'nct of delhi': 'delhi',
    'delhi nct': 'delhi',
    'orissa': 'odisha',
    'pondicherry': 'puducherry',
    'uttaranchal': 'uttarakhand',
    'andaman and nicobar island': 'andaman and nicobar islands',
    'dadara and nagar havelli': 'dadra and nagar haveli and daman and diu',
    'dadra and nagar havelli and daman and diu': 'dadra and nagar haveli and daman and diu',
    'arunanchal pradesh': 'arunachal pradesh',
    'daman and diu': 'dadra and nagar haveli and daman and diu',
}

def clean_name(value):
    value = unicodedata.normalize('NFKD', str(value))
    value = ''.join(ch for ch in value if not unicodedata.combining(ch))
    value = ' '.join(value.lower().replace('&', 'and').replace('.', '').split())
    return ALIASES.get(value, value)


def main():
    for path in (CSV, GEOJSON):
        if not path.is_file():
            raise FileNotFoundError(path)

    df = pd.read_csv(CSV)
    df = df.loc[
        (df['wave'] == 'NFHS-5') &
        (df['analysis'] == 'all_narrow') &
        df['metric'].isin(['observed_public_pct', 'observed_private_pct'])
    ].copy()
    df['key'] = df['state_name'].map(clean_name)
    if df.duplicated(['key', 'metric']).any():
        raise ValueError('Duplicate state/sector results')
    if len(df) != 72 or df['key'].nunique() != 36:
        raise ValueError(f'Expected 72 result rows across 36 geographies; got {len(df)} rows, {df["key"].nunique()} geographies')

    geo = gpd.read_file(GEOJSON)
    names = ['state_name', 'ST_NM', 'name', 'NAME', 'state', 'STATE', 'st_nm']
    name_col = next((c for c in names if c in geo.columns), None)
    if name_col is None:
        raise ValueError(f'Cannot identify boundary state column: {list(geo.columns)}')
    if geo.crs is None:
        raise ValueError('Government reference has no CRS')
    if str(geo.crs).upper() != 'EPSG:7755':
        print('WARNING: Government file CRS differs from inspected EPSG:7755:', geo.crs)
    from shapely.validation import make_valid
    # Repair invalid source shapes and reject empty geometries.
    if geo.geometry.isna().any() or geo.geometry.is_empty.any():
        raise ValueError('Government reference has null/empty shapes')
    invalid_n = int((~geo.geometry.is_valid).sum())
    print('Government invalid source geometries:', invalid_n)
    if invalid_n:
        geo['geometry'] = geo.geometry.map(lambda x: make_valid(x) if not x.is_valid else x)
    if not geo.geometry.is_valid.all():
        raise ValueError('Government geometries still invalid after repair')
    geo['key'] = geo[name_col].map(clean_name)
    # Verify expected labels; dissolve also handles multipart constituent regions.
    geo = geo[['key', 'geometry']].dissolve(by='key', as_index=False)
    if len(geo) != 36:
        raise ValueError(f'Government reference should have 36 distinct regions, found {len(geo)}')
    missing = sorted(set(df['key']) - set(geo['key']))
    extra = sorted(set(geo['key']) - set(df['key']))
    print('Results without matching boundaries:', missing)
    print('Boundaries without matching results:', extra)
    if missing or extra:
        raise ValueError('State/UT name or geography mismatch. Do not publish partial map.')

    if geo.crs is None:
        raise ValueError('Boundary CRS missing; verify boundary source before plotting.')
    geo = geo.to_crs('EPSG:4326')
    if geo.geometry.isna().any() or geo.geometry.is_empty.any():
        raise ValueError('Missing/empty boundaries')

    # Wide layout deliberately reserves dedicated, nonoverlapping footer axes.
    fig = plt.figure(figsize=(15, 8.8), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, height_ratios=[14, 0.65, 1.7],
                          left=0.035, right=0.965, top=0.87, bottom=0.055,
                          hspace=0.15, wspace=0.09)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[1, :])
    note_ax = fig.add_subplot(gs[2, :])
    note_ax.axis('off')
    panels = [
        ('observed_public_pct', 'A. Government / municipal hospitals', 'n_public'),
        ('observed_private_pct', 'B. Private hospitals / maternity homes / clinics', 'n_private'),
    ]
    norm = mpl.colors.Normalize(0, 100)
    cmap = mpl.colormaps['YlOrRd']
    audits = []
    for ax, (metric, heading, sector_count) in zip(axes, panels):
        sub = df.loc[df['metric'] == metric,
                     ['key', 'state_name', 'estimate', 'status', 'flags',
                      'minimum_weight_only_effective_n', 'n_public', 'n_private']].copy()
        merged = geo.merge(sub, on='key', how='left', validate='one_to_one')
        if merged['estimate'].isna().any() or not merged['estimate'].between(0, 100).all():
            raise ValueError(f'Invalid results for {metric}')
        merged['sparse'] = (
            (merged[sector_count] < 30) |
            (merged['minimum_weight_only_effective_n'] < 30) |
            merged['flags'].fillna('').str.contains('sparse_n_below_30|weight_only_effective_n_below_30')
        )
        # One recorded private birth in Ladakh yields an unstable 100% estimate.
        # Preserve its numerical value in the audit CSV, but do not encode it
        # with the same colour scale as larger-sample estimates.
        merged['suppressed_visual'] = (
            (metric == 'observed_private_pct') & (merged['n_private'] < 2)
        )
        merged['plot_estimate'] = merged['estimate'].mask(merged['suppressed_visual'])
        merged.plot(column='plot_estimate', ax=ax, cmap=cmap, norm=norm,
                    missing_kwds={'color': '#d9d9d9'},
                    edgecolor='#ffffff', linewidth=0.48, zorder=2)
        # Hatched overlay warns about sector samples <30 or low effective n.
        if merged['sparse'].any():
            merged.loc[merged['sparse'], 'geometry'].plot(
                ax=ax, facecolor='none', edgecolor='#323232',
                linewidth=0.55, hatch='///', zorder=3)
        ax.set_title(heading, fontsize=12.1, fontweight='bold', pad=9)
        ax.set_axis_off()
        ax.set_aspect('equal')
        print(f'{heading}: sparse/low-effective-n geographies =',
              merged.loc[merged['sparse'], 'state_name'].tolist())
        audits.append(merged.drop(columns=['geometry']).assign(metric=metric))

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal',
                      ticks=[0, 20, 40, 60, 80, 100])
    cb.ax.tick_params(labelsize=9)
    cb.set_label('Cesarean deliveries among births in selected facilities (%)',
                 fontsize=10, labelpad=2)
    note_ax.text(0.5, 0.74,
                 'Survey-weighted NFHS-5 descriptive estimates | All eligible narrow-facility births | m15 = 21 versus 31',
                 ha='center', va='center', fontsize=9.2, transform=note_ax.transAxes)
    note_ax.text(0.5, 0.42,
                 'Hatching: sector n or effective n < 30. Grey: Ladakh private estimate based on one birth (not colour-scaled).',
                 ha='center', va='center', fontsize=8.8, transform=note_ax.transAxes)
    note_ax.text(0.5, 0.10,
                 'Not adjusted or causal. Boundary layer: NWIC dataset (SOI attributed); exact vintage/legal status not certified.',
                 ha='center', va='center', fontsize=8.3, color='#555555', transform=note_ax.transAxes)
    fig.suptitle('State/UT-wise Cesarean proportions | NFHS-5',
                 fontsize=18, fontweight='bold', y=0.955)

    fig.savefig(OUTPUT_PNG, dpi=300, facecolor='white')
    fig.savefig(OUTPUT_PDF, facecolor='white')
    plt.close(fig)
    pd.concat(audits, ignore_index=True).to_csv(AUDIT_CSV, index=False)
    print('Saved:', OUTPUT_PNG)
    print('Saved:', OUTPUT_PDF)
    print('Saved:', AUDIT_CSV)
    print('IMPORTANT: Source attributed to SOI by NWIC; geometry vintage/legal boundary accuracy not certified.')

if __name__ == '__main__':
    main()