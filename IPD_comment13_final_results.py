"""Package reviewed aggregate results for Comment 13. No models or raw data.

UNTESTED: written for the user to run locally in VS Code.
This creates a new, separate results package. It does not certify inference,
edit the manuscript, or rewrite the legacy final-table notebook.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone

# Exact summaries reviewed in the user's Comment 13 audit. A different
# summary requires review rather than silently selecting a newer run.
SOURCES = [
    ('national', 'extensions_work/12_narrow_scope/outputs/national_associations',
     'national_association_summary.json',
     '849eb897f4e5dff258e82c9f0a6dfb6021c1b911bd6fde98a759d974cd54e2cf',
     ['national_association_intervals.csv', 'state_sector_support.csv'],
     'Own wave and birth subgroup reference; observed and adjusted associations'),
    ('support', 'extensions_work/12_narrow_scope/outputs/narrow_support_sensitivity',
     'support_sensitivity_summary.json',
     '0368589933c8961bdf89c13786a738c1b10ff9c40465e5c21c305428cdd7c51b',
     ['support_sensitivity_intervals.csv'],
     'Support sensitivity; distinguish reference restriction from model refitting'),
    ('comment15', 'extensions_work/12_narrow_scope/outputs/comment15_shared_reference',
     'comment15_summary.json',
     'd12a65be45bee2200f3bc0574a36de6478e45f8c20ad00ab04d5af7f97dffdf0',
     ['comment15_intervals.csv', 'comment15_state_mapping.csv'],
     'Aligned geography and shared reference; exploratory residence and wealth contrasts'),
    ('west_bengal', 'extensions_work/04_health_system_context/outputs/narrow_west_bengal_sensitivity',
     'west_bengal_sensitivity_summary.json',
     'af631562544e43697ac9bc55bce1150b6682adccad56cb5f92264a1e43b0a20d',
     ['west_bengal_sensitivity_intervals.csv'],
     'West Bengal exclusion; distinguish reference restriction from model refitting'),
    ('policy_periods', 'extensions_work/03_nfhs4_nfhs5_temporal/outputs/policy_period_state_roster_uncertainty',
     'household_uncertainty_review.json',
     '447d4c7841bb1c8769de939e7d14667cb8c3a601c478aa51fb956c8c49acecd9',
     ['household_period_intervals.csv', 'household_period_contrasts.csv',
      'point_estimate_verification.csv', 'singleton_sensitivity.csv'],
     'Selected 11 states and common period reference; approximate exploratory intervals; no policy effects'),
]

README = '''# Current narrow association results

Facility definition: raw m15 = 21 for public and 31 for private.
Birth recall rule: 0 <= v008 - b3 < 60 months.

This package copies aggregate outputs without changing estimates or fitting models.
Each component remains in its own folder with its original summary and column names.
Use manifest.csv to identify the source and reference for every copied file.

Use national results for observed and adjusted within-wave associations, and
Comment 15 for the shared-reference NFHS-4 versus NFHS-5 comparison.
Support and West Bengal sensitivity scenarios can change the reference,
training sample, or both. Preserve those distinctions when reporting them.
Policy-period results concern selected states and periods, with approximate
exploratory intervals. They are not causal policy effects.

The legacy notebooks/v2/10_final_tables_figures.ipynb reads old B1-B4 tables,
exports AIPW and risk-score results, and has stored broad-sector counts.
Keep that notebook and outputs/final_tables and outputs/final_figures as
historical material. Do not use them as the current narrow headline results.
Old prediction and SHAP results may be retained only with their broad cohort
explicitly identified. Prediction-score heterogeneity is omitted here.

Federated results and Comment 17 historical context are not included or
certified by this package. Their reviewed outputs require separate integration.
No professor document or manuscript is edited by this script.
The national script/run hash difference remains documented in the handoff.

Summary hashes are checked against the audit. CSV hashes are recorded for
traceability; no prior expected CSV hashes were available to verify against.
A successful package does not validate models, confidence intervals, or claims.
'''

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = (args.output or root / 'extensions_work/12_narrow_scope/outputs' /
           ('comment13_final_results_' + stamp)).resolve()
    if out.exists():
        raise FileExistsError('Choose a new output folder: ' + str(out))
    selected = []
    # Complete all input checks before creating the output directory.
    for component, relative, summary_name, expected, csv_names, reference in SOURCES:
        folder = root / relative
        summary_path = folder / summary_name
        if not summary_path.is_file():
            raise FileNotFoundError(str(summary_path))
        if digest(summary_path) != expected:
            raise ValueError('Summary differs from the reviewed audit; review before packaging: ' + str(summary_path))
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        for name in [summary_name] + csv_names:
            path = folder / name
            if not path.is_file():
                raise FileNotFoundError(str(path))
            if path.suffix == '.csv':
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    reader = csv.reader(stream)
                    header = next(reader, None)
                    first = next(reader, None)
                    if not header or not first:
                        raise ValueError('Empty aggregate table: ' + str(path))
            selected.append((component, path, reference, summary.get('status', 'unspecified')))
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for component, source, reference, status in selected:
        target = out / component / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        if digest(source) != digest(target):
            raise IOError('Copy verification failed: ' + str(target))
        rows.append(dict(component=component, source_path=str(source.relative_to(root)),
                         packaged_path=str(target.relative_to(out)), sha256=digest(target),
                         facility_codes='21 versus 31', reference=reference,
                         recorded_run_status=status))
    with (out / 'manifest.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'README.md').write_text(README, encoding='utf-8')
    (out / 'packaging_status.json').write_text(json.dumps(dict(
        status='aggregate_package_created_document_integration_pending',
        author_execution_status='Not run or tested by the assistant',
        created_at_utc=stamp, files_copied=len(rows), raw_data_read=False,
        model_refits_run=False, manuscript_changed=False,
        legacy_outputs_changed=False), indent=2), encoding='utf-8')
    print('Created aggregate package: ' + str(out))
    print('Share manifest.csv and packaging_status.json for review.')

if __name__ == '__main__':
    main()
