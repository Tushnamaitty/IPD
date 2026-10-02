"""Comment 13 local consistency audit. NOT EXECUTED OR TESTED by the author.

Reads aggregate JSON and source text only. Does not read raw/Parquet records,
fit models, import analysis modules, or modify any existing output or source.
Text matches are review hints, never certification of an executed cohort.
Requires Python standard library only. Run from the repository with --repo-root .
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', 'site-packages',
        'data', 'outputs', 'artifacts', '.ipynb_checkpoints'}
SPECS = [
    ('national_associations', 'IPD_narrow_facility_analysis.py',
     'extensions_work/12_narrow_scope/outputs/national_associations/national_association_summary.json',
     'Own wave and subgroup reference; additive model'),
    ('support_sensitivity', 'IPD_narrow_support_sensitivity.py',
     'extensions_work/12_narrow_scope/outputs/narrow_support_sensitivity/support_sensitivity_summary.json',
     'Support restrictions of the national analysis; some scenarios also refit'),
    ('comment15_comparison', 'IPD_comment15_analysis.py',
     'extensions_work/12_narrow_scope/outputs/comment15_shared_reference/comment15_summary.json',
     'Shared pooled reference; aligned geography; residence and wealth interactions'),
    ('west_bengal_sensitivity', 'IPD_west_bengal_sensitivity.py',
     'extensions_work/04_health_system_context/outputs/narrow_west_bengal_sensitivity/west_bengal_sensitivity_summary.json',
     'National reference restriction and refit excluding West Bengal'),
]
PATTERNS = {
    'broad_sector_selection_hint': r'between\s*\(\s*20\s*,\s*27|between\s*\(\s*30\s*,\s*33|isin\s*\(\s*\[\s*20\s*,\s*21',
    'narrow_code_hint': r'isin\s*\(\s*\[\s*21\s*,\s*31|[\"\x27]public[\"\x27]\s*:\s*21',
    'legacy_processed_input_hint': r'df_model_v2|df_model_nfhs4|df_model_nfhs5',
    'legacy_estimator_hint': r'\bAIPW\b|\bTMLE\b|super.?learner',
    'legacy_predictive_heterogeneity_hint': r'predicted.?effect|effect.?quintile|prediction.?score|CATE|causal.?forest',
    'historical_broad_count_hint': r'\b200794\b|\b195366\b|\b200,794\b|\b195,366\b',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def walk_values(value, key):
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                yield child
            yield from walk_values(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from walk_values(child, key)


def read_json(path):
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Expected an aggregate JSON object: ' + str(path))
    return value


def source_record(path, root):
    record = {'path': path.relative_to(root).as_posix(), 'sha256': sha(path)}
    text = path.read_text(encoding='utf-8-sig', errors='replace')
    if path.suffix.lower() == '.ipynb':
        notebook = json.loads(text)
        text = '\n'.join(''.join(c.get('source', []))
                         for c in notebook.get('cells', []) if c.get('cell_type') == 'code')
    record['source_only_not_notebook_outputs'] = True
    for name, pattern in PATTERNS.items():
        record[name] = len(re.findall(pattern, text, flags=re.I))
    record['narrow_base_dependency_hint'] = 'IPD_narrow_facility_analysis.py' in text
    record['python_parse_status'] = 'not_applicable'
    if path.suffix.lower() == '.py':
        try:
            ast.parse(text)
            record['python_parse_status'] = 'parsed_without_execution'
        except SyntaxError:
            record['python_parse_status'] = 'syntax_error_requires_review'
    record['scope_verdict'] = 'manual_source_review_required'
    if record['broad_sector_selection_hint'] and record['narrow_code_hint']:
        record['note'] = 'Both hints present; broad counts can be input checks before a valid narrow filter.'
    else:
        record['note'] = 'Hints do not establish selected data, execution path, or manuscript use.'
    return record


def component(root, name, script, summary, reference, baseline_hash, baseline_audits):
    path = root / summary
    result = {'component': name, 'summary_path': summary, 'script_path': script,
              'declared_reference': reference, 'issues': [], 'scope_evidence': []}
    if not path.is_file():
        result['issues'].append('Aggregate summary missing at the declared path; locate it before judging scope.')
        result['status'] = 'missing_summary'
        return result
    try:
        obj = read_json(path)
    except (ValueError, OSError) as error:
        result['issues'].append(str(error)); result['status'] = 'unreadable_summary'
        return result
    result['summary_sha256'] = sha(path)
    result['reported_run_status'] = obj.get('status', 'not_recorded')
    explicit = list(walk_values(obj, 'facility_codes'))
    if explicit:
        if all(c == {'public': 21, 'private': 31} for c in explicit):
            result['scope_evidence'].append('Explicit aggregate facility_codes public 21 and private 31.')
        else:
            result['issues'].append('Facility-code declaration differs from exact public 21 private 31.')
    links = []
    for key in ('baseline_summary_sha256', 'baseline_sha256', 'national_summary_sha256'):
        links.extend(v for v in walk_values(obj, key) if isinstance(v, str))
    result['recorded_baseline_links'] = links
    if links and baseline_hash:
        if baseline_hash in links:
            result['scope_evidence'].append('Recorded baseline hash matches the current national summary.')
        else:
            result['issues'].append('Recorded baseline link differs; inspect version/provenance before combining outputs.')
    if not explicit and not (baseline_hash and baseline_hash in links):
        result['issues'].append('No explicit code declaration or matching baseline hash; source/provenance review required.')
    recorded = obj.get('script_sha256')
    source = root / script
    if source.is_file():
        current = sha(source); result['current_script_sha256'] = current
        result['recorded_script_sha256'] = recorded
        result['script_hash_matches_run'] = current == recorded if recorded else None
        if recorded and current != recorded:
            result['issues'].append('Current script differs from run hash. A self-test-only edit may explain this; do not assume a model rerun is needed.')
    else:
        result['issues'].append('Declared analysis source missing; cannot review current dependency.')
    audits = obj.get('cohort_audit', {})
    result['cohort_counts'] = {}
    for wave in ('NFHS-4', 'NFHS-5'):
        a = audits.get(wave, {})
        if not isinstance(a, dict):
            result['issues'].append('Unexpected cohort-audit format for ' + wave)
            continue
        result['cohort_counts'][wave] = {k: a[k] for k in
            ('analytic_n', 'narrow_before_date_rule_n', 'date_or_recall_excluded_n',
             'public_n', 'private_n', 'geography_excluded_n', 'household_roster_available') if k in a}
        if a and baseline_audits.get(wave):
            previous = baseline_audits[wave]
            if a.get('birth_sha256') and a.get('birth_sha256') != previous.get('birth_sha256'):
                result['issues'].append(wave + ' raw birth fingerprint differs from national baseline.')
            if name in ('support_sensitivity', 'comment15_comparison') and a.get('analytic_n') != previous.get('analytic_n'):
                result['issues'].append(wave + ' pre-restriction cohort count differs from national baseline.')
    result['status'] = 'aggregate_scope_evidence_recorded' if not result['issues'] else 'review_items_present'
    result['certification'] = 'Metadata/provenance only; not raw-data or full source-path verification.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--policy-summary', type=Path)
    args = parser.parse_args(); root = args.repo_root.resolve()
    out = (args.output.resolve() if args.output else
           root / 'extensions_work/12_narrow_scope/outputs/comment13_consistency_audit')
    if out.exists():
        raise FileExistsError('Existing audit preserved; choose a new --output.')
    baseline_path = root / SPECS[0][2]
    baseline = read_json(baseline_path) if baseline_path.is_file() else {}
    baseline_hash = sha(baseline_path) if baseline_path.is_file() else None
    records = [component(root, *spec, baseline_hash, baseline.get('cohort_audit', {})) for spec in SPECS]
    sources = []; errors = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.py', '.ipynb', '.md'):
            continue
        rel = path.relative_to(root)
        if any(part.lower() in SKIP for part in rel.parts[:-1]):
            continue
        if path.stat().st_size > 4 * 1024 * 1024:
            errors.append({'path': rel.as_posix(), 'reason': 'Source exceeds 4 MiB; manual review required.'})
            continue
        try:
            sources.append(source_record(path, root))
        except (ValueError, OSError) as error:
            errors.append({'path': rel.as_posix(), 'reason': str(error)})
    policy = args.policy_summary or root / 'extensions_work/03_nfhs4_nfhs5_temporal/outputs/policy_period_state_roster_uncertainty/household_uncertainty_review.json'
    if not policy.is_absolute():
        policy = root / policy
    policy_record = {'path': str(policy.relative_to(root)) if policy.is_relative_to(root) else str(policy),
                     'source_scope_status': 'manual_review_required',
                     'note': 'Different 11-state and period reference is intentional. Confirm exact facility filter and corrected state design keys from source; a summary alone does not prove this.'}
    if policy.is_file():
        try:
            obj = read_json(policy)
            policy_record.update(summary_sha256=sha(policy), reported_status=obj.get('status'),
                state_design_keys_verified=obj.get('state_design_keys_verified'),
                reference_n=obj.get('reference_n'), reference_excluded_religion_codes=obj.get('reference_excluded_religion_codes'),
                script_sha256=obj.get('script_sha256'))
        except (ValueError, OSError) as error:
            policy_record['read_error'] = str(error)
    else:
        policy_record['read_error'] = 'Corrected policy aggregate not found at declared path.'
    checklist = [
        {'item': 'Main associations', 'decision': 'Use national narrow observed and adjusted results; keep wave-specific reference explicit.'},
        {'item': 'Cross-wave comparison', 'decision': 'Use Comment 15 shared-reference contrasts; do not subtract older broad-sector or wave-specific results as the primary temporal test.'},
        {'item': 'Residence and wealth heterogeneity', 'decision': 'Use completed Comment 15 narrow results with exploratory subgroup wording.'},
        {'item': 'First delivery', 'decision': 'Use narrow first-delivery outputs; identify reference and birth-record subgroup consistently.'},
        {'item': 'Support sensitivity', 'decision': 'Use linked narrow support outputs; distinguish reference-only restriction from refitting.'},
        {'item': 'West Bengal', 'decision': 'Use linked narrow exclusion results; this does not complete historical state-context work.'},
        {'item': 'Policy periods', 'decision': 'Retain corrected narrow period results with their distinct 11-state reference; source verification remains necessary.'},
        {'item': 'Legacy AIPW and TMLE', 'decision': 'Keep historical files for traceability; remove old broad-sector estimates from current narrow main findings under the prepared methods decision.'},
        {'item': 'Original prediction and SHAP', 'decision': 'Keep broad-sector predictive scope separate and explicitly labelled if retained; do not claim it is the narrow association analysis.'},
        {'item': 'Prediction-score heterogeneity', 'decision': 'Not rerun in Comment 15. Omit from the narrow association findings, or explicitly commission a narrow rerun if retained.'},
        {'item': 'Historical state context', 'decision': 'Separate ongoing work must use/declare the narrow birth cohort; not certified by this audit.'},
        {'item': 'Professor update and manuscript', 'decision': 'Apply definitions, counts, references and replacement tables together. Existing documents are not edited by this audit.'},
    ]
    summary = {'status': 'aggregate_and_source_inventory_ready_manual_review_required',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(Path(__file__)), 'author_execution_status': 'Not run or tested by the assistant before delivery',
        'facility_scope_required': {'public': 21, 'private': 31},
        'components': records, 'policy_period_check': policy_record,
        'source_files_count': len(sources), 'source_read_errors': errors,
        'scope_decisions_prepared_not_applied': checklist,
        'expected_count_differences': ['Date/recall restriction', 'First-delivery subgroup',
            'Shared-reference geography alignment', 'Reference support restrictions', 'West Bengal exclusion', 'Policy-period and common-state selection'],
        'remaining_review': 'Review flagged execution paths and manuscript use before closing Comment 13; regex hits are not bugs or verified scope.',
        'model_refits_run': False, 'raw_or_record_level_data_read': False,
        'existing_files_changed': False, 'professor_document_changed': False}
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'comment13_consistency_summary.json', summary)
    with (out / 'comment13_source_inventory.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['path', 'sha256', 'source_only_not_notebook_outputs'] + list(PATTERNS) + ['narrow_base_dependency_hint', 'python_parse_status', 'scope_verdict', 'note']
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(sources)
    with (out / 'comment13_scope_decisions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['item', 'decision']); writer.writeheader(); writer.writerows(checklist)
    print('Audit outputs saved: ' + str(out))
    print('Manual review required. No models or private records read; existing files preserved.')
    print('Share comment13_consistency_summary.json, comment13_source_inventory.csv, comment13_scope_decisions.csv.')


if __name__ == '__main__':
    main()
