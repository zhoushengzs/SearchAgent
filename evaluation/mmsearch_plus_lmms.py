"""Score saved MMSearch-Plus answers with the authors-linked lmms-eval metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.common import load_tasks, read_predictions, write_json_atomic
from evaluation.prepare import digest
from evaluation.run import fingerprint
from evaluation.score import content_answer
from evaluation.vendor.mmsearch_plus_lmms import UPSTREAM, mmsearch_plus_process_results


def score_answer(answer, alternatives):
    if not isinstance(answer, str) or not isinstance(alternatives, list) or any(not isinstance(v, str) for v in alternatives):
        raise ValueError('Answers and alternative references must be strings')
    # Preserve upstream normalization, max-over-alternatives and empty-reference
    # behavior. The original task body is unchanged; only decryption is bypassed.
    return mmsearch_plus_process_results({'answer': alternatives}, [answer])


def run_evaluation(run_dir, tasks_path, references_path, output, expected_count=66):
    run_dir, tasks_path, references_path, output = map(Path, [run_dir, tasks_path, references_path, output])
    run = json.loads((run_dir / 'run.json').read_text())
    if run.get('shard', {}).get('count', 1) != 1:
        raise ValueError('Use the merged run, not an individual GPU shard')
    if digest(tasks_path) != run['protocol']['tasks_sha256']:
        raise ValueError('Task bundle differs from the source run')
    all_tasks = {t['id']: t for t in load_tasks(tasks_path)}
    selected = run['protocol']['selected_ids']
    if len(selected) != len(set(selected)) or not set(selected) <= all_tasks.keys():
        raise ValueError('Invalid selected task IDs')
    tasks = [t for t in all_tasks.values() if t['source_dataset'] == 'mmsearch_plus']
    if len(tasks) != expected_count or not {t['id'] for t in tasks} <= set(selected):
        raise ValueError('Run does not cover the requested MMSearch-Plus subset')
    before = digest(run_dir / 'predictions.jsonl')
    predictions = read_predictions(run_dir / 'predictions.jsonl', run['run_id'], {k: all_tasks[k] for k in selected})
    if before != digest(run_dir / 'predictions.jsonl'):
        raise ValueError('Source predictions are changing; wait until inference stops')
    refs = {}
    for line in references_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        ref = json.loads(line)
        key = (ref['id'], ref['input_fingerprint'])
        if key in refs:
            raise ValueError('Duplicate reference identity')
        refs[key] = ref
    records, inputs, excluded, missing_gold = [], [], [], []
    for task in tasks:
        key = task['id']
        pred = predictions.get(key)
        if pred is None:
            excluded.append({'id': key, 'reason': 'missing_prediction'})
            continue
        if pred.get('infrastructure_error'):
            excluded.append({'id': key, 'reason': 'infrastructure_error'})
            continue
        ref = refs.get((key, task['input_fingerprint']))
        if ref is None:
            raise ValueError('Missing or mismatched reference: ' + key)
        if ref.get('answer_format') != 'alternatives':
            raise ValueError('MMSearch-Plus references must represent alternatives')
        alternatives = ref.get('answer_candidates')
        if alternatives is None:
            raise ValueError('Missing alternative answer list: ' + key)
        answer = content_answer(pred)
        metrics = score_answer(answer, alternatives)
        has_gold = any(value.strip() for value in alternatives)
        if not has_gold:
            missing_gold.append(key)
        inputs.append({'id': key, 'question': task['model_input']['question'], 'answer': alternatives,
                       'prediction': answer, 'already_decrypted': True})
        records.append({'id': key, 'source_dataset': 'mmsearch_plus',
                        'input_fingerprint': task['input_fingerprint'], 'prediction_sha256': fingerprint(pred),
                        'original_status': pred['status'], 'candidate': answer, 'reference_alternatives': alternatives,
                        'has_reference': has_gold, **metrics})
    if not records:
        raise ValueError('No evaluable predictions')
    def aggregate(values):
        count = len(values)
        return {'count': count,
                'exact_match_count': sum(r['exact_match'] for r in values),
                'exact_match_pct': 100 * sum(r['exact_match'] for r in values) / count if count else None,
                'f1_pct': 100 * sum(r['f1_score'] for r in values) / count if count else None}
    vendor = Path(__file__).parent / 'vendor/mmsearch_plus_lmms.py'
    config = {'source_run_id': run['run_id'], 'source_run_sha256': digest(run_dir / 'run.json'),
              'predictions_sha256': before, 'tasks_sha256': digest(tasks_path), 'references_sha256': digest(references_path),
              'adapter_sha256': digest(__file__), 'vendor_sha256': digest(vendor), 'upstream': UPSTREAM,
              'answer_extraction': 'evaluation.score.content_answer',
              'answer_extraction_source_sha256': digest(Path(__file__).parent / 'score.py'),
              'dataset': 'already-decrypted project holdout; no dataset redownload', 'api_requests': 0}
    summary = {'metric': 'authors-linked lmms-eval MMSearch-Plus EM/F1 on saved agent answers',
               'planned_count': len(tasks), 'scored_count': len(records),
               'excluded': excluded, 'missing_reference_ids': missing_gold,
               'raw_upstream_metrics': aggregate(records),
               'valid_reference_metrics': aggregate([r for r in records if r['has_reference']]),
               'provisional': bool(excluded or missing_gold), 'upstream_revision': UPSTREAM['revision'],
               'policy_id': fingerprint(config), 'api_requests': 0,
               'scope': 'Post-hoc EM/F1 only. Existing agent trajectories are retained; this is neither the no-search VQA inference protocol nor the paper GPT-4o judge evaluation.',
               'missing_reference_policy': 'Raw metrics retain upstream empty-gold behavior (zero); valid-reference metrics separately exclude missing local annotations. No source reference is edited.'}
    if output.exists():
        expected = output / 'score_config.json'
        if not expected.is_file() or json.loads(expected.read_text()) != config:
            raise ValueError('Output belongs to different inputs/code; choose another directory')
        if (json.loads((output / 'summary.json').read_text()) != summary
                or [json.loads(s) for s in (output / 'scores.jsonl').read_text().splitlines() if s.strip()] != records):
            raise ValueError('Saved scoring artifacts were changed')
        return summary
    output.mkdir(parents=True)
    write_json_atomic(output / 'score_config.json', config)
    write_json_atomic(output / 'summary.json', summary)
    for name, values in [('scores.jsonl', records), ('lmms_inputs.jsonl', inputs)]:
        with (output / name).open('w', encoding='utf-8') as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-count', type=int, default=66)
    a = p.parse_args()
    try:
        result = run_evaluation(a.run, a.tasks, a.references, a.output, a.expected_count)
    except (ValueError, KeyError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
