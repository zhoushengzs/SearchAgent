"""Merge only complete, non-overlapping shards of exactly the same model run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.common import load_tasks, read_predictions, shard_tasks, write_json_atomic
from evaluation.prepare import digest, dump, write_rows
from evaluation.run import fingerprint


def merge_shards(shards, output, tasks_path):
    shards, output = Path(shards), Path(output)
    configs = sorted(shards.glob('*/run.json'))
    if not configs:
        raise ValueError('No shard run.json files found')
    first = json.loads(configs[0].read_text())
    count = first['shard']['count']
    if len(configs) != count:
        raise ValueError(f'Expected {count} shards, found {len(configs)}')
    tasks = load_tasks(tasks_path)
    if digest(tasks_path) != first['protocol']['tasks_sha256']:
        raise ValueError('Task manifest differs from the inference run')
    by_id = {t['id']: t for t in tasks}
    expected_ids = first['protocol']['selected_ids']
    if len(set(expected_ids)) != len(expected_ids) or not set(expected_ids) <= set(by_id):
        raise ValueError('Invalid selected task IDs')
    selected = [by_id[key] for key in expected_ids]
    merged, seen_ranks = {}, set()
    for config in configs:
        run = json.loads(config.read_text())
        if run['run_id'] != first['run_id'] or run['protocol'] != first['protocol'] or run['model_identity'] != first['model_identity'] or run['model'] != first['model']:
            raise ValueError('Mixed models or evaluation protocols across shards')
        if run['protocol_sha256'] != fingerprint(run['protocol']):
            raise ValueError('Protocol hash differs')
        identity = {k: run[k] for k in ['model', 'model_identity', 'protocol', 'protocol_sha256', 'requested_device']}
        if fingerprint(identity) != run['run_id']:
            raise ValueError('Model run identity differs')
        shard = run['shard']
        rank = shard['index']
        if shard['count'] != count or rank in seen_ranks or not 0 <= rank < count:
            raise ValueError('Invalid or repeated shard rank')
        seen_ranks.add(rank)
        assigned = shard_tasks(selected, rank, count)
        expected = {t['id']: t for t in assigned}
        if shard['selected_ids'] != list(expected):
            raise ValueError('Shard assignments differ from the deterministic split')
        predictions = read_predictions(config.parent / 'predictions.jsonl', run['run_id'], expected)
        if set(predictions) != set(expected):
            raise ValueError(f'Shard {rank} is incomplete: {len(predictions)}/{len(expected)} tasks')
        if any(r.get('infrastructure_error') for r in predictions.values()):
            raise ValueError(f'Shard {rank} has infrastructure errors; resume with --retry-infrastructure-errors')
        if set(merged) & set(predictions):
            raise ValueError('Repeated tasks across shards')
        merged.update(predictions)
    if set(merged) != set(expected_ids):
        raise ValueError('Merged tasks do not cover the selected evaluation set')
    if output.resolve() == shards.resolve() or output.resolve().is_relative_to(shards.resolve()):
        raise ValueError('Merged output must be outside the shard directories')
    if (output / 'run.json').exists() and json.loads((output / 'run.json').read_text())['run_id'] != first['run_id']:
        raise ValueError('Merged output already belongs to another run')
    output.mkdir(parents=True, exist_ok=True)
    run = {k: v for k, v in first.items() if k != 'shard'}
    run['tasks_file'] = str(Path(tasks_path).resolve())
    run['merged_shards'] = count
    # Do not rewrite task evidence; artifacts remain in each shard directory.
    temporary = output / 'predictions.jsonl.tmp'
    write_rows(temporary, [merged[key] for key in expected_ids])
    temporary.replace(output / 'predictions.jsonl')
    write_json_atomic(output / 'run.json', run)
    return len(merged)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    a = p.parse_args()
    try:
        total = merge_shards(a.shards, a.output, a.tasks)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    print(f'Merged {total} tasks into {a.output.resolve()}')


if __name__ == '__main__':
    main()
