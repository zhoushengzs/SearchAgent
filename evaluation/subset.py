"""Derive an explicitly labelled task subset from existing predictions, without inference."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

from evaluation.common import load_tasks, read_predictions, write_json_atomic
from evaluation.prepare import digest, write_rows
from evaluation.remote import output_lock
from evaluation.run import fingerprint


def derive_subset_run(source, tasks_file, output):
    source, tasks_file, output = Path(source).resolve(), Path(tasks_file).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Subset output already exists; choose a new directory')
    with output_lock(source):
        run = json.loads((source / 'run.json').read_text())
        if run.get('shard', {}).get('count', 1) > 1:
            raise ValueError('Use the merged run, not an individual GPU shard')
        identity_fields = ['model', 'model_identity', 'protocol', 'protocol_sha256', 'requested_device']
        if fingerprint(run['protocol']) != run['protocol_sha256'] or fingerprint({k: run[k] for k in identity_fields}) != run['run_id']:
            raise ValueError('Source run metadata identity is invalid')
        source_tasks_file = Path(run['tasks_file'])
        if digest(source_tasks_file) != run['protocol']['tasks_sha256']:
            raise ValueError('Source task file changed after inference')
        source_tasks = {t['id']: t for t in load_tasks(source_tasks_file)}
        selected_ids = run['protocol']['selected_ids']
        if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= set(source_tasks):
            raise ValueError('Invalid source selection')
        predictions = read_predictions(source / 'predictions.jsonl', run['run_id'],
                                       {key: source_tasks[key] for key in selected_ids})
        tasks = load_tasks(tasks_file)
        new_ids = [t['id'] for t in tasks]
        if set(new_ids) == set(selected_ids) or not set(new_ids) < set(selected_ids):
            raise ValueError('New tasks must be a strict subset of the source run')
        for task in tasks:
            key = task['id']
            if task['input_fingerprint'] != source_tasks[key]['input_fingerprint'] or task['source_dataset'] != source_tasks[key]['source_dataset']:
                raise ValueError(f'{key}: source task identity differs')
            if key not in predictions or predictions[key].get('infrastructure_error'):
                raise ValueError(f'{key}: retained task is missing or has infrastructure errors')

        derived = copy.deepcopy(run)
        derived['protocol'].update(tasks_sha256=digest(tasks_file), selected_ids=new_ids)
        derived['protocol_sha256'] = fingerprint(derived['protocol'])
        derived['run_id'] = fingerprint({k: derived[k] for k in identity_fields})
        derived['tasks_file'] = str(tasks_file)
        if 'shard' in derived:
            derived['shard'] = {'index': 0, 'count': 1, 'selected_ids': new_ids}
        excluded = sorted(set(selected_ids) - set(new_ids))
        derived['derivation'] = {
            'type': 'post_hoc_task_subset_without_model_rerun',
            'source_run_id': run['run_id'], 'source_run_directory': str(source),
            'source_run_json_sha256': digest(source / 'run.json'),
            'source_predictions_file_sha256': digest(source / 'predictions.jsonl'),
            'source_tasks_sha256': run['protocol']['tasks_sha256'],
            'source_selected_count': len(selected_ids), 'excluded_task_ids': excluded,
            'inference_calls': 0, 'artifacts': 'Retained at original source paths; preserve the source run directory.'}
        records = []
        for key in new_ids:
            original = predictions[key]
            record = copy.deepcopy(original)
            record['run_id'] = derived['run_id']
            record['derived_from_prediction'] = {'run_id': run['run_id'], 'sha256': fingerprint(original)}
            records.append(record)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.subset-run-', dir=output.parent))
        try:
            write_json_atomic(temporary / 'run.json', derived)
            write_rows(temporary / 'predictions.jsonl', records)
            read_predictions(temporary / 'predictions.jsonl', derived['run_id'], {t['id']: t for t in tasks})
            write_json_atomic(temporary / 'completion.json', {
                'expected': len(tasks), 'recorded': len(records), 'missing_ids': [],
                'infrastructure_error_ids': [], 'complete': True,
                'derivation': derived['derivation']})
            temporary.rename(output)
        except BaseException:
            shutil.rmtree(temporary)
            raise
    return {'output': str(output), 'tasks': len(tasks), 'excluded_task_ids': excluded, 'inference_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = derive_subset_run(args.source_run, args.tasks, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
