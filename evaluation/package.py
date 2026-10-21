"""Export frozen original tasks, with optional deterministic per-benchmark sampling."""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from core.provenance import file_sha
from evaluation.common import load_tasks, resolve_image
from evaluation.prepare import dump, rows, write_rows

ROOT = Path(__file__).resolve().parents[1]


def select_benchmark_sample(tasks, source=None, count=None, seed=42):
    if source is None and count is None:
        return tasks, None
    if source is None or count is None or count < 1:
        raise ValueError('Provide both --sample-source and a positive --sample-count')
    candidates = [task for task in tasks if task['source_dataset'] == source]
    if len(candidates) < count:
        raise ValueError(f'{source}: requested {count} tasks, but only {len(candidates)} exist')
    # Rank IDs and input identities, never answers, success rates or model outputs.
    # SHA256 ranking is stable across Python versions and input file ordering.
    def rank(task):
        payload = json.dumps([seed, source, task['id'], task['input_fingerprint']], ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest(), task['id']
    selected = {task['id'] for task in sorted(candidates, key=rank)[:count]}
    subset = [task for task in tasks if task['source_dataset'] != source or task['id'] in selected]
    selection = {'source': source, 'source_population': len(candidates), 'sample_count': count,
                 'seed': seed, 'method': 'sha256_rank_seed_source_id_input_fingerprint_v1',
                 'parent_task_count': len(tasks), 'selected_source_task_ids': sorted(selected),
                 'other_benchmarks': 'all retained'}
    return subset, selection


def export_bundle(tasks_path, references_path, output, expected_count=3103,
                  image_root=None, original_root=None, sample_source=None, sample_count=None, seed=42):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new directory')
    tasks = load_tasks(tasks_path, expected_count)
    tasks, selection = select_benchmark_sample(tasks, sample_source, sample_count, seed)
    refs = list(rows(references_path))
    reference_map = {(r['id'], r['input_fingerprint']): r for r in refs}
    if len(reference_map) != len(refs):
        raise ValueError('Duplicate references')
    for task in tasks:
        if (task['id'], task['input_fingerprint']) not in reference_map:
            raise ValueError('Missing detached reference')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.eval-bundle-', dir=output.parent))
    try:
        (temporary / 'images').mkdir()
        exported, images = [], {}
        for task in tasks:
            task = copy.deepcopy(task)
            for image in task['model_input']['images']:
                source = resolve_image(image['path'], tasks_path, image_root, original_root)
                sha = image['sha256']
                if file_sha(source) != sha:
                    raise ValueError(f'{task["id"]}: image content changed')
                relative = 'images/' + sha + '.png'
                if sha not in images:
                    shutil.copyfile(source, temporary / relative)
                    images[sha] = {'path': relative, 'bytes': source.stat().st_size}
                image['path'] = relative
            exported.append(task)
        write_rows(temporary / 'tasks.jsonl', exported)
        write_rows(temporary / 'references.jsonl', [reference_map[(t['id'], t['input_fingerprint'])] for t in tasks])
        # Re-read to prove path rewriting has not changed question/image identities.
        load_tasks(temporary / 'tasks.jsonl', len(tasks))
        manifest = {'schema': 'portable-frozen-evaluation-v1', 'task_count': len(tasks),
                    'source_tasks_sha256': file_sha(tasks_path),
                    'tasks_sha256': file_sha(temporary / 'tasks.jsonl'),
                    'references_sha256': file_sha(temporary / 'references.jsonl'),
                    'by_source': dict(collections.Counter(t['source_dataset'] for t in tasks)),
                    'image_count': len(images), 'images': images,
                    'input_policy': 'Only original question and initial images; no teacher steps or prior tool results.',
                    'label': 'project_holdout_not_official_full_benchmark'}
        if selection is not None:
            manifest['selection'] = selection
        dump(temporary / 'manifest.json', manifest)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, default=ROOT / 'outputs/local_benchmark_eval/data/tasks.jsonl')
    p.add_argument('--references', type=Path, default=ROOT / 'outputs/local_benchmark_eval/data/references.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-count', type=int, default=3103)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--original-root', type=Path)
    p.add_argument('--sample-source', help='Benchmark to subsample; all other benchmarks are retained')
    p.add_argument('--sample-count', type=int)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    try:
        result = export_bundle(a.tasks, a.references, a.output, a.expected_count, a.image_root, a.original_root,
                               a.sample_source, a.sample_count, a.seed)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    print(json.dumps({'output': str(a.output.resolve()), 'tasks': result['task_count'],
                      'images': result['image_count'], 'by_source': result['by_source']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
