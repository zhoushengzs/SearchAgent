"""Export the frozen 3,103 tasks and only their original images for transfer."""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path
import shutil
import tempfile

from core.provenance import file_sha
from evaluation.common import load_tasks, resolve_image
from evaluation.prepare import dump, rows, write_rows

ROOT = Path(__file__).resolve().parents[1]


def export_bundle(tasks_path, references_path, output, expected_count=3103,
                  image_root=None, original_root=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new directory')
    tasks = load_tasks(tasks_path, expected_count)
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
        load_tasks(temporary / 'tasks.jsonl', expected_count)
        manifest = {'schema': 'portable-frozen-evaluation-v1', 'task_count': len(tasks),
                    'source_tasks_sha256': file_sha(tasks_path),
                    'tasks_sha256': file_sha(temporary / 'tasks.jsonl'),
                    'references_sha256': file_sha(temporary / 'references.jsonl'),
                    'by_source': dict(collections.Counter(t['source_dataset'] for t in tasks)),
                    'image_count': len(images), 'images': images,
                    'input_policy': 'Only original question and initial images; no teacher steps or prior tool results.',
                    'label': 'project_holdout_not_official_full_benchmark'}
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
    a = p.parse_args()
    try:
        result = export_bundle(a.tasks, a.references, a.output, a.expected_count, a.image_root, a.original_root)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    print(json.dumps({'output': str(a.output.resolve()), 'tasks': result['task_count'],
                      'images': result['image_count'], 'by_source': result['by_source']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
