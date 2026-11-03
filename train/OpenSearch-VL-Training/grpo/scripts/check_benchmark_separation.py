"""Independently compare merged training data against the frozen benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root, project = args.root.resolve(), args.project_root.resolve()
    benchmark = project / 'outputs/server_eval_778'
    tasks = [json.loads(l) for l in (benchmark / 'tasks.jsonl').open() if l.strip()]
    manifest = json.loads((benchmark / 'manifest.json').read_text())
    before = json.loads((root / 'audit/sft_preparation.json').read_text())['input_files'][str(benchmark / 'tasks.jsonl')]['sha256']
    current = sha(benchmark / 'tasks.jsonl')
    assert current == before == manifest['tasks_sha256'], 'Frozen benchmark task file changed'
    assert sha(benchmark / 'references.jsonl') == manifest['references_sha256'], 'Frozen benchmark reference file changed'
    ids = {t['id'] for t in tasks}
    benchmark_pixels = {}
    for task in tasks:
        for image in task['model_input']['images']:
            if image['sha256'] in benchmark_pixels:
                continue
            path = benchmark / image['path']
            assert sha(path) == image['sha256'], 'Benchmark image checksum differs'
            with Image.open(path) as opened:
                opened.load()
                rgb = opened.convert('RGB')
                benchmark_pixels[image['sha256']] = hashlib.sha256(str(rgb.size).encode() + b'\0' + rgb.tobytes()).hexdigest()
    final_manifest = json.loads((root / 'audit/image_manifest.json').read_text())
    provenance = {r['id']: r for r in map(json.loads, (root / 'audit/provenance.jsonl').open())}
    forbidden_pixels = set(benchmark_pixels.values())
    overlap_ids, overlap_images = [], []
    for row in map(json.loads, (root / 'merged.jsonl').open()):
        if provenance[row['id']].get('original_task_id') in ids:
            overlap_ids.append(row['id'])
        for relative in row['images']:
            if final_manifest[relative]['pixel_sha256'] in forbidden_pixels:
                overlap_images.append({'training_id': row['id'], 'image': relative})
    guard = {'benchmark_task_ids': sorted(ids), 'benchmark_decoded_image_sha256': sorted(forbidden_pixels),
             'benchmark_tasks_sha256': current, 'benchmark_references_sha256': manifest['references_sha256'],
             'benchmark_directory': str(benchmark)}
    (root / 'audit/benchmark_test_guard.json').write_text(json.dumps(guard, ensure_ascii=False, indent=2) + '\n')
    result = {'passed': not overlap_ids and not overlap_images, 'protected_benchmark_tasks': len(tasks),
              'protected_benchmark_image_files': len(benchmark_pixels), 'training_or_development_ids_in_benchmark': overlap_ids,
              'training_or_development_images_matching_benchmark_pixels': overlap_images,
              'benchmark_tasks_unchanged': True, 'benchmark_references_unchanged': True,
              'registry_test_means_grpo_development_split_not_benchmark': True,
              'method': 'Frozen IDs plus all exact decoded RGB image pixels, independent of file encoding; byte hashes and 512px previews also checked by the preparation pipeline.'}
    (root / 'audit/benchmark_separation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit('Benchmark separation failed; do not use these files')


if __name__ == '__main__':
    main()
