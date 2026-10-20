"""Freeze the existing holdout; exclude known initial-image overlap with SFT train."""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = ['agentvista', 'mmsearch_plus', 'mmbrowsecomp', 'browsecomp_v3',
           'm3vqa', 'tirbench', 'dynvqa']


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, values):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in values:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def initial_hashes(row, images):
    text = row['conversations'][0]['value']
    marker = 'Recorded task state follows.'
    if marker in text:
        start = text.index('{', text.index(marker))
        state, _ = json.JSONDecoder().raw_decode(text[start:])
        return {image['sha256'] for image in state['initial_views']}
    # Auxiliary answer examples use the original initial images.
    return {images[path]['sha256'] for path in row.get('images', [])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sft-data', type=Path, default=ROOT / 'OpenSearch-VL-SFT-Agent-NoImages')
    p.add_argument('--holdout', type=Path, default=ROOT / 'data/seeds/search_v1_holdout.jsonl')
    p.add_argument('--references', type=Path, default=ROOT / 'data/benchmarks/tasks_references.jsonl')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/local_benchmark_eval/data')
    p.add_argument('--pilot-per-source', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.pilot_per_source < 1:
        p.error('--pilot-per-source must be positive')
    if a.output.exists():
        raise SystemExit('Output already exists. Reuse the frozen data, or choose a new --output directory.')
    from core.views import _render_view

    manifest = json.loads((a.sft_data / 'configs/image_paths.json').read_text())['images']
    split_hashes, files = {}, {}
    for split in ['train', 'val']:
        source = a.sft_data / (split + '.jsonl')
        hashes = set()
        for row in rows(source):
            hashes.update(initial_hashes(row, manifest))
        split_hashes[split] = hashes
        files[split] = {'path': str(source.resolve()), 'sha256': digest(source)}
        print(f'Read SFT {split}: {len(hashes)} distinct initial-image hashes', flush=True)

    holdout = list(rows(a.holdout))
    if len({t['id'] for t in holdout}) != len(holdout):
        raise ValueError('Duplicate holdout IDs')
    references = {(r['id'], r['input_fingerprint']): r for r in rows(a.references)}
    kept, excluded, seen_val, preview_cache = [], [], [], {}
    for index, task in enumerate(holdout, 1):
        if task.get('trajectory_derivation', {}).get('partition') != 'holdout':
            raise ValueError('Input contains a task outside the original holdout')
        if 'reference' in task:
            raise ValueError('Answers must not be stored in model input')
        if (task['id'], task['input_fingerprint']) not in references:
            raise ValueError('Missing detached reference')
        identities = set()
        for image in task['model_input']['images']:
            path = Path(image['path'])
            if not path.is_absolute():
                path = ROOT / path
            actual = digest(path)
            if actual != image['sha256']:
                raise ValueError(f'Initial image changed: {path}')
            if actual not in preview_cache:
                rendered, _ = _render_view(str(path), None, 512)
                buffer = io.BytesIO()
                rendered.save(buffer, format='PNG')
                preview_cache[actual] = hashlib.sha256(buffer.getvalue()).hexdigest()
            identities.update([actual, preview_cache[actual]])
            image['path'] = str(path.resolve())
        train_overlap = identities & split_hashes['train']
        if train_overlap:
            excluded.append({'id': task['id'], 'source_dataset': task['source_dataset'],
                             'reason': 'initial_image_or_512px_preview_in_sft_train',
                             'matched_hashes': sorted(train_overlap)})
        else:
            kept.append(task)
        if identities & split_hashes['val']:
            seen_val.append(task['id'])
        if index % 250 == 0:
            print(f'Checked {index}/{len(holdout)} holdout tasks', flush=True)
    if not kept:
        raise ValueError('No tasks remain after overlap removal')

    pilot = []
    for source in SOURCES:
        group = sorted((t for t in kept if t['source_dataset'] == source), key=lambda t: t['id'])
        random.Random(a.seed).shuffle(group)
        pilot.extend(group[:a.pilot_per_source])
    a.output.mkdir(parents=True)
    write_rows(a.output / 'tasks.jsonl', kept)
    write_rows(a.output / 'pilot_tasks.jsonl', pilot)
    write_rows(a.output / 'references.jsonl', [references[(t['id'], t['input_fingerprint'])] for t in kept])
    dump(a.output / 'excluded_train_overlap.json', excluded)
    counts = collections.Counter(t['source_dataset'] for t in kept)
    report = {'label': 'project_development_holdout_not_official_full_benchmark',
              'input_count': len(holdout), 'kept_count': len(kept),
              'excluded_count': len(excluded), 'kept_by_source': dict(counts),
              'excluded_by_source': dict(collections.Counter(t['source_dataset'] for t in excluded)),
              'requested_sources': SOURCES, 'missing_sources': [s for s in SOURCES if not counts[s]],
              'pilot_count': len(pilot), 'seed': a.seed,
              'validation_image_overlap_count_before_train_filter': len(seen_val),
              'validation_image_overlap_task_ids': seen_val,
              'sft_files': files,
              'original_holdout_sha256': digest(a.holdout),
              'tasks_sha256': digest(a.output / 'tasks.jsonl'),
              'pilot_tasks_sha256': digest(a.output / 'pilot_tasks.jsonl'),
              'references_sha256': digest(a.output / 'references.jsonl'),
              'limitations': [
                  'Train-overlap filter compares original initial-image file hashes and 512px rendered preview hashes.',
                  'It does not detect all re-encoding, crop, perceptual or semantic near duplicates.',
                  'A training validation set is development data when used for checkpoint or parameter selection.',
                  'Actual training run provenance and whether this checkpoint used this data are not established.',
                  'DynVQA is not added or randomly partitioned when absent from the existing holdout.',
              ]}
    dump(a.output / 'manifest.json', report)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('_ids')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
