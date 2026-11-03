"""Build original-question GRPO data; never export SFT histories as prompts."""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import html
import io
import json
import random
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for n, line in enumerate(stream, 1):
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized_question(text):
    if text.startswith('[') and text.endswith(']'):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], str):
                text = parsed[0]
        except (ValueError, SyntaxError):
            pass
    return re.sub(r'\s+', ' ', html.unescape(text).replace('<image>', '')).strip()


def prompt_question(text):
    text = html.unescape(text).replace('<image>', '').strip()
    return re.sub(r'^(?:\\n|\s)+', '', text)


def row_signature(row, manifest):
    text = row['conversations'][0]['value']
    marker = 'Recorded task state follows.'
    if marker in text:
        state, _ = json.JSONDecoder().raw_decode(text[text.index('{', text.index(marker)):])
        hashes = tuple(v['sha256'] for v in state['initial_views'])
        actual = tuple(manifest[p]['sha256'] for p in row.get('images', [])[:len(hashes)])
        if hashes != actual:
            raise ValueError('SFT initial image order/hash differs: ' + row['id'])
        return 'preview', (row['source'], normalized_question(state['question']), hashes)
    question = re.sub(r'^(?:img_\d+: <image>\s*\n)+', '', text)
    return 'original', (row['source'], normalized_question(question), tuple(manifest[p]['sha256'] for p in row.get('images', [])))


def inspect_image(path):
    digest = sha(path)
    with Image.open(path) as image:
        image.load()
        image = image.convert('RGB')
        dimensions = image.size
        pixel = hashlib.sha256(str(dimensions).encode() + b'\0' + image.tobytes()).hexdigest()
        side = min(512, max(image.size))
        preview = ImageOps.contain(image, (side, side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        preview.save(buffer, format='PNG')
        preview_sha = hashlib.sha256(buffer.getvalue()).hexdigest()
    return {'sha256': digest, 'pixel_sha256': pixel, 'preview_sha256': preview_sha, 'width': dimensions[0], 'height': dimensions[1]}


def prepare_sft(project, sft, output):
    state_dir = output / 'audit'
    state_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = sft / 'configs/image_paths.json'
    manifest = json.loads(source_manifest_path.read_text())['images']
    tasks_path = project / 'data/benchmarks/tasks.jsonl'
    references_path = project / 'data/benchmarks/tasks_references.jsonl'
    eval_path = project / 'outputs/server_eval_778/tasks.jsonl'
    tasks = list(read_jsonl(tasks_path))
    by_id = {r['id']: r for r in tasks}
    references = {r['id']: r for r in read_jsonl(references_path)}
    evaluation = list(read_jsonl(eval_path))
    frozen_ids = {r['id'] for r in evaluation}
    frozen_hashes = {im['sha256'] for r in evaluation for im in r['model_input']['images']}
    frozen_text = {normalized_question(r['model_input']['question']) for r in evaluation if not r['model_input']['images']}
    cache_path = project / '.codex/sft_benchmark_comparison/preview_hash_cache.json'
    cache = json.loads(cache_path.read_text())
    if cache['render_code_sha256'] != sha(project / 'core/views.py') or cache['errors']:
        raise ValueError('The existing original-to-preview map is incompatible')
    previews = {r['original_sha256']: r['preview_sha256'] for r in cache['images']}
    original_index, preview_index = collections.defaultdict(set), collections.defaultdict(set)
    for task in tasks:
        images = task['model_input']['images']
        a = tuple(im['sha256'] for im in images)
        b = tuple(previews[im['sha256']] for im in images)
        base = task['source_dataset'], normalized_question(task['model_input']['question'])
        original_index[(*base, a)].add(task['id'])
        preview_index[(*base, b)].add(task['id'])
    frozen_hashes.update(previews[h] for h in list(frozen_hashes) if h in previews)
    train_signatures = set()
    train_hashes = set()
    train_questions = set()
    train_groups = set()
    train_rows = 0
    for row in read_jsonl(sft / 'train.jsonl'):
        train_rows += 1
        train_groups.add(row['id'].split(':', 1)[0])
        kind, sig = row_signature(row, manifest)
        train_signatures.add((kind, sig))
        train_questions.add(sig[1])
        train_hashes.update(manifest[p]['sha256'] for p in row.get('images', []))
        matches = (preview_index if kind == 'preview' else original_index).get(sig, set())
        for tid in matches:
            train_hashes.update(im['sha256'] for im in by_id[tid]['model_input']['images'])
    print(f'Training exclusion index: {train_rows} rows / {len(train_groups)} tasks', flush=True)
    groups = {}
    val_rows = 0
    for row in read_jsonl(sft / 'val.jsonl'):
        val_rows += 1
        key = row['id'].split(':', 1)[0]
        kind, sig = row_signature(row, manifest)
        group = groups.setdefault(key, {'source': row['source'], 'signatures': set(), 'rows': 0})
        group['signatures'].add((kind, sig))
        group['rows'] += 1
    candidates, all_rows, audit, image_cache = [], [], [], {}
    for group_id, group in sorted(groups.items()):
        sets = [(preview_index if kind == 'preview' else original_index).get(sig, set()) for kind, sig in group['signatures']]
        matches = set.intersection(*sets)
        if len(matches) != 1:
            raise ValueError(f'Val task mapping is not unique: {group_id}: {matches}')
        tid = next(iter(matches))
        task, ref = by_id[tid], references[tid]
        if ref['input_fingerprint'] != task['input_fingerprint']:
            raise ValueError('Reference/task input mismatch: ' + tid)
        image_hashes = set()
        paths, verified = [], []
        for image in task['model_input']['images']:
            path = Path(image['path'])
            if not path.is_absolute():
                path = project / path
            if image['sha256'] not in image_cache:
                info = inspect_image(path)
                if info['sha256'] != image['sha256'] or info['preview_sha256'] != previews[image['sha256']]:
                    raise ValueError('Original/preview image changed: ' + str(path))
                image_cache[image['sha256']] = info
            info = image_cache[image['sha256']]
            paths.append(str(path))
            verified.append(info)
            image_hashes.update([info['sha256'], info['preview_sha256']])
        status = 'included'
        if any(sig in train_signatures for sig in group['signatures']) or image_hashes & train_hashes:
            status = 'excluded_sft_training_overlap'
        elif tid in frozen_ids or image_hashes & frozen_hashes or (not paths and normalized_question(task['model_input']['question']) in frozen_text):
            status = 'excluded_frozen_evaluation_overlap'
        answer = ref['answer']
        if not isinstance(answer, str) or not answer.strip():
            if status == 'included':
                status = 'excluded_missing_original_reference'
            answer = answer if isinstance(answer, str) else ''
        sample = {'id': 'sft_val:' + tid, 'question': prompt_question(task['model_input']['question']), 'answer': answer, 'images': paths,
                  '_source': 'sft_val/' + task['source_dataset'], '_images': verified,
                  '_provenance': {'dataset': 'sft_val', 'source_dataset': task['source_dataset'], 'original_task_id': tid, 'val_group': group_id,
                                  'val_rows': group['rows'], 'dataset_provenance': task['dataset_provenance'], 'input_fingerprint': task['input_fingerprint'],
                                  'answer_origin': 'original_dataset_reference', 'reference': ref, 'official_benchmark_repurposed_for_training': True}}
        all_rows.append(sample)
        if status == 'included':
            candidates.append(sample)
        audit.append({'id': sample['id'], 'val_group': group_id, 'source': task['source_dataset'], 'rows': group['rows'], 'status': status})
    write_jsonl(output / 'sources/sft_val/normalized_all.jsonl', all_rows)
    write_jsonl(output / 'sources/sft_val/normalized_eligible.jsonl', candidates)
    write_jsonl(state_dir / 'sft_val_selection.jsonl', audit)
    save_json(state_dir / 'exclusion_keys.json', {'sft_image_sha256': sorted(train_hashes), 'sft_normalized_questions': sorted(train_questions),
                                               'frozen_image_sha256': sorted(frozen_hashes), 'frozen_text_questions': sorted(frozen_text)})
    input_files = [sft / 'train.jsonl', sft / 'val.jsonl', source_manifest_path, tasks_path, references_path, eval_path, cache_path]
    summary = {'sft_root': str(sft), 'sft_train_rows': train_rows, 'sft_train_tasks': len(train_groups), 'sft_val_rows': val_rows,
               'sft_val_tasks': len(groups), 'sft_val_eligible': len(candidates), 'sft_val_status': dict(collections.Counter(r['status'] for r in audit)),
               'sft_val_eligible_by_source': dict(collections.Counter(r['_source'] for r in candidates)), 'frozen_evaluation_tasks': len(evaluation),
               'input_files': {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in input_files}}
    save_json(state_dir / 'sft_preparation.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'input_files'}, ensure_ascii=False, indent=2), flush=True)


def merge(output):
    raw = output / 'sources/Search-VL-RL-8K'
    metadata = json.loads((raw / 'upstream_metadata.json').read_text())
    entry = next(x for x in metadata['siblings'] if x['rfilename'] == 'images.zip')
    if not (raw / 'images.zip.sha256').exists() or (raw / 'images.zip').stat().st_size != entry['size']:
        raise ValueError('RL image download is not complete/verified')
    exclusions = json.loads((output / 'audit/exclusion_keys.json').read_text())
    train_hashes, frozen_hashes = set(exclusions['sft_image_sha256']), set(exclusions['frozen_image_sha256'])
    frozen_text = set(exclusions['frozen_text_questions'])
    rl_rows, excluded, image_manifest = [], [], {}
    with zipfile.ZipFile(raw / 'images.zip') as archive:
        members = {n for n in archive.namelist() if not n.endswith('/') and not n.startswith('__MACOSX/')}
        for index, item in enumerate(read_jsonl(raw / 'rl_data.jsonl')):
            images, information = [], []
            for source_path in item['images']:
                if Path(source_path).is_absolute() or '..' in Path(source_path).parts:
                    raise ValueError('Unsafe upstream image path: ' + source_path)
                matches = [n for n in members if n == source_path or n.endswith('/' + source_path)]
                if len(matches) != 1:
                    raise ValueError('Archive image does not resolve uniquely: ' + source_path)
                target = output / 'images/rl8k' / Path(source_path).name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(matches[0]) as stream, target.open('wb') as dest:
                    shutil.copyfileobj(stream, dest)
                info = inspect_image(target)
                relative = str(target.relative_to(output))
                images.append(relative)
                information.append(info)
                image_manifest[relative] = {**info, 'bytes': target.stat().st_size}
            record = {'id': f'rl8k:{index:06d}', 'question': item['question'], 'answer': item['answer'], 'images': images,
                      '_source': 'rl8k/' + item['dataset'], '_images': information,
                      '_provenance': {'dataset': 'OpenSearch-VL/Search-VL-RL-8K', 'revision': metadata['sha'], 'upstream_row_zero_based': index,
                                      'source_subset': item['dataset'], 'answer_origin': 'upstream_reference_unchanged'}}
            hashes = {h for im in information for h in [im['sha256'], im['preview_sha256']]}
            reason = None
            if hashes & train_hashes:
                reason = 'excluded_sft_training_image_overlap'
            elif hashes & frozen_hashes or (not images and normalized_question(item['question']) in frozen_text):
                reason = 'excluded_frozen_evaluation_overlap'
            if reason:
                excluded.append({'id': record['id'], 'reason': reason, 'record': record})
            else:
                rl_rows.append(record)
            if (index + 1) % 1000 == 0:
                print(f'RL images decoded and checked: {index + 1}', flush=True)
    sft = list(read_jsonl(output / 'sources/sft_val/normalized_eligible.jsonl'))
    for row in sft:
        new_images = []
        for original, info in zip(row['images'], row['_images']):
            source = Path(original)
            target = output / 'images/sft_val' / (info['sha256'] + source.suffix.lower())
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if sha(target) != info['sha256']:
                raise ValueError('Copied SFT original image differs: ' + str(target))
            relative = str(target.relative_to(output))
            new_images.append(relative)
            image_manifest[relative] = {**info, 'bytes': target.stat().st_size}
        row['images'] = new_images
    # Identical RGB pixels catch byte-level re-encodings without guessing semantic similarity.
    inputs = collections.defaultdict(list)
    for row in rl_rows + sft:
        key = normalized_question(row['question']), tuple(i['pixel_sha256'] for i in row['_images'])
        inputs[key].append(row)
    unique, duplicate_records = [], []
    for group in inputs.values():
        answers = {re.sub(r'\s+', ' ', r['answer']).strip().casefold() for r in group}
        if len(answers) > 1:
            excluded.extend({'id': r['id'], 'reason': 'conflicting_references_for_identical_input', 'record': r} for r in group)
            continue
        first = group[0]
        if len(group) > 1:
            first['_provenance']['duplicate_record_ids'] = [r['id'] for r in group[1:]]
            duplicate_records.extend({'id': r['id'], 'kept_id': first['id'], 'reason': 'identical_question_image_and_answer'} for r in group[1:])
        unique.append(first)
    # Split connected image groups, so the same picture cannot appear in train and test.
    parents = list(range(len(unique)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(unique):
        tokens = ['image:' + im['pixel_sha256'] for im in row['_images']]
        if not tokens:
            tokens = ['text:' + normalized_question(row['question'])]
        for token in tokens:
            if token in seen:
                parents[find(i)] = find(seen[token])
            else:
                seen[token] = i
    groups = collections.defaultdict(list)
    for i, row in enumerate(unique):
        groups[find(i)].append(row)
    ordered = sorted(groups.values(), key=lambda g: min(r['id'] for r in g))
    random.Random(42).shuffle(ordered)
    target = round(len(unique) * 0.1)
    test, train = [], []
    for group in ordered:
        if len(test) < target:
            test.extend(group)
        else:
            train.extend(group)
    train.sort(key=lambda r: r['id'])
    test.sort(key=lambda r: r['id'])
    unique.sort(key=lambda r: r['id'])
    def plain(r):
        return {k: r[k] for k in ['id', 'question', 'answer', 'images']}
    for name, data in [('merged', unique), ('train', train), ('test', test)]:
        write_jsonl(output / (name + '.jsonl'), [plain(r) for r in data])
    split_map = {r['id']: split for split, rows in [('train', train), ('test', test)] for r in rows}
    write_jsonl(output / 'audit/provenance.jsonl', [{'id': r['id'], 'split': split_map[r['id']], 'source': r['_source'], **r['_provenance']} for r in unique])
    write_jsonl(output / 'audit/excluded_rl_records.jsonl', excluded)
    write_jsonl(output / 'audit/duplicate_records.jsonl', duplicate_records)
    save_json(output / 'audit/image_manifest.json', image_manifest)
    state = json.loads((output / 'audit/sft_preparation.json').read_text())
    summary = {**{k: v for k, v in state.items() if k != 'input_files'}, 'rl_upstream_rows': index + 1, 'rl_revision': metadata['sha'],
               'merged_tasks': len(unique), 'train_tasks': len(train), 'test_tasks': len(test),
               'by_source': dict(collections.Counter(r['_source'] for r in unique)),
               'train_by_source': dict(collections.Counter(r['_source'] for r in train)), 'test_by_source': dict(collections.Counter(r['_source'] for r in test)),
               'excluded_rl_reasons': dict(collections.Counter(r['reason'] for r in excluded)), 'exact_duplicates_removed': len(duplicate_records),
               'image_files': len(image_manifest), 'image_bytes': sum(im['bytes'] for im in image_manifest.values()),
               'multi_image_tasks': sum(len(r['images']) > 1 for r in unique),
               'file_sha256': {name: sha(output / name) for name in ['merged.jsonl', 'train.jsonl', 'test.jsonl']},
               'split_method': 'seed=42; connected components of shared initial RGB image pixels; approximately 90/10',
               'limitations': ['No independent factual verification of upstream answers.', 'SFT unseen refers to this SFT training file only, not base-model pretraining.',
                               'Overlap checks use known original/preview hashes and exact decoded pixels; arbitrary lossy or semantic near-duplicates are not guaranteed.',
                               'SFT val contains official benchmark-derived questions; those used here become training/development data and must not be reported as untouched benchmark test questions.']}
    save_json(output / 'audit/summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['sft', 'merge'], required=True)
    p.add_argument('--project-root', type=Path)
    p.add_argument('--sft-root', type=Path)
    p.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    if args.stage == 'sft':
        if not args.project_root or not args.sft_root:
            p.error('sft stage requires --project-root and --sft-root')
        prepare_sft(args.project_root.resolve(), args.sft_root.resolve(), args.output_root.resolve())
    else:
        merge(args.output_root.resolve())


if __name__ == '__main__':
    main()
