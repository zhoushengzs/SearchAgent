"""Check every final row and image without loading a model or calling a tool."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def check(root):
    summary = json.loads((root / 'audit/summary.json').read_text())
    manifest = json.loads((root / 'audit/image_manifest.json').read_text())
    exclusions = json.loads((root / 'audit/exclusion_keys.json').read_text())
    sft_hashes = set(exclusions['sft_image_sha256'])
    frozen_hashes = set(exclusions['frozen_image_sha256'])
    guard_path = root / 'audit/benchmark_test_guard.json'
    guard = json.loads(guard_path.read_text()) if guard_path.exists() else {}
    forbidden_pixels = set(guard.get('benchmark_decoded_image_sha256', []))
    image_sets = {}
    id_sets = {}
    records = {}
    count = {}
    used_paths = set()
    for split in ['merged', 'train', 'test']:
        rows = [json.loads(l) for l in (root / (split + '.jsonl')).open() if l.strip()]
        ids, pixels = set(), set()
        for row in rows:
            if set(row) != {'id', 'question', 'answer', 'images'}:
                raise ValueError('Unexpected fields in final training data')
            if row['id'] in ids or not row['question'].strip() or not row['answer'].strip():
                raise ValueError('Duplicate ID/empty question/empty answer')
            ids.add(row['id'])
            if split == 'merged':
                records[row['id']] = row
            elif records.get(row['id']) != row:
                raise ValueError('Train/test record differs from merged data')
            for relative in row['images']:
                path = (root / relative).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError('Image is missing or outside the portable bundle')
                if relative not in manifest:
                    raise ValueError('Image is not in verified manifest')
                used_paths.add(relative)
                pixels.add(manifest[relative]['pixel_sha256'])
                if manifest[relative]['pixel_sha256'] in forbidden_pixels:
                    raise ValueError('Frozen benchmark image pixels appear in final data')
                if manifest[relative]['sha256'] in sft_hashes or manifest[relative]['preview_sha256'] in sft_hashes:
                    raise ValueError('Known SFT image overlap in final data')
                if manifest[relative]['sha256'] in frozen_hashes or manifest[relative]['preview_sha256'] in frozen_hashes:
                    raise ValueError('Frozen evaluation image overlap in final data')
        id_sets[split], image_sets[split], count[split] = ids, pixels, len(rows)
        if sha(root / (split + '.jsonl')) != summary['file_sha256'][split + '.jsonl']:
            raise ValueError('Output file checksum differs')
    if id_sets['train'] & id_sets['test'] or id_sets['train'] | id_sets['test'] != id_sets['merged']:
        raise ValueError('Train/test ID overlap or missing records')
    if image_sets['train'] & image_sets['test']:
        raise ValueError('Train/test images overlap')
    for n, relative in enumerate(sorted(used_paths), 1):
        path = root / relative
        if sha(path) != manifest[relative]['sha256']:
            raise ValueError('Image bytes changed: ' + relative)
        with Image.open(path) as image:
            image.load()
            rgb = image.convert('RGB')
            fingerprint = hashlib.sha256(str(rgb.size).encode() + b'\0' + rgb.tobytes()).hexdigest()
            if fingerprint != manifest[relative]['pixel_sha256']:
                raise ValueError('Decoded image pixels changed: ' + relative)
        if n % 2000 == 0:
            print(f'Checked final images: {n}/{len(used_paths)}', flush=True)
    expected = {'merged': summary['merged_tasks'], 'train': summary['train_tasks'], 'test': summary['test_tasks']}
    if count != expected:
        raise ValueError('Output row counts differ from summary')
    report = {'passed': True, 'counts': count, 'used_image_files': len(used_paths), 'train_test_shared_ids': 0,
              'train_test_shared_decoded_images': 0, 'all_used_images_decoded_and_checksum_verified': True,
              'all_rows_have_only_official_input_fields': True, 'known_sft_or_frozen_evaluation_image_overlaps': 0}
    (root / 'audit/final_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    check(args.root.resolve())
