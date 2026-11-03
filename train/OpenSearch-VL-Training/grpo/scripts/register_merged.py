"""Register portable merged JSONL with the real upstream DatasetRegistry."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def load_registry(root, rllm_root=None):
    # Load the actual, unchanged registry file without importing unrelated
    # agent/search packages through rllm.__init__ during CPU-only registration.
    module_path = (rllm_root / 'rllm/data/dataset.py') if rllm_root else (root / 'upstream/OpenSearch-VL/RL/rllm/rllm/data/dataset.py')
    if not module_path.is_file():
        raise FileNotFoundError('DatasetRegistry source not found: ' + str(module_path))
    spec = importlib.util.spec_from_file_location('grpo_upstream_dataset', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DatasetRegistry


def bind_registry(cls, root):
    cls._REGISTRY_DIR = str(root / 'registry')
    cls._REGISTRY_FILE = str(root / 'registry/dataset_registry.json')
    cls._DATASET_DIR = str(root / 'registered')


def register(root, rllm_root=None):
    guard = json.loads((root / 'audit/benchmark_test_guard.json').read_text())
    forbidden_pixels = set(guard['benchmark_decoded_image_sha256'])
    forbidden_ids = set(guard['benchmark_task_ids'])
    manifest = json.loads((root / 'audit/image_manifest.json').read_text())
    provenance = {r['id']: r for r in map(json.loads, (root / 'audit/provenance.jsonl').open())}
    summary = json.loads((root / 'audit/summary.json').read_text())
    cls = load_registry(root, rllm_root)
    bind_registry(cls, root)
    results = {}
    verified_images = set()
    for split in ['train', 'test']:
        checksum = hashlib.sha256((root / (split + '.jsonl')).read_bytes()).hexdigest()
        if checksum != summary['file_sha256'][split + '.jsonl']:
            raise ValueError('Input file changed after separation checks: ' + split)
        records = []
        with (root / (split + '.jsonl')).open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if provenance[item['id']].get('original_task_id') in forbidden_ids:
                    raise ValueError('Benchmark test question cannot enter the GRPO registry')
                images = []
                for relative in item['images']:
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root) or not path.is_file():
                        raise ValueError('Image path missing/outside bundle: ' + relative)
                    if manifest[relative]['pixel_sha256'] in forbidden_pixels:
                        raise ValueError('Benchmark test image cannot enter the GRPO registry')
                    if relative not in verified_images:
                        image_hash = hashlib.sha256()
                        with path.open('rb') as stream:
                            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                                image_hash.update(chunk)
                        if image_hash.hexdigest() != manifest[relative]['sha256']:
                            raise ValueError('Image changed after benchmark separation checks: ' + relative)
                        verified_images.add(relative)
                    images.append(str(path))
                image_ids = ', '.join(str(i + 1) for i in range(len(images)))
                question = f"image_id: {image_ids} \n Question: {item['question']}" if images else item['question']
                records.append({'id': item['id'], 'question': question, 'answer': item['answer'], 'images': images})
        cls.register_dataset('Vision-DeepResearch-QA', records, split)
        loaded = cls.load_dataset('Vision-DeepResearch-QA', split)
        if loaded is None or len(loaded) != len(records):
            raise ValueError('Upstream registry readback count mismatch')
        for expected, actual in zip(records, loaded.get_data()):
            if expected != actual:
                raise ValueError('Upstream readback content mismatch: ' + expected['id'])
        import pandas as pd
        verl = pd.read_parquet(loaded.get_verl_data_path()).to_dict('records')
        if len(verl) != len(records):
            raise ValueError('Verl parquet count mismatch')
        for expected, actual in zip(records, verl):
            extra = actual['extra_info']
            if extra['id'] != expected['id'] or extra['question'] != expected['question'] or extra['answer'] != expected['answer'] or list(extra['images']) != expected['images']:
                raise ValueError('Verl parquet content mismatch')
        results[split] = {'rows': len(records), 'native_parquet': loaded.get_data_path(), 'verl_parquet': loaded.get_verl_data_path()}
    result = {'registry_name': 'Vision-DeepResearch-QA', 'registry_isolated_from_upstream_installation': True,
              'bundle_root': str(root), 'python_executable': sys.executable, 'verified_images': len(verified_images),
              'registry_source': str(rllm_root / 'rllm/data/dataset.py') if rllm_root else str(root / 'upstream/OpenSearch-VL/RL/rllm/rllm/data/dataset.py'),
              'verified_by': 'Unmodified upstream DatasetRegistry.register_dataset and load_dataset; full record readback', 'splits': results}
    (root / 'audit/registration_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--rllm-root', type=Path)
    args = parser.parse_args()
    register(args.root.resolve(), args.rllm_root.resolve() if args.rllm_root else None)
