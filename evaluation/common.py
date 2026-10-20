"""Portable task inputs, deterministic sharding and strict prediction validation."""
from __future__ import annotations

import json
import re
from pathlib import Path
import tempfile

from core.provenance import file_sha, fingerprint as content_fingerprint


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def task_fingerprint(task):
    return content_fingerprint({
        'question': task['model_input']['question'],
        'images': [{k: v for k, v in image.items() if k != 'path'}
                   for image in task['model_input']['images']],
        'dataset_provenance': task['dataset_provenance'],
    })


def load_tasks(path, expected_count=None, limit=None):
    tasks = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                tasks.append(json.loads(line))
    if expected_count is not None and len(tasks) != expected_count:
        raise ValueError(f'Expected {expected_count} tasks, found {len(tasks)}')
    ids = set()
    forbidden = {'reference', 'answer', 'answers', 'answer_candidates', 'conversations',
                 'steps', 'search_trace', 'execution', 'annotation', 'initial_views',
                 'teacher_decision_json', 'teacher_steps', 'tool_results'}
    for task in tasks:
        key = task['id']
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', key) or key in {'.', '..'} or key in ids:
            raise ValueError('Task IDs must be unique safe filenames')
        ids.add(key)
        if forbidden & task.keys():
            raise ValueError(f'{key}: task input contains answers or historical execution')
        model_input = task['model_input']
        if set(model_input) != {'question', 'images'} or not isinstance(model_input['question'], str) or not model_input['question'].strip():
            raise ValueError(f'{key}: supply only original question and initial images')
        for index, image in enumerate(model_input['images'], 1):
            if set(image) != {'index', 'path', 'sha256'} or image['index'] != index or not isinstance(image['path'], str):
                raise ValueError(f'{key}: invalid initial image')
            if not re.fullmatch(r'[a-f0-9]{64}', image['sha256']):
                raise ValueError(f'{key}: invalid image SHA256')
        if task_fingerprint(task) != task['input_fingerprint']:
            raise ValueError(f'{key}: original task fingerprint changed')
    if not tasks:
        raise ValueError('Empty task manifest')
    return tasks[:limit] if limit is not None else tasks


def resolve_image(path, tasks_path, image_root=None, original_root=None):
    path = Path(path).expanduser()
    if path.is_absolute():
        if original_root is not None:
            if image_root is None:
                raise ValueError('--original-root requires --image-root')
            try:
                relative = path.relative_to(Path(original_root))
            except ValueError as exc:
                raise ValueError('Image is outside --original-root') from exc
            return (Path(image_root) / relative).resolve()
        return path.resolve()
    base = Path(image_root) if image_root is not None else Path(tasks_path).resolve().parent
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError('Relative image path escapes the image root')
    return resolved


def materialize_task_paths(task, tasks_path, image_root=None, original_root=None, check_images=True):
    # Preserve the task identity: paths are excluded from input_fingerprint.
    result = {**task, 'model_input': {**task['model_input'], 'images': []}}
    for image in task['model_input']['images']:
        path = resolve_image(image['path'], tasks_path, image_root, original_root)
        if not path.is_file():
            raise ValueError(f'{task["id"]}: missing image {path}')
        if check_images and file_sha(path) != image['sha256']:
            raise ValueError(f'{task["id"]}: image SHA256 differs')
        result['model_input']['images'].append({**image, 'path': str(path)})
    return result


def shard_tasks(tasks, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError('Invalid shard index/count')
    return tasks[index::count]


def read_predictions(path, run_id, expected, recover_tail=False):
    """Last infrastructure retry wins; reject mixed runs, IDs and fingerprints."""
    path = Path(path)
    if not path.exists():
        return {}
    latest = {}
    offset = 0
    size = path.stat().st_size
    final_line = b''
    with path.open('rb') as source:
        while line := source.readline():
            final_line = line
            if not line.strip():
                offset += len(line)
                continue
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if not recover_tail or source.tell() != size:
                    raise ValueError('Malformed prediction JSON; only an interrupted final line is recoverable')
                backup = path.with_name(path.name + '.interrupted-tail')
                if backup.exists():
                    backup = path.with_name(path.name + '.interrupted-tail.' + str(size))
                backup.write_bytes(line)
                with path.open('r+b') as stream:
                    stream.truncate(offset)
                return latest
            key = record['id']
            task = expected.get(key)
            if task is None or record.get('run_id') != run_id:
                raise ValueError('Predictions contain an unexpected task or model run')
            if record.get('input_fingerprint') != task['input_fingerprint'] or record.get('source_dataset') != task['source_dataset']:
                raise ValueError('Prediction task identity differs')
            if key in latest and not latest[key].get('infrastructure_error'):
                raise ValueError('Repeated prediction for a task that was already completed')
            latest[key] = record
            offset += len(line)
    if recover_tail and final_line and not final_line.endswith(b'\n'):
        with path.open('ab') as stream:
            stream.write(b'\n')
    return latest
