"""Verify the uploaded frozen bundle, then run and score base165 on allocated GPUs.

This entrypoint deliberately retains the evaluator's existing inference and
scoring policy. It installs nothing and never prints credentials or references.
"""
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PERSONAL = Path('/cluster/home/shenqunlun/zhousheng')
TRAINING = PERSONAL / 'visual-search-agent'
ROOT = PERSONAL / 'benchmark-vs778'
DATA = TRAINING / 'datasets/grpo'
BUNDLE = ROOT / 'outputs/server_eval_778'
MODEL = PERSONAL / 'results/base_grpo/checkpoints/global_step_165/actor/huggingface'
ENV_FILE = DATA / 'integration/local_training.env'
STATE = DATA / '.codex/serper_key_state.json'
EXPECTED_MANIFEST_SHA256 = '29372ecd3937af72c2a45f5350bb0ff36663a3b5a57147bad583a89e625bcaa6'
EXPECTED_COUNT = 778
EXPECTED_IMAGES = 773
METRIC = 'project_reference_judge_accuracy_not_official_benchmark_score'


def inside_personal(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PERSONAL.resolve()):
        raise ValueError('A workflow path resolves outside the personal directory')
    return path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def validate_bundle():
    from evaluation.common import load_tasks
    from evaluation.prepare import rows
    for path in [ROOT, BUNDLE, MODEL, ENV_FILE, STATE]:
        inside_personal(path)
    if digest(BUNDLE / 'manifest.json') != EXPECTED_MANIFEST_SHA256:
        raise ValueError('Uploaded manifest differs from the frozen local bundle')
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    if (manifest.get('schema') != 'portable-frozen-evaluation-v1' or
            manifest.get('task_count') != EXPECTED_COUNT or manifest.get('image_count') != EXPECTED_IMAGES):
        raise ValueError('Wrong benchmark schema or count')
    for name, field in [('tasks.jsonl', 'tasks_sha256'), ('references.jsonl', 'references_sha256')]:
        if digest(BUNDLE / name) != manifest[field]:
            raise ValueError(f'Uploaded {name} SHA256 differs')
    tasks = load_tasks(BUNDLE / 'tasks.jsonl', EXPECTED_COUNT)
    task_ids = {(item['id'], item['input_fingerprint']) for item in tasks}
    references = list(rows(BUNDLE / 'references.jsonl'))
    reference_ids = [(item['id'], item['input_fingerprint']) for item in references]
    if len(reference_ids) != EXPECTED_COUNT or set(reference_ids) != task_ids:
        raise ValueError('Task/reference counts or identities differ')
    if dict(Counter(item['source_dataset'] for item in tasks)) != manifest['by_source']:
        raise ValueError('Per-source task counts differ')
    images = manifest['images']
    if len(images) != EXPECTED_IMAGES:
        raise ValueError('Image manifest count differs')
    referenced = set()
    for task in tasks:
        for image in task['model_input']['images']:
            info = images.get(image['sha256'])
            if info is None or image['path'] != info['path']:
                raise ValueError('Task image differs from the manifest')
            referenced.add(image['sha256'])
    if referenced != set(images):
        raise ValueError('Task images do not cover the frozen image manifest')
    total_bytes = 0
    for index, (expected_hash, info) in enumerate(images.items(), 1):
        path = inside_personal(BUNDLE / info['path'])
        if not path.is_relative_to(BUNDLE.resolve()):
            raise ValueError('Image path escapes the benchmark bundle')
        if path.stat().st_size != info['bytes'] or digest(path) != expected_hash:
            raise ValueError('Uploaded image size or SHA256 differs')
        total_bytes += info['bytes']
        if index % 100 == 0:
            print(f'Checked image SHA256: {index}/{EXPECTED_IMAGES}', flush=True)
    result = {'tasks': len(tasks), 'references': len(references), 'images': len(images),
              'image_bytes': total_bytes, 'manifest_sha256': EXPECTED_MANIFEST_SHA256,
              'tasks_sha256': manifest['tasks_sha256'], 'references_sha256': manifest['references_sha256']}
    print(json.dumps({'bundle_verified': result}), flush=True)
    return result


def validate_model():
    success = json.loads((PERSONAL / 'results/base_grpo/SUCCESS.json').read_text())
    if success.get('stage') != 'base' or success.get('smoke') is not False or success.get('step') != 165:
        raise ValueError('The completed formal base165 model is required')
    if Path(success['model']).resolve() != MODEL.resolve():
        raise ValueError('Base SUCCESS points to a different model')
    checkpoint = MODEL.parents[1]
    for path in [checkpoint / 'data.pt', checkpoint / 'actor/dist_ckpt/.metadata', MODEL / 'config.json']:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Base165 checkpoint is missing a required saved file')
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        path = inside_personal(MODEL / name)
        if not path.is_relative_to(MODEL.resolve()) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Base165 model shard is missing')


def runtime_environment():
    from training_control import environment
    from project_runtime.serper_keys import configured_keys
    env = environment(TRAINING / 'train/OpenSearch-VL-Training/RL/rllm')
    env.update(VS_SERPER_BACKEND='training', SERPER_KEY_STATE_FILE=str(STATE),
               PYTHONDONTWRITEBYTECODE='1')
    env.pop('VS_DEBUG', None)
    keys = configured_keys(env)
    if len(keys) < 212:
        raise ValueError('The complete training Serper pool is required')
    # Read under the same file lock used by live training; do not edit state.
    with STATE.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        try:
            state = json.load(stream)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    expected_pool = hashlib.sha256(json.dumps(keys).encode()).hexdigest()
    if state.get('pool') != expected_pool or state.get('key_count') != len(keys):
        raise ValueError('Training key configuration and shared state differ; do not reset them')
    if type(state.get('index')) is not int or not 0 <= state['index'] < len(keys):
        raise ValueError('Training search pool has no available key position')
    judge_key = 'JUDGE_API_KEY' if env.get('JUDGE_API_KEY') else 'OPENAI_API_KEY'
    if not env.get(judge_key) or not env.get('JUDGE_MODEL') or not env.get('JUDGE_BASE_URL'):
        raise ValueError('The existing training judge service is not fully configured')
    print(json.dumps({'search_key_position': state['index'] + 1, 'search_key_count': len(keys),
                      'judge_configured': True}), flush=True)
    return env, judge_key


def run_stage(label, count, gpus, env, judge_key):
    from evaluation.common import load_tasks, read_predictions
    output = inside_personal(ROOT / 'outputs' / f'base165-{label}')
    score_output = inside_personal(ROOT / 'outputs' / f'base165-{label}-score')
    command = [sys.executable, '-u', '-m', 'evaluation.launch', '--model', str(MODEL),
               '--tasks', str(BUNDLE / 'tasks.jsonl'), '--output', str(output), '--gpus', gpus,
               '--expected-count', str(EXPECTED_COUNT), '--mode', 'agent', '--protocol', 'training',
               '--dtype', 'bfloat16', '--max-steps', '10', '--max-new-tokens', '4096',
               '--context-tokens', '32768', '--max-pixels', '262144', '--image-side', '1536',
               '--no-tool-summaries', '--env-file', str(ENV_FILE)]
    if count != EXPECTED_COUNT:
        command += ['--limit', str(count)]
    if env.get('VS_BENCHMARK_RETRY_INFRASTRUCTURE') == '1':
        command += ['--retry-infrastructure-errors']
    print(f'Starting {label}: {count} tasks', flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    # The merger already rejects infrastructure errors; independently check
    # the selected count before spending judge requests or advancing to 778.
    merged = output / 'merged'
    run = json.loads((merged / 'run.json').read_text())
    tasks = load_tasks(BUNDLE / 'tasks.jsonl', EXPECTED_COUNT)[:count]
    predictions = read_predictions(merged / 'predictions.jsonl', run['run_id'], {t['id']: t for t in tasks})
    if len(predictions) != count or any(row.get('infrastructure_error') for row in predictions.values()):
        raise ValueError('Inference is incomplete or contains infrastructure errors')
    # Empty or incorrect final answers are valid evaluation outcomes, not a
    # reason to repeat generation. The unchanged scorer marks them incorrect.
    subprocess.run([sys.executable, '-u', '-m', 'evaluation.score', '--run', str(merged),
                    '--tasks', str(BUNDLE / 'tasks.jsonl'), '--references', str(BUNDLE / 'references.jsonl'),
                    '--output', str(score_output), '--judge-model', env['JUDGE_MODEL'],
                    '--base-url', env['JUDGE_BASE_URL'], '--key-env', judge_key,
                    '--env-file', str(ENV_FILE), '--workers', '1'], cwd=ROOT, env=env, check=True)
    summary = json.loads((score_output / 'summary.json').read_text())
    if summary.get('count') != count or summary.get('metric') != METRIC or summary.get('scoring_scope'):
        raise ValueError('Scoring did not produce a complete reference-judge summary')
    return summary


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_NUM_NODES') != '1':
        raise SystemExit('A single-node Slurm GPU allocation is required')
    if os.environ.get('VS_BENCHMARK_IN_CONTAINER') != '1':
        raise SystemExit('Use evaluation/server_benchmark.sh to enter the existing training container')
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit('Run from the personal benchmark directory')
    import torch
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpus or not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit('Allocated CUDA GPUs are unavailable')
    devices = gpus.split(',')
    if len(devices) != torch.cuda.device_count():
        raise SystemExit('Visible GPU mask and device count differ')
    output_root = inside_personal(ROOT / 'outputs')
    output_root.mkdir(exist_ok=True)
    with (output_root / '.base165-benchmark.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another base165 benchmark workflow is already running')
        verified = validate_bundle()
        validate_model()
        env, judge_key = runtime_environment()
        run_stage('smoke', 2, devices[0], env, judge_key)
        summary = run_stage('778', EXPECTED_COUNT, gpus, env, judge_key)
        from evaluation.common import write_json_atomic
        success_path = inside_personal(output_root / 'base165-778/SERVER_SUCCESS.json')
        write_json_atomic(success_path, {'model': str(MODEL), 'checkpoint_step': 165,
                          'job_id': os.environ['SLURM_JOB_ID'], 'bundle_verified': verified,
                          'metric': METRIC, 'count': summary['count'],
                          'accuracy_pct': summary['micro_accuracy_pct'], 'completed_at': time.time()})
        print(f'Benchmark and scoring complete: {success_path}', flush=True)


if __name__ == '__main__':
    main()
