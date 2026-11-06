"""Run base165 with a frozen snapshot of its native six-tool GRPO agent."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluation import server_benchmark as base
from evaluation.common import write_json_atomic
from evaluation.grpo_native import runtime_identity
from evaluation.merge import merge_shards
from evaluation.prepare import digest
from evaluation.remote import output_lock
from evaluation.run import fingerprint

OUTPUT_NAME = 'base165-native-training-778-20261103'


def snapshot_runtime(output):
    source = base.TRAINING / 'datasets/grpo/integration'
    target = output / 'runtime'
    manifest = output / 'runtime_snapshot.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if runtime_identity(target) != saved['sha256']:
            raise ValueError('Frozen native runtime changed')
        return target
    identity = runtime_identity(source)
    for relative, expected in identity.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) != expected:
            raise ValueError('An unfinished runtime copy differs; inspect before continuing')
        shutil.copy2(source / relative, destination)
    if runtime_identity(target) != identity:
        raise ValueError('Native runtime snapshot verification failed')
    write_json_atomic(manifest, {'source': str(source), 'sha256': identity})
    return target


def native_commands(args, runtime, output, devices, limit=None):
    result = []
    for index, device in enumerate(devices):
        cmd = [sys.executable, '-u', '-m', 'evaluation.grpo_native', '--model', str(base.MODEL),
               '--runtime', str(runtime), '--tasks', str(base.BUNDLE/'tasks.jsonl'),
               '--output', str(output/'shards'/f'{index:03d}'), '--env-file', str(base.ENV_FILE),
               '--num-shards', str(len(devices)), '--shard-index', str(index),
               '--expected-count', '778', '--max-steps', str(args.max_steps),
               '--prompt-tokens', str(args.prompt_tokens), '--max-new-tokens', '4096',
               '--temperature', str(args.temperature), '--seed', '42', '--max-pixels', '262144']
        if limit is not None:
            cmd += ['--limit', str(limit)]
        if args.retry_infrastructure_errors:
            cmd.append('--retry-infrastructure-errors')
        result.append((device, cmd))
    return result


def run_workers(commands, output, env):
    output.mkdir(parents=True, exist_ok=True)
    processes, streams = [], []
    try:
        for index, (device, command) in enumerate(commands):
            stream = (output/f'worker-{index:03d}.log').open('a')
            streams.append(stream)
            child_env = dict(env, CUDA_VISIBLE_DEVICES=device, PYTHONHASHSEED='42',
                             CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
            processes.append(subprocess.Popen(command, cwd=base.ROOT, env=child_env,
                                               stdout=stream, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Native worker failed; inspect its own worker log')
            time.sleep(2)
        if any(p.returncode for p in processes):
            raise RuntimeError('Native worker failed')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in streams:
            stream.close()


def validate_merged(output, expected):
    merged = output / 'merged'
    count = merge_shards(output/'shards', merged, base.BUNDLE/'tasks.jsonl')
    if count != expected:
        raise ValueError('Native inference count differs')
    run = json.loads((merged/'run.json').read_text())
    if run['protocol']['version'] != 'grpo-native-agent-evaluation-v1' or len(run['protocol']['tools']) != 6:
        raise ValueError('This is not the native six-tool agent')
    return run


def score(args, output, env, judge_key):
    run = validate_merged(output/'full', 778)
    cmd = [sys.executable, '-u', '-m', 'evaluation.score', '--run', str(output/'full/merged'),
           '--tasks', str(base.BUNDLE/'tasks.jsonl'), '--references', str(base.BUNDLE/'references.jsonl'),
           '--output', str(output/'scores'), '--judge-model', env['JUDGE_MODEL'],
           '--base-url', env['JUDGE_BASE_URL'], '--key-env', judge_key, '--env-file', str(base.ENV_FILE),
           '--judge-thinking', 'disabled', '--judge-max-tokens', '4096', '--workers', '1']
    subprocess.run(cmd, cwd=base.ROOT, env=env, check=True)
    summary = json.loads((output/'scores/summary.json').read_text())
    if summary.get('count') != 778 or summary.get('scoring_scope'):
        raise ValueError('Native scores incomplete')
    write_json_atomic(output/'SERVER_SUCCESS.json', {
        'checkpoint_step': 165, 'count': 778, 'native_protocol_sha256': run['protocol_sha256'],
        'metric': base.METRIC, 'accuracy_pct': summary['micro_accuracy_pct'], 'completed_at': time.time()})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'infer', 'score', 'all'], default='prepare')
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--prompt-tokens', type=int, default=28672)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--retry-infrastructure-errors', action='store_true')
    args = parser.parse_args()
    if min(args.max_steps, args.prompt_tokens) < 1 or args.temperature < 0:
        parser.error('Budgets must be positive and temperature nonnegative')
    output = base.inside_personal(base.ROOT/'outputs'/OUTPUT_NAME)
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        verified = base.validate_bundle()
        base.validate_model()
        env, judge_key = base.runtime_environment()
        runtime = snapshot_runtime(output)
        plan = {'schema': 'native-grpo-778-v1', 'model': str(base.MODEL),
                'checkpoint_step': 165, 'bundle_verified': verified, 'native_runtime': runtime_identity(runtime),
                'max_steps': args.max_steps, 'prompt_tokens': args.prompt_tokens, 'max_new_tokens': 4096,
                'max_pixels': 262144, 'image_side': 1536, 'temperature': args.temperature, 'seed': 42,
                'inference_backend': 'transformers-sdpa-bfloat16',
                'scope': 'native GRPO prompt, tools, state, parser and action loop with explicit evaluation budgets',
                'reference_visibility': 'scorer only; no references passed to the native agent',
                'scoring': {'metric': base.METRIC, 'model': env['JUDGE_MODEL'], 'base_url': env['JUDGE_BASE_URL'],
                            'thinking': 'disabled', 'max_tokens': 4096, 'answer_extraction': 'native_answered_status'},
                'code_sha256': {str(p.relative_to(base.ROOT)): digest(p) for p in
                                [Path(__file__), base.ROOT/'evaluation/grpo_native.py', base.ROOT/'evaluation/run.py',
                                 base.ROOT/'evaluation/score.py', base.ROOT/'evaluation/common.py', base.ROOT/'evaluation/merge.py']}}
        plan_file = output/'plan.json'
        if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
            raise ValueError('Native benchmark settings changed; preserve old run and choose a new experiment')
        write_json_atomic(plan_file, plan)
        if args.stage == 'prepare':
            print('Native plan verified. No inference or external judge requests made.', flush=True)
            return
        if args.stage in {'infer', 'all'}:
            if not os.environ.get('SLURM_JOB_ID') or os.environ.get('VS_BENCHMARK_IN_CONTAINER') != '1':
                raise ValueError('Native inference needs its own allocated GPU container job')
            import torch
            devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
            if not torch.cuda.is_available() or not all(devices) or len(devices) != torch.cuda.device_count():
                raise ValueError('Allocated GPUs are unavailable or inconsistent')
            write_json_atomic(output/'hardware.json', {'job_id': os.environ['SLURM_JOB_ID'],
                              'host': os.uname().nodename, 'gpu_names': [torch.cuda.get_device_name(i) for i in range(len(devices))]})
            if not (output/'SMOKE_SUCCESS.json').exists():
                commands = native_commands(args, runtime, output/'smoke', devices[:1], limit=2)
                run_workers(commands, output/'smoke', env)
                validate_merged(output/'smoke', 2)
                write_json_atomic(output/'SMOKE_SUCCESS.json', {'count': 2, 'plan_sha256': fingerprint(plan)})
                print('NATIVE_SMOKE_SUCCESS 2/2; no judge requests made.', flush=True)
            commands = native_commands(args, runtime, output/'full', devices)
            run_workers(commands, output/'full', env)
            run = validate_merged(output/'full', 778)
            write_json_atomic(output/'INFERENCE_SUCCESS.json', {
                'count': 778, 'protocol_sha256': run['protocol_sha256'], 'completed_at': time.time()})
            print('NATIVE_INFERENCE_SUCCESS 778/778; scoring is a separate stage.', flush=True)
        if args.stage in {'score', 'all'}:
            score(args, output, env, judge_key)


if __name__ == '__main__':
    main()
