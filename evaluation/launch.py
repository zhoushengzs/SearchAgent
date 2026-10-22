"""Run one independent full-model replica per GPU, then validate and merge."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from evaluation.common import load_tasks, materialize_task_paths
from evaluation.merge import merge_shards

ROOT = Path(__file__).resolve().parents[1]


def worker_commands(a):
    devices = [part.strip() for part in a.gpus.split(',')]
    if len(set(devices)) != len(devices) or not all(re.fullmatch(r'\d+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/.-]+', part) for part in devices):
        raise ValueError('--gpus must list distinct GPU numbers or UUIDs')
    model_path = Path(a.model).expanduser()
    model = str(model_path.resolve()) if model_path.exists() else a.model
    common = ['--model', model, '--tasks', str(a.tasks.resolve()), '--device', 'cuda',
              '--num-shards', str(len(devices)), '--expected-count', str(a.expected_count),
              '--env-file', str(a.env_file.resolve())]
    for name in ['mode', 'protocol', 'dtype', 'max_steps', 'max_new_tokens', 'context_tokens', 'max_pixels', 'image_side', 'repetition_penalty', 'repeat_limit']:
        common += ['--' + name.replace('_', '-'), str(getattr(a, name))]
    for name in ['limit', 'image_root', 'original_root']:
        if getattr(a, name) is not None:
            value = getattr(a, name)
            if name == 'image_root':
                value = Path(value).resolve()
            common += ['--' + name.replace('_', '-'), str(value)]
    for name in ['retry_infrastructure_errors', 'allow_download']:
        if getattr(a, name):
            common.append('--' + name.replace('_', '-'))
    commands = []
    for index, device in enumerate(devices):
        output = a.output.resolve() / 'shards' / f'{index:03d}'
        command = [sys.executable, '-u', '-m', 'evaluation.run', *common,
                   '--shard-index', str(index), '--output', str(output)]
        commands.append((device, command))
    return commands


def launch_workers(a, commands):
    a.output.mkdir(parents=True, exist_ok=True)
    # Prevent accidentally starting a second launcher against the same outputs.
    with (a.output / '.launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another launcher is already using this output') from exc
        logs = a.output / 'logs'
        logs.mkdir(exist_ok=True)
        processes, streams = [], []
        try:
            for index, (device, command) in enumerate(commands):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, TOKENIZERS_PARALLELISM='false')
                env.setdefault('OMP_NUM_THREADS', '2')
                stream = (logs / f'shard-{index:03d}.log').open('ab')
                streams.append(stream)
                processes.append(subprocess.Popen(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
                print(f'Started worker {index} on GPU {device}; log: {logs / f"shard-{index:03d}.log"}', flush=True)
            while any(process.poll() is None for process in processes):
                failed = [index for index, process in enumerate(processes) if process.poll() not in (None, 0)]
                if failed:
                    raise ValueError(f'Workers failed: {failed}. See logs; completed records are resumable.')
                time.sleep(1)
            if any(process.returncode != 0 for process in processes):
                raise ValueError('A worker failed. Inspect logs before resuming.')
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for stream in streams:
                stream.close()
        count = merge_shards(a.output / 'shards', a.output / 'merged', a.tasks)
        print(f'Complete: {count} tasks; merged output: {(a.output / "merged").resolve()}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES') or '0,1,2,3,4,5,6,7')
    p.add_argument('--expected-count', type=int, help='Optional extra count check; portable bundles declare their full count')
    p.add_argument('--mode', choices=['agent', 'direct'], default='agent')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--max-steps', type=int, default=10)
    p.add_argument('--protocol', choices=['eval', 'training'], default='training')
    p.add_argument('--repetition-penalty', type=float, default=1.0)
    p.add_argument('--repeat-limit', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--context-tokens', type=int, default=32768)
    p.add_argument('--max-pixels', type=int, default=262144)
    p.add_argument('--image-side', type=int, default=1536)
    p.add_argument('--limit', type=int, help='Same global prefix for both models; use 16 for an eight-GPU smoke run')
    p.add_argument('--image-root', type=Path)
    p.add_argument('--original-root', type=Path)
    p.add_argument('--env-file', type=Path, default=ROOT / '.env')
    p.add_argument('--retry-infrastructure-errors', action='store_true')
    p.add_argument('--allow-download', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Print assignments and commands without loading models')
    a = p.parse_args()
    if (a.expected_count is not None and a.expected_count < 1) or (a.limit is not None and a.limit < 1) or a.max_steps < 1 or a.max_new_tokens < 1 or a.context_tokens <= a.max_new_tokens or not 64 <= a.image_side <= 2048 or a.max_pixels < 1:
        p.error('Invalid task count or model budget')
    if not 0.9 <= a.repetition_penalty <= 2.0 or a.repeat_limit < 0:
        p.error('repetition_penalty must be 0.9..2.0 and repeat_limit nonnegative')
    try:
        tasks = load_tasks(a.tasks, a.expected_count)
        a.expected_count = len(tasks)
        if a.limit is not None:
            tasks = tasks[:a.limit]
        commands = worker_commands(a)
        if a.dry_run:
            for index, (device, command) in enumerate(commands):
                print(f'GPU {device}: {len(tasks[index::len(commands)])} tasks')
                print('CUDA_VISIBLE_DEVICES=' + shlex.quote(device) + ' ' + shlex.join(command))
            return
        # Catch missing/moved/corrupt images before loading eight model copies.
        for task in tasks:
            materialize_task_paths(task, a.tasks, a.image_root, a.original_root)
        from evaluation.run import load_env
        load_env(a.env_file)
        if a.mode == 'agent' and not any(os.environ.get(key) for key in ['SERPER_API_KEY', *[f'SERPER_API_KEY_{i}' for i in range(1, 9)]]):
            raise ValueError('Set a real Serper API key in the server environment or --env-file')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < len(commands):
            raise ValueError(f'Requires {len(commands)} visible CUDA GPUs')
        launch_workers(a, commands)
    except (ValueError, KeyError, OSError) as exc:
        p.error(str(exc))
    except KeyboardInterrupt:
        raise SystemExit('Interrupted. Completed task records are retained; resume with the same command.')


if __name__ == '__main__':
    main()
