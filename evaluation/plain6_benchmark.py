"""Evaluate five conditions, including an ordinary-prompt tool control."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from evaluation import server_benchmark as base
from evaluation import unified_benchmark as parent
from evaluation.common import write_json_atomic
from evaluation.grpo_native import load_runtime, runtime_identity
from evaluation.merge import merge_shards
from evaluation.native_benchmark import run_workers
from evaluation.plain6_run import extension_identity, ordinary_system
from evaluation.prepare import digest
from evaluation.remote import output_lock
from evaluation.run import fingerprint
from evaluation.unified_common import CONDITIONS, LABELS, asset_identity, weight_identity
from evaluation.unified_run import code_identity

NAME = 'unified6_fiveway_778_20261103'
PLAIN = 'base_plain_tools'
ALL_CONDITIONS = (CONDITIONS[0], PLAIN, *CONDITIONS[1:])
ALL_LABELS = {**LABELS, PLAIN: 'Base＋普通工具提示词（6工具）'}


def link_once(target, source):
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError('Existing experiment link differs: ' + str(target))
    elif target.exists():
        raise ValueError('Refuse to replace an existing path: ' + str(target))
    else:
        target.symlink_to(source, target_is_directory=True)


def prepare(output, previous):
    prior = json.loads((previous / 'plan.json').read_text())
    if code_identity() != prior['code_sha256']:
        raise ValueError('The running four-condition evaluator changed')
    if asset_identity(previous / 'shared_assets') != prior['shared_assets']:
        raise ValueError('Shared processing/generation assets changed')
    if runtime_identity(previous / 'runtime') != prior['native_runtime']:
        raise ValueError('Native six-tool runtime changed')
    identity, _ = weight_identity(base.PERSONAL / 'Qwen3-VL-8B-Instruct')
    if identity != prior['weights']['base_native']['files']:
        raise ValueError('Base weights changed')
    for condition in CONDITIONS:
        link_once(output / condition, previous / condition)
    for name in ['evidence_cache', 'judge_cache']:
        (previous / name).mkdir(exist_ok=True)
        link_once(output / name, previous / name)
    _, session = load_runtime(previous / 'runtime')
    plan = {'schema': 'unified-six-tool-five-way-v1', 'conditions': list(ALL_CONDITIONS),
            'labels': ALL_LABELS, 'previous_output': str(previous),
            'previous_plan_sha256': digest(previous / 'plan.json'),
            'shared_code_sha256': prior['code_sha256'], 'extension_code_sha256': extension_identity(),
            'system': ordinary_system(session), 'weights': prior['weights']['base_native'],
            'generation': prior['generation'], 'scoring': prior['scoring'],
            'shared_assets': prior['shared_assets'], 'native_runtime': prior['native_runtime'],
            'bundle_verified': prior['bundle_verified'],
            'rules': 'ordinary saved system prompt; identical six schemas, task state, parser, native loop and generation; no legacy format retries or forced answers',
            'placement': 'additional dependent job; same gnode05 host, same two-GPU type and count; separate allocation',
            'reuse_historical_answers': False}
    path = output / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('The added condition changed; preserve this run')
    write_json_atomic(path, plan)
    return plan


def commands(output, previous, devices, limit=None, retry=False):
    phase = output / PLAIN / ('smoke' if limit is not None else 'full')
    cmds = []
    for index, device in enumerate(devices):
        cmd = [sys.executable, '-u', '-m', 'evaluation.plain6_run',
               '--model', str(base.PERSONAL / 'Qwen3-VL-8B-Instruct'),
               '--assets', str(previous / 'shared_assets'), '--runtime', str(previous / 'runtime'),
               '--tasks', str(base.BUNDLE / 'tasks.jsonl'),
               '--output', str(phase / 'shards' / f'{index:03d}'),
               '--evidence-cache', str(previous / 'evidence_cache'), '--env-file', str(base.ENV_FILE),
               '--mode', 'agent', '--num-shards', str(len(devices)), '--shard-index', str(index)]
        if limit is not None:
            cmd += ['--limit', str(limit)]
        if retry:
            cmd.append('--retry-infrastructure-errors')
        cmds.append((device, cmd))
    return phase, cmds


def compare_protocol(ordinary, native, extension):
    if ordinary['system'] == native['system']:
        raise ValueError('Ordinary and native prompts unexpectedly match')
    if ordinary['code_sha256'] != {**native['code_sha256'], **extension}:
        raise ValueError('The added worker does not use the same shared evaluator')
    a = {k: v for k, v in ordinary.items() if k not in {'system', 'code_sha256'}}
    b = {k: v for k, v in native.items() if k not in {'system', 'code_sha256'}}
    if a != b:
        raise ValueError('Ordinary and native conditions differ besides the system prompt')


def verify(output, previous, phase, count):
    root = output / PLAIN / phase
    if merge_shards(root / 'shards', root / 'merged', base.BUNDLE / 'tasks.jsonl') != count:
        raise ValueError('The additional condition is incomplete')
    run = json.loads((root / 'merged/run.json').read_text())
    native = json.loads((previous / 'base_native' / phase / 'merged/run.json').read_text())
    compare_protocol(run['protocol'], native['protocol'], extension_identity())
    if run['model'] != native['model'] or run['model_identity'] != native['model_identity']:
        raise ValueError('The prompt comparison uses different Base weights')
    if phase == 'smoke':
        for task in run['protocol']['selected_ids']:
            inputs = []
            for condition_root in [output / PLAIN, previous / 'base_native']:
                matches = list((condition_root / 'smoke/shards').glob(f'*/artifacts/{task}/raw_calls.jsonl'))
                if len(matches) != 1:
                    raise ValueError('Missing unique initial model input')
                messages = json.loads(matches[0].read_text().splitlines()[0])['messages']
                normalized = []
                for message in messages[1:]:
                    row = {k: v for k, v in message.items() if k != 'images'}
                    if 'images' in message:
                        row['images'] = [image['sha256'] for image in message['images']]
                    normalized.append(row)
                inputs.append(normalized)
            if inputs[0] != inputs[1]:
                raise ValueError('Actual initial state/images differ between prompt conditions')
    return run


def score_all(output, plan, env, key):
    # Reuse the exact same scoring implementation/policy for every condition.
    # All changes below are local to this scorer process and only extend the
    # condition list and report labels; no inference files are rewritten.
    from evaluation import unified_score as scorer
    original = (scorer.CONDITIONS, scorer.LABELS, scorer.render_results, scorer.write_json_atomic, sys.argv)
    render, write = scorer.render_results, scorer.write_json_atomic
    def render_five(directory, summaries):
        render(directory, summaries)
        path = directory / '.codex/RESULTS.md'
        path.write_text(path.read_text().replace('四组', '五组'))
        summary = json.loads((directory / 'summary.json').read_text())
        summary['schema'] = 'unified-six-tool-five-way-v1'
        write(directory / 'summary.json', summary)
    def write_five(path, data):
        if Path(path) == output / 'SERVER_SUCCESS.json':
            data = {**data, 'conditions': len(ALL_CONDITIONS), 'plan_sha256': fingerprint(plan)}
        write(path, data)
    scorer.CONDITIONS, scorer.LABELS = ALL_CONDITIONS, ALL_LABELS
    scorer.render_results, scorer.write_json_atomic = render_five, write_five
    original_key = os.environ.get(key)
    os.environ[key] = env[key]
    sys.argv = ['unified_score', '--output', str(output), '--tasks', str(base.BUNDLE / 'tasks.jsonl'),
                '--references', str(base.BUNDLE / 'references.jsonl'),
                '--env-file', str(base.ENV_FILE), '--key-env', key]
    try:
        scorer.main()
    finally:
        scorer.CONDITIONS, scorer.LABELS, scorer.render_results, scorer.write_json_atomic, sys.argv = original
        if original_key is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'infer', 'score'], default='prepare')
    parser.add_argument('--retry-infrastructure-errors', action='store_true')
    args = parser.parse_args()
    output = base.inside_personal(base.ROOT / 'outputs' / NAME)
    previous = base.inside_personal(base.ROOT / 'outputs' / parent.NAME)
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        plan = prepare(output, previous)
        if args.stage == 'prepare':
            print('FIVE_CONDITION_PLAN_VERIFIED; original four-condition code/results unchanged.', flush=True)
            return
        if not (previous / 'INFERENCE_SUCCESS.json').is_file():
            raise ValueError('Wait for all four existing conditions to finish successfully')
        env, key = base.runtime_environment()
        if args.stage == 'score':
            parent.verify_comparison(previous, 'full', 778)
            verify(output, previous, 'full', 778)
            score_all(output, plan, env, key)
            return
        if not os.environ.get('SLURM_JOB_ID') or os.environ.get('VS_BENCHMARK_IN_CONTAINER') != '1':
            raise ValueError('The added condition must run in its allocated GPU container')
        import torch
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        if not torch.cuda.is_available() or not all(devices) or len(devices) != torch.cuda.device_count():
            raise ValueError('Allocated GPUs unavailable')
        hardware = {'job_id': os.environ['SLURM_JOB_ID'], 'host': os.uname().nodename,
                    'gpu_names': [torch.cuda.get_device_name(i) for i in range(len(devices))],
                    'same_allocation_as_first_four': False}
        previous_hardware = json.loads((previous / 'hardware.json').read_text())
        if hardware['gpu_names'] != previous_hardware['gpu_names'] or hardware['host'] != previous_hardware['host']:
            raise ValueError('The added condition needs the same GPU count/type and host')
        write_json_atomic(output / 'hardware.json', hardware)
        if not (output / PLAIN / 'SMOKE_SUCCESS.json').exists():
            phase, cmds = commands(output, previous, devices[:1], limit=2, retry=args.retry_infrastructure_errors)
            run_workers(cmds, phase, env)
            verify(output, previous, 'smoke', 2)
            write_json_atomic(output / PLAIN / 'SMOKE_SUCCESS.json',
                              {'count': 2, 'only_system_prompt_differs': True, 'plan_sha256': fingerprint(plan)})
            print('PLAIN_SIX_TOOL_SMOKE_SUCCESS', flush=True)
        phase, cmds = commands(output, previous, devices, retry=args.retry_infrastructure_errors)
        run_workers(cmds, phase, env)
        run = verify(output, previous, 'full', 778)
        write_json_atomic(output / PLAIN / 'INFERENCE_SUCCESS.json', {'count': 778, 'run_id': run['run_id']})
        parent.verify_comparison(previous, 'full', 778)
        write_json_atomic(output / 'INFERENCE_SUCCESS.json',
                          {'conditions': 5, 'count_each': 778, 'completed_at': time.time()})
        print('FIVE_CONDITION_INFERENCE_SUCCESS 5 x 778; no historical answers reused.', flush=True)


if __name__ == '__main__':
    main()
