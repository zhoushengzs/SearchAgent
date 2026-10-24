"""Run fixed evaluation tasks through OpenRouter using the existing SFT agent loop."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from evaluation.common import load_tasks, materialize_task_paths, read_predictions, write_json_atomic
from evaluation.prepare import digest
from evaluation.run import ROOT, SYSTEM, VALIDATION_ATTEMPTS, fingerprint, load_env, run_task, tool_definitions
from evaluation.protocol import training_system, training_tools
from evaluation.execution import execution_policy
from evaluation.remote import APIClient, OpenRouterModel, parallel_results, output_lock


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='qwen/qwen3-vl-8b-instruct',
                   choices=['qwen/qwen3-vl-8b-instruct'], help='Image token accounting currently targets this model')
    p.add_argument('--tokenizer', type=Path, required=True, help='Local base-model tokenizer folder; no weights loaded')
    p.add_argument('--tasks', type=Path, default=ROOT / 'outputs/local_benchmark_eval/data/tasks.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-count', type=int, help='Optional count check; portable bundles declare their full count')
    p.add_argument('--limit', type=int)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--provider', help='Optional exact OpenRouter provider slug; fixes routing via provider.only')
    p.add_argument('--mode', choices=['agent', 'direct'], default='agent')
    p.add_argument('--tool-summaries', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-steps', type=int, default=10)
    p.add_argument('--protocol', choices=['eval', 'training'], default='training')
    p.add_argument('--repetition-penalty', type=float, default=1.0)
    p.add_argument('--repeat-limit', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--context-tokens', type=int, default=32768)
    p.add_argument('--image-side', type=int, default=1536)
    p.add_argument('--max-pixels', type=int, default=262144)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--original-root', type=Path)
    p.add_argument('--env-file', type=Path, default=ROOT / '.env')
    p.add_argument('--retry-infrastructure-errors', action='store_true')
    p.add_argument('--check-only', action='store_true', help='Validate inputs and dependencies without API requests')
    return p


def main():
    p = parser()
    a = p.parse_args()
    if not 1 <= a.workers <= 64 or (a.expected_count is not None and a.expected_count < 1) or (a.limit is not None and a.limit < 1):
        p.error('Workers must be 1..64 and task counts positive')
    if a.max_steps < 1 or a.max_new_tokens < 1 or a.context_tokens <= a.max_new_tokens or not 64 <= a.image_side <= 2048 or a.max_pixels < 65536:
        p.error('Invalid step/token/image budget; Qwen image minimum is 65536 pixels')
    if not 0.9 <= a.repetition_penalty <= 2.0 or a.repeat_limit < 0:
        p.error('repetition_penalty must be 0.9..2.0 and repeat_limit nonnegative')
    load_env(a.env_file)
    if not a.check_only and not os.environ.get('OPENROUTER_API_KEY'):
        p.error('Missing OPENROUTER_API_KEY in environment or .env')
    search_ready = any(os.environ.get(k) for k in ['SERPER_API_KEY', *[f'SERPER_API_KEY_{i}' for i in range(1, 9)]])
    if not a.check_only and a.mode == 'agent' and not search_ready:
        p.error('Missing SERPER_API_KEY for real search')
    try:
        tasks = load_tasks(a.tasks, a.expected_count, a.limit)
        tasks = [materialize_task_paths(t, a.tasks, a.image_root, a.original_root) for t in tasks]
        config = json.loads((a.tokenizer / 'config.json').read_text())
        if config.get('model_type') != 'qwen3_vl':
            p.error('Use the Qwen3-VL base model tokenizer folder')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(a.tokenizer), local_files_only=True)
    except (OSError, ValueError, ImportError) as exc:
        p.error(f'Input/dependency check failed: {type(exc).__name__}: {exc}')
    schemas = [] if a.mode != 'agent' else (training_tools() if a.protocol == 'training' else tool_definitions())
    system = training_system() if a.protocol == 'training' else SYSTEM
    shared_files = {ROOT / 'evaluation/run.py', ROOT / 'evaluation/common.py', ROOT / 'evaluation/protocol.py', ROOT / 'evaluation/execution.py',
                    ROOT / 'evaluation/training_protocol/tools.json', ROOT / 'evaluation/training_protocol/system.txt',
                    ROOT / 'trajectory_sources.json',
                    ROOT / 'sources/registry.py', ROOT / 'sources/download.py',
                    *list((ROOT / 'agent').glob('*.py')), *list((ROOT / 'core').glob('*.py'))}
    protocol = {'version': 'openrouter-shared-agent-v4', 'mode': a.mode, 'device': 'openrouter',
                'dtype': 'provider_managed', 'system': system, 'tools': schemas,
                'protocol': a.protocol, 'repetition_penalty': a.repetition_penalty,
                'repeat_limit': a.repeat_limit, 'validation_attempts': VALIDATION_ATTEMPTS,
                'execution_policy': execution_policy(a.tool_summaries),
                'tasks_sha256': digest(a.tasks), 'selected_ids': [t['id'] for t in tasks],
                **{k: getattr(a, k) for k in ['max_steps', 'max_new_tokens', 'context_tokens', 'max_pixels', 'image_side']},
                'code_sha256': {str(f.relative_to(ROOT)): digest(f) for f in sorted(shared_files)},
                'versions': {name: importlib.metadata.version(name) for name in ['transformers', 'Pillow', 'requests']},
                'api': {'temperature': 0, 'provider': a.provider, 'allow_fallbacks': False,
                        'context_guard': 'local_qwen_template_estimate_then_actual_usage_check',
                        'image_transport': 'PNG_RGB_area_cap_32px_grid_provider_preprocessing_unknown',
                        'tool_transport': 'native_API_envelopes_to_existing_SFT_tags',
                        'limitations': ['Provider chat template, image encoding and precision may differ from local SFT.'],
                        'adapter_sha256': digest(ROOT / 'evaluation/remote.py'),
                        'runner_sha256': digest(Path(__file__))}}
    identity = {'openrouter_id': a.model, 'tokenizer_sha256': {
        f.name: digest(f) for f in sorted(a.tokenizer.iterdir())
        if f.is_file() and (f.name.startswith(('tokenizer', 'chat_template')) or f.name in {'config.json', 'merges.txt', 'vocab.json'})}}
    run = {'model': a.model, 'model_identity': identity, 'protocol': protocol,
           'protocol_sha256': fingerprint(protocol), 'requested_device': 'openrouter'}
    run['run_id'] = fingerprint(run)
    run.update(tasks_file=str(a.tasks.resolve()), shard={'index': 0, 'count': 1, 'selected_ids': [t['id'] for t in tasks]})
    if a.check_only:
        checker = OpenRouterModel(a.model, tokenizer, APIClient(''), a.provider)
        first = tasks[0]['model_input']
        _, token_count = checker.prepare([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': [
            *[{'type': 'image', 'image': image['path'], 'max_pixels': a.max_pixels} for image in first['images']],
            {'type': 'text', 'text': first['question']}]}], schemas)
        print(json.dumps({'checked_tasks': len(tasks), 'workers': a.workers, 'api_requests': 0,
                          'model': a.model, 'output': str(a.output.resolve()),
                          'openrouter_key_present': bool(os.environ.get('OPENROUTER_API_KEY')),
                          'search_key_present': search_ready,
                          'tokenizer_probe_tokens': token_count,
                          'budgets': {k: protocol[k] for k in ['max_steps', 'max_new_tokens', 'context_tokens', 'image_side', 'max_pixels']}}, ensure_ascii=False))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    with output_lock(a.output):
        config_path = a.output / 'run.json'
        if config_path.exists() and json.loads(config_path.read_text()).get('run_id') != run['run_id']:
            p.error('Output belongs to a different run. Use a new directory; do not overwrite records.')
        write_json_atomic(config_path, run)
        path = a.output / 'predictions.jsonl'
        expected = {t['id']: t for t in tasks}
        previous = read_predictions(path, run['run_id'], expected, recover_tail=True)
        pending = [t for t in tasks if t['id'] not in previous or
                   (a.retry_infrastructure_errors and previous[t['id']].get('infrastructure_error'))]
        client = APIClient(os.environ['OPENROUTER_API_KEY'])
        model = OpenRouterModel(a.model, tokenizer, client, a.provider)
        def evaluate(task):
            record = run_task(task, model, a, schemas, run['run_id'])
            if record.get('error_type') == 'SerperKeysExhausted':
                client.stop.set()
            return record
        print(f'{len(pending)} tasks pending; {a.workers} workers', flush=True)
        with path.open('a', encoding='utf-8') as stream:
            for record in parallel_results(evaluate, pending, a.workers, client.stop):
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                previous[record['id']] = record
                print(json.dumps({'id': record['id'], 'status': record['status'],
                                  'error_type': record.get('error_type'),
                                  'infrastructure_error': record.get('infrastructure_error', False),
                                  'tool_errors': record.get('tool_errors', 0),
                                  'recorded': len(previous), 'total': len(tasks)}), flush=True)
        missing = set(expected) - set(previous)
        errors = [key for key, record in previous.items() if record.get('infrastructure_error')]
        summary = {'expected': len(tasks), 'recorded': len(previous), 'missing_ids': sorted(missing),
                   'infrastructure_error_ids': errors, 'complete': not missing and not errors}
        summary['failure_details'] = [
            {'id': key, 'error_type': previous[key].get('error_type'),
             'tool_errors': [{'step': turn.get('step'), 'tool': turn.get('tool_call', {}).get('name'),
                              'error_type': turn['result'].get('error_type'),
                              'http_status': turn['result'].get('http_status')}
                             for turn in previous[key].get('turns', [])
                             if turn.get('result', {}).get('status') == 'error']}
            for key in errors]
        write_json_atomic(a.output / 'completion.json', summary)
        if not summary['complete']:
            raise SystemExit(f'Incomplete run: {len(missing)} missing tasks, {len(errors)} tasks with infrastructure errors. '
                             f'See {a.output / "completion.json"} for failure_details. Fix the reported issue, then '
                             'resume with --retry-infrastructure-errors using the same configuration. Do not score yet.')
        print('Complete: all selected tasks recorded without infrastructure errors. Ready to score.')


if __name__ == '__main__':
    main()
