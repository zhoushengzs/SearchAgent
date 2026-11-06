"""Evaluate a checkpoint with the GRPO agent and tool environment."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image

from evaluation.common import load_tasks, materialize_task_paths, read_predictions, shard_tasks, write_json_atomic
from evaluation.prepare import digest
from evaluation.remote import output_lock
from evaluation.run import LocalModel, fingerprint, load_env

ROOT = Path(__file__).resolve().parents[1]


def load_runtime(directory):
    directory = Path(directory).resolve()
    if not (directory / 'merged_agent.py').is_file():
        raise ValueError('The actual GRPO integration directory is required')
    sys.path.insert(0, str(directory))
    agent = importlib.import_module('merged_agent')
    session = importlib.import_module('project_runtime.session')
    if Path(agent.__file__).resolve() != directory / 'merged_agent.py' or Path(session.__file__).resolve().parent != directory / 'project_runtime':
        raise ValueError('Another GRPO runtime was already imported; start a fresh worker')
    expected = {'vs_web_search', 'vs_image_search', 'vs_visit', 'vs_inspect_image', 'vs_crop', 'vs_calculate'}
    if set(session.get_all_tools()) != expected:
        raise ValueError('Native GRPO requires exactly its six original tools')
    return agent, session


def runtime_identity(directory):
    directory = Path(directory)
    files = [directory / 'merged_agent.py', *[p for p in (directory / 'project_runtime').iterdir()
                                             if p.is_file() and p.suffix in {'.py', '.json', '.txt'}]]
    return {str(p.relative_to(directory)): digest(p) for p in sorted(files)}


class NativeEngine:
    """Adapt native PIL-image messages to local inference without extra instructions."""
    def __init__(self, model, directory, prompt_tokens, response_tokens, max_pixels, temperature=0.0):
        self.model, self.directory = model, Path(directory)
        self.max_prompt_length, self.max_response_length = prompt_tokens, response_tokens
        self.max_pixels, self.temperature = max_pixels, temperature
        self.calls, self.error = [], None
        self.directory.mkdir(parents=True, exist_ok=True)

    def image_record(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError('Native runtime must provide decoded PIL images')
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format='PNG')
        data = buffer.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        path = self.directory / (sha + '.png')
        if not path.exists():
            path.write_bytes(data)
        return {'path': str(path.resolve()), 'sha256': sha}

    def audit_messages(self, messages):
        return [{k: ([self.image_record(im) for im in value] if k == 'images' else value)
                 for k, value in message.items()} for message in messages]

    async def get_model_response(self, messages, max_tokens, **kwargs):
        converted = []
        for message in messages:
            blocks = [{'type': 'image', 'image': self.image_record(im)['path'], 'max_pixels': self.max_pixels}
                      for im in message.get('images', [])]
            blocks.append({'type': 'text', 'text': message['content']})
            converted.append({'role': message['role'], 'content': blocks})
        options = {'repetition_penalty': 1.0, 'do_sample': self.temperature > 0}
        if self.temperature > 0:
            options.update(temperature=self.temperature, top_p=0.95, top_k=0)
        try:
            generated = self.model.generate(converted, [], max_tokens,
                                            self.max_prompt_length + max_tokens, **options)
        except Exception as exc:
            self.error = type(exc).__name__
            raise
        row = {'messages': self.audit_messages(messages), 'generation': generated,
               'tools_in_system_prompt_only': True}
        self.calls.append(row)
        with (self.directory.parent / 'raw_calls.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        if generated.get('status') == 'context_limit':
            self.error = 'ContextBudgetExceeded'
            raise ValueError('Native prompt token budget exceeded')
        # The native parser decides whether generated text contains a valid action.
        # Do not add format retries, forced answer prefixes, or post-hoc recovery.
        return SimpleNamespace(text=generated.get('text', ''),
                               prompt_length=generated.get('input_tokens', 0),
                               completion_length=generated.get('new_tokens', 0))


@contextmanager
def keep_native_artifacts(session_module, directory):
    original = session_module.ProjectSession

    class SavedSession(original):
        def __init__(self, question, images, temporary_directory, max_steps):
            super().__init__(question, images, str(directory), max_steps)

    session_module.ProjectSession = SavedSession
    try:
        yield
    finally:
        session_module.ProjectSession = original


def result_record(task, result, engine, session_module, run_id):
    turns, tool_errors, duplicates = [], 0, 0
    assistant_messages = [(i, m) for i, m in enumerate(result['messages']) if m['role'] == 'assistant']
    for step, (index, message) in enumerate(assistant_messages, 1):
        turn = {'step': step, 'text': message['content'], 'status': 'ok'}
        try:
            kind, value = session_module.parse_action(message['content'])
        except (ValueError, KeyError, TypeError):
            turn['status'] = 'invalid_output'
        else:
            if kind == 'tool':
                turn['tool_call'] = value
        if index + 1 < len(result['messages']):
            observation = result['messages'][index + 1].get('content', '')
            if '<tool_response>' in observation and '</tool_response>' in observation:
                raw = observation.split('<tool_response>', 1)[1].split('</tool_response>', 1)[0]
                tool_result = json.loads(raw)
                turn['result'] = tool_result
                tool_errors += tool_result.get('status') != 'ok'
                duplicates += tool_result.get('error_type') == 'RepeatedAction'
        turns.append(turn)
    termination = result['termination']
    infrastructure = engine.error not in (None, 'ContextBudgetExceeded')
    for turn in turns:
        if turn.get('result', {}).get('error_type') in {'SerperKeysExhausted', 'OutOfMemoryError'}:
            infrastructure = True
    status = ('answered' if termination == 'answer' else 'context_limit' if engine.error == 'ContextBudgetExceeded'
              else 'inference_error' if infrastructure else 'step_limit' if termination == 'max_rounds_reached_no_answer'
              else 'invalid_output')
    return {'id': task['id'], 'source_dataset': task['source_dataset'],
            'input_fingerprint': task['input_fingerprint'], 'run_id': run_id,
            'status': status, 'answer': result.get('prediction', '') if termination == 'answer' else '',
            'native_termination': termination, 'native_rounds': result['rounds'],
            'tool_calls': sum('tool_call' in t for t in turns), 'tool_errors': tool_errors,
            'duplicate_calls': duplicates, 'turns': turns,
            'infrastructure_error': infrastructure, 'error_type': engine.error,
            'elapsed_seconds': result['time_taken']}


def evaluate_task(task, model, agent_module, session_module, args, run_id):
    directory = args.output / 'artifacts' / task['id']
    directory.mkdir(parents=True, exist_ok=True)
    engine = NativeEngine(model, directory / 'model_images', args.prompt_tokens,
                          args.max_new_tokens, args.max_pixels, args.temperature)
    agent = agent_module.DeepResearchAgent(engine, tools=session_module.get_all_tools())
    agent.max_llm_calls = args.max_steps
    agent.max_prompt_tokens, agent.max_response_tokens = args.prompt_tokens, args.max_new_tokens
    with keep_native_artifacts(session_module, directory / 'evidence'):
        # Detached references must never enter the agent's run() call.
        result = asyncio.run(agent.run(question=task['model_input']['question'],
                                       images=[im['path'] for im in task['model_input']['images']]))
    write_json_atomic(directory / 'native_result.json',
                      {**result, 'messages': engine.audit_messages(result['messages'])})
    return result_record(task, result, engine, session_module, run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--prompt-tokens', type=int, default=28672)
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--max-pixels', type=int, default=262144)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expected-count', type=int, default=778)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--retry-infrastructure-errors', action='store_true')
    args = parser.parse_args()
    if min(args.max_steps, args.prompt_tokens, args.max_new_tokens, args.max_pixels) < 1 or args.temperature < 0:
        parser.error('Budgets must be positive; temperature must be nonnegative')
    if not 0 <= args.seed < 2 ** 32:
        parser.error('Seed must be 0..4294967295')
    if args.limit is not None and args.limit < 1:
        parser.error('Limit must be positive')
    load_env(args.env_file)
    agent_module, session_module = load_runtime(args.runtime)
    source_identity = runtime_identity(args.runtime)
    tasks = load_tasks(args.tasks, args.expected_count, args.limit)
    assigned = [materialize_task_paths(t, args.tasks) for t in shard_tasks(tasks, args.shard_index, args.num_shards)]
    model_path = args.model.resolve()
    model_identity = {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                      for p in sorted(model_path.glob('*.safetensors'))}
    if not model_identity or not (model_path / 'config.json').exists():
        parser.error('Complete local model weights and configuration are required')
    for name in ['config.json', 'generation_config.json', 'tokenizer_config.json', 'preprocessor_config.json',
                 'tokenizer.json', 'vocab.json', 'merges.txt', 'chat_template.json', 'chat_template.jinja']:
        if (model_path / name).exists():
            model_identity[name] = digest(model_path / name)
    protocol = {'version': 'grpo-native-agent-evaluation-v1', 'mode': 'agent', 'protocol': 'grpo-native',
                'system': session_module.system_prompt(), 'tools': list(session_module.get_all_tools().values()),
                'native_runtime_sha256': source_identity, 'selected_ids': [t['id'] for t in tasks],
                'tasks_sha256': digest(args.tasks), 'max_steps': args.max_steps,
                'prompt_tokens': args.prompt_tokens, 'max_new_tokens': args.max_new_tokens,
                'max_pixels': args.max_pixels, 'temperature': args.temperature, 'seed': args.seed,
                'seed_policy': 'seed_plus_task_id_hash', 'dtype': 'bfloat16', 'inference_backend': 'transformers-sdpa',
                'execution_policy': 'actual DeepResearchAgent.run; no evaluator answer/format recovery',
                'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                                [Path(__file__), ROOT/'evaluation/run.py', ROOT/'evaluation/common.py']},
                'versions': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'Pillow']}}
    run = {'model': str(model_path), 'model_identity': model_identity, 'protocol': protocol,
           'protocol_sha256': fingerprint(protocol), 'requested_device': 'cuda'}
    run['run_id'] = fingerprint(run)
    run.update(tasks_file=str(args.tasks.resolve()), shard={'index': args.shard_index, 'count': args.num_shards,
                                                          'selected_ids': [t['id'] for t in assigned]})
    args.output.mkdir(parents=True, exist_ok=True)
    with output_lock(args.output):
        config = args.output / 'run.json'
        if config.exists() and json.loads(config.read_text()) != run:
            raise ValueError('Native run changed; use a new output directory')
        write_json_atomic(config, run)
        previous = read_predictions(args.output / 'predictions.jsonl', run['run_id'], {t['id']: t for t in assigned}, recover_tail=True)
        pending = [t for t in assigned if t['id'] not in previous or
                   (args.retry_infrastructure_errors and previous[t['id']].get('infrastructure_error'))]
        if not pending:
            print('All native predictions already saved; no model loaded.', flush=True)
            return
        import torch
        model = LocalModel(str(model_path), 'cuda', 'bfloat16')
        for task in pending:
            if runtime_identity(args.runtime) != source_identity:
                raise ValueError('Native agent source changed during evaluation')
            task_seed = (args.seed + int(hashlib.sha256(task['id'].encode()).hexdigest()[:8], 16)) % 2 ** 32
            torch.manual_seed(task_seed)
            torch.cuda.manual_seed_all(task_seed)
            record = evaluate_task(task, model, agent_module, session_module, args, run['run_id'])
            record['task_seed'] = task_seed
            with (args.output / 'predictions.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
            print(json.dumps({'id': record['id'], 'status': record['status'], 'rounds': record['native_rounds'],
                              'seconds': record['elapsed_seconds']}, ensure_ascii=False), flush=True)
            if record['infrastructure_error']:
                raise SystemExit('Native model/service infrastructure failed; inspect records before resuming')


if __name__ == '__main__':
    main()
