"""Bounded paired diagnosis, separate from formal benchmark scores."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from PIL import Image
from agent.tools import EvidenceTools
from evaluation.common import load_tasks, materialize_task_paths, write_json_atomic
from evaluation.execution import final_content
from evaluation.protocol import state_message, training_system, training_tools
from evaluation.run import LocalModel, load_env, parse_output, run_task, tool_definitions

ROOT = Path(__file__).resolve().parents[1]
IDS = ['agentvista_train_0', 'agentvista_train_36', 'agentvista_train_42']


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def action_of(generation):
    text = generation.get('text', '')
    if generation.get('status') != 'ok':
        return {'action': 'unfinished', 'stop_reason': generation.get('stop_reason')}
    answer = final_content(text)
    if answer:
        return {'action': 'answer', 'answer': answer}
    import re
    answer = re.search(r'<answer>(.*?)</answer>', text, re.S)
    if answer:
        return {'action': 'answer', 'answer': answer.group(1).strip()}
    try:
        action, data = parse_output(text)
        return {'action': action, 'proposal': data}
    except ValueError:
        return {'action': 'invalid_format'}


class AuditedModel:
    """No hidden observer; record the exact generated text before adapters."""
    def __init__(self, model, output):
        self.model, self.output, self.context = model, output, {}

    def invoke(self, method, messages, tools, cap, context, **kwargs):
        kwargs.update(repetition_guard=True, max_time=120)
        start = time.monotonic()
        if method == 'generate':
            result = self.model.generate(messages, tools, cap, context, **kwargs)
        else:
            result = self.model.generate_prefill(messages, cap, context, **kwargs)
        row = {**self.context, 'method': method, 'max_new_tokens': cap,
               'seconds': time.monotonic() - start, 'messages': messages,
               'tool_names': [t['function']['name'] for t in tools], 'generation': result}
        append(self.output / 'raw_calls.jsonl', row)
        print(json.dumps({**self.context, 'tokens': result.get('new_tokens'),
                          'stop_reason': result.get('stop_reason'), **action_of(result)},
                         ensure_ascii=False), flush=True)
        return result

    def generate(self, messages, tools, cap, context, **kwargs):
        return self.invoke('generate', messages, tools, cap, context, **kwargs)

    def generate_prefill(self, messages, cap, context, **kwargs):
        return self.invoke('prefill', messages, [], cap, context, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    load_env(ROOT / '.env')
    os.environ['SERPER_API_KEY'] = os.environ['SERPER_API_KEY_4']
    bundle = ROOT / 'outputs/sft_778_pilot20_v2_no_summary/data/agentvista/tasks.jsonl'
    tasks = {t['id']: materialize_task_paths(t, bundle) for t in load_tasks(bundle)}
    official_system = (ROOT / 'outputs/sft_778_pilot20_v2_no_summary/runs/agentvista/run/agentvista_train_0/0_injected_system_prompt.txt').read_text()
    sys.path.insert(0, str(ROOT / 'official_eval/AgentVista'))
    from utils.tool_schema_builder import build_openai_tools_schema
    official_tools = build_openai_tools_schema(['web_search', 'image_search', 'visit', 'code_interpreter'])
    if len(official_tools) != 4:
        raise ValueError('Cannot load the four pinned official tool schemas')
    manifest = {'schema': 'paired-protocol-diagnosis-v1', 'model': str(a.model.resolve()),
                'ids': IDS, 'first_decision_cap': 512, 'flow_cap': 256, 'flow_steps': 3,
                'max_time_seconds_soft': 120, 'repetition_guard': {'span': 32, 'occurrences': 3},
                'repetition_penalty': 1.1, 'do_sample': False, 'extra_summaries': False,
                'probe_max_pixels': 1000000, 'seed': 42,
                'code_sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in [Path(__file__), ROOT/'evaluation/run.py',
                                         ROOT/'evaluation/generation_guard.py', ROOT/'evaluation/protocol.py']},
                'official_system': official_system, 'official_tools': official_tools,
                'training_tools': training_tools(),
                'scope': 'Diagnostic only. First-decision arms use identical pixels/budgets; prompt, state and tool interfaces change together. Follow-up tool flows use the shared training runner and its preview images, separately from the first-decision comparison. No judge or reference answers supplied.'}
    write_json_atomic(a.output / 'manifest.json', manifest)
    import torch
    torch.manual_seed(42)
    model = AuditedModel(LocalModel(a.model, 'mps', 'bfloat16'), a.output)
    summaries = []
    for key in IDS:
        task = tasks[key]
        tools = EvidenceTools(a.output / 'probe_assets' / key, image_side=1536)
        previews = tools.initialize(task['model_input']['images'])
        attached = []
        for index, source in enumerate(task['model_input']['images'], 1):
            path = a.output / 'probe_assets' / key / f'controlled_{index}.png'
            with Image.open(source['path']) as image:
                image.convert('RGB').save(path)
                width, height = image.size
            attached.append(dict(previews[index-1], path=str(path.resolve()), width=width,
                                 height=height, image_index=index, observed_at_step=0,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        plain = [{'type': 'image', 'image': im['path'], 'max_pixels': 1000000} for im in attached]
        plain.append({'type': 'text', 'text': task['model_input']['question'].replace('<image>', '').strip()})
        states = state_message(task, tools, [{k:v for k,v in im.items() if k!='path'} for im in attached],
                               attached, [], 1, 10, 1000000)
        for arm, system, user, schemas in [
            ('official_interface', official_system, {'role': 'user', 'content': plain}, official_tools),
            ('training_interface', training_system(), states, training_tools()),
        ]:
            model.context = {'phase': 'first_decision', 'id': key, 'arm': arm}
            result = model.generate([{'role':'system','content':system}, user], schemas, 512, 32768,
                                    repetition_penalty=1.1)
            summaries.append({**model.context, **result, **action_of(result)})
            write_json_atomic(a.output / 'first_decisions.json', summaries)
    # Real tools and bounded correction/final-answer recovery; no official backend is impersonated.
    for key in [IDS[0], IDS[2]]:
        model.context = {'phase': 'real_tool_flow', 'id': key, 'arm': 'training_interface'}
        args = SimpleNamespace(output=a.output/'flows', image_side=1536, mode='agent',
                               max_steps=3, max_pixels=262144, max_new_tokens=256,
                               context_tokens=32768, protocol='training', tool_summaries=False,
                               repeat_limit=2, repetition_penalty=1.1)
        record = run_task(tasks[key], model, args, tool_definitions(), 'diagnostic_only')
        append(a.output/'flow_results.jsonl', record)
    write_json_atomic(a.output/'completion.json', {'complete': True, 'first_decisions': len(summaries),
                                                  'real_tool_flows': 2, 'formal_score': False})


if __name__ == '__main__':
    main()
