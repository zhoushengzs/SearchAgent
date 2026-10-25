"""Freeze a per-benchmark pilot and route local SFT inference to benchmark suites."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from evaluation.common import load_tasks, write_json_atomic, read_predictions
from evaluation.package import export_bundle
from evaluation.prepare import digest
from evaluation.run import load_env
from evaluation.execution import execution_policy
from evaluation.suites.setup import verify
from evaluation.suites.runtime import official_root

ROOT = Path(__file__).resolve().parents[1]
ROUTES = {'agentvista': 'official_agentvista', 'mmbrowsecomp': 'paper_reconstruction_mmbrowsecomp'}


def source_identity():
    files = {f for folder in ['evaluation','agent','core','sources'] for f in (ROOT/folder).glob('*.py')}
    files.update((ROOT/'evaluation/training_protocol').glob('*'))
    files.update([ROOT/'trajectory_sources.json',ROOT/'evaluation/tool_schemas.json'])
    return {str(f.relative_to(ROOT)):digest(f) for f in sorted(files) if f.is_file()}


def selection(tasks, fraction=.2, seed=42):
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    groups = collections.defaultdict(list)
    for row in tasks:
        groups[row['source_dataset']].append(row)
    chosen = {}
    for source, rows in sorted(groups.items()):
        def rank(row):
            return hashlib.sha256(json.dumps([seed, source, row['id'], row['input_fingerprint']],
                                              ensure_ascii=False).encode()).hexdigest(), row['id']
        ids = {r['id'] for r in sorted(rows, key=rank)[:math.ceil(len(rows) * fraction)]}
        chosen[source] = [r['id'] for r in rows if r['id'] in ids]
    return chosen


def prepare(a):
    tasks = load_tasks(a.tasks, a.expected_count)
    chosen = selection(tasks, a.fraction, a.seed)
    plan = {'schema': 'routed-benchmark-pilot-v3', 'tasks_sha256': digest(a.tasks),
            'routing': getattr(a, 'routing', 'training'),
            'code_sha256': source_identity(),
            'generation_budget': {k:getattr(a,k,default) for k,default in [
                ('max_steps',10),('max_new_tokens',1024),('repetition_guard',True),
                ('max_generation_seconds',120.0)]},
            'shared_execution_policy': execution_policy(False),
            'references_sha256': digest(a.references), 'fraction': a.fraction, 'seed': a.seed,
            'rounding': 'ceil_per_benchmark', 'sampling': 'sha256_seed_source_id_input_fingerprint',
            'selection_before_inference': True, 'model': str(a.model.resolve()),
            'device': a.device, 'routes': []}
    counts = collections.Counter(t['source_dataset'] for t in tasks)
    for source, ids in chosen.items():
        bundle = a.output / 'data' / source
        plan['routes'].append({'benchmark': source, 'population': counts[source], 'count': len(ids),
                               'ids': ids, 'runner': (ROUTES.get(source, 'shared_agent')
                                   if getattr(a, 'routing', 'training') == 'benchmark' else 'shared_agent'),
                               'tasks': str((bundle / 'tasks.jsonl').resolve()),
                               'output': str((a.output / 'runs' / source).resolve())})
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'pilot.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('Pilot selection/configuration differs; choose a new output directory')
        for route in plan['routes']:
            if [r['id'] for r in load_tasks(route['tasks'])] != route['ids']:
                raise ValueError('Pilot bundle changed')
        return plan
    for route in plan['routes']:
        bundle = Path(route['tasks']).parent
        ids = set(route['ids'])
        manifest = export_bundle(a.tasks, a.references, bundle, expected_count=len(tasks),
                                 exclude_ids=[t['id'] for t in tasks if t['id'] not in ids],
                                 exclusion_reason='Pre-inference per-benchmark seeded pilot selection')
        manifest['exclusions']['selection_stage'] = 'pre_inference_sampling'
        manifest['pilot_sampling'] = {k: plan[k] for k in ['fraction', 'seed', 'rounding', 'sampling']}
        write_json_atomic(bundle / 'manifest.json', manifest)
    write_json_atomic(path, plan)
    return plan


def child_command(a, route):
    common = ['--tasks', route['tasks'], '--output', route['output']]
    if getattr(a, 'routing', 'training') == 'benchmark' and route['benchmark'] in ROUTES:
        script = 'run_agentvista.py' if route['benchmark'] == 'agentvista' else 'run_mmbrowsecomp.py'
        command = [sys.executable, '-m', 'evaluation.suites.' + script.removesuffix('.py'),
                   *common, '--all', '--model', str(a.model.resolve()), '--inference-only',
                   '--search-key-env', a.search_key_env]
        if route['benchmark'] == 'mmbrowsecomp':
            command += ['--search-backend', 'serper']
        return command
    command = [sys.executable, '-u', '-m', 'evaluation.run', *common, '--model', str(a.model.resolve()),
            '--device', a.device, '--expected-count', str(route['count']), '--protocol', 'training',
            '--repetition-penalty', '1.1', '--repeat-limit', '2',
            '--max-steps', str(getattr(a, 'max_steps', 10)),
            '--max-new-tokens', str(getattr(a, 'max_new_tokens', 1024)),
            '--no-tool-summaries', '--retry-infrastructure-errors']
    if getattr(a, 'repetition_guard', True):
        command.append('--repetition-guard')
    if getattr(a, 'max_generation_seconds', 120.0) is not None:
        command += ['--max-generation-seconds', str(getattr(a, 'max_generation_seconds', 120.0))]
    return command


def audit_shared_route(route):
    output = Path(route['output'])
    config = json.loads((output/'run.json').read_text())
    tasks = {t['id']:t for t in load_tasks(route['tasks'])}
    records = read_predictions(output/'predictions.jsonl', config['run_id'], tasks)
    audit = {'benchmark':route['benchmark'], 'expected':len(tasks), 'recorded':len(records),
             'missing_ids':sorted(set(tasks)-set(records)),
             'infrastructure_error_ids':[k for k,r in records.items() if r.get('infrastructure_error')],
             'unanswered_ids':[k for k,r in records.items() if r.get('status')!='answered' or not str(r.get('answer','')).strip()]}
    audit['complete'] = not any(audit[k] for k in ['missing_ids','infrastructure_error_ids','unanswered_ids'])
    write_json_atomic(output/'completion_audit.json', audit)
    return audit


def saved_agentvista_attempts(route, model):
    """Resume after an attempted suite without silently retrying model failures."""
    output = Path(route['output'])
    audit_file, records_file = output / 'trial_audit.json', output / 'run/results.jsonl'
    if not audit_file.exists() or not records_file.exists():
        return None
    config = json.loads((output / 'trial_config.json').read_text())
    records = [json.loads(line) for line in records_file.read_text().splitlines() if line.strip()]
    ids = [r['question_id'] for r in records]
    if (config['selected_ids'] != route['ids'] or config['model'] != str(Path(model).resolve())
            or config['tasks_sha256'] != digest(route['tasks']) or len(ids) != len(set(ids))
            or set(ids) != set(route['ids'])):
        raise ValueError('Saved AgentVista attempts differ from this pilot')
    audit = json.loads(audit_file.read_text())
    return {'benchmark': 'agentvista', 'resumed_saved_attempts': True,
            'failed_ids': audit.get('failed_ids', []), 'complete': bool(audit.get('inference_complete'))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, default=ROOT / 'outputs/server_eval_778/tasks.jsonl')
    p.add_argument('--references', type=Path)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--expected-count', type=int, default=778)
    p.add_argument('--device', choices=['mps', 'cuda', 'cpu'], default='mps')
    p.add_argument('--routing', choices=['training','benchmark'], default='training',
                   help='training: all benchmarks use the same SFT-style agent; benchmark: historical suite routing')
    p.add_argument('--max-steps', type=int, default=10)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--repetition-guard', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-generation-seconds', type=float, default=120.0)
    p.add_argument('--official-root', type=Path, default=official_root())
    p.add_argument('--search-key-env', default='SERPER_API_KEY')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='Retain attempted AgentVista results, including flagged model failures')
    a = p.parse_args()
    if min(a.max_steps,a.max_new_tokens,a.max_generation_seconds) <= 0:
        p.error('Generation budgets must be positive')
    a.tasks, a.output, a.model = a.tasks.resolve(), a.output.resolve(), a.model.resolve()
    a.references = (a.references or a.tasks.parent / 'references.jsonl').resolve()
    if not (a.model / 'config.json').is_file() or not list(a.model.glob('*.safetensors')):
        p.error('Supply complete merged SFT model weights')
    plan = prepare(a)
    print(json.dumps({'routes': [{k: r[k] for k in ['benchmark', 'count', 'runner']} for r in plan['routes']],
                      'total': sum(r['count'] for r in plan['routes']), 'output': str(a.output)}, ensure_ascii=False), flush=True)
    if a.prepare_only:
        return
    versions = verify(a.official_root) if a.routing == 'benchmark' else {}
    load_env(ROOT / '.env')
    if not os.environ.get(a.search_key_env):
        p.error('Missing selected Serper search key')
    import requests
    response = requests.post('https://google.serper.dev/search',
                             headers={'X-API-KEY': os.environ[a.search_key_env]},
                             json={'q': 'visual search benchmark', 'num': 1}, timeout=30)
    if response.status_code != 200:
        raise SystemExit('Search preflight failed: HTTP ' + str(response.status_code))
    env = dict(os.environ, EVAL_LOCAL_MODEL=str(a.model), EVAL_LOCAL_DEVICE=a.device,
               EVAL_OFFICIAL_ROOT=str(a.official_root.resolve()), PYTHONPATH=str(ROOT),
               PLAYWRIGHT_BROWSERS_PATH=str(a.official_root.resolve() / 'browsers'),
               PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    env['SERPER_API_KEY'] = os.environ[a.search_key_env]
    (a.output / 'logs').mkdir(exist_ok=True)
    execution = {'official_versions': versions, 'local_model': str(a.model), 'status': 'running',
                 'pid': os.getpid(), 'commands': [child_command(a, r) for r in plan['routes']]}
    write_json_atomic(a.output / 'execution.json', execution)
    write_json_atomic(a.output / 'completion.json', {'complete': False, 'status': 'running',
                      'inference_only': True, 'all_suites_attempted': False})
    outcomes = []
    for route in plan['routes']:
        if source_identity() != plan['code_sha256']:
            execution['status']='needs_review'
            write_json_atomic(a.output/'execution.json',execution)
            write_json_atomic(a.output/'completion.json',{'complete':False,'status':'needs_review',
                'error':'Evaluation code changed during inference; do not mix protocols.','outcomes':outcomes})
            raise SystemExit('Evaluation code changed; retain saved results and start a new run for new rules.')
        if a.resume and a.routing == 'benchmark' and route['benchmark'] == 'agentvista':
            saved = saved_agentvista_attempts(route, a.model)
            if saved is not None:
                outcomes.append(saved)
                write_json_atomic(a.output / 'suite_progress.json', {'outcomes': outcomes})
                print('Retained saved AgentVista attempts; flagged failures:', saved['failed_ids'], flush=True)
                continue
        log = a.output / 'logs' / (route['benchmark'] + '.log')
        print('Starting', route['benchmark'], 'log:', log, flush=True)
        with log.open('a') as stream:
            result = subprocess.run(child_command(a, route), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        outcome = {'benchmark': route['benchmark'], 'complete': result.returncode == 0,
                   'exit_code': result.returncode, 'log': str(log)}
        if route['runner'] == 'shared_agent':
            try:
                outcome.update(audit_shared_route(route))
                outcome['complete'] = outcome['complete'] and result.returncode == 0
            except (OSError,ValueError,KeyError) as exc:
                outcome.update(complete=False,audit_error=str(exc))
        outcomes.append(outcome)
        write_json_atomic(a.output / 'suite_progress.json', {'outcomes': outcomes})
        if not outcome['complete']:
            print('Suite needs review; continuing other benchmarks:', route['benchmark'], flush=True)
    complete = all(row['complete'] for row in outcomes)
    write_json_atomic(a.output / 'completion.json', {'complete': complete,
                       'status': 'complete' if complete else 'needs_review', 'inference_only': True,
                       'all_suites_attempted': True, 'outcomes': outcomes,
                       'count': sum(r['count'] for r in plan['routes'])})
    execution['status'] = 'complete' if complete else 'needs_review'
    write_json_atomic(a.output / 'execution.json', execution)
    if a.routing == 'training':
        merged = []
        for route in plan['routes']:
            folder = Path(route['output'])
            if not (folder/'run.json').exists():continue
            config = json.loads((folder/'run.json').read_text())
            expected = {t['id']:t for t in load_tasks(route['tasks'])}
            records = read_predictions(folder/'predictions.jsonl',config['run_id'],expected)
            merged.extend(records[k] for k in route['ids'] if k in records)
        (a.output/'all_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
        write_json_atomic(a.output/'merge_manifest.json',{'recorded':len(merged),'expected':sum(r['count'] for r in plan['routes']),
            'ready_to_score':complete,'model':str(a.model),'scope':'shared training agent; per-benchmark scoring is separate'})


if __name__ == '__main__':
    main()
