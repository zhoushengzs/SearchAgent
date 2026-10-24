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

from evaluation.common import load_tasks, write_json_atomic
from evaluation.package import export_bundle
from evaluation.prepare import digest
from evaluation.run import load_env
from evaluation.suites.setup import verify
from evaluation.suites.runtime import official_root

ROOT = Path(__file__).resolve().parents[1]
ROUTES = {'agentvista': 'official_agentvista', 'mmbrowsecomp': 'paper_reconstruction_mmbrowsecomp'}


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
    plan = {'schema': 'routed-benchmark-pilot-v1', 'tasks_sha256': digest(a.tasks),
            'references_sha256': digest(a.references), 'fraction': a.fraction, 'seed': a.seed,
            'rounding': 'ceil_per_benchmark', 'sampling': 'sha256_seed_source_id_input_fingerprint',
            'selection_before_inference': True, 'model': str(a.model.resolve()),
            'device': a.device, 'routes': []}
    counts = collections.Counter(t['source_dataset'] for t in tasks)
    for source, ids in chosen.items():
        bundle = a.output / 'data' / source
        plan['routes'].append({'benchmark': source, 'population': counts[source], 'count': len(ids),
                               'ids': ids, 'runner': ROUTES.get(source, 'shared_agent'),
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
    if route['benchmark'] in ROUTES:
        script = 'run_agentvista.py' if route['benchmark'] == 'agentvista' else 'run_mmbrowsecomp.py'
        command = [sys.executable, '-m', 'evaluation.suites.' + script.removesuffix('.py'),
                   *common, '--all', '--model', str(a.model.resolve()), '--inference-only',
                   '--search-key-env', a.search_key_env]
        if route['benchmark'] == 'mmbrowsecomp':
            command += ['--search-backend', 'serper']
        return command
    return [sys.executable, '-u', '-m', 'evaluation.run', *common, '--model', str(a.model.resolve()),
            '--device', a.device, '--expected-count', str(route['count']), '--protocol', 'training',
            '--repetition-penalty', '1.1', '--repeat-limit', '2', '--max-steps', '10',
            '--tool-summaries', '--retry-infrastructure-errors']


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
    p.add_argument('--official-root', type=Path, default=official_root())
    p.add_argument('--search-key-env', default='SERPER_API_KEY')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    a.tasks, a.output, a.model = a.tasks.resolve(), a.output.resolve(), a.model.resolve()
    a.references = (a.references or a.tasks.parent / 'references.jsonl').resolve()
    if not (a.model / 'config.json').is_file() or not list(a.model.glob('*.safetensors')):
        p.error('Supply complete merged SFT model weights')
    plan = prepare(a)
    print(json.dumps({'routes': [{k: r[k] for k in ['benchmark', 'count', 'runner']} for r in plan['routes']],
                      'total': sum(r['count'] for r in plan['routes']), 'output': str(a.output)}, ensure_ascii=False), flush=True)
    if a.prepare_only:
        return
    versions = verify(a.official_root)
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
    write_json_atomic(a.output / 'execution.json', {'official_versions': versions,
                      'local_model': str(a.model), 'status': 'running', 'commands': [child_command(a, r) for r in plan['routes']]})
    for route in plan['routes']:
        log = a.output / 'logs' / (route['benchmark'] + '.log')
        print('Starting', route['benchmark'], 'log:', log, flush=True)
        with log.open('a') as stream:
            result = subprocess.run(child_command(a, route), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json_atomic(a.output / 'completion.json', {'complete': False, 'stopped_benchmark': route['benchmark'],
                                                          'exit_code': result.returncode, 'log': str(log)})
            raise SystemExit('Pilot incomplete. Inspect ' + str(log))
    write_json_atomic(a.output / 'completion.json', {'complete': True, 'inference_only': True,
                                                   'count': sum(r['count'] for r in plan['routes'])})


if __name__ == '__main__':
    main()
