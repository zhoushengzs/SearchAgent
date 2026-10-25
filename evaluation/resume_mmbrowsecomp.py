"""Resume the pinned local MM agent using isolated worker processes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from evaluation.common import load_tasks, materialize_task_paths, write_json_atomic
from evaluation.prepare import digest
from evaluation.remote import output_lock
from evaluation.run import load_env

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.codex/official_eval'


def adapter():
    sys.path.insert(0, str(BASE))
    sys.path.insert(0, str(BASE / 'Agent-R1'))
    spec = importlib.util.spec_from_file_location('saved_mm_adapter', BASE / 'run_mmbrowsecomp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def work(task, config, output):
    from openai import OpenAI
    runner = adapter()
    args = SimpleNamespace(output=Path(output), model=config['model'],
                           max_turns=config['max_turns'], search_backend=config['search_backend'])
    try:
        client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1',
                        timeout=180, max_retries=2)
        return runner.infer_task(task, args, client)
    except Exception as exc:
        return {'id': task['id'], 'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, default=ROOT/'outputs/server_eval_778/tasks.jsonl')
    parser.add_argument('--workers', type=int, default=4)
    args=parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('workers must be 1..8')
    output=args.output.resolve()
    config=json.loads((output/'trial_config.json').read_text())
    if digest(args.tasks) != config['tasks_sha256']:
        raise ValueError('Task bundle changed')
    for name, sha in config['adapter_sha256'].items():
        if digest(BASE/name) != sha:
            raise ValueError('Saved inference implementation changed: '+name)
    load_env(ROOT/'.env')
    os.environ['SERPER_API_KEY']=os.environ.get(config['search_key_variable'], '')
    os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(BASE/'browsers')
    if not os.environ.get('OPENROUTER_API_KEY') or not os.environ.get('SERPER_API_KEY'):
        raise ValueError('Missing model/search credentials')
    tasks=[materialize_task_paths(t,args.tasks) for t in load_tasks(args.tasks) if t['id'] in set(config['selected_ids'])]
    with output_lock(output):
        # Preserve the original official-style handling of terminal length outputs.
        sys.path.insert(0,str(BASE))
        from finalize_mm_outputs import finalize
        finalize(output)
        path=output/'answers.jsonl'
        completed={json.loads(s)['id'] for s in path.read_text().splitlines() if s.strip()}
        pending=[t for t in tasks if t['id'] not in completed]
        for task in pending:
            trace_dir=output/'artifacts'/task['id']
            if trace_dir.exists():
                archive=output/'interrupted_artifacts'/task['id']
                archive.parent.mkdir(parents=True,exist_ok=True)
                suffix=1
                while archive.exists():
                    archive=archive.with_name(task['id']+f'.attempt-{suffix}');suffix+=1
                shutil.move(trace_dir,archive)
        write_json_atomic(output/'parallel_resume.json', {'workers':args.workers,
            'isolation':'One task at a time per isolated process; unchanged pinned infer_task implementation.',
            'runner_sha256':digest(__file__), 'retained':len(completed), 'pending_ids':[t['id'] for t in pending]})
        print(json.dumps({'retained':len(completed),'remaining':len(pending),'workers':args.workers}),flush=True)
        failures=[]
        iterator=iter(pending)
        fatal=False
        with path.open('a',encoding='utf-8') as stream, ProcessPoolExecutor(
                max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=set()
            def fill():
                while len(futures)<args.workers and not fatal:
                    task=next(iterator,None)
                    if task is None:break
                    futures.add(pool.submit(work,task,config,str(output)))
            fill()
            while futures:
                done, remaining=wait(futures,return_when=FIRST_COMPLETED)
                futures=remaining
                for future in done:
                    record=future.result()
                    if record.get('error_type'):
                        failures.append(record)
                        if record.get('http_status') in (401,402,403):fatal=True
                        print(json.dumps(record),flush=True)
                    else:
                        stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                        completed.add(record['id'])
                        print(json.dumps({'id':record['id'],'status':'answered','recorded':len(completed),'total':len(tasks)}),flush=True)
                fill()
        write_json_atomic(output/'failures.json',failures)
        audit=finalize(output)
        print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
        if not audit['all_tasks_recorded']:
            raise SystemExit('Inference incomplete; inspect failures before resuming.')


if __name__ == '__main__':
    main()
