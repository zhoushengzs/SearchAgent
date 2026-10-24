"""Prepare held-out inputs and launch the pinned, unmodified official AgentVista agent."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path(os.environ.get('EVAL_OFFICIAL_ROOT', str(ROOT / 'outputs/official_eval_dependencies')))
sys.path.insert(0, str(ROOT))
from evaluation.run import load_env
from evaluation.common import load_tasks, materialize_task_paths


def valid_saved_answer(row):
    answer = row.get('final_answer')
    return isinstance(answer,str) and bool(answer.strip()) and not answer.lstrip().startswith('Error:')


def prepare_resume(output, selected_ids):
    run = output/'run'
    path = run/'results.jsonl'
    rows = [json.loads(line) for line in path.read_text().split('\n') if line.strip()] if path.exists() else []
    ids = [row['question_id'] for row in rows]
    if len(ids)!=len(set(ids)) or set(ids)-set(selected_ids):
        raise SystemExit('Saved AgentVista results contain duplicate or unexpected task IDs.')
    valid = {row['question_id']:row for row in rows if valid_saved_answer(row)}
    retry_ids = [key for key in selected_ids if key not in valid and (run/key).exists()]
    if rows or retry_ids:
        archive = output/'previous_attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive.mkdir(parents=True)
        for name in ['results.jsonl','summary_metrics.json']:
            if (run/name).exists():shutil.copy2(run/name,archive/name)
        for key in retry_ids:
            shutil.move(str(run/key),str(archive/key))
    return valid


def merge_saved_answers(output, previous, selected_ids):
    path = output/'run/results.jsonl'
    rows = [json.loads(line) for line in path.read_text().split('\n') if line.strip()] if path.exists() else []
    merged = dict(previous)
    for row in rows:
        merged[row['question_id']] = row
    if set(merged)-set(selected_ids):raise SystemExit('Unexpected AgentVista task ID after inference.')
    ordered = [merged[key] for key in selected_ids if key in merged]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in ordered))
    return [key for key in selected_ids if key not in merged or not valid_saved_answer(merged[key])]


def check_judge(args, env):
    from openai import OpenAI
    try:
        probe = OpenAI(api_key=env.get(args.judge_key_env,''),base_url=args.judge_base_url,max_retries=0).chat.completions.create(
            model=args.judge,temperature=0.2,max_tokens=128,
            messages=[{'role':'user','content':'Question: What is 2+2? Reference: 4. Candidate: 4. Return Score: 1 and a brief explanation.'}])
        text = probe.choices[0].message.content or ''
        return {'ready':'score: 1' in text.lower(),'model':args.judge}
    except Exception as exc:
        return {'ready':False,'model':args.judge,'error_type':type(exc).__name__,
                'http_status':getattr(exc,'status_code',None)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=2)
    p.add_argument('--all', action='store_true', help='Run all selected AgentVista tasks')
    p.add_argument('--tasks',type=Path,default=ROOT/'outputs/server_eval_778/tasks.jsonl')
    p.add_argument('--references',type=Path)
    p.add_argument('--output', type=Path, default=BASE / 'results/agentvista_trial')
    p.add_argument('--model', default='qwen/qwen3-vl-8b-instruct')
    p.add_argument('--judge', default='openai/gpt-4.1')
    p.add_argument('--judge-base-url', default='https://openrouter.ai/api/v1')
    p.add_argument('--judge-key-env', default='OPENROUTER_API_KEY')
    p.add_argument('--search-key-env', default='SERPER_API_KEY', help='Environment variable containing a usable Serper key')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--inference-only',action='store_true',help='Run the official agent without requesting grades')
    args = p.parse_args()
    if args.limit < 1:
        p.error('--limit must be positive')
    repo = BASE / 'AgentVista'
    sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    tasks_file = args.tasks.resolve()
    reference_file = args.references or tasks_file.parent/'references.jsonl'
    references = {r['id']: r for r in map(json.loads, reference_file.read_text().splitlines())}
    tasks = [t for t in load_tasks(tasks_file) if t['source_dataset'] == 'agentvista']
    if not args.all:tasks=tasks[:args.limit]
    if not tasks:raise SystemExit('No AgentVista tasks in the supplied task file.')
    inputs = []
    for task in tasks:
        task = materialize_task_paths(task, tasks_file)
        inputs.append({'question_id': task['id'], 'problem': task['model_input']['question'],
                       'images': [image['path'] for image in task['model_input']['images']],
                       'solution': references[task['id']]['answer']})
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'suite': 'AgentVista official infer.py', 'commit': sha, 'model': args.model,
                'judge': args.judge, 'judge_base_url': args.judge_base_url,
                'selected_ids': [t['question_id'] for t in inputs],
                'search_key_variable': args.search_key_env,
                'tasks_sha256': hashlib.sha256(tasks_file.read_bytes()).hexdigest(),
                'references_sha256':hashlib.sha256(reference_file.read_bytes()).hexdigest(),
                'inference_only':args.inference_only,
                'max_turns': 30, 'max_images': 100, 'max_total_tokens': 65536,
                'max_completion_tokens': 8192, 'max_pixels': 2000000,
                'scope': 'fixed project holdout subset, not the official full-dataset score',
                'local_model': os.environ.get('EVAL_LOCAL_MODEL'),
                'dependencies': 'minimal inference dependencies on macOS; not the complete pinned CUDA training environment',
                'visit_backend': 'official Trafilatura; Jina fallback only if a key is configured'}
    path = args.output / 'trial_config.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise SystemExit('Trial configuration differs; choose a new --output directory.')
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    data = args.output / 'inputs.json'
    data.write_text(json.dumps(inputs, ensure_ascii=False, indent=2))
    load_env(ROOT / '.env')
    env = dict(os.environ)
    key = 'local-inference' if env.get('EVAL_LOCAL_MODEL') else env.get('OPENROUTER_API_KEY')
    search_key = env.get(args.search_key_env)
    env.update(REASONING_MODEL_NAME=args.model, REASONING_API_KEY=key or '',
               REASONING_END_POINT=('http://local-sft.invalid/v1/chat/completions' if env.get('EVAL_LOCAL_MODEL') else 'https://openrouter.ai/api/v1/chat/completions'),
               SERPAPI_KEY=search_key or '', ENABLED_TOOLS='web_search,image_search,visit,code_interpreter',
               VERIFIER_MODEL_NAME=args.judge, VERIFIER_API_KEY=env.get(args.judge_key_env,''),
               VERIFIER_END_POINT=args.judge_base_url, PYTHONDONTWRITEBYTECODE='1')
    if args.inference_only:
        env['VERIFIER_API_KEY']=''
        env['VERIFIER_END_POINT']=''
    print(json.dumps({'commit': sha, 'selected_ids': manifest['selected_ids'],
                      'model_key_present': bool(key), 'search_key_present': bool(search_key),
                      'jina_fallback_present': bool(env.get('JINA_API_KEY')), 'output': str(args.output)}, ensure_ascii=False), flush=True)
    if args.check_only:
        return
    if not key or not search_key:
        raise SystemExit('Configure OPENROUTER_API_KEY and SERPER_API_KEY in the original .env or terminal.')
    previous = prepare_resume(args.output,manifest['selected_ids'])
    if args.inference_only and set(previous)==set(manifest['selected_ids']):
        print('All selected AgentVista answers are already complete; no new inference.',flush=True)
        return
    import requests
    probe = requests.post('https://google.serper.dev/search', headers={'X-API-KEY':search_key},
                          json={'q':'AgentVista benchmark','num':1}, timeout=30)
    (args.output / 'search_preflight.json').write_text(json.dumps({'http_status':probe.status_code,
        'ready':probe.status_code==200, 'key_variable':args.search_key_env},indent=2))
    if probe.status_code != 200:
        raise SystemExit('Search preflight failed; check Serper credits/key before paying for model inference.')
    # The official verifier silently maps request failures to 0. Check service access first.
    if not args.inference_only:
        audit=check_judge(args,env)
        (args.output/'judge_preflight.json').write_text(json.dumps(audit,indent=2))
        if not audit['ready']:
            raise SystemExit('Official judge preflight failed. No inference started; do not report default-zero official scores.')
    env['AGENTVISTA_TRANSPORT_AUDIT'] = str(args.output/'model_http.jsonl')
    command = [sys.executable, str(HERE / 'agentvista_transport.py'), '--input-file', str(data),
               '--image-folder', '/', '--output-dir', str(args.output / 'run'), '--max-turns', '30',
               '--max-images', '100', '--max-total-tokens', '65536', '--max-completion-tokens', '8192',
               '--max-pixels', '2000000', '--skip-completed']
    with (args.output / 'run.log').open('a') as log:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    failed_ids = merge_saved_answers(args.output,previous,manifest['selected_ids'])
    if args.inference_only:
        (args.output/'trial_audit.json').write_text(json.dumps({'official_score_valid':False,
            'scoring_requested':False,'process_exit_code':result.returncode,
            'inference_complete':not failed_ids and result.returncode==0,'failed_ids':failed_ids,
            'note':'Inference only. Upstream default-zero metrics are placeholders, not measured accuracy.'},indent=2))
    if failed_ids:raise SystemExit('AgentVista inference incomplete; failed task IDs: '+', '.join(failed_ids))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
