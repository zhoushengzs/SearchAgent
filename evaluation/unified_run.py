"""One fresh four-way experiment: shared inference assets, native agent and budgets."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from evaluation.common import load_tasks, materialize_task_paths, read_predictions, shard_tasks, write_json_atomic
from evaluation.grpo_native import NativeEngine, evaluate_task, load_runtime, result_record, runtime_identity
from evaluation.prepare import digest
from evaluation.remote import output_lock
from evaluation.run import fingerprint, load_env
from evaluation.unified_cache import evidence_snapshots
from evaluation.unified_common import SharedLocalModel, asset_identity, direct_system, weight_identity

ROOT = Path(__file__).resolve().parents[1]


def code_identity():
    files = {ROOT/'evaluation'/name for name in ['unified_common.py','unified_cache.py','unified_run.py',
             'unified_benchmark.py','unified_score.py','grpo_native.py','run.py','common.py','merge.py','prepare.py','remote.py','score.py']}
    for folder in ['core','agent','sources']:
        files.update((ROOT/folder).glob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def direct_task(task, model, session_module, args, run_id):
    started = time.monotonic()
    directory = args.output/'artifacts'/task['id']
    engine = NativeEngine(model, directory/'model_images', 28672, 4096, 262144, 0.0)
    original = session_module.ProjectSession

    class NoToolsSession(original):
        def state(self, step):
            value = super().state(step)
            value['tool_configuration']['available_tools'] = []
            value['tool_configuration']['backend'] = 'disabled'
            return value

    session = NoToolsSession(task['model_input']['question'],
                             [im['path'] for im in task['model_input']['images']], directory/'evidence', 1)
    messages = [{'role':'system','content':direct_system(session_module)}, session.initial_message()]
    answer, termination = '', 'direct_invalid_action'
    try:
        response = asyncio.run(engine.get_model_response(messages, 4096))
        messages.append({'role':'assistant','content':response.text})
        kind, value = session_module.parse_action(response.text)
        if kind == 'answer':
            answer, termination = value, 'answer'
    except (ValueError, KeyError, TypeError):
        pass
    result = {'question':task['model_input']['question'],'answer':None,'messages':messages,
              'prediction':answer,'termination':termination,'rounds':1,'time_taken':time.monotonic()-started}
    write_json_atomic(directory/'native_result.json', {**result,'messages':engine.audit_messages(messages)})
    record = result_record(task,result,engine,session_module,run_id)
    record['tools_enabled'] = False
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence-cache',type=Path,required=True)
    p.add_argument('--env-file',type=Path)
    p.add_argument('--mode',choices=['direct','agent'],required=True)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--limit',type=int)
    p.add_argument('--retry-infrastructure-errors',action='store_true')
    args=p.parse_args()
    load_env(args.env_file)
    agent,session=load_runtime(args.runtime)
    tasks=load_tasks(args.tasks,778,args.limit)
    selected=[materialize_task_paths(t,args.tasks) for t in shard_tasks(tasks,args.shard_index,args.num_shards)]
    weights,shapes=weight_identity(args.model)
    identity=asset_identity(args.assets)
    protocol={'version':'unified-native-six-tools-v1','mode':args.mode,
              'system':session.system_prompt() if args.mode=='agent' else direct_system(session),
              'tools':list(session.get_all_tools().values()) if args.mode=='agent' else [],
              'native_runtime_sha256':runtime_identity(args.runtime),'shared_assets_sha256':identity,
              'tasks_sha256':digest(args.tasks),'selected_ids':[t['id'] for t in tasks],
              'max_steps':10 if args.mode=='agent' else 1,'prompt_tokens':28672,'max_new_tokens':4096,
              'context_tokens':32768,'max_pixels':262144,'image_min_pixels':65536,'image_side':1536,
              'temperature':0.0,'do_sample':False,'repetition_penalty':1.0,'seed':42,
              'seed_policy':'seed_plus_task_id_hash','dtype':'bfloat16','attention':'sdpa',
              'model_input_wall_clock':False,'shared_evidence_cache':str(args.evidence_cache.resolve()),
              'evidence_policy':'first successful identical search or URL request reused across conditions',
              'execution_policy':'actual native loop; strict native final answer; no extra format retries or forced answers',
              'code_sha256':code_identity(),
              'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','Pillow','requests']}}
    run={'model':str(args.model.resolve()),'model_identity':weights,'protocol':protocol,
         'protocol_sha256':fingerprint(protocol),'requested_device':'cuda'}
    run['run_id']=fingerprint(run)
    run.update(tasks_file=str(args.tasks.resolve()),shard={'index':args.shard_index,'count':args.num_shards,
                                                       'selected_ids':[t['id'] for t in selected]})
    args.output.mkdir(parents=True,exist_ok=True)
    with output_lock(args.output):
        path=args.output/'run.json'
        if path.exists() and json.loads(path.read_text())!=run:
            raise ValueError('Unified run settings changed; preserve this run and use a new output')
        write_json_atomic(path,run)
        saved=read_predictions(args.output/'predictions.jsonl',run['run_id'],{t['id']:t for t in selected},recover_tail=True)
        pending=[t for t in selected if t['id'] not in saved or (args.retry_infrastructure_errors and saved[t['id']].get('infrastructure_error'))]
        if not pending:
            print('All selected unified predictions already saved; no model loaded.',flush=True)
            return
        model=SharedLocalModel(args.model,args.assets)
        args.max_steps,args.prompt_tokens,args.max_new_tokens,args.max_pixels,args.temperature=10,28672,4096,262144,0.0
        with evidence_snapshots(args.evidence_cache):
            for task in pending:
                if asset_identity(args.assets)!=identity or runtime_identity(args.runtime)!=protocol['native_runtime_sha256']:
                    raise ValueError('Shared inference assets or native runtime changed during inference')
                seed=(42+int(hashlib.sha256(task['id'].encode()).hexdigest()[:8],16))%(2**32)
                model.torch.manual_seed(seed);model.torch.cuda.manual_seed_all(seed)
                record=(direct_task(task,model,session,args,run['run_id']) if args.mode=='direct'
                        else evaluate_task(task,model,agent,session,args,run['run_id']))
                record['task_seed']=seed
                with (args.output/'predictions.jsonl').open('a') as stream:
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                print(json.dumps({'id':record['id'],'status':record['status'],'rounds':record.get('native_rounds'),
                                  'seconds':record['elapsed_seconds']},ensure_ascii=False),flush=True)
                if record['infrastructure_error']:
                    raise SystemExit('Unified model/service failure; preserve records before resuming')


if __name__=='__main__':
    main()
