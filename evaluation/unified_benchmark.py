"""Fresh, frozen four-condition 778-task comparison; historical runs are read-only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluation import server_benchmark as base
from evaluation.common import write_json_atomic
from evaluation.grpo_native import runtime_identity
from evaluation.merge import merge_shards
from evaluation.native_benchmark import run_workers
from evaluation.prepare import digest
from evaluation.remote import output_lock
from evaluation.run import fingerprint
from evaluation.unified_common import ASSETS, CONDITIONS, LABELS, asset_identity, weight_identity
from evaluation.unified_run import code_identity

NAME='unified6_fourway_778_20261103'


def models():
    return {'base_direct':base.PERSONAL/'Qwen3-VL-8B-Instruct',
            'base_native':base.PERSONAL/'Qwen3-VL-8B-Instruct',
            'sft_native':base.PERSONAL/'qwen3-8b-sft',
            'base_grpo_native':base.MODEL}


def freeze_inputs(output):
    target=output/'runtime'
    metadata=output/'runtime_snapshot.json'
    if metadata.exists():
        saved=json.loads(metadata.read_text())
        if runtime_identity(target)!=saved['sha256']:
            raise ValueError('The unified native runtime snapshot changed')
    else:
        source=base.ROOT/'outputs/base165-native-training-778-20261103/runtime'
        identity=runtime_identity(source)
        for name in identity:
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,path)
        write_json_atomic(metadata,{'source':str(source),'sha256':identity})
    assets=output/'shared_assets'
    manifest=output/'assets_snapshot.json'
    if manifest.exists():
        if asset_identity(assets)!=json.loads(manifest.read_text())['sha256']:
            raise ValueError('The shared text/image/generation assets changed')
    else:
        assets.mkdir(exist_ok=True)
        source=models()['base_native']
        for name in ASSETS:
            if (source/name).is_file():shutil.copy2(source/name,assets/name)
        generation=json.loads((assets/'generation_config.json').read_text())
        generation.update(do_sample=False,temperature=1.0,top_p=1.0,top_k=50,repetition_penalty=1.0)
        write_json_atomic(assets/'generation_config.json',generation)
        write_json_atomic(manifest,{'source':str(source),'sha256':asset_identity(assets),
                                  'changes':'common greedy defaults; preserve Base token IDs and stop tokens'})
    return target,assets


def make_plan(output,verified,runtime,assets,env):
    weights={};reference_shapes=None
    for condition,path in models().items():
        identity,shapes=weight_identity(path)
        if reference_shapes is None:reference_shapes=shapes
        if shapes!=reference_shapes:
            raise ValueError('Weight tensor names/shapes differ from Base: '+condition)
        weights[condition]={'path':str(path),'files':identity,'tensor_count':len(shapes)}
    return {'schema':'unified-six-tool-four-way-v1','conditions':list(CONDITIONS),'labels':LABELS,
            'weights':weights,'bundle_verified':verified,'shared_assets':asset_identity(assets),
            'native_runtime':runtime_identity(runtime),'code_sha256':code_identity(),
            'generation':{'agent_steps':10,'direct_calls':1,'max_new_tokens':4096,'context_tokens':32768,
                          'prompt_tokens':28672,'max_pixels':262144,'min_pixels':65536,'image_side':1536,
                          'do_sample':False,'repetition_penalty':1.0,'seed':42,'dtype':'bfloat16','attention':'sdpa'},
            'rules':'one shared native parser and agent loop; six tools for agent conditions; no tool execution in direct condition; no extra format recovery or forced answers',
            'evidence':'shared first-successful real search/page/image snapshots; transient errors logged, not synthesized',
            'scoring':{'judge_model':env['JUDGE_MODEL'],'base_url':env['JUDGE_BASE_URL'],
                       'thinking':'disabled','max_tokens':4096,'temperature':0,'answer_extraction':'native_answered_status',
                       'same_question_reference_candidate_reuses_same_judgment':True},
            'placement':'all four conditions sequentially use the same allocated GPU set on gnode05',
            'reference_visibility':'detached references are read only by the scorer'}


def commands(output,condition,devices,limit=None,retry=False):
    phase=output/condition/('smoke' if limit is not None else 'full')
    result=[]
    for index,device in enumerate(devices):
        cmd=[sys.executable,'-u','-m','evaluation.unified_run','--model',str(models()[condition]),
             '--assets',str(output/'shared_assets'),'--runtime',str(output/'runtime'),
             '--tasks',str(base.BUNDLE/'tasks.jsonl'),'--output',str(phase/'shards'/f'{index:03d}'),
             '--evidence-cache',str(output/'evidence_cache'),'--env-file',str(base.ENV_FILE),
             '--mode','direct' if condition=='base_direct' else 'agent',
             '--num-shards',str(len(devices)),'--shard-index',str(index)]
        if limit is not None:cmd+=['--limit',str(limit)]
        if retry:cmd.append('--retry-infrastructure-errors')
        result.append((device,cmd))
    return phase,result


def merged(output,condition,phase,count):
    root=output/condition/phase
    n=merge_shards(root/'shards',root/'merged',base.BUNDLE/'tasks.jsonl')
    if n!=count:raise ValueError('Incomplete unified condition: '+condition)
    run=json.loads((root/'merged/run.json').read_text())
    expected=0 if condition=='base_direct' else 6
    if len(run['protocol']['tools'])!=expected:
        raise ValueError('Unexpected tool set in '+condition)
    return run


def verify_comparison(output,phase,count):
    runs={c:merged(output,c,phase,count) for c in CONDITIONS}
    protocols=[runs[c]['protocol'] for c in CONDITIONS if c!='base_direct']
    if any(p!=protocols[0] for p in protocols[1:]):
        raise ValueError('Agent conditions differ in settings besides model weights')
    baseline=runs['base_direct']['protocol']
    for field in ['shared_assets_sha256','native_runtime_sha256','tasks_sha256','selected_ids','max_new_tokens',
                  'context_tokens','prompt_tokens','max_pixels','do_sample','repetition_penalty','seed','dtype','attention','versions','code_sha256']:
        if baseline[field]!=protocols[0][field]:
            raise ValueError('Direct condition unexpectedly differs: '+field)
    if phase=='smoke':
        first_inputs={}
        for condition in CONDITIONS:
            for path in (output/condition/'smoke/shards').glob('*/artifacts/*/raw_calls.jsonl'):
                row=json.loads(path.read_text().splitlines()[0])
                normalized=[]
                for m in row['messages']:
                    n={k:v for k,v in m.items() if k!='images'}
                    if 'images' in m:n['images']=[x['sha256'] for x in m['images']]
                    normalized.append(n)
                first_inputs[(condition,path.parent.name)]=fingerprint(normalized)
        for task in protocols[0]['selected_ids']:
            values=[first_inputs[(c,task)] for c in CONDITIONS if c!='base_direct']
            if len(set(values))!=1:
                raise ValueError('The actual initial model input differs between agent conditions')
    return runs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','infer','score'],default='prepare')
    p.add_argument('--retry-infrastructure-errors',action='store_true')
    args=p.parse_args()
    output=base.inside_personal(base.ROOT/'outputs'/NAME);output.mkdir(parents=True,exist_ok=True)
    with output_lock(output):
        verified=base.validate_bundle();base.validate_model()
        env,key=base.runtime_environment()
        runtime,assets=freeze_inputs(output)
        plan=make_plan(output,verified,runtime,assets,env)
        path=output/'plan.json'
        if path.exists() and json.loads(path.read_text())!=plan:
            raise ValueError('Unified experiment inputs changed; do not mix results')
        write_json_atomic(path,plan)
        if args.stage=='prepare':
            print('UNIFIED_PLAN_VERIFIED; four fresh conditions; no inference or judge requests.',flush=True)
            return
        if args.stage=='score':
            verify_comparison(output,'full',778)
            subprocess.run([sys.executable,'-u','-m','evaluation.unified_score','--output',str(output),
                            '--tasks',str(base.BUNDLE/'tasks.jsonl'),'--references',str(base.BUNDLE/'references.jsonl'),
                            '--env-file',str(base.ENV_FILE),'--key-env',key],cwd=base.ROOT,env=env,check=True)
            return
        if not os.environ.get('SLURM_JOB_ID') or os.environ.get('VS_BENCHMARK_IN_CONTAINER')!='1':
            raise ValueError('Unified inference must use its own GPU job/container')
        import torch
        devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        if not torch.cuda.is_available() or not all(devices) or len(devices)!=torch.cuda.device_count():
            raise ValueError('Allocated GPUs are unavailable')
        names=[torch.cuda.get_device_name(i) for i in range(len(devices))]
        if len(set(names))!=1:raise ValueError('All comparison GPUs must be the same type')
        write_json_atomic(output/'hardware.json',{'job_id':os.environ['SLURM_JOB_ID'],'host':os.uname().nodename,
                                                'gpu_names':names,'same_allocation_for_all_conditions':True})
        if not (output/'SMOKE_SUCCESS.json').exists():
            for condition in CONDITIONS:
                phase,cmds=commands(output,condition,devices[:1],limit=2,retry=args.retry_infrastructure_errors)
                run_workers(cmds,phase,env)
                merged(output,condition,'smoke',2)
            verify_comparison(output,'smoke',2)
            write_json_atomic(output/'SMOKE_SUCCESS.json',{'conditions':4,'count_each':2,
                              'agent_protocols_identical':True,'actual_initial_agent_inputs_identical':True,
                              'plan_sha256':fingerprint(plan)})
            print('UNIFIED_SMOKE_SUCCESS four conditions x two tasks; only weights differ among agents.',flush=True)
        for condition in CONDITIONS:
            if code_identity()!=plan['code_sha256']:
                raise ValueError('Evaluation code changed between conditions')
            phase,cmds=commands(output,condition,devices,retry=args.retry_infrastructure_errors)
            print('START_CONDITION',condition,flush=True)
            run_workers(cmds,phase,env)
            run=merged(output,condition,'full',778)
            write_json_atomic(output/condition/'INFERENCE_SUCCESS.json',{'count':778,'run_id':run['run_id']})
            print('CONDITION_INFERENCE_SUCCESS',condition,'778/778',flush=True)
        verify_comparison(output,'full',778)
        write_json_atomic(output/'INFERENCE_SUCCESS.json',{'conditions':4,'count_each':778,'completed_at':time.time()})
        print('UNIFIED_INFERENCE_SUCCESS 4 x 778; no historical answers reused.',flush=True)


if __name__=='__main__':main()
