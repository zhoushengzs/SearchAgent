"""Allocated multi-node workers; identical protocol, global disjoint shards."""
import argparse,json,os,subprocess,sys,time
from types import SimpleNamespace
from evaluation import server_benchmark as base
from evaluation.launch import worker_commands
from evaluation.merge import merge_shards
from evaluation.common import write_json_atomic

def commands(node, nodes, devices, output):
    if nodes<1 or not 0<=node<nodes or not devices:raise ValueError('Invalid allocation')
    a=SimpleNamespace(model=str(base.MODEL),tasks=base.BUNDLE/'tasks.jsonl',output=output,gpus=','.join(devices),expected_count=778,mode='agent',protocol='training',dtype='bfloat16',max_steps=10,max_new_tokens=4096,context_tokens=32768,max_pixels=262144,image_side=1536,repetition_penalty=1.0,repeat_limit=2,tool_summaries=False,env_file=base.ENV_FILE,limit=None,image_root=None,original_root=None,retry_infrastructure_errors=False,allow_download=False)
    result=[]
    for local,(device,cmd) in enumerate(worker_commands(a)):
        rank=node*len(devices)+local
        cmd[cmd.index('--num-shards')+1]=str(nodes*len(devices))
        cmd[cmd.index('--shard-index')+1]=str(rank)
        cmd[cmd.index('--output')+1]=str(output/'shards'/f'{rank:03d}')
        result.append((rank,device,cmd))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','workers','finish']);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('VS_BENCHMARK_IN_CONTAINER')!='1':raise ValueError('Allocated container required')
    env,key=base.runtime_environment();output=base.ROOT/'outputs/base165-778'
    if a.stage=='prepare':
        verified=base.validate_bundle();base.validate_model()
        gpus=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        base.run_stage('smoke',2,gpus[0],env,key)
        write_json_atomic(base.ROOT/'outputs/multinode-preflight.json',{'job_id':os.environ['SLURM_JOB_ID'],'bundle_verified':verified})
    elif a.stage=='workers':
        import torch
        nodes=int(os.environ['SLURM_JOB_NUM_NODES']);node=int(os.environ['SLURM_NODEID']);devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        expected=int(os.environ['VS_BENCH_GPUS_PER_NODE'])
        if len(devices)!=expected or torch.cuda.device_count()!=expected:raise ValueError('GPU allocation differs')
        output.mkdir(exist_ok=True,parents=True);procs=[];streams=[]
        try:
            for rank,device,cmd in commands(node,nodes,devices,output):
                stream=(output/f'worker-{rank:03d}.log').open('a');streams.append(stream)
                procs.append(subprocess.Popen(cmd,cwd=base.ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=device),stdout=stream,stderr=subprocess.STDOUT))
            while any(c.poll() is None for c in procs):
                if any(c.poll() not in (None,0) for c in procs):raise RuntimeError('Worker failed; inspect worker logs')
                time.sleep(2)
            if any(c.returncode for c in procs):raise RuntimeError('Worker failed')
        finally:
            for c in procs:
                if c.poll() is None:c.terminate()
            for c in procs:
                try:c.wait(timeout=20)
                except subprocess.TimeoutExpired:c.kill();c.wait()
            for stream in streams:stream.close()
    else:
        verified=json.loads((base.ROOT/'outputs/multinode-preflight.json').read_text())
        if verified['job_id']!=os.environ['SLURM_JOB_ID']:raise ValueError('Preflight belongs to another job')
        merged=output/'merged';count=merge_shards(output/'shards',merged,base.BUNDLE/'tasks.jsonl')
        if count!=778:raise ValueError('Incomplete merged answers')
        score=base.ROOT/'outputs/base165-778-score'
        subprocess.run([sys.executable,'-u','-m','evaluation.score','--run',str(merged),'--tasks',str(base.BUNDLE/'tasks.jsonl'),'--references',str(base.BUNDLE/'references.jsonl'),'--output',str(score),'--judge-model',env['JUDGE_MODEL'],'--base-url',env['JUDGE_BASE_URL'],'--key-env',key,'--env-file',str(base.ENV_FILE),'--workers','1'],cwd=base.ROOT,env=env,check=True)
        summary=json.loads((score/'summary.json').read_text())
        if summary.get('count')!=778 or summary.get('metric')!=base.METRIC or summary.get('scoring_scope'):raise ValueError('Scoring incomplete')
        write_json_atomic(output/'SERVER_SUCCESS.json',{'job_id':os.environ['SLURM_JOB_ID'],'checkpoint_step':165,'model':str(base.MODEL),'bundle_verified':verified['bundle_verified'],'count':778,'metric':base.METRIC,'accuracy_pct':summary['micro_accuracy_pct'],'completed_at':time.time()})
if __name__=='__main__':main()
