"""Scheduling-only extension: resume unchanged two-shard model evaluations."""
import argparse,json,os,time
from evaluation import unified_benchmark as parent
from evaluation import server_benchmark as base
from evaluation.common import write_json_atomic
from evaluation.remote import output_lock
from evaluation.native_benchmark import run_workers
from evaluation.unified_run import code_identity
from evaluation.unified_common import asset_identity,weight_identity
from evaluation.grpo_native import runtime_identity

def main():
 p=argparse.ArgumentParser();p.add_argument('condition',choices=['base_native','sft_native','base_grpo_native','finalize']);a=p.parse_args()
 out=base.ROOT/'outputs'/parent.NAME
 plan=json.loads((out/'plan.json').read_text())
 assert code_identity()==plan['code_sha256'],'Original evaluator changed'
 assert asset_identity(out/'shared_assets')==plan['shared_assets']
 assert runtime_identity(out/'runtime')==plan['native_runtime']
 if a.condition=='finalize':
  parent.verify_comparison(out,'full',778)
  write_json_atomic(out/'INFERENCE_SUCCESS.json',{'conditions':4,'count_each':778,'completed_at':time.time(),'placement_override':'parallel_execution.json'})
  print('ALL_FOUR_CONDITIONS_COMPLETE',flush=True);return
 assert os.environ.get('VS_BENCHMARK_IN_CONTAINER')=='1'
 import torch
 devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
 assert torch.cuda.is_available() and len(devices)==torch.cuda.device_count()==2
 names=[torch.cuda.get_device_name(i) for i in range(2)]
 expected=json.loads((out/'hardware.json').read_text())['gpu_names']
 assert names==expected,(names,expected)
 assert weight_identity(parent.models()[a.condition])[0]==plan['weights'][a.condition]['files']
 condition=out/a.condition
 condition.mkdir(exist_ok=True)
 with output_lock(condition):
  write_json_atomic(condition/'parallel_hardware.json',{'job_id':os.environ['SLURM_JOB_ID'],'host':os.uname().nodename,'gpu_names':names,'resume_existing_answers':True,'same_generation_and_sharding':True})
  env,_=base.runtime_environment()
  phase,commands=parent.commands(out,a.condition,devices)
  run_workers(commands,phase,env)
  run=parent.merged(out,a.condition,'full',778)
  write_json_atomic(condition/'INFERENCE_SUCCESS.json',{'count':778,'run_id':run['run_id']})
  print('CONDITION_COMPLETE',a.condition,flush=True)
if __name__=='__main__':main()
