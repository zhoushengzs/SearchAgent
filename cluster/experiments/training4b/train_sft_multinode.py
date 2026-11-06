"""Run the existing project on an allocated 16/24 GPU Ray cluster."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from training_control import environment,profile,prepare_registry,choose_debug_rows
BASE=Path('/cluster/home/shenqunlun/zhousheng')
ROOT=BASE/'visual-search-agent/datasets/grpo'
CODE=BASE/'visual-search-agent/train/OpenSearch-VL-Training/RL/rllm'

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--stage',choices=['base','sft'],required=True)
 p.add_argument('--smoke',action='store_true')
 p.add_argument('--dry-run',action='store_true')
 a=p.parse_args()
 nodes=int(os.environ['SLURM_JOB_NUM_NODES'])
 if nodes not in (1,2,3): raise SystemExit('Need 1, 2 or 3 allocated nodes')
 model=BASE/('Qwen3-VL-4B-Instruct' if a.stage=='base' else 'results4b/sft')
 if a.stage == 'sft' and not a.dry_run:
  assert (model/'SUCCESS.json').is_file(), '4B SFT has not completed'
  config=json.loads((model/'config.json').read_text())
  base_config=json.loads((BASE/'Qwen3-VL-4B-Instruct/config.json').read_text())
  assert config['text_config']['hidden_size']==base_config['text_config']['hidden_size'], 'Wrong model size'
 run=BASE/'results4b'/((a.stage+'_grpo')+('_smoke' if a.smoke else ''))
 env=environment(CODE)
 env['PYTHONPATH']=str(BASE/'training4b/compat')+os.pathsep+env.get('PYTHONPATH','')
 overrides=profile('smoke8' if a.smoke else 'train8',run)
 # Both layouts use identical batch sizes, divisible by their GPU groups.
 values={
 'trainer.nnodes':nodes,'data.train_batch_size':24 if a.smoke else 48,
 'data.val_batch_size':24,'actor_rollout_ref.actor.ppo_mini_batch_size':24 if a.smoke else 12,
 'rllm.workflow.n_parallel_tasks':8 if a.smoke else 16,
 'rllm.workflow.n_parallel_tools':16 if a.smoke else 32,
 'trainer.max_actor_ckpt_to_keep':2,
 'trainer.resume_mode':'auto',
 'actor_rollout_ref.actor.checkpoint.save_contents':'[model,optimizer,extra,hf_model]',
 'actor_rollout_ref.actor.checkpoint.load_contents':'[model,optimizer,extra]',
 }
 overrides=[x for x in overrides if x.split('=',1)[0] not in values]
 overrides += [f'{k}={v}' for k,v in values.items()]
 command=['bash',str(ROOT/'integration/official_8gpu_grpo.sh'),*overrides]
 if a.dry_run:
  print(json.dumps({'nodes':nodes,'model':str(model),'run':str(run),'command':command},indent=2));return
 if (run/'SUCCESS.json').exists(): print('Already completed:',run);return
 if not (model/'config.json').is_file():raise SystemExit('Model upload is incomplete')
 env.update(GRPO_MODEL_PATH=str(model),GRPO_MAX_PROMPT_LENGTH='8192',
            GRPO_MAX_RESPONSE_LENGTH='2048' if a.smoke else '4096',
            GRPO_AGENT_MAX_STEPS='6' if a.smoke else '20')
 if not env.get('JUDGE_MODEL') or not env.get('JUDGE_BASE_URL') or not (env.get('JUDGE_API_KEY') or env.get('OPENAI_API_KEY')):
  raise SystemExit('Judge service is not configured')
 subprocess.run([sys.executable,str(ROOT/'scripts/check_project_tools.py'),'--live'],env=env,check=True)
 run.mkdir(parents=True,exist_ok=True)
 registry_root=prepare_registry(CODE,env,run,False)
 if a.smoke:
  from register_merged import load_registry,bind_registry
  cls=load_registry(ROOT,CODE)
  bind_registry(cls,ROOT)
  selected={split:choose_debug_rows(cls.load_dataset('Vision-DeepResearch-QA',split).get_data(),limit) for split,limit in [('train',96),('test',24)]}
  registry_root=run/'debug_data'
  bind_registry(cls,registry_root)
  for split,rows in selected.items():cls.register_dataset('Vision-DeepResearch-QA',rows,split)
 env['GRPO_REGISTRY_ROOT']=str(registry_root)
 (run/'launch.json').write_text(json.dumps({'nodes':nodes,'model':str(model),'command':command,'job_id':os.environ['SLURM_JOB_ID']},indent=2))
 subprocess.run(command,env=env,check=True)
 tracker=run/'checkpoints/latest_checkpointed_iteration.txt'
 step=int(tracker.read_text().strip())
 hf=run/'checkpoints'/f'global_step_{step}'/'actor/huggingface'
 from safetensors import safe_open
 from transformers import AutoConfig, AutoTokenizer, AutoProcessor
 AutoConfig.from_pretrained(hf,local_files_only=True)
 weights=list(hf.glob('*.safetensors'))
 if not weights:raise RuntimeError('Training exited without exported model weights')
 keys=set()
 for f in weights:
  with safe_open(f,framework='pt',device='cpu') as tensors:
   names=list(tensors.keys())
   if not names:raise RuntimeError('Empty weights: '+str(f))
   keys.update(names)
   for name in names:tensors.get_slice(name).get_shape()
 index=hf/'model.safetensors.index.json'
 if index.exists():
  mapping=json.loads(index.read_text())['weight_map']
  if not set(mapping)<=keys or not all((hf/f).is_file() for f in mapping.values()):raise RuntimeError('Incomplete model shards')
 # Some upstream exporters save only weights/config; preserve the original processor.
 import shutil
 for pattern in ['*token*','*processor*','vocab.json','merges.txt','chat_template*','generation_config.json']:
  for f in model.glob(pattern):
   if f.is_file() and not (hf/f.name).exists():shutil.copy2(f,hf/f.name)
 AutoTokenizer.from_pretrained(hf,local_files_only=True)
 AutoProcessor.from_pretrained(hf,local_files_only=True)
 link=run/'final_model'
 if link.is_symlink():link.unlink()
 link.symlink_to(hf,target_is_directory=True)
 (run/'SUCCESS.json').write_text(json.dumps({'stage':a.stage,'smoke':a.smoke,'step':step,'model':str(hf),'completed_at':time.time()},indent=2))
if __name__=='__main__':main()
