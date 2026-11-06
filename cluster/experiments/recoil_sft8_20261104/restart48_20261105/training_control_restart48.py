import os,runpy
from pathlib import Path
D=Path(__file__).resolve().parent
mod=runpy.run_path('/cluster/home/shenqunlun/zhousheng/recoil_sft8_20261104/multinode_20261104/training_control_multinode_16gpu.py',run_name='restart_configuration')
ns=mod['ns'];original=ns['profile'];old_environment=ns['environment']
def environment(code):
 env=old_environment(code)
 env['PYTHONPATH']=str(D/'overlay')+os.pathsep+env.get('PYTHONPATH','')
 env['GRPO_RLLM_ROOT']=str(D/'runtime')
 return env
def profile(name,run):
 values=original(name,run)
 changes={'data.train_batch_size':'64','actor_rollout_ref.actor.ppo_mini_batch_size':'64','trainer.resume_mode':'disable'}
 result=[v.split('=',1)[0]+'='+changes[v.split('=',1)[0]] if v.split('=',1)[0] in changes else v for v in values]
 assert 'actor_rollout_ref.rollout.n=4' in result
 assert 64%8==0 and (64*4)%8==0
 return result
ns['profile']=profile
ns['environment']=environment
if __name__=='__main__':ns['main']()
