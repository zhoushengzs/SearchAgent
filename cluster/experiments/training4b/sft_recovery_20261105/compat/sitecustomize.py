import os
if os.environ.get('SFT_CHUNKED_LOSS')=='1':
 from chunked_sft_loss import install
 install()
 import torch,json
 import transformers.trainer as tr
 original=tr.set_rng_state_for_device
 def restore(device_name,device,checkpoint_rng_state,is_distributed):
  if device_name!='CUDA':return original(device_name,device,checkpoint_rng_state,is_distributed)
  saved=checkpoint_rng_state['cuda'];rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
  if isinstance(saved,(list,tuple)):
   index=rank if len(saved)==8 else local
   assert index<len(saved),(rank,local,len(saved))
   state=saved[index]
  else:state=saved
  device.set_rng_state(state,device=local)
  assert torch.equal(device.get_rng_state(local).cpu(),state.cpu())
  from pathlib import Path
  out=Path(os.environ['SFT_RNG_AUDIT']);out.mkdir(exist_ok=True,parents=True)
  (out/('rank-'+str(rank)+'.json')).write_text(json.dumps({'passed':True,'global_rank':rank,'local_rank':local,'saved_devices':len(saved) if isinstance(saved,(list,tuple)) else 1}))
  print('SFT_RNG_RESTORED_AND_VERIFIED rank='+str(rank),flush=True)
 tr.set_rng_state_for_device=restore
