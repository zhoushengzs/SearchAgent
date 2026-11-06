import os,json,socket,datetime
from pathlib import Path
import torch,torch.distributed as dist
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
assert int(os.environ['WORLD_SIZE'])==8 and torch.cuda.device_count()==4
assert 'A100' in torch.cuda.get_device_name(local)
assert torch.cuda.get_device_properties(local).total_memory>75*1024**3
torch.cuda.set_device(local)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
for size in [1,1024,1048576]:
 t=torch.full((size,),float(rank+1),device='cuda')
 dist.all_reduce(t)
 assert torch.all(t==36).item()
rows=[None]*8
dist.all_gather_object(rows,{'rank':rank,'node':socket.gethostname(),'local_gpu':local,'gpu':torch.cuda.get_device_name(local)})
if rank==0:
 assert len(set(r['node'] for r in rows))==2
 (Path(os.environ['CHECK_DIR'])/'nccl.json').write_text(json.dumps({'passed':True,'ranks':rows},indent=2))
dist.barrier();dist.destroy_process_group()
