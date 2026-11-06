import os,time,json,socket
from pathlib import Path
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
ray.init(address=os.environ['RAY_ADDRESS'],_node_ip_address=os.environ['TRAIN_HEAD_IP'],logging_level='ERROR')
for _ in range(180):
 nodes=[n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU',0)==4]
 if len(nodes)==2: break
 time.sleep(2)
else: raise RuntimeError('Expected two Ray nodes each with four allocated GPUs')
@ray.remote(num_gpus=4,num_cpus=1)
class Probe:
 def check(self):
  import torch
  assert torch.cuda.device_count()==4
  return {'node':socket.gethostname(),'visible_gpu_count':torch.cuda.device_count(),'models':[torch.cuda.get_device_name(i) for i in range(4)]}
actors=[Probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n['NodeID'],soft=False)).remote() for n in nodes]
rows=ray.get([a.check.remote() for a in actors])
assert len(set(x['node'] for x in rows))==2
(Path(os.environ['JOB_DIR'])/'ray_allocation.json').write_text(json.dumps({'passed':True,'nodes':rows},indent=2))
for a in actors:ray.kill(a)
ray.shutdown()
