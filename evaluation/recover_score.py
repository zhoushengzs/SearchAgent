import os,runpy,json
from pathlib import Path
p=Path('/cluster/home/shenqunlun/zhousheng/benchmark-vs778/outputs/multinode-preflight.json')
original=json.loads(p.read_text())['job_id']
os.environ['SLURM_JOB_ID']=str(original)
import sys
sys.argv=['evaluation.server_benchmark_multinode','finish']
runpy.run_module('evaluation.server_benchmark_multinode',run_name='__main__')
