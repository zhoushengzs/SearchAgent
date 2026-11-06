#!/bin/bash
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
source "$P/visual-search-agent/cluster/runtime.sh"
export PATH="$P/envs/qwen4-sft-runtime/bin:$PATH"
unset PYTHONPATH
export PYTHONPATH="$P/visual-search-agent/train/OpenSearch-VL-Training/SFT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FORCE_TORCHRUN=1 NPROC_PER_NODE=8
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
cd "$P/datasets/sft/downloads_selected_20261103"
test -f UPLOAD_VERIFIED.json
test -f "$P/training4b/SFT_ENV_READY.json"
python - <<'PY'
import torch,json,os,time
from pathlib import Path
import sys
wrapper=Path(sys.prefix)/'bin/torchrun'
wrapper.write_text('#!'+sys.executable+'\nfrom torch.distributed.run import main\nif __name__ == '__main__': main()\n');wrapper.chmod(0o755)
assert torch.cuda.device_count()==8,torch.cuda.device_count()
assert all('A100' in torch.cuda.get_device_name(i) for i in range(8))
p=Path('/cluster/home/shenqunlun/zhousheng/results4b/sft');p.mkdir(parents=True,exist_ok=True)
(p/'launch.json').write_text(json.dumps({'job_id':os.environ['SLURM_JOB_ID'],'node':os.uname().nodename,'gpus':[torch.cuda.get_device_name(i) for i in range(8)],'time':time.time(),'phase':'smoke'}))
PY
llamafactory-cli train "$P/training4b/sft_smoke.yaml"
python - <<'PY'
import json,time
from pathlib import Path
p=Path('/cluster/home/shenqunlun/zhousheng/results4b/sft_smoke')
s=json.loads((p/'trainer_state.json').read_text());assert s['global_step']==2,s
(p/'SUCCESS.json').write_text(json.dumps({'steps':2,'time':time.time()}))
PY
llamafactory-cli train "$P/training4b/sft_full.yaml"
python - <<'PY'
import json,time
from pathlib import Path
p=Path('/cluster/home/shenqunlun/zhousheng/results4b/sft');s=json.loads((p/'trainer_state.json').read_text());assert s['global_step']>2
(p/'SUCCESS.json').write_text(json.dumps({'steps':s['global_step'],'time':time.time()}))
PY
