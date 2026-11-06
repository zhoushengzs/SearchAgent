#!/bin/bash
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
source "$P/visual-search-agent/cluster/runtime.sh"
export PATH="$P/envs/qwen4-sft-runtime/bin:$PATH"
export PYTHONPATH="$P/visual-search-agent/train/OpenSearch-VL-Training/SFT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FORCE_TORCHRUN=1 NPROC_PER_NODE=8
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export TMPDIR="$P/tmp/sft-early-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
python - <<'PY'
import sys,torch,json,os,time
from pathlib import Path
w=Path(sys.prefix)/'bin/torchrun';w.write_text('#!'+sys.executable+'\nfrom torch.distributed.run import main\nif __name__ == "__main__": main()\n');w.chmod(0o755)
assert torch.cuda.device_count()==8
p=Path('/cluster/home/shenqunlun/zhousheng/results4b/sft_early_smoke');p.mkdir(parents=True,exist_ok=True)
(p/'launch.json').write_text(json.dumps({'job_id':os.environ['SLURM_JOB_ID'],'node':os.uname().nodename,'gpus':[torch.cuda.get_device_name(i) for i in range(8)],'time':time.time()}))
PY
llamafactory-cli train "$P/training4b/sft_early_smoke.yaml"
python - <<'PY'
import json,time,math
from pathlib import Path
p=Path('/cluster/home/shenqunlun/zhousheng/results4b/sft_early_smoke');s=json.loads((p/'trainer_state.json').read_text());assert s['global_step']==2,s
losses=[x['loss'] for x in s['log_history'] if 'loss' in x];assert losses and all(math.isfinite(x) for x in losses)
(p/'SUCCESS.json').write_text(json.dumps({'steps':2,'losses':losses,'time':time.time(),'scope':'first16 verified training samples, eight GPUs, same SFT configuration except smoke step count and accumulation'}))
print('SFT_GPU_SMOKE_SUCCESS')
PY
