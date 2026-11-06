#!/bin/bash
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
PAIR=$P/training4b/sft_a100_recovery_20261103
VARIANT=$1
case "$VARIANT" in
 original) DATA=$P/datasets/sft/downloads_selected_20261103; OUT=$P/results4b/sft ;;
 v4) DATA=$P/datasets/sft/sft_v4_tidied_20261026; OUT=$P/results4b/sft_v4 ;;
 *) exit 2 ;;
esac
export VARIANT OUT
source "$P/visual-search-agent/cluster/runtime.sh"
export PATH="$P/envs/qwen4-sft-runtime/bin:$PATH"
unset PYTHONPATH
export PYTHONPATH="$P/visual-search-agent/train/OpenSearch-VL-Training/SFT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FORCE_TORCHRUN=1 NPROC_PER_NODE=4
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export TMPDIR="$P/tmp/sft-pair-${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="$P/cache/sft-pair-${SLURM_JOB_ID}/datasets"
export TORCH_EXTENSIONS_DIR="$P/cache/sft-pair-${SLURM_JOB_ID}/torch-extensions"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$TORCH_EXTENSIONS_DIR"
cd "$DATA"
test -f UPLOAD_VERIFIED.json
test -f "$P/training4b/SFT_ENV_READY.json"
python - <<'CHECK'
import torch,json,os,time
from pathlib import Path
assert torch.cuda.device_count()==4,torch.cuda.device_count()
names=[torch.cuda.get_device_name(i) for i in range(4)]
assert all('A100' in n for n in names),names
assert torch.cuda.is_bf16_supported()
p=Path(os.environ['OUT']);p.mkdir(parents=True,exist_ok=True)
f=p/'launch.json';assert not f.exists(),'Output already started; refusing duplicate'
f.write_text(json.dumps({'job_id':os.environ['SLURM_JOB_ID'],'node':os.uname().nodename,'gpus':names,'time':time.time(),'phase':'smoke','variant':os.environ['VARIANT'],'global_train_batch_size':64}))
print('SFT_PAIR_HARDWARE_READY',names,flush=True)
CHECK
llamafactory-cli train "$PAIR/${VARIANT}_smoke.yaml"
python - <<'SMOKE'
import json,time,os
from pathlib import Path
p=Path(os.environ['OUT']+'_a100_long_smoke');s=json.loads((p/'trainer_state.json').read_text());assert s['global_step']==2,s
(p/'SUCCESS.json').write_text(json.dumps({'steps':2,'time':time.time()}))
p=Path(os.environ['OUT']);f=p/'launch.json';x=json.loads(f.read_text());x['phase']='formal';x['formal_started_at']=time.time();f.write_text(json.dumps(x))
print('SFT_PAIR_SMOKE_SUCCESS',flush=True)
SMOKE
llamafactory-cli train "$PAIR/${VARIANT}_full.yaml"
python - <<'DONE'
import json,time,os
from pathlib import Path
p=Path(os.environ['OUT']);s=json.loads((p/'trainer_state.json').read_text());assert s['global_step']>=s['max_steps']>2,s
(p/'SUCCESS.json').write_text(json.dumps({'steps':s['global_step'],'time':time.time(),'variant':os.environ['VARIANT']}))
print('SFT_PAIR_FORMAL_SUCCESS',flush=True)
DONE
