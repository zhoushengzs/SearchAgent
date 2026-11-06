#!/bin/bash
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
D=$P/training4b/sft_resume_20261104
source "$P/visual-search-agent/cluster/runtime.sh"
export PATH="$P/envs/qwen4-sft-runtime/bin:$PATH"
export PYTHONPATH="$P/training4b/sft_8gpu_sequential_20261103/compat:$P/visual-search-agent/train/OpenSearch-VL-Training/SFT/src"
export SFT_CHUNKED_LOSS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$P/tmp/sft-resume-$SLURM_JOB_ID-$SLURM_PROCID"
export HF_DATASETS_CACHE="$P/cache/sft-pair-4412139/datasets"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
cd "$P/datasets/sft/downloads_selected_20261103"
python - <<'CHECK'
import torch,json,os,socket
from pathlib import Path
from transformers.loss.loss_utils import LOSS_MAPPING
assert torch.cuda.device_count()==4
assert LOSS_MAPPING['ForCausalLM'].__name__=='chunked_causal_loss'
p=Path('/cluster/home/shenqunlun/zhousheng/training4b/sft_resume_20261104')
(p/('hardware-'+os.environ['SLURM_JOB_ID']+'-'+os.environ['SLURM_PROCID']+'.json')).write_text(json.dumps({'host':socket.gethostname(),'gpus':[torch.cuda.get_device_name(i) for i in range(4)]}))
CHECK
torchrun --nnodes=2 --nproc_per_node=4 --node_rank="$SLURM_PROCID" --master_addr="$SFT_HEAD" --master_port="$SFT_PORT" "$P/visual-search-agent/train/OpenSearch-VL-Training/SFT/src/train.py" "$D/resume.yaml"
