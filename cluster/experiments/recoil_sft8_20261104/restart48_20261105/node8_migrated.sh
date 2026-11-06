#!/usr/bin/env bash
set -euo pipefail
source /cluster/home/shenqunlun/projects/zhousheng/runtime_migration_20261105/grpo_runtime.sh
R="$BASE/SearchAgent-recoil-20261104"
D="$BASE/recoil_sft8_20261104/restart48_20261105"
C="$R/train/OpenSearch-VL-Training/RL/rllm"
export GRPO_PROJECT_ROOT="$R"
export PYTHONPATH="$D/overlay:$R/datasets/grpo/scripts:$R/datasets/grpo/integration:$C:$C/verl:$R/train/OpenSearch-VL-Training/RL/Megatron-LM:$R/train/OpenSearch-VL-Training/RL/mbridge"
export RECOIL_NNODES=2 RECOIL_GPUS_PER_NODE=4
export JOB_DIR="$D/job-$SLURM_JOB_ID"
mkdir -p "$JOB_DIR"
export TMPDIR="$BASE/tmp/$SLURM_JOB_ID-$SLURM_PROCID"
mkdir -p "$TMPDIR"
python - <<'PREFLIGHT'
from pathlib import Path
import os
assert Path('/proc/meminfo').is_file()
with open('/dev/null','w') as f:f.write('')
p=Path('/dev/shm')/('zs-preflight-'+str(os.getpid()))
p.write_text('ok');p.unlink()
print('CONTAINER_PROC_DEV_SHM_WRITABLE',flush=True)
PREFLIGHT

export RAY_USAGE_STATS_ENABLED=0
export RAY_TMPDIR=/tmp
export RAY_ADDRESS="$TRAIN_HEAD_IP:$TRAIN_PORT"
export RECOIL_NNODES=2 RECOIL_GPUS_PER_NODE=4
IP=$(getent ahostsv4 "$(hostname)" | awk 'NR==1 {print $1}')
if [ "$SLURM_PROCID" = 0 ]; then
 ray start --head --temp-dir=/tmp/r --node-ip-address="$IP" --port="$TRAIN_PORT" --num-gpus=4 --num-cpus=24 --include-dashboard=false --disable-usage-stats --block > "$JOB_DIR/ray-$SLURM_PROCID.log" 2>&1 &
else
 sleep 12
 ray start --address="$RAY_ADDRESS" --node-ip-address="$IP" --num-gpus=4 --num-cpus=24 --disable-usage-stats --block > "$JOB_DIR/ray-$SLURM_PROCID.log" 2>&1 &
fi
RAY_PID=$!
trap 'kill "$RAY_PID" 2>/dev/null || true' EXIT
if [ "$SLURM_PROCID" = 0 ]; then
 sleep 20
 python "$D/ray_check_8gpu.py"
 cd "$R"
 python "$D/training_control_restart48.py" train8 --model-path /cluster/home/shenqunlun/projects/zhousheng/qwen3-8b-sft-transformers4 --config-only > "$JOB_DIR/expanded_config.txt" 2>&1
 python "$D/training_control_restart48.py" train8 --model-path /cluster/home/shenqunlun/projects/zhousheng/qwen3-8b-sft-transformers4 --tag priority_sft8_restart64_20261105 --seed 42
 touch "$JOB_DIR/TRAINING_PROCESS_EXIT_ZERO"
else
 while kill -0 "$RAY_PID" 2>/dev/null && [ ! -f "$JOB_DIR/TRAINING_PROCESS_EXIT_ZERO" ]; do sleep 5; done
fi
