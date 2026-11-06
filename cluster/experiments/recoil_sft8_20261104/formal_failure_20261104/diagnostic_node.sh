#!/usr/bin/env bash
set -euo pipefail
source /cluster/home/shenqunlun/zhousheng/training4b/runtime.sh
R="$BASE/SearchAgent-recoil-20261104"
D="$BASE/recoil_sft8_20261104"
C="$R/train/OpenSearch-VL-Training/RL/rllm"
export GRPO_PROJECT_ROOT="$R"
export PYTHONPATH="$R/datasets/grpo/scripts:$R/datasets/grpo/integration:$C:$C/verl:$R/train/OpenSearch-VL-Training/RL/Megatron-LM:$R/train/OpenSearch-VL-Training/RL/mbridge"
export TMPDIR="$D/tmp" RAY_TMPDIR="$BASE" HF_HOME="$BASE/cache/huggingface" XDG_CACHE_HOME="$BASE/cache"
mkdir -p "$TMPDIR"
cd "$R"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$D/hardware.txt"
python "$BASE/visual-search-agent/cluster/prepare_sft_compat.py" --validate
python "$R/datasets/grpo/scripts/build_error_states.py" --sft-root "$D/sft_source"
python "$R/datasets/grpo/scripts/build_error_states.py" --sft-root "$D/sft_source" --held-out
python "$R/datasets/grpo/scripts/training_control.py" register
export RAY_PORT=$((20000 + SLURM_JOB_ID % 20000))
IP=$(getent ahostsv4 "$(hostname)" | awk 'NR==1 {print $1}')
export RAY_ADDRESS="$IP:$RAY_PORT" RAY_USAGE_STATS_ENABLED=0
ray start --head --temp-dir="$BASE/r" --node-ip-address="$IP" --port="$RAY_PORT" --num-gpus=8 --num-cpus=24 --include-dashboard=false --disable-usage-stats --block > "$D/ray-$SLURM_JOB_ID.log" 2>&1 &
RAY_PID=$!
trap 'kill "$RAY_PID" 2>/dev/null || true' EXIT
python - <<'PY'
import os,time,ray
for _ in range(120):
 try:
  ray.init(address=os.environ['RAY_ADDRESS'],logging_level='ERROR');break
 except Exception:time.sleep(2)
else:raise RuntimeError('Ray unavailable')
for _ in range(120):
 if ray.cluster_resources().get('GPU',0)==8:break
 time.sleep(2)
else:raise RuntimeError('8 allocated GPUs not registered')
ray.shutdown()
PY
export CUDA_LAUNCH_BLOCKING=1
python /cluster/home/shenqunlun/zhousheng/SearchAgent-recoil-20261104/datasets/grpo/scripts/training_control.py train8 --tag priority_sft8_failure_diagnostic --seed 42 --steps 1
