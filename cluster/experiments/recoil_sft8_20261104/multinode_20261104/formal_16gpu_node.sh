#!/usr/bin/env bash
set -euo pipefail
source /cluster/home/shenqunlun/zhousheng/training4b/runtime.sh
R="$BASE/SearchAgent-recoil-20261104"
D="$BASE/recoil_sft8_20261104/multinode_20261104"
C="$R/train/OpenSearch-VL-Training/RL/rllm"
export GRPO_PROJECT_ROOT="$R"
export PYTHONPATH="$R/datasets/grpo/scripts:$R/datasets/grpo/integration:$C:$C/verl:$R/train/OpenSearch-VL-Training/RL/Megatron-LM:$R/train/OpenSearch-VL-Training/RL/mbridge"
export JOB_DIR="$D/job-$SLURM_JOB_ID"
mkdir -p "$JOB_DIR"
export TMPDIR="$BASE/tmp/$SLURM_JOB_ID-$SLURM_PROCID"
mkdir -p "$TMPDIR"
export RAY_USAGE_STATS_ENABLED=0
export RAY_ADDRESS="$TRAIN_HEAD_IP:$TRAIN_PORT"
export RECOIL_NNODES=2 RECOIL_GPUS_PER_NODE=8
IP=$(getent ahostsv4 "$(hostname)" | awk 'NR==1 {print $1}')
if [ "$SLURM_PROCID" = 0 ]; then
 ray start --head --temp-dir="$BASE/rm" --node-ip-address="$IP" --port="$TRAIN_PORT" --num-gpus=8 --num-cpus=24 --include-dashboard=false --disable-usage-stats --block > "$JOB_DIR/ray-$SLURM_PROCID.log" 2>&1 &
else
 sleep 12
 ray start --address="$RAY_ADDRESS" --node-ip-address="$IP" --num-gpus=8 --num-cpus=24 --disable-usage-stats --block > "$JOB_DIR/ray-$SLURM_PROCID.log" 2>&1 &
fi
RAY_PID=$!
trap 'kill "$RAY_PID" 2>/dev/null || true' EXIT
if [ "$SLURM_PROCID" = 0 ]; then
 sleep 20
 python "$D/ray_check_16gpu.py"
 cd "$R"
 python "$D/training_control_multinode_16gpu.py" train8 --config-only > "$JOB_DIR/expanded_config.txt" 2>&1
 python "$D/training_control_multinode_16gpu.py" train8 --tag priority_sft8_formal_multinode --seed 42
 touch "$JOB_DIR/TRAINING_PROCESS_EXIT_ZERO"
else
 while kill -0 "$RAY_PID" 2>/dev/null && [ ! -f "$JOB_DIR/TRAINING_PROCESS_EXIT_ZERO" ]; do sleep 5; done
fi
