#!/usr/bin/env bash
set -euo pipefail
HERE=/cluster/home/shenqunlun/zhousheng/training4b
source "$HERE/runtime.sh"
export PYTHONPATH="$HERE/compat:${PYTHONPATH:-}"
export HF_HOME="/cluster/home/shenqunlun/zhousheng/cache/huggingface"
export XDG_CACHE_HOME="/cluster/home/shenqunlun/zhousheng/cache"
IP=$(getent ahostsv4 "$(hostname)" | awk 'NR==1 {print $1}')
RAY_PID=
cleanup() { if [ -n "$RAY_PID" ]; then kill "$RAY_PID" 2>/dev/null || true; wait "$RAY_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
if [ "$SLURM_PROCID" = 0 ]; then
  ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --num-gpus=8 --num-cpus="$SLURM_CPUS_PER_TASK" --include-dashboard=false --disable-usage-stats --block > "$BASE/logs/ray-${SLURM_JOB_ID}-0.log" 2>&1 &
else
  for i in $(seq 1 120); do
    if timeout 2 bash -c "</dev/tcp/$HEAD_IP/$RAY_PORT" 2>/dev/null; then break; fi
    sleep 2
  done
  ray start --address="$HEAD_IP:$RAY_PORT" --node-ip-address="$IP" --num-gpus=8 --num-cpus="$SLURM_CPUS_PER_TASK" --disable-usage-stats --block > "$BASE/logs/ray-${SLURM_JOB_ID}-${SLURM_PROCID}.log" 2>&1 &
fi
RAY_PID=$!
if [ "$SLURM_PROCID" = 0 ]; then
 python - <<'READY'
import os,time,ray
for _ in range(120):
 try:
  ray.init(address=os.environ['RAY_ADDRESS'],logging_level='ERROR')
  break
 except Exception:time.sleep(2)
else:raise SystemExit('Ray head did not start')
for _ in range(180):
 nodes=[n for n in ray.nodes() if n['Alive']]
 if len(nodes)==int(os.environ['SLURM_JOB_NUM_NODES']) and all(n['Resources'].get('GPU',0)==8 for n in nodes):break
 time.sleep(2)
else:raise SystemExit('Not all allocated nodes have joined with 8 GPUs')
print('Connected nodes:',[(n['NodeManagerAddress'],n['Resources'].get('GPU')) for n in nodes])
ray.shutdown()
READY
 stage="${GRPO_STAGE:?Set base or sft}"
 python "$HERE/train_sft_multinode.py" --stage "$stage" --smoke
 python "$HERE/train_sft_multinode.py" --stage "$stage"
 touch "$BASE/results4b/job-${SLURM_JOB_ID}.done"
else
 while kill -0 "$RAY_PID" 2>/dev/null; do
  [ ! -f "$BASE/results4b/job-${SLURM_JOB_ID}.done" ] || exit 0
  sleep 5
 done
 exit 1
fi
