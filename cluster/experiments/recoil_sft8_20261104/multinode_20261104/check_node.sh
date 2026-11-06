#!/usr/bin/env bash
set -euo pipefail
source /cluster/home/shenqunlun/zhousheng/training4b/runtime.sh
D="$BASE/recoil_sft8_20261104/multinode_20261104"
export CHECK_DIR="$D/check-$SLURM_JOB_ID"
mkdir -p "$CHECK_DIR"
python -m torch.distributed.run --nnodes=2 --nproc-per-node=4 --node-rank="$SLURM_PROCID" --master-addr="$CHECK_HEAD" --master-port="$CHECK_PORT" "$D/communication.py"
