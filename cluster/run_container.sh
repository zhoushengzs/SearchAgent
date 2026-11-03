#!/usr/bin/env bash
set -euo pipefail
BASE=/cluster/home/shenqunlun/zhousheng
source /etc/profile.d/modules.sh
module load singularity/3.11.5 squashfs-tools/4.6.1
export SINGULARITY_CACHEDIR="$BASE/cache/singularity"
export SINGULARITY_TMPDIR="$BASE/tmp"
export XDG_CACHE_HOME="$BASE/cache"
export HF_HOME="$BASE/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch-extensions"
export TRITON_CACHE_DIR="$BASE/cache/triton/${SLURM_JOB_ID:-check}-${SLURM_PROCID:-0}"
export LC_ALL=C LANG=C
mkdir -p "$BASE/runtime-home" "$BASE/tmp/${SLURM_JOB_ID:-check}-${SLURM_PROCID:-0}"
cd "$BASE"
exec singularity exec --nv --contain --home "$BASE/runtime-home" \
  --bind "$BASE:$BASE" \
  --bind /lib64/libcuda.so:/usr/lib/x86_64-linux-gnu/libcuda.so:ro \
  --bind /lib64/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
  --bind /lib64/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
  --bind /lib64/libnvidia-ptxjitcompiler.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro \
  --bind /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  --bind "$BASE/tmp/${SLURM_JOB_ID:-check}-${SLURM_PROCID:-0}:/tmp" \
  "$BASE/containers/ubuntu22-rootfs" "$@"
