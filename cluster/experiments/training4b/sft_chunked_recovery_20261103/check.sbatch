#!/bin/bash
#SBATCH --partition=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/sft-loss-check-%j.log
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
source /etc/profile.d/modules.sh
module load singularity/3.11.5 squashfs-tools/4.6.1
export SINGULARITY_CACHEDIR="$P/cache/singularity" SINGULARITY_TMPDIR="$P/tmp" OMP_NUM_THREADS=2
singularity exec --contain --home "$P/runtime-home" --bind "$P:$P" "$P/containers/ubuntu22-rootfs" "$P/envs/qwen4-sft-runtime/bin/python" "$P/training4b/sft_chunked_recovery_20261103/compat/test_chunked_sft_loss.py" > "$P/training4b/sft_chunked_recovery_20261103/numerical_check.json"
