#!/bin/bash
#SBATCH --partition=low
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/download-fast-%j.log
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
source /etc/profile.d/modules.sh
module load singularity/3.11.5 squashfs-tools/4.6.1
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
exec singularity exec --contain --home "$P/runtime-home" --bind "$P:$P" "$P/containers/ubuntu22-rootfs" "$P/envs/search-grpo-a100/bin/python" "$P/training4b/download_fast.py" "$@"
