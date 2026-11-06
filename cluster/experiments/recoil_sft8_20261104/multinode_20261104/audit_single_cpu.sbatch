#!/bin/bash
#SBATCH --partition=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=zs-recoil-input-audit
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/recoil_sft8_20261104/multinode_20261104/input-audit-%j.log
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
source /etc/profile.d/modules.sh
module load singularity/3.11.5 squashfs-tools/4.6.1
export SINGULARITY_CACHEDIR="$P/cache/singularity"
export SINGULARITY_TMPDIR="$P/tmp"
mkdir -p "$P/tmp/input-audit-$SLURM_JOB_ID"
exec singularity exec --contain --home "$P/runtime-home" --bind "$P:$P" --bind "$P/tmp/input-audit-$SLURM_JOB_ID:/tmp" "$P/containers/ubuntu22-rootfs" bash "$P/recoil_sft8_20261104/multinode_20261104/audit_single_node.sh"
