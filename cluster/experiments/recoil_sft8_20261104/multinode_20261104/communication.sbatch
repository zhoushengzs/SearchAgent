#!/usr/bin/env bash
#SBATCH --job-name=zs-recoil-multinode-check
#SBATCH --partition=hgx
#SBATCH --nodelist=gnode04,gnode05
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/recoil_sft8_20261104/multinode_20261104/check-%j.log
set -euo pipefail
export CHECK_HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export CHECK_PORT=$((30000 + SLURM_JOB_ID % 10000))
srun --kill-on-bad-exit=1 /cluster/home/shenqunlun/zhousheng/visual-search-agent/cluster/run_container.sh bash /cluster/home/shenqunlun/zhousheng/recoil_sft8_20261104/multinode_20261104/check_node.sh
