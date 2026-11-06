#!/usr/bin/env bash
#SBATCH --job-name=zs-recoil-formal-multinode
#SBATCH --partition=hgx
#SBATCH --nodelist=gnode04,gnode07
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --mem=600G
#SBATCH --time=3-00:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/recoil_sft8_20261104/multinode_20261104/formal-%j.log
set -euo pipefail
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export TRAIN_HEAD_IP=$(getent ahostsv4 "$HEAD" | awk 'NR==1 {print $1}')
export TRAIN_PORT=$((20000 + SLURM_JOB_ID % 10000))
srun --kill-on-bad-exit=1 /cluster/home/shenqunlun/zhousheng/visual-search-agent/cluster/run_container.sh bash /cluster/home/shenqunlun/zhousheng/recoil_sft8_20261104/multinode_20261104/formal_16gpu_node.sh
