#!/usr/bin/env bash
#SBATCH --job-name=zs-recoil-64x4-8gpu
#SBATCH --partition=hgx
#SBATCH --nodelist=gnode04,gnode07
#SBATCH --exclude=gnode05,gnode06
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=600G
#SBATCH --time=3-00:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/recoil_sft8_20261104/restart48_20261105/formal-%j.log
set -euo pipefail
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export TRAIN_HEAD_IP=$(getent ahostsv4 "$HEAD" | awk 'NR==1 {print $1}')
export TRAIN_PORT=$((20000 + SLURM_JOB_ID % 10000))
srun --kill-on-bad-exit=1 /cluster/home/shenqunlun/zhousheng/visual-search-agent/cluster/run_container.sh bash /cluster/home/shenqunlun/zhousheng/recoil_sft8_20261104/restart48_20261105/node8.sh
