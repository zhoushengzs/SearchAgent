#!/usr/bin/env bash
#SBATCH --job-name=zs-base-sft-grpo
#SBATCH --partition=hgx
#SBATCH --nodes=2
#SBATCH --mem=512G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/train-%j.log
set -euo pipefail
BASE=/cluster/home/shenqunlun/zhousheng
[ -f "$BASE/DEPLOYMENT_READY" ] || { echo 'Deployment checks have not passed'; exit 1; }
cd "$BASE"
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export HEAD_IP=$(getent ahostsv4 "$HEAD" | awk 'NR==1 {print $1}')
export RAY_PORT=$((20000 + SLURM_JOB_ID % 20000))
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
export RAY_USAGE_STATS_ENABLED=0
srun --kill-on-bad-exit=1 --export=ALL "$BASE/visual-search-agent/cluster/run_container.sh" bash "$BASE/visual-search-agent/cluster/node.sh"
