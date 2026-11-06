#!/bin/bash
#SBATCH --job-name=zs4-sft-resume
#SBATCH --partition=hgx
#SBATCH --nodes=2
#SBATCH --nodelist=gnode05,gnode06
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --time=7-00:00:00
#SBATCH --output=/cluster/home/shenqunlun/projects/zhousheng/logs/sft4-resume-%j.log
set -euo pipefail
P=/cluster/home/shenqunlun/projects/zhousheng
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export SFT_HEAD=$(getent ahostsv4 "$HEAD" | awk 'NR==1 {print $1}')
export SFT_PORT=$((20000 + SLURM_JOB_ID % 20000))
srun --kill-on-bad-exit=1 "$P/runtime_migration_20261105/run_container.sh" bash "$P/training4b/sft_recovery_20261105/node.sh"
