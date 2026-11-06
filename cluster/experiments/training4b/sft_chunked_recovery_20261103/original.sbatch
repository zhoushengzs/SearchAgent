#!/bin/bash
#SBATCH --job-name=zs4-sft-original-a100
#SBATCH --partition=hgx
#SBATCH --nodelist=gnode07
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/sft4-original-chunked-%j.log
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
srun --kill-on-bad-exit=1 "$P/visual-search-agent/cluster/run_container.sh" bash "$P/training4b/sft_chunked_recovery_20261103/node.sh" original
