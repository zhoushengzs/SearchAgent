#!/bin/bash
#SBATCH --partition=hgx
#SBATCH --nodelist=gnode07
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --job-name=zs4-sft-early-smoke
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/sft-early-%j.log
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
srun --kill-on-bad-exit=1 "$P/visual-search-agent/cluster/run_container.sh" bash "$P/training4b/sft_early_node.sh"
