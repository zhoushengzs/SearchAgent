#!/bin/bash
#SBATCH --job-name=zs4-sft-original-a100
#SBATCH --partition=hgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/sft4-original-8gpu-%j.log
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
srun --kill-on-bad-exit=1 "$P/visual-search-agent/cluster/run_container.sh" bash "$P/training4b/sft_8gpu_sequential_20261103/node.sh" original
