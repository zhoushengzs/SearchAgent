#!/usr/bin/env bash
#SBATCH --job-name=zs-five6-score
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/five6-score-%j.log
set -euo pipefail
umask 077
source /cluster/home/shenqunlun/zhousheng/visual-search-agent/cluster/runtime.sh
cd /cluster/home/shenqunlun/zhousheng/benchmark-vs778
export PYTHONDONTWRITEBYTECODE=1
exec python -u -m evaluation.plain6_benchmark --stage score
