#!/usr/bin/env bash
#SBATCH --partition=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/unified6-finalize-%j.log
set -euo pipefail
source /cluster/home/shenqunlun/zhousheng/visual-search-agent/cluster/runtime.sh
cd /cluster/home/shenqunlun/zhousheng/benchmark-vs778
export PYTHONDONTWRITEBYTECODE=1
exec python -u -m evaluation.parallel_unified6 finalize
