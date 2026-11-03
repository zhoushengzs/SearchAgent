#!/usr/bin/env bash
#SBATCH --job-name=zs-base165-native-score
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/native778-score-%j.log
set -euo pipefail
umask 077
PERSONAL=/cluster/home/shenqunlun/zhousheng
source "$PERSONAL/visual-search-agent/cluster/runtime.sh"
cd "$PERSONAL/benchmark-vs778"
export PYTHONDONTWRITEBYTECODE=1
# Submit only after authorization for the explicit external judge data recipient.
exec python -u -m evaluation.native_benchmark --stage score
