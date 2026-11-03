#!/usr/bin/env bash
#SBATCH --job-name=zs-base165-native778
#SBATCH --partition=hgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/native778-%j.log
set -euo pipefail
umask 077
PERSONAL=/cluster/home/shenqunlun/zhousheng
ROOT="$PERSONAL/benchmark-vs778"
if [[ ${VS_BENCHMARK_IN_CONTAINER:-0} != 1 ]]; then
  export VS_BENCHMARK_IN_CONTAINER=1
  exec "$PERSONAL/visual-search-agent/cluster/run_container.sh" bash "$ROOT/evaluation/native_benchmark.sh" "$@"
fi
source "$PERSONAL/visual-search-agent/cluster/runtime.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
# Inference uses native real search tools. External reference judging is separate.
exec python -u -m evaluation.native_benchmark --stage infer "$@"
