#!/usr/bin/env bash
#SBATCH --job-name=zs-plain6-778
#SBATCH --partition=hgx
#SBATCH --nodelist=gnode05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/plain6-778-%j.log
set -euo pipefail
umask 077
PERSONAL=/cluster/home/shenqunlun/zhousheng
ROOT="$PERSONAL/benchmark-vs778"
if [[ ${VS_BENCHMARK_IN_CONTAINER:-0} != 1 ]]; then
  export VS_BENCHMARK_IN_CONTAINER=1
  exec "$PERSONAL/visual-search-agent/cluster/run_container.sh" bash "$ROOT/evaluation/plain6_benchmark.sh" "$@"
fi
source "$PERSONAL/visual-search-agent/cluster/runtime.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$ROOT/outputs/unified6_fiveway_778_20261103/tmp"
mkdir -p "$TMPDIR"
exec python -u -m evaluation.plain6_benchmark --stage infer "$@"
