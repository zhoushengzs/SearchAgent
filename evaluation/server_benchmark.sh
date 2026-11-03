#!/usr/bin/env bash
#SBATCH --job-name=zs-base165-bench778
#SBATCH --partition=hgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/benchmark778-%j.log
set -euo pipefail
umask 077

BENCH_PERSONAL=/cluster/home/shenqunlun/zhousheng
BENCH_TRAINING="$BENCH_PERSONAL/visual-search-agent"
BENCH_ROOT="$BENCH_PERSONAL/benchmark-vs778"
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_NUM_NODES:-}" != 1 ]]; then
  echo 'Run this script only in a single-node Slurm GPU allocation.' >&2
  exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == -1 ]]; then
  echo 'Slurm has not exposed an allocated GPU.' >&2
  exit 1
fi
if [[ "${VS_BENCHMARK_IN_CONTAINER:-0}" != 1 ]]; then
  export VS_BENCHMARK_IN_CONTAINER=1
  exec "$BENCH_TRAINING/cluster/run_container.sh" bash "$BENCH_ROOT/evaluation/server_benchmark.sh"
fi
source "$BENCH_TRAINING/cluster/runtime.sh"
cd "$BENCH_ROOT"
export PYTHONPATH="$BENCH_ROOT:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export VS_SERPER_BACKEND=training
export SERPER_KEY_STATE_FILE="$BENCH_TRAINING/datasets/grpo/.codex/serper_key_state.json"
unset VS_DEBUG
exec python -u -m evaluation.server_benchmark
