#!/usr/bin/env bash
#SBATCH --job-name=zs-base165-bench778
#SBATCH --partition=hgx
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=24
#SBATCH --mem=384G
#SBATCH --time=2-00:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/benchmark778-%j.log
set -euo pipefail
umask 077
ROOT=/cluster/home/shenqunlun/zhousheng/benchmark-vs778
mkdir -p "$ROOT/outputs"
exec 9>"$ROOT/outputs/.base165-benchmark.lock"
flock -n 9 || { echo 'Another benchmark workflow holds the lock'; exit 1; }
export VS_BENCH_GPUS_PER_NODE=6
STEP="$ROOT/evaluation/server_benchmark_step.sh"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:6 --kill-on-bad-exit=1 bash "$STEP" prepare
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --gres=gpu:6 --kill-on-bad-exit=1 bash "$STEP" workers
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:6 --kill-on-bad-exit=1 bash "$STEP" finish
