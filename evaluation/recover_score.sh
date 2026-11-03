#!/bin/bash
#SBATCH --job-name=zs-base165-score
#SBATCH --partition=hgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/logs/score-recovery-%j.log
set -euo pipefail
B=/cluster/home/shenqunlun/zhousheng
if [[ ${VS_SCORE_CONTAINER:-0} != 1 ]]; then
 export VS_SCORE_CONTAINER=1
 exec "$B/visual-search-agent/cluster/run_container.sh" bash "$B/benchmark-vs778/evaluation/recover_score.sh"
fi
source "$B/visual-search-agent/cluster/runtime.sh"
cd "$B/benchmark-vs778"
export VS_BENCHMARK_IN_CONTAINER=1
exec 9>outputs/.base165-benchmark.lock
flock -n 9
python -u -m evaluation.recover_score
