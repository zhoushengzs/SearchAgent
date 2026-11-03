#!/usr/bin/env bash
set -euo pipefail
BASE=/cluster/home/shenqunlun/zhousheng
if [[ ${VS_BENCHMARK_IN_CONTAINER:-0} != 1 ]]; then
 export VS_BENCHMARK_IN_CONTAINER=1
 exec "$BASE/visual-search-agent/cluster/run_container.sh" bash "$BASE/benchmark-vs778/evaluation/server_benchmark_step.sh" "$@"
fi
source "$BASE/visual-search-agent/cluster/runtime.sh"
cd "$BASE/benchmark-vs778"
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
exec python -u -m evaluation.server_benchmark_multinode "$@"
