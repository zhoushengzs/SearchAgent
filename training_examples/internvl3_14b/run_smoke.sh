#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SFT_WORK="${SFT_WORK:-$HOME/internvl3-sft-work}"
export USE_RAY=0
python "$HERE/make_config.py"
if test -e "$SFT_WORK/outputs/smoke-lora"; then
  echo '试跑输出目录已经存在。请检查旧结果，或换一个 SFT_WORK，避免覆盖。' >&2
  exit 1
fi
python "$HERE/check_cpu.py" --model "$SFT_WORK/model" --data "$SFT_WORK/data" --output "$SFT_WORK/prepared"
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-8}" "$HERE/train.py" "$SFT_WORK/smoke.json"
