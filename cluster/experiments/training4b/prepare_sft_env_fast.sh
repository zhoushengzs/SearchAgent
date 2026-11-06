#!/bin/bash
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
source /etc/profile.d/modules.sh
module load singularity/3.11.5 squashfs-tools/4.6.1
exec singularity exec --contain --home "$P/runtime-home" --bind "$P:$P" "$P/containers/ubuntu22-rootfs" bash -s <<'INNER'
set -euo pipefail
P=/cluster/home/shenqunlun/zhousheng
"$P/envs/search-grpo-a100/bin/python" -m venv --system-site-packages "$P/envs/qwen4-sft-runtime"
export PATH="$P/envs/qwen4-sft-runtime/bin:$P/conda-new/bin:$PATH"
export TMPDIR="$P/tmp/sft-install" PIP_CACHE_DIR="$P/runtime-home/.cache/pip"
export HF_HOME="$P/cache/huggingface" XDG_CACHE_HOME="$P/cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"
export MAX_JOBS=4 DS_BUILD_OPS=0
export CUDA_HOME="$P/cuda-toolkit-12.8" CUDA_PATH="$P/cuda-toolkit-12.8"
export CC="$P/envs/search-grpo-a100/bin/x86_64-conda-linux-gnu-cc" CXX="$P/envs/search-grpo-a100/bin/x86_64-conda-linux-gnu-c++"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple PIP_DEFAULT_TIMEOUT=120
python - <<'PY'
import importlib.metadata as m
from pathlib import Path
names=['torch','torchvision','torchaudio','transformers']
Path('/cluster/home/shenqunlun/zhousheng/training4b/sft_constraints.txt').write_text('\n'.join(n+'=='+m.version(n) for n in names)+'\n')
print(Path('/cluster/home/shenqunlun/zhousheng/training4b/sft_constraints.txt').read_text())
PY
python -m pip install -c "$P/training4b/sft_constraints.txt" -e "$P/visual-search-agent/train/OpenSearch-VL-Training/SFT" 'deepspeed>=0.16,<=0.18.8'
python - <<'PY'
import torch,transformers,deepspeed,llamafactory
from llamafactory.train.tuner import run_exp
from pathlib import Path
import json,datetime
Path('/cluster/home/shenqunlun/zhousheng/training4b/SFT_ENV_READY.json').write_text(json.dumps({'environment':'/cluster/home/shenqunlun/zhousheng/envs/qwen4-sft-runtime','torch':torch.__version__,'transformers':transformers.__version__,'checked_at':datetime.datetime.now().isoformat()}))
print('SFT_ENV_READY')
PY
INNER
