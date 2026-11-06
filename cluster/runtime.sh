#!/usr/bin/env bash
set -euo pipefail
export BASE=/cluster/home/shenqunlun/zhousheng
export GRPO_PROJECT_ROOT="$BASE/visual-search-agent"
export PATH="$BASE/envs/search-grpo-a100/bin:$BASE/conda-new/bin:$PATH"
export PYTHONPATH="$GRPO_PROJECT_ROOT/datasets/grpo/scripts:$GRPO_PROJECT_ROOT/datasets/grpo/integration:$GRPO_PROJECT_ROOT/train/OpenSearch-VL-Training/RL/rllm:$GRPO_PROJECT_ROOT/train/OpenSearch-VL-Training/RL/rllm/verl:$GRPO_PROJECT_ROOT/train/OpenSearch-VL-Training/RL/Megatron-LM:$GRPO_PROJECT_ROOT/train/OpenSearch-VL-Training/RL/mbridge"
export LD_LIBRARY_PATH="$(python - <<'LIBS'
import sysconfig
from pathlib import Path
s=Path(sysconfig.get_paths()['purelib'])
p=sorted((s/'nvidia').glob('*/lib'))+[s/'torch/lib',Path('/cluster/home/shenqunlun/zhousheng/envs/search-grpo-a100/lib')]
print(':'.join(str(x) for x in p if x.is_dir())+':/.singularity.d/libs')
LIBS
)"
export GRPO_SKIP_FRAMEWORK_ENV=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export CUDA_HOME="$BASE/cuda-toolkit-12.8"
export CUDA_PATH="$CUDA_HOME"

# Slurm also exports AMD masks; this project uses NVIDIA CUDA devices.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Raise only this job process soft limit; server hard limit stays unchanged.
ulimit -u 16384

export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export CC="$BASE/envs/search-grpo-a100/bin/x86_64-conda-linux-gnu-cc"
export CXX="$BASE/envs/search-grpo-a100/bin/x86_64-conda-linux-gnu-c++"

export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$BASE/envs/search-grpo-a100/lib/python3.12/site-packages/nvidia/cudnn/include${CPATH:+:$CPATH}"
export MAX_JOBS=2

export FLASHINFER_WORKSPACE_BASE="$BASE/cache/cuda128-compiler"

export LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

export TRITON_CACHE_MANAGER=process_triton_cache:ProcessFileCacheManager
export TORCHINDUCTOR_COMPILE_THREADS=1

# Disable graph compilation for this runtime.
export TORCH_COMPILE_DISABLE=1
