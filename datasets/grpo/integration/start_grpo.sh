#!/usr/bin/env bash
set -euo pipefail
GRPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GRPO_DATA_ROOT="$(cd "${GRPO_SCRIPT_DIR}/.." && pwd)"
export GRPO_RLLM_ROOT="${1:-${GRPO_DATA_ROOT}/../../train/OpenSearch-VL/RL/rllm}"
export GRPO_MODEL_PATH="${2:-}"
export GRPO_PYTHON="${GRPO_PYTHON:-python}"
if [ -z "${GRPO_MODEL_PATH}" ]; then
    echo '用法：bash integration/start_grpo.sh "OpenSearch-VL/RL/rllm 的地址" "SFT 模型地址"'
    echo '请先在服务器激活已经安装 OpenSearch-VL 训练依赖的 Conda 环境。'
    exit 2
fi
if [ ! -f "${GRPO_RLLM_ROOT}/rllm/data/dataset.py" ]; then
    echo "找不到官方训练程序：${GRPO_RLLM_ROOT}"
    exit 2
fi
export GRPO_RLLM_ROOT="$(cd "${GRPO_RLLM_ROOT}" && pwd)"
export PYTHONPATH="${GRPO_SCRIPT_DIR}:${GRPO_RLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${GRPO_PYTHON}" "${GRPO_DATA_ROOT}/scripts/register_merged.py" --rllm-root "${GRPO_RLLM_ROOT}"
exec bash "${GRPO_SCRIPT_DIR}/official_8gpu_grpo.sh"
