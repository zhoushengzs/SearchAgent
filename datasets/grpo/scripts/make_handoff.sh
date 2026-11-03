#!/usr/bin/env bash
set -euo pipefail
GRPO_PACK_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_PACK_ROOT="$(cd "${GRPO_PACK_SCRIPT}/.." && pwd)"
GRPO_PACK_DEST="${1:-${GRPO_PACK_ROOT}/grpo_training_handoff.tar.gz}"
tar --exclude='local_training.env' --exclude='__pycache__' --exclude='*.pyc' \
    -czf "${GRPO_PACK_DEST}" -C "${GRPO_PACK_ROOT}" \
    merged.jsonl train.jsonl test.jsonl images scripts integration audit upstream \
    .gitignore .codex/数据与训练接入说明.md .codex/双卡调试与八卡训练教程.md .codex/单卡调试教程.md
echo "已打包：${GRPO_PACK_DEST}"
echo '不包含本机地址设置、真实服务密钥、调试保存文件或 benchmark 测试题文件。'
echo '朋友仍需准备官方训练代码、原始 SFT 模型和完整训练环境。'
