#!/bin/bash
set -euo pipefail
source /cluster/home/shenqunlun/zhousheng/training4b/runtime.sh
export PYTHONDONTWRITEBYTECODE=1
python /cluster/home/shenqunlun/zhousheng/recoil_sft8_20261104/multinode_20261104/audit_single.py
