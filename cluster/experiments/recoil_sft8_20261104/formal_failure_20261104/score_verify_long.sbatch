#!/usr/bin/env bash
#SBATCH --job-name=zs-recoil-score-verify
#SBATCH --partition=hgx
#SBATCH --nodelist=gnode04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --mem=700G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/home/shenqunlun/zhousheng/recoil_sft8_20261104/formal_failure_20261104/score-verify-long-%j.log
set -euo pipefail
srun --kill-on-bad-exit=1 /cluster/home/shenqunlun/zhousheng/visual-search-agent/cluster/run_container.sh bash /cluster/home/shenqunlun/zhousheng/recoil_sft8_20261104/formal_failure_20261104/score_verify_long_node.sh
