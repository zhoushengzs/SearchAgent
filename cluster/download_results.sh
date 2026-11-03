#!/usr/bin/env bash
# Run on the local Mac after each completed training stage.
set -euo pipefail
REMOTE=shenqunlun@10.50.27.186
BASE=/cluster/home/shenqunlun/zhousheng
SSH=(ssh -S /private/tmp/zhousheng-training-ssh.sock)
for stage in base sft; do
  if ! "${SSH[@]}" "$REMOTE" "test -f '$BASE/results/${stage}_grpo/SUCCESS.json'"; then
    echo "$stage: training has not completed; leaving local files untouched"
    continue
  fi
  destination="/Users/ss/models/${stage}-grpo"
  mkdir -p "$destination"
  rsync -a --partial -e 'ssh -S /private/tmp/zhousheng-training-ssh.sock' \
    "$REMOTE:$BASE/results/${stage}_grpo/final_model/" "$destination/"
  "${SSH[@]}" "$REMOTE" "cd '$BASE/results/${stage}_grpo/final_model' && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum" > "$destination/SHA256SUMS"
  (cd "$destination" && shasum -a 256 -c SHA256SUMS)
  echo "Verified model: $destination"
done
