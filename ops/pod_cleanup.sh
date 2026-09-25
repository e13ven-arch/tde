#!/usr/bin/env bash
# Free pod disk: delete best.pt of every run except the ones named as arguments (reports and logs are kept).
#   bash ops/pod_cleanup.sh exp020_large_set_v07 exp010_fullk_v05
set -euo pipefail
cd "$(dirname "$0")/.."
keep=" $* "
for d in runs/*/; do
  n=$(basename "$d")
  if [[ "$keep" != *" $n "* ]] && [ -f "$d/best.pt" ]; then rm -v "$d/best.pt"; fi
done
rm -rf ~/.cache/huggingface/datasets
df -h / | tail -1
