#!/usr/bin/env bash
# Runs ON the GPU host: train + evaluate a list of jobs sequentially. Each job is "config|readout|out_dir|extra args".
#   nohup bash scripts/run_queue.sh jobs/stage1.txt > runs/queue_stage1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/usr/lib/wsl/lib:$PATH" HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
       PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONDA_GCC="$HOME/miniconda3/envs/py312/bin/x86_64-conda-linux-gnu-gcc"
if command -v gcc >/dev/null; then export CC=gcc; elif [ -x "$CONDA_GCC" ]; then export CC="$CONDA_GCC" PATH="$HOME/miniconda3/envs/py312/bin:$PATH"; else export TORCH_COMPILE_DISABLE=1; fi
JOBS="${1:?job file}"
while IFS='|' read -r cfg readout out extra; do
  [ -z "$cfg" ] || [[ "$cfg" == \#* ]] && continue
  if [ -f "$out/eval_test.json" ]; then echo "[queue] $out already done, skipping"; continue; fi
  echo "[queue] === $out === $(date)"
  .venv/bin/python scripts/train.py --config "$cfg" --readout "$readout" --out_dir "$out" $extra
  .venv/bin/python scripts/evaluate.py --run "$out" --split test --limit 6000 --batch_size 32
done < "$JOBS"
echo "[queue] ALL DONE $(date)"
