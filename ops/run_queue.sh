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
  .venv/bin/python scripts/evaluate.py --run "$out" --data_dir "data/${DATA_DIR:-v0.1}" --split test --limit 20000 --batch_size 32
  [ "${DATA_DIR:-v0.1}" != "v0.1" ] && .venv/bin/python scripts/evaluate.py --run "$out" --data_dir data/v0.1 --split test --limit 6000 --batch_size 32 --no-controls --out "$out/eval_test_v0.1.json"
  for f in data/${DATA_DIR:-v0.1}/eval_only/*.test.jsonl; do   # full-label-set, OOD and comparability sets (no controls, no TS refit)
    [ -f "$f" ] && .venv/bin/python scripts/evaluate.py --run "$out" --split "$f" --limit 6000 --batch_size 16 --no-controls
  done
done < "$JOBS"
echo "[queue] ALL DONE $(date)"
