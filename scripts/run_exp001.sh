#!/usr/bin/env bash
# Runs ON the GPU host: Exp 001 readout ablation, three readouts sequentially (8 GB VRAM), each followed by evaluation.
#   nohup bash scripts/run_exp001.sh > runs/exp001.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/usr/lib/wsl/lib:$PATH" HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
READOUTS="${READOUTS:-joint branch biencoder}"
EXTRA="${EXTRA:-}"
for r in $READOUTS; do
  out="runs/exp001_${r}_full"
  if [ -f "$out/eval_test.json" ]; then echo "[exp001] $r already done, skipping"; continue; fi
  echo "[exp001] === $r === $(date)"
  .venv/bin/python scripts/train.py --config configs/exp001_readout.yaml --readout "$r" --out_dir "$out" $EXTRA
  .venv/bin/python scripts/evaluate.py --run "$out" --split test --limit 6000 --batch_size 32
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
done
echo "[exp001] ALL DONE $(date)"
