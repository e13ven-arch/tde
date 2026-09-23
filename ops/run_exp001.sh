#!/usr/bin/env bash
# Runs ON the GPU host: Exp 001 readout ablation, three readouts sequentially (8 GB VRAM), each followed by evaluation.
#   nohup bash scripts/run_exp001.sh > runs/exp001.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/usr/lib/wsl/lib:$PATH" HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Triton (torch.compile) needs a C compiler: prefer the conda-forge gcc installed without sudo, else disable compile.
CONDA_GCC="$HOME/miniconda3/envs/py312/bin/x86_64-conda-linux-gnu-gcc"
if command -v gcc >/dev/null; then export CC=gcc
elif [ -x "$CONDA_GCC" ]; then export CC="$CONDA_GCC" PATH="$HOME/miniconda3/envs/py312/bin:$PATH"
else export TORCH_COMPILE_DISABLE=1; echo "[exp001] no C compiler found; torch.compile disabled"
fi
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
