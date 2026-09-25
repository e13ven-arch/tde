#!/usr/bin/env bash
# One-shot setup of a fresh RunPod pod (or any CUDA box) with a persistent /workspace.
#   bash <(curl -fsSL https://raw.githubusercontent.com/e13ven-arch/tde/main/ops/pod_bootstrap.sh)
# Idempotent: re-running only pulls code and fills in whatever is missing under /workspace/jev.
set -euo pipefail
ROOT="${ROOT:-/workspace/jev}"
REPO="${REPO:-https://github.com/e13ven-arch/tde}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if [ ! -d "$ROOT/.git" ]; then git clone -q "$REPO" "$ROOT"; else git -C "$ROOT" pull -q --ff-only; fi
cd "$ROOT"
[ -x .venv/bin/python ] || python3 -m venv --system-site-packages .venv   # reuse the image's torch+CUDA
.venv/bin/pip install -q -e ".[dev]" 2>&1 | grep -v -i "warning\|notice" || true
.venv/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
mkdir -p runs data checkpoints
[ -f data/v0.1/train.jsonl ] || .venv/bin/python scripts/build_data.py --stage 0 --out data/v0.1
[ -f checkpoints/gliclass-modern-base-encoder/model.safetensors ] || .venv/bin/python scripts/extract_gliclass_encoder.py --out checkpoints/gliclass-modern-base-encoder
.venv/bin/python -m pytest -q tests 2>&1 | tail -1
echo "[bootstrap] ready: $ROOT  data: $(ls data | tr '\n' ' ')  checkpoints: $(ls checkpoints | tr '\n' ' ')"
