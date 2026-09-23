#!/usr/bin/env bash
# Runs ON the GPU host: create a Python 3.12 env via miniconda, a project venv, and install deps.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/miniconda3/bin:/usr/lib/wsl/lib:$PATH"
if [ -x "$HOME/miniconda3/bin/conda" ] && [ ! -x "$HOME/miniconda3/envs/py312/bin/python" ]; then
  conda create -y -q -n py312 -c conda-forge --override-channels python=3.12
fi
PYB="$HOME/miniconda3/envs/py312/bin/python"; [ -x "$PYB" ] || PYB=python3
[ -x .venv/bin/python ] || "$PYB" -m venv .venv
.venv/bin/pip install -U pip 2>&1 | tail -1
.venv/bin/pip install --progress-bar off -e '.[dev,baselines]' 2>&1 | grep -E "Downloading|Installing|Successfully|ERROR|error" 
.venv/bin/python -c 'import torch,transformers,datasets;print("torch",torch.__version__,"cuda",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else "");print("transformers",transformers.__version__,"datasets",datasets.__version__)'
.venv/bin/python -m pytest -q
echo SETUP_DONE
