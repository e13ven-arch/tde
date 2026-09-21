#!/usr/bin/env bash
# Helper for running experiments on the GPU host (default bot@192.168.0.102).
#   scripts/remote.sh sync                 # rsync code (no data, no venv) to $REMOTE:$REMOTE_DIR
#   scripts/remote.sh setup                # create venv + install deps on the host
#   scripts/remote.sh data                 # build the stage-0 corpus on the host (downloads from HF there)
#   scripts/remote.sh push-data            # or rsync the locally built data/v0.1 instead
#   scripts/remote.sh train configs/exp001_readout.yaml --readout branch --out_dir runs/exp001_branch
#   scripts/remote.sh pull runs/exp001_branch   # fetch a run directory back
#   scripts/remote.sh probe                # GPU / python / disk on the host
set -euo pipefail
REMOTE="${REMOTE:-bot@192.168.0.102}"
REMOTE_DIR="${REMOTE_DIR:-~/jev}"
PY="${REMOTE_PY:-python3}"
SSH_OPTS="${SSH_OPTS:--i $HOME/.ssh/id_ed25519_jevbot -o BatchMode=yes}"
SSH="ssh $SSH_OPTS"
export RSYNC_RSH="ssh $SSH_OPTS"
RSYNC="rsync -az"
cmd="${1:-}"; shift || true
case "$cmd" in
  sync)
    $RSYNC --delete --exclude .venv --exclude data --exclude runs --exclude .git --exclude '__pycache__' ./ "$REMOTE:$REMOTE_DIR/" ;;
  setup)
    # Host has Python 3.14 (no torch wheels yet) but miniconda: build a 3.12 env, then a project venv from it.
    $SSH "$REMOTE" "set -e; cd $REMOTE_DIR; CONDA=\$HOME/miniconda3/bin/conda; if [ -x \$CONDA ] && [ ! -x \$HOME/miniconda3/envs/py312/bin/python ]; then \$CONDA create -y -q -n py312 python=3.12 >/dev/null; fi; PYB=\$HOME/miniconda3/envs/py312/bin/python; [ -x \$PYB ] || PYB=$PY; [ -x .venv/bin/python ] || \$PYB -m venv .venv; .venv/bin/pip install -q -U pip; .venv/bin/pip install -q -e '.[dev,baselines]'; .venv/bin/python -c 'import torch;print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")'" ;;
  setup-bg)
    $SSH "$REMOTE" "mkdir -p $REMOTE_DIR/runs; cd $REMOTE_DIR && nohup bash -c 'bash scripts/remote_setup.sh' > runs/setup.log 2>&1 &" && echo "setup started; log: $REMOTE_DIR/runs/setup.log" ;;
  data)
    $SSH "$REMOTE" "cd $REMOTE_DIR && HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python scripts/build_data.py --stage 0 --out data/v0.1" ;;
  push-data)
    $SSH "$REMOTE" "mkdir -p $REMOTE_DIR/data/v0.1" && $RSYNC data/v0.1/ "$REMOTE:$REMOTE_DIR/data/v0.1/" ;;
  train)
    cfg="$1"; shift
    $SSH "$REMOTE" "mkdir -p $REMOTE_DIR/runs; cd $REMOTE_DIR && nohup .venv/bin/python scripts/train.py --config $cfg $* > runs/\$(basename $cfg .yaml)_\$(date +%s).log 2>&1 &" && echo "started; tail with: ssh $REMOTE 'tail -f $REMOTE_DIR/runs/*.log'" ;;
  probe)
    $SSH "$REMOTE" "hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; python3 --version; nproc; free -g | head -2; df -h ~ | tail -1" ;;
  logs)
    $SSH "$REMOTE" "cd $REMOTE_DIR && ls -t runs/*.log | head -3; tail -n ${1:-20} \$(ls -t runs/*.log | head -1)" ;;
  gpu)
    $SSH "$REMOTE" "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; ps aux | grep scripts/ | grep -v grep | awk '{print \$2, \$11, \$12, \$13, \$14, \$15}'" ;;
  pull)
    $RSYNC "$REMOTE:$REMOTE_DIR/$1/" "$1/" ;;
  *)
    sed -n 2,9p "$0"; exit 1 ;;
esac
