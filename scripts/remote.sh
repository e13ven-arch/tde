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
RSYNC="rsync -az -e \"ssh $SSH_OPTS\""
cmd="${1:-}"; shift || true
case "$cmd" in
  sync)
    $RSYNC --delete --exclude .venv --exclude data --exclude runs --exclude .git --exclude '__pycache__' ./ "$REMOTE:$REMOTE_DIR/" ;;
  setup)
    $SSH "$REMOTE" "cd $REMOTE_DIR && $PY -m venv .venv && .venv/bin/pip install -q -U pip && .venv/bin/pip install -q -e '.[dev,baselines]' && .venv/bin/python -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")'" ;;
  data)
    $SSH "$REMOTE" "cd $REMOTE_DIR && HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python scripts/build_data.py --stage 0 --out data/v0.1" ;;
  push-data)
    $RSYNC data/v0.1/ "$REMOTE:$REMOTE_DIR/data/v0.1/" ;;
  train)
    cfg="$1"; shift
    $SSH "$REMOTE" "cd $REMOTE_DIR && nohup .venv/bin/python scripts/train.py --config $cfg $* > runs/\$(basename $cfg .yaml)_\$(date +%s).log 2>&1 &" && echo "started; tail with: ssh $REMOTE 'tail -f $REMOTE_DIR/runs/*.log'" ;;
  probe)
    $SSH "$REMOTE" "hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; python3 --version; nproc; free -g | head -2; df -h ~ | tail -1" ;;
  pull)
    $RSYNC "$REMOTE:$REMOTE_DIR/$1/" "$1/" ;;
  *)
    sed -n 2,9p "$0"; exit 1 ;;
esac
