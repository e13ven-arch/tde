#!/usr/bin/env bash
# Push a finished run's checkpoint + reports to the private Hub repo (checkpoints live on the Hub, not the pod).
#   bash ops/hf_push_run.sh runs/exp020_large_set_v06            # -> tdelab/tde-checkpoints/runs/exp020_large_set_v06/
set -euo pipefail
run="${1:?run dir}"; name="$(basename "$run")"
.venv/bin/python - "$run" "$name" <<'PY'
import sys, os
from huggingface_hub import HfApi
run, name = sys.argv[1], sys.argv[2]
api = HfApi()
api.upload_folder(folder_path=run, path_in_repo=f"runs/{name}", repo_id="tdelab/tde-checkpoints", repo_type="model",
                  ignore_patterns=["log.jsonl.bak", "*.tmp"], commit_message=f"run {name}")
print("pushed", name)
PY
