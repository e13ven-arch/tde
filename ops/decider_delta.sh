#!/usr/bin/env bash
# Continue decider-4b on our mixture (delta stage), then score the public JevBench items through decider's own server.
#   bash ops/decider_delta.sh /workspace/runs/delta_mix_v1.pkl /workspace/runs/delta_v1 [extra decider.train args]
# Runs inside the decider checkout (/workspace/decider, its own venv with torch 2.14). Same effective batch as the
# reference recipe (16384 x 2 = 8192 x 4 tokens per optimizer step), lr 8e-6 as in `train.sh delta`.
set -euo pipefail
MIX="${1:?mixture pkl}"; OUT="${2:?out dir}"; shift 2
ROOT="${ROOT:-/workspace}"   # /root/work on the H100 pod
cd "$ROOT/decider"
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH="/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH"
.venv/bin/python -m decider.train --model "${INIT:-Mapika/decider-4b}" --data "$MIX" --out "$OUT" --epochs "${EPOCHS:-1}" --lr "${LR:-8e-6}" --warmup 150 \
    --max_tokens 8192 --accum 4 --max_options 255 --max_ctx 4096 --none_prob 0.1 --schema_first_prob 0.5 --eval_every 400 --eval_limit 100 "$@"
# copy decider_config.json (temperature, layout) next to the saved weights so the server can load the folder as a model
SNAP=$(ls -d ~/.cache/huggingface/hub/models--Mapika--decider-4b/snapshots/*/ | head -1)
cp -n "$SNAP/decider_config.json" "$OUT/model/" 2>/dev/null || true
# serve and score
DECIDER_MODEL="$OUT/model" setsid nohup .venv/bin/python -m uvicorn decider.serve:app --host 127.0.0.1 --port 8001 > "$OUT/serve.log" 2>&1 < /dev/null &
SP=$!
for i in $(seq 1 120); do curl -sf -o /dev/null localhost:8001/health && break; sleep 5; done
cd "$ROOT/jevbench" && rm -rf "$OUT/jevbench" && mkdir -p "$OUT/jevbench"
"$ROOT/jev/.venv/bin/python" -m jevbench.cli run --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
    --adapter typesafe --endpoint http://127.0.0.1:8001 --model "$(basename "$OUT")" --key-env "" --cost-basis local_gpu_no_provider_tariff --reserve-usd 0 --cap-usd 1 \
    --results "$OUT/jevbench/results.jsonl" --raw-dir "$OUT/jevbench/raw" --ledger "$OUT/jevbench/ledger.jsonl" 2>&1 | tail -1
kill $SP 2>/dev/null || true; rm -rf "$OUT/jevbench/raw"
"$ROOT/jev/.venv/bin/python" - "$OUT/jevbench/results.jsonl" <<'PY'
import json, sys, collections
rows=[json.loads(l) for l in open(sys.argv[1])]
def tier(t): p=t.split("-")[0]; return {"easy":"easy","hard":"hard"}.get(p,"standard")
by=collections.defaultdict(list)
for j in rows: by[tier(j["task_id"])].append(int(bool(j["correct"])))
print("[jevbench]", {k: f"{sum(v)}/{len(v)}" for k,v in by.items()}, "total", sum(int(bool(j["correct"])) for j in rows), "/", len(rows), "ok", sum(1 for j in rows if j["ok"]))
fam=collections.defaultdict(list)
for j in rows:
    if tier(j["task_id"])=="hard": fam[j["family"]].append(int(bool(j["correct"])))
print("[jevbench] hard by family:", {f: f"{sum(v)}/{len(v)}" for f,v in sorted(fam.items())})
PY
echo "[delta] done $OUT"
