#!/usr/bin/env bash
# Serve a decider-format model folder with decider's own server and score the JevBench public items (231).
#   ROOT=/root/work bash ops/decider_score.sh /root/work/runs/lora_v1/model [port]
set -euo pipefail
MODEL="${1:?model dir}"; PORT="${2:-8002}"; ROOT="${ROOT:-/root/work}"; OUT="$(dirname "$MODEL")/jevbench"
cd "$ROOT/decider"; export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
DECIDER_MODEL="$MODEL" setsid nohup .venv/bin/python -m uvicorn decider.serve:app --host 127.0.0.1 --port "$PORT" > "$(dirname "$MODEL")/serve.log" 2>&1 < /dev/null &
SP=$!
for i in $(seq 1 120); do curl -sf -o /dev/null "localhost:$PORT/health" && break; sleep 5; done
cd "$ROOT/jevbench"; rm -rf "$OUT"; mkdir -p "$OUT"
"$ROOT/jev/.venv/bin/python" -m jevbench.cli run --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
    --adapter typesafe --endpoint "http://127.0.0.1:$PORT" --model "$(basename "$(dirname "$MODEL")")" --key-env "" --cost-basis local_gpu_no_provider_tariff \
    --reserve-usd 0 --cap-usd 1 --results "$OUT/results.jsonl" --raw-dir "$OUT/raw" --ledger "$OUT/ledger.jsonl" 2>&1 | tail -1
kill $SP 2>/dev/null || true; rm -rf "$OUT/raw"
"$ROOT/jev/.venv/bin/python" - "$OUT/results.jsonl" "$ROOT/runs/decider_base/results.jsonl" <<'PY'
import json, sys, collections, os
rows = {j["task_id"]: j for j in map(json.loads, open(sys.argv[1]))}
def tier(t): p=t.split("-")[0]; return {"easy":"easy","hard":"hard"}.get(p,"standard")
by = collections.defaultdict(list)
for t, j in rows.items(): by[tier(t)].append(int(bool(j["correct"])))
print("[score]", {k: f"{sum(v)}/{len(v)}" for k, v in by.items()}, "total", sum(int(bool(j["correct"])) for j in rows.values()), "ok", sum(1 for j in rows.values() if j["ok"]))
if os.path.exists(sys.argv[2]):
    base = {j["task_id"]: j for j in map(json.loads, open(sys.argv[2]))}
    fam = collections.defaultdict(lambda: [0, 0, 0])
    for t, j in rows.items():
        if tier(t) == "hard": f = fam[j["family"]]; f[0] += int(bool(base[t]["correct"])); f[1] += int(bool(j["correct"])); f[2] += 1
    print("[score] hard by family (base->this/n):", {k: f"{v[0]}->{v[1]}/{v[2]}" for k, v in sorted(fam.items())})
    ids = [t for t in rows if tier(t) == "hard"]
    print("[score] hard flips: fixed", sum(1 for t in ids if rows[t]["correct"] and not base[t]["correct"]), "broken", sum(1 for t in ids if base[t]["correct"] and not rows[t]["correct"]))
PY
