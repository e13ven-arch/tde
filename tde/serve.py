"""TypeSafe-compatible `/v1/systemone` server for a TDE run directory or Hugging Face model id.

    tde-serve --model tdelab/tde-general-v0.1 --port 8000
    curl -X POST localhost:8000/v1/systemone -H 'Content-Type: application/json' \
         -d '{"state": "...", "questions": {"decision": {"type": "noul", "instructions": "..."}}}'

Request  {"state": str | object, "model": str (ignored), "questions": {id: {"type": "noul"|"choice"|"score",
          "instructions": str, "criteria": {...} | [...]}}}
Response {"model": str, "answers": {id: {"type": ..., "noul": p_yes | "choice": label, "probabilities": {...},
          "confidence": float}}, "usage": {"decisions": n, "latency_ms": ...}}

Probabilities are the model's own softmax over the supplied options in the supplied order (no temperature unless
the directory ships a temperature.json and --temperature is given). Score levels are keyed by their index as strings.
Stdlib only; one request at a time (a lock serialises the model), which is what a benchmark run needs.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _answer(rec: dict) -> dict:
    out = {"type": rec["type"], "probabilities": rec["probabilities"], "confidence": rec["confidence"]}
    if rec["type"] == "noul":
        out["noul"] = rec["noul"]
    elif rec["type"] == "choice":
        out["choice"] = rec["label"]
    else:
        out["score"] = rec["score"]
    return out


def make_handler(decider, model_name: str):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "tde-serve/0.1"

        def log_message(self, fmt, *args):  # quiet; the harness keeps its own log
            pass

        def _send(self, status: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/health", "/v1/health"):
                return self._send(200, {"ok": True, "model": model_name})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                questions = req.get("questions")
                if not isinstance(questions, dict) or not questions:
                    return self._send(400, {"error": "questions must be a non-empty object"})
                t0 = time.perf_counter()
                with lock:
                    recs = decider.decide_batch(req.get("state", ""), questions)
                ms = (time.perf_counter() - t0) * 1000
                self._send(200, {"model": model_name, "answers": {qid: _answer(r) for qid, r in recs.items()},
                                 "usage": {"decisions": len(recs), "latency_ms": round(ms, 2)}})
            except (KeyError, TypeError, ValueError) as e:
                self._send(400, {"error": f"bad request: {e}"})
            except Exception as e:  # noqa: BLE001 - report, keep serving
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="run / release directory or Hugging Face model id")
    ap.add_argument("--revision", default=None, help="Hugging Face revision to pin")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--temperature", action="store_true", help="apply temperature.json if the directory ships one")
    ap.add_argument("--name", default=None, help="model name reported in responses")
    a = ap.parse_args(argv)
    from tde.inference import Decider
    t0 = time.perf_counter()
    decider = Decider.from_run(a.model, device=a.device, use_temperature=a.temperature, revision=a.revision)
    name = a.name or str(a.model)
    print(f"[tde-serve] loaded {name} on {decider.device} in {time.perf_counter() - t0:.1f}s", flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(decider, name))
    print(f"[tde-serve] POST http://{a.host}:{a.port}/v1/systemone", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
