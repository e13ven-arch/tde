"""The /v1/systemone wire server, exercised with a fake decider (no model load)."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from tde.serve import make_handler


class FakeDecider:
    device = "cpu"

    def decide_batch(self, state, questions):
        out = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                out[qid] = {"type": "noul", "probabilities": {"yes": 0.7, "no": 0.3}, "label": "yes", "confidence": 0.4, "noul": 0.7}
            elif q["type"] == "score":
                k = len(q["criteria"]); p = [1.0 / k] * k
                out[qid] = {"type": "score", "probabilities": {str(i): x for i, x in enumerate(p)}, "label": "0", "confidence": 0.0, "score": (k - 1) / 2}
            else:
                names = list(q["criteria"]); p = [0.0] * len(names); p[0] = 1.0
                out[qid] = {"type": "choice", "probabilities": dict(zip(names, p)), "label": names[0], "confidence": 1.0}
        return out


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_wire_contract():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeDecider(), "fake"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        status, body = _post(base + "/v1/systemone", {"state": "s", "questions": {
            "a": {"type": "noul", "instructions": "?"},
            "b": {"type": "choice", "instructions": "?", "criteria": {"x": "", "y": ""}},
            "c": {"type": "score", "instructions": "?", "criteria": ["lo", "hi"]}}})
        assert status == 200 and body["model"] == "fake"
        a, b, c = body["answers"]["a"], body["answers"]["b"], body["answers"]["c"]
        assert a["type"] == "noul" and a["noul"] == 0.7 and a["probabilities"] == {"yes": 0.7, "no": 0.3}
        assert b["type"] == "choice" and b["choice"] == "x" and set(b["probabilities"]) == {"x", "y"}
        assert c["type"] == "score" and set(c["probabilities"]) == {"0", "1"}
        assert body["usage"]["decisions"] == 3
        status, body = _post(base + "/v1/systemone", {"state": "s"})
        assert status == 400
        status, body = _post(base + "/nope", {})
        assert status == 404
        with urllib.request.urlopen(base + "/health") as r:
            assert json.loads(r.read())["ok"] is True
    finally:
        srv.shutdown()
