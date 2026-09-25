"""Score a /v1/systemone server on a TDE decision jsonl (e.g. the typed-decisions test set) as a sealed-set proxy.

    python ops/wire_eval.py --endpoint http://127.0.0.1:8006 --data data/v0.7/eval_only/typed_decisions.test.jsonl --limit 2000
Prints accuracy, Brier (over the returned distribution), ECE and per-primitive accuracy. No training, no temperature.
"""
import argparse, json, math, urllib.request, time, collections


def post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def to_question(r):
    prim, cands = r["primitive"], r["candidates"]
    if prim == "noul":
        crit = {}
        for c in cands:
            if c["name"] == "yes" and c.get("description"): crit["true"] = c["description"]
            if c["name"] == "no" and c.get("description"): crit["false"] = c["description"]
        return {"type": "noul", "instructions": r["question"], **({"criteria": crit} if crit else {})}
    if prim == "score":
        return {"type": "score", "instructions": r["question"], "criteria": [c.get("description") or c["name"] for c in cands]}
    return {"type": "choice", "instructions": r["question"], "criteria": {c["name"]: (c.get("description") or "") for c in cands}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--endpoint", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=2000); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data)][: a.limit]
    stats = collections.defaultdict(lambda: [0, 0]); brier = []; conf = []; cor = []; recs = []; t0 = time.time(); fails = 0
    for r in rows:
        q = to_question(r); names = [c["name"] for c in r["candidates"]]; gold = max(range(len(r["target"])), key=lambda i: r["target"][i])
        try:
            ans = post(a.endpoint.rstrip("/") + "/v1/systemone", {"state": r["state"], "questions": {"q": q}})["answers"]["q"]
        except Exception as e:
            fails += 1; continue
        if r["primitive"] == "noul":
            p_yes = float(ans["noul"]); probs = {"yes": p_yes, "no": 1.0 - p_yes}
        elif r["primitive"] == "score":
            probs = {names[i]: float(ans["probabilities"].get(str(i), 0.0)) for i in range(len(names))}
        else:
            probs = {n: float(ans["probabilities"].get(n, 0.0)) for n in names}
        p = [probs.get(n, 0.0) for n in names]; pred = max(range(len(p)), key=lambda i: p[i]); ok = int(pred == gold)
        stats[r["primitive"]][0] += ok; stats[r["primitive"]][1] += 1
        brier.append(sum((p[i] - r["target"][i]) ** 2 for i in range(len(p)))); conf.append(max(p)); cor.append(ok)
        recs.append({"id": r["id"], "primitive": r["primitive"], "correct": ok, "probs": probs, "names": names, "target": r["target"]})
    n = len(cor); acc = sum(cor) / n
    bins = [[] for _ in range(10)]
    for c, y in zip(conf, cor): bins[min(9, int(c * 10))].append((c, y))
    ece = sum(len(b) / n * abs(sum(y for _, y in b) / len(b) - sum(c for c, _ in b) / len(b)) for b in bins if b)
    print(f"[wire_eval] n={n} fails={fails} acc={acc:.3f} brier={sum(brier)/n:.3f} ece={ece:.3f} " + " ".join(f"{k}={v[0]/v[1]:.3f}({v[1]})" for k, v in sorted(stats.items())) + f" {time.time()-t0:.0f}s")
    if a.out:
        json.dump({"n": n, "acc": acc, "brier": sum(brier) / n, "ece": ece, "by_primitive": {k: v[0] / v[1] for k, v in stats.items()}, "rows": recs}, open(a.out, "w"))


if __name__ == "__main__":
    main()
