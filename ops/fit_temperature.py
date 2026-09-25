"""Fit one temperature per question type from a wire_eval json taken at T=1 (probabilities -> log-probs -> softmax(logp/T)).

    python ops/fit_temperature.py --evals cal_typed.json cal_synth.json --write runs/x/model/decider_config.json
Minimises NLL against the (possibly soft) targets on a grid, prints NLL/ECE before and after, and writes
`temperature_by_type` (plus the mean as `temperature`) into the decider_config.json if --write is given.
"""
import argparse, json, math, collections


def nll_ece(rows, T):
    nll = 0.0; conf = []; cor = []
    for r in rows:
        lp = [math.log(max(r["probs"].get(n, 0.0), 1e-9)) / T for n in r["names"]]
        m = max(lp); z = [math.exp(x - m) for x in lp]; s = sum(z); p = [x / s for x in z]
        nll -= sum(t * math.log(max(pi, 1e-9)) for t, pi in zip(r["target"], p))
        g = max(range(len(p)), key=lambda i: r["target"][i]); conf.append(max(p)); cor.append(int(max(range(len(p)), key=lambda i: p[i]) == g))
    bins = [[] for _ in range(10)]
    for c, y in zip(conf, cor): bins[min(9, int(c * 10))].append((c, y))
    ece = sum(len(b) / len(rows) * abs(sum(y for _, y in b) / len(b) - sum(c for c, _ in b) / len(b)) for b in bins if b)
    return nll / len(rows), ece


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--evals", nargs="+", required=True); ap.add_argument("--write", default=None)
    a = ap.parse_args()
    by = collections.defaultdict(list)
    for f in a.evals:
        for r in json.load(open(f))["rows"]: by[r["primitive"]].append(r)
    grid = [x / 100 for x in range(50, 401, 2)]
    out = {}
    for prim, rows in sorted(by.items()):
        n0, e0 = nll_ece(rows, 1.0)
        best = min(grid, key=lambda T: nll_ece(rows, T)[0]); n1, e1 = nll_ece(rows, best)
        out[prim] = best; print(f"[fit] {prim:6s} n={len(rows)} T={best:.2f}  nll {n0:.3f}->{n1:.3f}  ece {e0:.3f}->{e1:.3f}")
    if a.write:
        cfg = json.load(open(a.write)); cfg["temperature_by_type"] = out; cfg["temperature"] = round(sum(out.values()) / len(out), 3)
        json.dump(cfg, open(a.write, "w"), indent=1); print("[fit] wrote", a.write, out)


if __name__ == "__main__":
    main()
