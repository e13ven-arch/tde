"""JevBench adapter for a TDE run directory (in-process, native probabilities).

Drop into `jevbench/adapters/` of https://github.com/fstandhartinger/jevbench or use scripts/run_jevbench.py,
which registers it without modifying the benchmark. Mirrors the `verdict_local` / `local_openjev` adapters:
one forward pass per decision, probabilities over the task's exact label list, no generation.
"""
from __future__ import annotations

import json
import time


class TdeLocalAdapter:
    name = "tde_local"
    cost_basis = "local_gpu_no_provider_tariff"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None, price_input_per_m=None,
                 price_output_per_m=None, device=None, revision=None, use_temperature=True, **kwargs):
        self.path = endpoint  # TDE run directory (config.json, tokenizer/, best.pt)
        self.model = model or str(endpoint)
        self.key_env = key_env
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.device = device
        self.revision = revision
        self.use_temperature = use_temperature
        self._decider = None

    def load(self):
        if self._decider is None:
            from tde.inference import Decider
            t0 = time.perf_counter()
            self._decider = Decider.from_run(self.path, device=self.device, use_temperature=self.use_temperature)
            self.load_s = time.perf_counter() - t0
        return self._decider

    def build_question(self, task) -> dict:
        q = task.question
        qtype = q["type"]
        crit = q.get("criteria")
        if qtype == "choice":
            crit = crit if isinstance(crit, dict) else {}
            # exact label list from the task, descriptions from the rubric when present
            criteria = {lab: (crit.get(lab) or "") for lab in task.labels}
            return {"type": "choice", "instructions": q["instructions"], "criteria": criteria}
        if qtype == "score":
            levels = list(crit) if isinstance(crit, list) else [str(i) for i in range(len(task.labels))]
            return {"type": "score", "instructions": q["instructions"], "criteria": levels}
        return {"type": "noul", "instructions": q["instructions"], "criteria": crit if isinstance(crit, dict) else {}}

    def run(self, task):
        from jevbench.adapters.base import DecisionResult

        res = DecisionResult(adapter=self.name, ok=False, probs_source="native", model=self.model)
        try:
            d = self.load()
            q = self.build_question(task)
        except Exception as e:  # noqa: BLE001
            res.error = f"load/query failed: {type(e).__name__}: {str(e)[:250]}"
            return res
        state = task.state if isinstance(task.state, str) else json.dumps(task.state, ensure_ascii=False)
        res.request_body = {"state": state, "question": q}
        t0 = time.perf_counter()
        try:
            out = d.decide(state, q)
        except Exception as e:  # noqa: BLE001
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res
        res.latency_s = time.perf_counter() - t0
        p = out["probabilities"]
        if task.question["type"] == "noul":
            probs = {"no": p.get("no", 0.0), "yes": p.get("yes", 0.0)}
        else:
            probs = {lab: float(p.get(lab, 0.0)) for lab in task.labels}
        total = sum(probs.values())
        if total <= 0:
            res.error = "no probability mass on the label set"
            return res
        res.probs = {k: v / total for k, v in probs.items()}
        res.raw = {"response": out, "runtime": {"device": str(d.device), "temperature_scaled": d.temperature is not None,
                                                 "readout": d.cfg.get("readout"), "backbone": d.cfg.get("backbone"),
                                                 "probability_origin": "native softmax over the label set"}}
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0
