"""JevBench adapter for a TDE run directory (in-process, native probabilities).

Drop into `jevbench/adapters/` of https://github.com/fstandhartinger/jevbench or use scripts/run_jevbench.py,
which registers it without modifying the benchmark. Mirrors the `verdict_local` / `local_openjev` adapters:
one forward pass per decision, probabilities over the task's exact label list, no generation.

Measured method: the model's own softmax, no post-hoc temperature (`use_temperature=False` by default; the TDE
loader would otherwise apply a `temperature.json` if a run directory contained one). The adapter never fills in
missing labels and never renormalises: the returned distribution is passed to the shared scorer as is, and a
label set that differs from the task's declared labels, or a non-finite / negative value, is reported as an error.
"""
from __future__ import annotations

import json
import math
import time


class TdeLocalAdapter:
    name = "tde_local"
    cost_basis = "local_gpu_no_provider_tariff"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None, price_input_per_m=None,
                 price_output_per_m=None, device=None, revision=None, use_temperature=False, **kwargs):
        self.path = endpoint  # TDE run / release directory (config.json, tokenizer/, best.pt)
        self.model = model or str(endpoint)
        self.key_env = key_env
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.device = device
        self.revision = revision
        self.use_temperature = bool(use_temperature)
        self._decider = None

    def load(self):
        if self._decider is None:
            from tde.inference import Decider
            t0 = time.perf_counter()
            self._decider = Decider.from_run(self.path, device=self.device, use_temperature=self.use_temperature)
            self.load_s = time.perf_counter() - t0
        return self._decider

    def build_question(self, task) -> dict:
        """Map a JevBench task to a TDE question; candidates follow the task's declared label order exactly."""
        q = task.question
        qtype = q["type"]
        crit = q.get("criteria")
        if qtype == "choice":
            crit = crit if isinstance(crit, dict) else {}
            return {"type": "choice", "instructions": q["instructions"],
                    "criteria": {lab: (crit.get(lab) or "") for lab in task.labels}}
        if qtype == "score":
            levels = list(crit) if isinstance(crit, list) else [""] * len(task.labels)
            if len(levels) != len(task.labels):
                raise ValueError(f"score criteria has {len(levels)} levels but the task declares {len(task.labels)} labels")
            return {"type": "score", "instructions": q["instructions"], "criteria": levels}
        return {"type": "noul", "instructions": q["instructions"], "criteria": crit if isinstance(crit, dict) else {}}

    @staticmethod
    def to_label_probs(out: dict, task) -> dict:
        """Native probabilities keyed by the task's declared labels; no filling, no renormalisation.

        Raises ValueError when the returned label set differs from the declared one or a value is not a finite
        non-negative number, so the caller reports an error instead of a repaired distribution."""
        p = out.get("probabilities")
        if not isinstance(p, dict):
            raise ValueError("model returned no probability mapping")
        declared = list(task.labels)
        returned = set(p)
        if returned != set(declared):
            raise ValueError(f"label set mismatch: returned {sorted(returned)} vs declared {declared}")
        probs = {}
        for lab in declared:
            v = p[lab]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"invalid probability for {lab!r}: {v!r}")
            probs[lab] = float(v)
        return probs

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
        res.raw = {"response": out, "runtime": {"device": str(getattr(d, "device", "")), "temperature_scaled": bool(getattr(d, "temperature", None) is not None),
                                                 "readout": getattr(d, "cfg", {}).get("readout"), "backbone": getattr(d, "cfg", {}).get("backbone"),
                                                 "probability_origin": "native softmax over the declared label set; passed through unmodified"}}
        try:
            res.probs = self.to_label_probs(out, task)
        except ValueError as e:
            res.error = str(e)
            return res
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0
