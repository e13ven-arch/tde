"""NLI zero-shot baseline (MoritzLaurer/deberta-v3-*-zeroshot-v2.0 family).

Each candidate becomes a hypothesis "<question> <candidate>" scored for entailment
against the state; probabilities are the softmax over candidate entailment logits.
Zero training. Track B only.
"""
from __future__ import annotations

import numpy as np
import torch

from tde.schema import DecisionExample


class NLIZeroShot:
    def __init__(self, model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c", device: str | None = None, max_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.model.to(self.device)
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail = labels.get("entailment", 0)
        self.max_length = max_length

    @torch.no_grad()
    def predict(self, examples: list[DecisionExample], batch_size: int = 16) -> list[np.ndarray]:
        pairs, owners = [], []
        for i, e in enumerate(examples):
            for c in e.candidates:
                hyp = f"{e.question} {c.text()}" if e.primitive != "noul" else (f"{e.question} The answer is {c.name}.")
                pairs.append((e.state, hyp)); owners.append(i)
        scores = np.zeros(len(pairs))
        for s in range(0, len(pairs), batch_size):
            chunk = pairs[s : s + batch_size]
            enc = self.tok([a for a, _ in chunk], [b for _, b in chunk], truncation="only_first", max_length=self.max_length,
                           padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.float()
            scores[s : s + len(chunk)] = logits[:, self.entail].cpu().numpy()
        out, cursor = [], 0
        for e in examples:
            z = scores[cursor : cursor + e.k]; cursor += e.k
            z = z - z.max(); p = np.exp(z); out.append(p / p.sum())
        return out
