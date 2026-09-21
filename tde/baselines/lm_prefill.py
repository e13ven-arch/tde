"""Causal-LM prefill baseline (SemIf / simple-jev protocol): constrained-choice option log-prob, no decoding.

Prompt = state + question + enumerated options; the score of option i is the
summed log-prob of its label letter (A, B, C, ...) as the next token. One
prefill per example; the letter distribution is renormalised over offered
options. Zero training. Track B only.
"""
from __future__ import annotations

import string

import numpy as np
import torch

from tde.schema import DecisionExample

LETTERS = string.ascii_uppercase


class LMPrefill:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str | None = None, max_length: int = 2048):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.model.to(self.device)
        self.max_length = max_length
        self.letter_ids = [self.tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]

    def prompt(self, e: DecisionExample) -> str:
        opts = "\n".join(f"{LETTERS[i]}. {c.text()}" for i, c in enumerate(e.candidates))
        return (f"Context:\n{e.state}\n\nQuestion: {e.question}\nOptions:\n{opts}\n\n"
                f"Answer with the letter of the best option.\nAnswer:")

    @torch.no_grad()
    def predict(self, examples: list[DecisionExample], batch_size: int = 4) -> list[np.ndarray]:
        out = []
        for s in range(0, len(examples), batch_size):
            chunk = examples[s : s + batch_size]
            enc = self.tok([self.prompt(e) for e in chunk], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length, padding_side="left").to(self.device)
            logits = self.model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
            for j, e in enumerate(chunk):
                z = np.array([lp[j, self.letter_ids[i]] for i in range(e.k)])
                z = z - z.max(); p = np.exp(z); out.append(p / p.sum())
        return out
