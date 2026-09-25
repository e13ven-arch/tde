"""Extract the ModernBERT encoder (and tokenizer) from a GLiClass checkpoint into a plain ModernBertModel folder.

    python scripts/extract_gliclass_encoder.py --out checkpoints/gliclass-modern-base-encoder
Used as `backbone:` in the release configs (vocab 50,370 = ModernBERT-base + GLiClass's added tokens).
"""
from __future__ import annotations
import argparse, json, os
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="knowledgator/gliclass-modern-base-v3.0")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", default="checkpoints/gliclass-modern-base-encoder")
    a = ap.parse_args()
    src = snapshot_download(a.repo, revision=a.revision)
    cfg = json.load(open(os.path.join(src, "config.json")))
    enc_cfg = cfg["encoder_config"]
    enc_cfg.pop("architectures", None)
    state = load_file(os.path.join(src, "model.safetensors"))
    anchor = next(k for k in state if k.endswith("embeddings.tok_embeddings.weight"))
    prefix = anchor[: -len("embeddings.tok_embeddings.weight")]
    enc_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    model = ModernBertModel(ModernBertConfig(**enc_cfg))
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    unexpected = [k for k in unexpected]  # e.g. GLiClass heads under the same prefix
    print(f"prefix={prefix!r} tensors={len(enc_state)} missing={missing} unexpected={unexpected[:5]}")
    assert not missing, missing
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(a.out)
    print("saved", a.out, "vocab", model.config.vocab_size, "params", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
