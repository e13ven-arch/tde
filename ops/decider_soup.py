"""Weight-average (model soup) several decider-4b revisions into one checkpoint.
    .venv/bin/python ops/decider_soup.py --revisions v2 main --out runs/soup_v2_v21/model
"""
import argparse, os, shutil, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="Mapika/decider-4b"); ap.add_argument("--revisions", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
paths = [snapshot_download(a.repo, revision=r) for r in a.revisions]
print("snapshots:", paths)
sds = [load_file(os.path.join(p, "model.safetensors")) for p in paths]
keys = set(sds[0]); assert all(set(s) == keys for s in sds), "key sets differ"
avg = {k: (sum(s[k].float() for s in sds) / len(sds)).to(sds[0][k].dtype) for k in keys}
os.makedirs(a.out, exist_ok=True); save_file(avg, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
for f in os.listdir(paths[-1]):
    if f != "model.safetensors" and os.path.isfile(os.path.join(paths[-1], f)): shutil.copy(os.path.join(paths[-1], f), os.path.join(a.out, f))
print("saved soup of", a.revisions, "to", a.out)
