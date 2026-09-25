"""LoRA delta stage for decider-4b: same data pipeline, batching, loss and eval as decider.train, but the base stays
frozen in bf16 and only fp32 LoRA adapters (r64, alpha 128, every linear layer except lm_head) are trained.
Why: decider.train updates bf16 parameters directly; at lr ~1e-5 most updates fall below bf16 resolution (2^-8 relative)
and are rounded away. decider's own v2 stage used LoRA. The adapters are merged and a plain checkpoint is saved.

    .venv/bin/python /root/work/jev/ops/decider_lora_train.py --model Mapika/decider-4b --data mix.pkl --out runs/lora_v1 \
        --epochs 1 --lr 1e-4 --r 64 --alpha 128 --max_tokens 16384 --accum 2
"""
import argparse, json, math, os, random, time
import torch
from decider.model import DecisionModel, collate
from decider.train import make_items, batches_by_tokens, loss_fn
from decider.evaluate import run_eval, aggregate
from decider import data as D
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Mapika/decider-4b"); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--r", type=int, default=64); ap.add_argument("--alpha", type=int, default=128); ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=16384); ap.add_argument("--accum", type=int, default=2); ap.add_argument("--max_ctx", type=int, default=4096)
    ap.add_argument("--max_options", type=int, default=255); ap.add_argument("--none_prob", type=float, default=0.1); ap.add_argument("--schema_first_prob", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=400); ap.add_argument("--eval_limit", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad_ckpt", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    logf = open(f"{a.out}/train.log", "a")
    def log(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    log("[args]", json.dumps(vars(a)))
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    train, evals = D.load_cache(a.data); evals_small = {k: v[:a.eval_limit] for k, v in evals.items()}
    model = DecisionModel(a.model, grad_ckpt=bool(a.grad_ckpt)).cuda(); tok = model.tok
    for p in model.parameters(): p.requires_grad_(False)
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules="all-linear", bias="none")
    model.lm = inject_adapter_in_model(cfg, model.lm)          # in place: DecisionModel.forward keeps working
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in lora_params: p.data = p.data.float(); p.requires_grad_(True)
    n_lora = sum(p.numel() for p in lora_params); log(f"[lora] r={a.r} alpha={a.alpha} trainable {n_lora/1e6:.1f}M params (fp32), base frozen bf16")
    items = make_items(train, tok, rng, a.max_ctx, a.none_prob, a.max_options, a.schema_first_prob)
    log(f"[data] {len(items)} items, {sum(len(it['ids']) for it in items)/1e6:.1f}M tokens")
    opt = torch.optim.AdamW(lora_params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(batches_by_tokens(items, a.max_tokens, random.Random(0))) / a.accum); total = int(steps_per_epoch * a.epochs)
    log(f"[sched] {steps_per_epoch} steps/epoch, {total} total")
    lr_at = lambda s: a.lr * s / a.warmup if s < a.warmup else a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, total - a.warmup))))
    step, micro, ep, hist = 0, 0, 0, []; model.train(); t0 = time.time(); ce_acc, n_acc, t_last, tok_acc = 0.0, 0, t0, 0
    while step < total:
        for bidx in batches_by_tokens(items, a.max_tokens, rng):
            if step >= total: break
            b = collate([items[i] for i in bidx], tok.pad_token_id); b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b)
            loss, ce = loss_fn(logits.float(), b["golds"], b["nopts"])
            (loss / a.accum).backward(); ce_acc += ce.item(); n_acc += 1; micro += 1; tok_acc += b["input_ids"].numel()
            if micro % a.accum == 0:
                for g in opt.param_groups: g["lr"] = lr_at(step)
                gn = torch.nn.utils.clip_grad_norm_(lora_params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    now = time.time(); log(f"[train] step {step}/{total} ep {ep} ce {ce_acc/n_acc:.4f} gn {gn:.2f} lr {lr_at(step):.2e} {(now-t0)/60:.1f}min eta {(total-step)*(now-t_last)/20/60:.0f}min {tok_acc/(now-t_last):.0f}tok/s mem {torch.cuda.max_memory_allocated()/2**30:.0f}GB")
                    ce_acc, n_acc, t_last, tok_acc = 0.0, 0, now, 0
                if step % a.eval_every == 0 or step == total:
                    model.eval()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        res, _ = run_eval(model, evals_small, log=log)
                    agg = aggregate(res); log(f"[eval-agg] step {step} " + json.dumps(agg)); hist.append(dict(step=step, agg=agg, results=res))
                    json.dump(hist, open(f"{a.out}/hist.json", "w"), indent=1); model.train()
        ep += 1
    # merge adapters into the base weights and save a plain checkpoint
    model.eval()
    for m in model.lm.modules():
        if isinstance(m, LoraLayer): m.merge()
    sd = {k.replace(".base_layer", ""): v for k, v in model.lm.state_dict().items() if "lora_" not in k}
    from transformers import AutoModelForCausalLM
    clean = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    missing, unexpected = clean.load_state_dict(sd, strict=False); log(f"[merge] missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing, missing[:5]
    clean.save_pretrained(f"{a.out}/model"); tok.save_pretrained(f"{a.out}/model")
    from huggingface_hub import snapshot_download; import shutil
    src = a.model if os.path.isdir(a.model) else snapshot_download(a.model, allow_patterns=["decider_config.json", "chat_template.jinja"])
    for f in ("decider_config.json", "chat_template.jinja"):
        if os.path.exists(f"{src}/{f}"): shutil.copy(f"{src}/{f}", f"{a.out}/model/{f}")
    log("[done] saved", f"{a.out}/model")


if __name__ == "__main__":
    main()
