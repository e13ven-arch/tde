"""Device-agnostic trainer for the three readout models.

Checkpoint layout (out_dir/):
    config.json      backbone, readout, hidden, tiny, finetune, loss weights
    tokenizer/       HF tokenizer with [OPT]/[DECIDE] added
    best.pt          state_dict selected by calibration NLL
    log.jsonl        one line per logging step
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from tde.calibration.metrics import Predictions, summarize
from tde.losses import log_probs, total_loss
from tde.model.encoding import collate
from tde.model.factory import apply_finetune_mode, build_model, pick_device
from tde.schema import DecisionExample, load_jsonl


@dataclass
class TrainConfig:
    backbone: str = "answerdotai/ModernBERT-base"
    readout: str = "joint"            # joint | branch | biencoder
    finetune: str = "full"            # full | frozen80 | lora
    data_dir: str = "data/v0.1"
    out_dir: str = "runs/debug"
    train_limit: int | None = None
    per_dataset_cap: int | None = None  # balance the mix: keep at most N training examples per dataset
    group_by_state: bool = False        # keep rewrites of the same source item adjacent so a batch shares states (branch/biencoder)
    eval_limit: int | None = 2000
    epochs: float = 1.0
    max_steps: int | None = None
    batch_size: int = 16
    grad_accum: int = 2
    lr_backbone: float = 2e-5
    lr_head: float = 1e-4
    layer_decay: float = 0.9
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    w_ce: float = 1.0
    w_brier: float = 0.0
    w_rps: float = 0.0
    w_perm: float = 0.0
    w_conf: float = 0.0
    w_pg: float = 0.0                 # RLCD-style paired proper-reward policy gradient (control arm)
    w_correct_pg: float = 0.0         # improper correctness-only REINFORCE (negative control)
    pg_samples: int = 32
    use_confidence_head: bool = False
    branch_layers: int = 3
    branch_through_backbone: bool = True
    marker: str = "mask"              # mask | new | eos (decoders)
    pool: str = "marker+span"         # marker | span | marker+span
    max_state_tokens: int = 448
    eval_every: int = 500
    log_every: int = 25
    seed: int = 0
    device: str = "auto"
    bf16: bool = True
    tiny: bool = False                # random tiny backbone for smoke tests
    notes: str = ""
    extra: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _param_groups(model, cfg: TrainConfig):
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    named = list(model.backbone.named_parameters())
    # layer-wise lr decay: later layers get higher lr; assign by order of appearance
    n = len(named)
    groups = []
    for i, (name, p) in enumerate(named):
        if not p.requires_grad:
            continue
        depth = i / max(n - 1, 1)
        lr = cfg.lr_backbone * (cfg.layer_decay ** ((1 - depth) * 12))
        wd = 0.0 if (p.ndim == 1 or "bias" in name or "norm" in name.lower()) else cfg.weight_decay
        groups.append({"params": [p], "lr": lr, "weight_decay": wd})
    heads = [p for p in model.parameters() if id(p) not in backbone_ids and p.requires_grad]
    groups.append({"params": heads, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay})
    return groups


def _to(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _permute(examples: list[DecisionExample], rng: random.Random) -> tuple[list[DecisionExample], torch.Tensor]:
    kmax = max(e.k for e in examples)
    perms, out = torch.arange(kmax).unsqueeze(0).repeat(len(examples), 1), []
    for i, e in enumerate(examples):
        order = list(range(e.k))
        rng.shuffle(order)
        perms[i, : e.k] = torch.tensor(order)
        out.append(e.with_candidate_order(order))
    return out, perms


@torch.no_grad()
def predict(model, dtok, examples: list[DecisionExample], device: torch.device, batch_size: int = 32, bf16: bool = True) -> list[np.ndarray]:
    """Return raw logits (length K each) for every example, in order."""
    model.eval()
    mode = model.mode
    out: list[np.ndarray] = []
    use_amp = bf16 and device.type == "cuda"
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        batch = _to(collate([dtok.encode(e, mode) for e in chunk], dtok.pad_id, mode), device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            res = model(batch)
        logits = res["logits"].float().cpu()
        for j, e in enumerate(chunk):
            out.append(logits[j, : e.k].numpy())
    return out


@torch.no_grad()
def predict_chunked(model, dtok, examples: list[DecisionExample], device: torch.device, chunk_k: int = 12,
                    batch_size: int = 32, bf16: bool = True, tournament: bool = True) -> list[np.ndarray]:
    """Score candidate sets larger than `chunk_k` in chunks (state repeated), concatenating chunk logits.

    Approximates a pointwise readout at inference so a model trained with K <= chunk_k can be queried with
    hundreds of options; exact when the readout is order/set independent."""
    pieces: list[tuple[int, list[int], DecisionExample]] = []
    for i, e in enumerate(examples):
        if e.k <= chunk_k:
            pieces.append((i, list(range(e.k)), e))
        else:
            order = list(range(e.k))
            for s in range(0, e.k, chunk_k):
                idx = order[s : s + chunk_k]
                if len(idx) < 2:  # a trailing singleton gets a partner so the listwise softmax is well defined
                    idx = order[s - 1 : s + chunk_k]
                pieces.append((i, idx, e.with_candidate_order(idx)))
    logits = predict(model, dtok, [p[2] for p in pieces], device, batch_size, bf16)
    out: list[np.ndarray] = [np.full(e.k, np.nan) for e in examples]
    winners: dict[int, list[tuple[int, float]]] = {}  # example -> [(candidate, round-1 logit)] per chunk
    for (i, idx, _), z in zip(pieces, logits):
        for j, c in enumerate(idx):
            if np.isnan(out[i][c]):
                out[i][c] = z[j]
        if examples[i].k > chunk_k:
            w = int(np.argmax(z[: len(idx)]))
            winners.setdefault(i, []).append((idx[w], float(z[w])))
    if tournament:
        # round 2: chunk winners compete listwise; each chunk's logits are shifted so its winner takes its round-2 logit
        finals = []
        for i, ws in winners.items():
            seen = {}
            for c, z1 in ws:
                seen.setdefault(c, z1)
            cands = list(seen)
            if len(cands) >= 2:
                finals.append((i, cands, [seen[c] for c in cands], examples[i].with_candidate_order(cands)))
        if finals:
            z2s = predict(model, dtok, [f[3] for f in finals], device, batch_size, bf16)
            for (i, cands, z1s, _), z2 in zip(finals, z2s):
                # every candidate belongs to exactly one chunk; find its chunk winner and apply that shift
                chunk_of = {}
                for (ii, idx, _) in pieces:
                    if ii == i:
                        w = max(idx, key=lambda c: out[i][c])
                        for c in idx:
                            chunk_of[c] = w
                shift = {w: z2[k] - z1 for k, (w, z1) in enumerate(zip(cands, z1s))}
                out[i] = np.array([out[i][c] + shift.get(chunk_of[c], 0.0) for c in range(examples[i].k)])
    return out


def evaluate(model, dtok, examples: list[DecisionExample], device, batch_size=32, bf16=True) -> dict:
    logits = predict(model, dtok, examples, device, batch_size, bf16)
    probs = []
    for z in logits:
        z = z - z.max()
        p = np.exp(z)
        probs.append(p / p.sum())
    P = Predictions(probs, [np.array(e.target) for e in examples], groups=[e.source_id for e in examples],
                    ordinal=[e.primitive == "score" for e in examples])
    return summarize(P)


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    train_ex = load_jsonl(Path(cfg.data_dir) / "train.jsonl", cfg.train_limit if not cfg.per_dataset_cap else None)
    if cfg.per_dataset_cap:
        seen: dict[str, int] = {}
        kept = []
        for e in train_ex:
            if seen.get(e.dataset, 0) < cfg.per_dataset_cap:
                kept.append(e); seen[e.dataset] = seen.get(e.dataset, 0) + 1
        train_ex = kept[: cfg.train_limit] if cfg.train_limit else kept
        print(f"[train] per-dataset cap {cfg.per_dataset_cap}: {seen}")
    cal_ex = load_jsonl(Path(cfg.data_dir) / "calibration.jsonl", cfg.eval_limit)
    dtok, model = build_model(cfg.backbone, cfg.readout, tiny=cfg.tiny, use_confidence_head=cfg.use_confidence_head,
                              branch_layers=cfg.branch_layers, max_state_tokens=cfg.max_state_tokens,
                              branch_through_backbone=cfg.branch_through_backbone, marker=cfg.marker, pool=cfg.pool)
    summary = apply_finetune_mode(model, cfg.finetune)
    dtok.tok.save_pretrained(out_dir / "tokenizer")
    (out_dir / "config.json").write_text(json.dumps({**asdict(cfg), "hidden": model.scorer.q.in_features if hasattr(model, "scorer") else None, **summary}, indent=2))
    model.to(device)
    print(f"[train] device={device} readout={cfg.readout} finetune={cfg.finetune} params={summary['total_params']/1e6:.1f}M trainable={summary['trainable_params']/1e6:.1f}M train={len(train_ex)} cal={len(cal_ex)}")

    steps_per_epoch = math.ceil(len(train_ex) / (cfg.batch_size * cfg.grad_accum))
    total_steps = cfg.max_steps or max(1, int(steps_per_epoch * cfg.epochs))
    warmup = max(1, int(total_steps * cfg.warmup_ratio))
    opt = torch.optim.AdamW(_param_groups(model, cfg), betas=(0.9, 0.98), eps=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    use_amp = cfg.bf16 and device.type == "cuda"
    rng = random.Random(cfg.seed)
    log_f = (out_dir / "log.jsonl").open("a")
    best_nll, step, t0 = float("inf"), 0, time.time()
    order = list(range(len(train_ex)))
    groups: list[list[int]] | None = None
    if cfg.group_by_state:
        by_src: dict[str, list[int]] = {}
        for i, e in enumerate(train_ex):
            by_src.setdefault(e.source_id, []).append(i)
        groups = list(by_src.values())
        print(f"[train] group_by_state: {len(groups)} source items, {len(train_ex)/len(groups):.1f} examples each")
    mode = model.mode
    micro = 0
    running: dict[str, float] = {}
    done = False
    while not done:
        if groups is not None:
            rng.shuffle(groups)
            order = [i for g in groups for i in g]
        else:
            rng.shuffle(order)
        for i in range(0, len(order), cfg.batch_size):
            chunk = [train_ex[j] for j in order[i : i + cfg.batch_size]]
            model.train()
            batch = _to(collate([dtok.encode(e, mode) for e in chunk], dtok.pad_id, mode), device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(batch)
                out_perm, perm = None, None
                if cfg.w_perm > 0:
                    pchunk, perm = _permute(chunk, rng)
                    pbatch = _to(collate([dtok.encode(e, mode) for e in pchunk], dtok.pad_id, mode), device)
                    out_perm = model(pbatch)
                    perm = perm.to(device)
                loss, parts = total_loss(out, batch, w_ce=cfg.w_ce, w_brier=cfg.w_brier, w_rps=cfg.w_rps, w_perm=cfg.w_perm,
                                         out_perm=out_perm, perm=perm, w_conf=cfg.w_conf,
                                         w_pg=cfg.w_pg, w_correct_pg=cfg.w_correct_pg, pg_samples=cfg.pg_samples)
            (loss / cfg.grad_accum).backward()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            running["loss"] = running.get("loss", 0.0) + float(loss.detach())
            micro += 1
            if micro % cfg.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0:
                rec = {"step": step, "elapsed_s": round(time.time() - t0, 1), "lr_head": sched.get_last_lr()[-1],
                       **{k: v / (cfg.log_every * cfg.grad_accum) for k, v in running.items()}}
                running = {}
                print("[train]", json.dumps(rec)); log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if step % cfg.eval_every == 0 or step >= total_steps:
                ev = evaluate(model, dtok, cal_ex, device, cfg.batch_size * 2, cfg.bf16) if cal_ex else {}
                rec = {"step": step, "eval": ev}
                print("[eval]", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()}))
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                if ev and ev["nll"] < best_nll:
                    best_nll = ev["nll"]
                    torch.save(model.state_dict(), out_dir / "best.pt")
                    (out_dir / "best_eval.json").write_text(json.dumps({"step": step, **ev}, indent=2))
            if step >= total_steps:
                done = True
                break
    if not (out_dir / "best.pt").exists():
        torch.save(model.state_dict(), out_dir / "best.pt")
    log_f.close()
    return {"steps": step, "best_nll": best_nll, "out_dir": str(out_dir), **summary}


def load_checkpoint(out_dir: str | Path, device: torch.device | None = None, max_total: int | None = None):
    """Rebuild tokenizer + model from a run directory."""
    out_dir = Path(out_dir)
    cfg = json.loads((out_dir / "config.json").read_text())
    device = device or pick_device(cfg.get("device", "auto"))
    tiny = cfg.get("tiny", False)
    dtok, model = build_model(cfg["backbone"], cfg["readout"], tiny=tiny, use_confidence_head=cfg.get("use_confidence_head", False),
                              branch_layers=cfg.get("branch_layers", 3), max_state_tokens=cfg.get("max_state_tokens", 448),
                              branch_through_backbone=cfg.get("branch_through_backbone", True),
                              marker=cfg.get("marker", "new"), pool=cfg.get("pool", "marker"))  # old checkpoints predate these keys
    if not tiny:
        from transformers import AutoTokenizer
        from tde.model.encoding import DecisionTokenizer
        tok = AutoTokenizer.from_pretrained(out_dir / "tokenizer")
        dtok = DecisionTokenizer(tok, max_state_tokens=cfg.get("max_state_tokens", 448), marker=cfg.get("marker", "new"))
    if max_total:
        dtok.max_total = max_total  # inference-time sequence budget (ModernBERT supports 8k); training used 1024
    apply_finetune_mode(model, cfg.get("finetune", "full"))
    state = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return dtok, model, cfg, device
