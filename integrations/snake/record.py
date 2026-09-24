#!/usr/bin/env python
"""Render one demo game to an animated GIF in the demo page's style, for the README.

    PYTHONPATH=. python -m integrations.snake.record --model release/tde-general-v0.2 --seed 103

Same play as demo.py (the model's top move with the anti-trap shield), one frame per move.
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
from PIL import Image, ImageDraw, ImageFont

from integrations.snake.demo import model_dir
from integrations.snake.game import DIRS, SnakeGame
from integrations.snake.play import choose, probs

C = dict(bg="#f5f1ea", panel="#fffdf9", line="#e7e0d4", grid="#fbf8f2", dot="#e3dccf", text="#1d2230", muted="#6f7482",
         snake="#4f46e5", tail="#a5b4fc", head="#312e81", eye="#fffdf9", food="#f97316", veto="#db2777", bar="#efe9df")
CELL, PAD, PANEL = 20, 18, 240


def rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))


def mix(a: str, b: str, t: float) -> tuple[int, ...]:
    return tuple(round(x + (y - x) * t) for x, y in zip(rgb(a), rgb(b)))


def palette() -> Image.Image:
    """Exact page colours, the body gradient and the ramps of antialiased text, so no colour drifts in the GIF."""
    cols = [rgb(v) for v in C.values()] + [mix(C["snake"], C["tail"], i / 31) for i in range(32)]
    for fg, bg in (("text", "bg"), ("muted", "bg"), ("text", "panel"), ("muted", "panel"), ("snake", "panel"),
                   ("veto", "panel"), ("text", "grid")):
        cols += [mix(C[fg], C[bg], i / 7) for i in range(8)]
    flat = [c for col in dict.fromkeys(cols) for c in col]
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat + [0] * (768 - len(flat)))
    return pal


def font(size: int, bold: bool = False):
    f = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", size)
    try:
        f.set_variation_by_name("Bold" if bold else "Regular")
    except (OSError, ValueError):
        pass
    return f


def frame(g: SnakeGame, p, move: str, veto: bool, ms: float, moves: int, name: str, fonts: dict, done: bool):
    bw, bh = g.w * CELL, g.h * CELL
    top = 64
    im = Image.new("RGB", (PAD * 3 + bw + PANEL, top + bh + PAD), C["bg"])
    d = ImageDraw.Draw(im)
    d.text((PAD, 14), "TDE plays Snake", font=fonts["title"], fill=C["text"])
    d.text((PAD, 40), f"{name} · a 150M decision model reads the board as text and scores four moves",
           font=fonts["small"], fill=C["muted"])
    x0, y0 = PAD, top
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), 10, fill=C["grid"], outline=C["line"])
    for y in range(g.h):
        for x in range(g.w):
            cx, cy = x0 + x * CELL + CELL // 2, y0 + y * CELL + CELL // 2
            d.rectangle((cx - 1, cy - 1, cx, cy), fill=C["dot"])
    if g.food:
        fx, fy = x0 + g.food[0] * CELL + CELL / 2, y0 + g.food[1] * CELL + CELL / 2
        d.ellipse((fx - 6, fy - 6, fx + 6, fy + 6), fill=C["food"])
    n = len(g.body)
    for i in range(n - 1, -1, -1):
        x, y = g.body[i]
        color = rgb(C["head"]) if i == 0 else mix(C["snake"], C["tail"], i / max(1, n - 1))
        cx, cy = x0 + x * CELL, y0 + y * CELL
        d.rounded_rectangle((cx + 2, cy + 2, cx + CELL - 3, cy + CELL - 3), 4, fill=color)
    hx, hy = g.body[0]
    dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[move]
    for s in (-1, 1):
        ex = x0 + hx * CELL + CELL / 2 + dx * 3 + (s * 4 if dy else 0)
        ey = y0 + hy * CELL + CELL / 2 + dy * 3 + (s * 4 if dx else 0)
        d.ellipse((ex - 1.6, ey - 1.6, ex + 1.6, ey + 1.6), fill=C["eye"])
    if done:
        d.rounded_rectangle((x0 + bw / 2 - 160, y0 + bh / 2 - 22, x0 + bw / 2 + 160, y0 + bh / 2 + 22), 12,
                            fill=C["panel"], outline=C["line"])
        msg = f"Survived {moves} moves · {g.score} food" if g.alive else f"Game over · {g.score} food"
        d.text((x0 + bw / 2, y0 + bh / 2), msg, font=fonts["row_b"], fill=C["text"], anchor="mm")
    px = x0 + bw + PAD
    d.rounded_rectangle((px, y0, px + PANEL, y0 + bh), 10, fill=C["panel"], outline=C["line"])
    d.text((px + 16, y0 + 14), "SCORE", font=fonts["label"], fill=C["muted"])
    d.text((px + 16, y0 + 30), str(g.score), font=fonts["score"], fill=C["text"])
    for k, (a, b) in enumerate((("Move", f"{g.steps} / {moves}"), ("Model latency (p50)", f"{ms:.0f} ms"))):
        d.text((px + 16, y0 + 84 + 22 * k), a, font=fonts["row"], fill=C["text"])
        d.text((px + PANEL - 16, y0 + 84 + 22 * k), b, font=fonts["row"], fill=C["text"], anchor="ra")
    d.line((px + 16, y0 + 140, px + PANEL - 16, y0 + 140), fill=C["line"])
    d.text((px + 16, y0 + 154), "MODEL'S MOVE PROBABILITIES", font=fonts["label"], fill=C["muted"])
    top_move = DIRS[max(range(4), key=lambda i: p[i])]
    for k, m in enumerate(DIRS):
        y = y0 + 178 + 26 * k
        col = C["snake"] if m == move else C["veto"] if veto and m == top_move else C["muted"]
        d.text((px + 16, y), m, font=fonts["row_b" if m == move else "row"], fill=col if m == move else C["text"])
        d.rounded_rectangle((px + 66, y + 4, px + PANEL - 58, y + 12), 4, fill=C["bar"])
        w = (PANEL - 124) * float(p[k])
        if w >= 1:
            d.rounded_rectangle((px + 66, y + 4, px + 66 + w, y + 12), 4, fill=col)
        d.text((px + PANEL - 16, y), f"{100 * float(p[k]):.0f}%", font=fonts["row"], fill=C["muted"], anchor="ra")
    if veto:
        d.text((px + 16, y0 + 290), f"Shield: {top_move} would trap the snake", font=fonts["small"], fill=C["veto"])
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tdelab/tde-general-v0.2")
    ap.add_argument("--seed", type=int, default=103)
    ap.add_argument("--moves", type=int, default=600)
    ap.add_argument("--frame_ms", type=int, default=30, help="per move; GIF delays under 20 ms are slowed by browsers")
    ap.add_argument("--out", default="integrations/snake/demo.gif")
    args = ap.parse_args()
    from pathlib import Path

    from transformers import AutoTokenizer

    from integrations.snake.encode import FastEncoder
    from tde.mlx.model import load_joint
    from tde.model.encoding import DecisionTokenizer
    path = model_dir(args.model)
    enc = FastEncoder(DecisionTokenizer(AutoTokenizer.from_pretrained(f"{path}/tokenizer"), max_state_tokens=480,
                                        marker="mask", max_total=1024))
    model = load_joint(f"{path}/model.safetensors")
    mx.set_cache_limit(1 << 30)
    fonts = {"title": font(22, True), "small": font(13), "label": font(11, True), "score": font(40, True),
             "row": font(14), "row_b": font(14, True)}
    name = Path(args.model).name
    g, frames, lat = SnakeGame(24, 16, 6, args.seed), [], []
    while g.alive and g.food is not None and g.steps < args.moves:
        t0 = time.perf_counter()
        p = probs(model, enc, [g])[0]
        lat.append(1000 * (time.perf_counter() - t0))
        ms = sorted(lat)[len(lat) // 2]
        move, veto = choose(p, g, "trap")
        g.step(move)
        done = not (g.alive and g.food is not None and g.steps < args.moves)
        frames.append(frame(g, p, move, veto, ms, args.moves, name, fonts, done))
    pal = palette()
    gif = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    gif[0].save(args.out, save_all=True, append_images=gif[1:], loop=0, optimize=False,
                duration=[args.frame_ms] * (len(gif) - 1) + [3000])
    print(f"{args.out}: {len(gif)} frames, score {g.score}, survived {g.alive}, "
          f"{Path(args.out).stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
