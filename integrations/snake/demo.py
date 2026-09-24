#!/usr/bin/env python
"""Live Snake demo: the model plays game after game and every move streams to a web page.

    PYTHONPATH=. python -m integrations.snake.demo    # then open http://localhost:8765

Each move is the model's top choice over the board text, with the anti-trap shield of integrations.snake.play (a move
that would cut the head off from its tail gives way to the model's next choice). Games run on the laya-mlx board
(24x16, initial length 6, 600 moves) from seed 101 up; the page is demo.html, fed by server-sent events.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx

from integrations.snake.game import DIRS, SnakeGame
from integrations.snake.play import choose, probs

PAGE = Path(__file__).with_name("demo.html")


class Feed:
    """The latest frame; every published move wakes the open streams."""

    def __init__(self):
        self.cond, self.frame, self.n = threading.Condition(), None, 0

    def publish(self, frame: dict):
        with self.cond:
            self.frame, self.n = frame, self.n + 1
            self.cond.notify_all()

    def wait(self, seen: int) -> tuple[int, dict | None]:
        with self.cond:
            self.cond.wait_for(lambda: self.n != seen, timeout=15)
            return self.n, self.frame


def model_dir(model: str) -> str:
    """A local folder with the weights as is; a Hugging Face id (tdelab/tde-general-v0.2) is fetched into the cache."""
    if (Path(model) / "model.safetensors").exists():
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model, allow_patterns=["model.safetensors", "config.json", "tokenizer/*"])


def play_forever(args, feed: Feed):
    from transformers import AutoTokenizer

    from integrations.snake.encode import FastEncoder
    from tde.mlx.model import load_joint
    from tde.model.encoding import DecisionTokenizer
    mx.set_cache_limit(1 << 30)
    path = model_dir(args.model)
    enc = FastEncoder(DecisionTokenizer(AutoTokenizer.from_pretrained(f"{path}/tokenizer"), max_state_tokens=480,
                                        marker="mask", max_total=1024))
    model = load_joint(f"{path}/model.safetensors")
    w, h, length = map(int, args.board.split("x"))
    name = Path(args.model).name
    latency, results, seed, n_game = deque(maxlen=300), [], args.seed, 0
    while True:
        g, vetoes, n_game = SnakeGame(w, h, length, seed), 0, n_game + 1
        while g.alive and g.food is not None and g.steps < args.moves:
            t0 = time.perf_counter()
            p = probs(model, enc, [g])[0]
            latency.append(1000 * (time.perf_counter() - t0))
            move, veto = choose(p, g, "trap")
            vetoes += veto
            g.step(move)
            done = not (g.alive and g.food is not None and g.steps < args.moves)
            if done:
                results.append({"seed": seed, "score": g.score, "survived": g.alive})
            feed.publish({"model": name, "w": w, "h": h, "body": list(g.body), "food": g.food, "alive": g.alive,
                          "done": done, "score": g.score, "step": g.steps, "moves": args.moves, "game": n_game,
                          "seed": seed, "dirs": DIRS, "p": [round(float(x), 4) for x in p], "move": move, "veto": veto,
                          "vetoes": vetoes, "ms": round(latency[-1], 1), "p50": round(statistics.median(latency), 1),
                          "results": results[-10:]})
            if args.speed:
                time.sleep(max(0.0, 1.0 / args.speed - (time.perf_counter() - t0)))
        time.sleep(args.pause)
        seed += 1


class Handler(BaseHTTPRequestHandler):
    feed: Feed

    def do_GET(self):
        if self.path == "/":
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seen = -1
            try:
                while True:
                    seen, frame = self.feed.wait(seen)
                    self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode() if frame else b": waiting\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_error(404)

    def log_message(self, *_):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tdelab/tde-general-v0.2", help="Hugging Face id or a local release folder")
    ap.add_argument("--board", default="24x16x6", help="WxHxInitialLength")
    ap.add_argument("--moves", type=int, default=600, help="moves per game")
    ap.add_argument("--speed", type=float, default=20, help="moves per second shown; 0 = as fast as the model runs")
    ap.add_argument("--pause", type=float, default=3, help="seconds between games")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    Handler.feed = Feed()
    threading.Thread(target=play_forever, args=(args, Handler.feed), daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"Snake demo on http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
