"""
Experiment C: In-Flight RPC Queue Buildup Under Load
=====================================================
Hypothesis:
  As write concurrency increases, LLM and embedding calls pile up as
  independent in-flight requests — no batching or queuing is happening.

How it works:
  - Wraps LLM and embedder methods with atomic counters
  - A background thread samples (in_flight_llm, in_flight_embed) every 50ms
  - Runs a high-concurrency write burst and records the time-series
  - Repeats at several concurrency levels for comparison

Plots (saved to benchmarks/graphs/):
  1. In-flight LLM calls over time at each concurrency level
  2. In-flight embedding calls over time at each concurrency level
  3. Peak in-flight counts vs concurrency (bar chart)

Usage:
    cd /Users/saraagarwal/mem0
    python benchmarks/exp_queue.py
    python benchmarks/exp_queue.py --requests 20 --concurrency 1 4 16 32
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.common import build_seeded_memory as build_memory, make_messages

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)

GRAPHS_DIR = "benchmarks/graphs"
SAMPLE_INTERVAL = 0.05   # seconds between samples


# ─────────────────────────────────────────────────────────────────────────────
# Instrument a Memory instance with in-flight counters
# ─────────────────────────────────────────────────────────────────────────────
def instrument(mem):
    """
    Patches llm.generate_response and embedding_model.embed/embed_batch
    with atomic in-flight counters. Returns (llm_counter, embed_counter).
    Both are threading.local-friendly ints wrapped in a list for mutability.
    """
    llm_counter   = [0]
    embed_counter = [0]
    lock = threading.Lock()

    def tracked(method, counter):
        orig = method
        def wrapper(*args, **kwargs):
            with lock:
                counter[0] += 1
            try:
                return orig(*args, **kwargs)
            finally:
                with lock:
                    counter[0] -= 1
        return wrapper

    mem.llm.generate_response       = tracked(mem.llm.generate_response,       llm_counter)
    mem.embedding_model.embed       = tracked(mem.embedding_model.embed,        embed_counter)
    mem.embedding_model.embed_batch = tracked(mem.embedding_model.embed_batch,  embed_counter)

    return llm_counter, embed_counter


# ─────────────────────────────────────────────────────────────────────────────
# Background sampler
# ─────────────────────────────────────────────────────────────────────────────
class Sampler:
    def __init__(self, llm_counter, embed_counter):
        self.llm_counter   = llm_counter
        self.embed_counter = embed_counter
        self.samples: List[Dict] = []
        self._stop = threading.Event()
        self._t0 = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t0 = time.perf_counter()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            self.samples.append({
                "t":     round(time.perf_counter() - self._t0, 3),
                "llm":   self.llm_counter[0],
                "embed": self.embed_counter[0],
            })
            time.sleep(SAMPLE_INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
# Run one level
# ─────────────────────────────────────────────────────────────────────────────
def run_level(concurrency: int, n_requests: int) -> Dict:
    mem = build_memory()
    llm_c, embed_c = instrument(mem)
    sampler = Sampler(llm_c, embed_c)

    sampler.start()
    t_wall = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(mem.add, make_messages(i),
                               **{"user_id": f"bench_user_{i % 5}"})
                   for i in range(n_requests)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logging.error(e)

    wall_time = time.perf_counter() - t_wall
    sampler.stop()

    peak_llm   = max((s["llm"]   for s in sampler.samples), default=0)
    peak_embed = max((s["embed"] for s in sampler.samples), default=0)

    print(f"  c={concurrency:<4}  wall={wall_time:.1f}s"
          f"  peak_llm={peak_llm}  peak_embed={peak_embed}"
          f"  samples={len(sampler.samples)}")

    return {
        "concurrency": concurrency,
        "wall_time_s": round(wall_time, 2),
        "peak_llm":    peak_llm,
        "peak_embed":  peak_embed,
        "samples":     sampler.samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#b07aa1"]

def plot(all_results: List[Dict], output: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Mem0 — In-Flight LLM Calls Over Time\n"
        "C: no batching — LLM requests pile up as independent remote calls",
        fontsize=13, fontweight="bold",
    )

    for i, result in enumerate(all_results):
        ts  = [s["t"]   for s in result["samples"]]
        llm = [s["llm"] for s in result["samples"]]
        ax.plot(ts, llm, linewidth=1.8, color=COLORS[i % len(COLORS)],
                label=f"c={result['concurrency']}", alpha=0.85)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("In-flight LLM calls", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Exp C: in-flight RPC queue buildup")
    parser.add_argument("--requests",    type=int, default=20)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--output", type=str, default=f"{GRAPHS_DIR}/exp_queue_results.json")
    parser.add_argument("--plot",   type=str, default=f"{GRAPHS_DIR}/exp_queue.png")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment C: In-Flight RPC Queue Buildup")
    print(f"  Concurrency levels : {args.concurrency}")
    print(f"  Requests per level : {args.requests}")
    print("=" * 60)

    all_results: List[Dict] = []
    for c in sorted(set(args.concurrency)):
        result = run_level(concurrency=c, n_requests=args.requests)
        all_results.append(result)

    # save without full sample arrays (too large)
    summary = [{k: v for k, v in r.items() if k != "samples"} for r in all_results]
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {args.output}")

    plot(all_results, args.plot)


if __name__ == "__main__":
    main()
