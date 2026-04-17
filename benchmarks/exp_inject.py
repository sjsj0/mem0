"""
Experiment F: Sensitivity to Remote API Latency
================================================
Hypothesis H4:
  Small increases in remote inference latency disproportionately hurt
  write latency and tail behavior — the architecture is network-bound.

How it works:
  - Injects artificial sleep into LLM and/or embedding calls
  - Measures single-write latency and throughput under concurrency
  - Injected delays: 0, 50, 100, 200, 500 ms

Plots (saved to benchmarks/graphs/):
  1. p95 latency vs injected delay  (linear relationship expected)
  2. Throughput vs injected delay
  3. Latency fan (avg/p50/p95/p99) vs injected delay

Usage:
    cd /Users/saraagarwal/mem0
    python benchmarks/exp_inject.py
    python benchmarks/exp_inject.py --requests 15 --concurrency 8 --delays 0 50 100 200 500
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.common import build_seeded_memory as build_memory, make_messages

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)

GRAPHS_DIR = "benchmarks/graphs"


# ─────────────────────────────────────────────────────────────────────────────
# Inject a fixed delay into LLM and embedding calls
# ─────────────────────────────────────────────────────────────────────────────
def inject_delay(mem, delay_ms: float):
    """Wraps LLM and embedder methods with an extra sleep of delay_ms."""
    delay_s = delay_ms / 1000.0

    def delayed(method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            time.sleep(delay_s)
            return method(*args, **kwargs)
        return wrapper

    mem.llm.generate_response       = delayed(mem.llm.generate_response)
    mem.embedding_model.embed       = delayed(mem.embedding_model.embed)
    mem.embedding_model.embed_batch = delayed(mem.embedding_model.embed_batch)


# ─────────────────────────────────────────────────────────────────────────────
# Run one (delay, concurrency) combination
# ─────────────────────────────────────────────────────────────────────────────
def run_scenario(delay_ms: float, concurrency: int, n_requests: int) -> Dict:
    mem = build_memory()
    inject_delay(mem, delay_ms)

    latencies: List[float] = []
    lock = threading.Lock()

    t_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_add_one, mem, i) for i in range(n_requests)]
        for fut in as_completed(futures):
            lat = fut.result()
            if lat >= 0:
                with lock:
                    latencies.append(lat)
    wall_time = time.perf_counter() - t_wall

    n = len(latencies)
    stats = {
        "delay_ms":       delay_ms,
        "concurrency":    concurrency,
        "n":              n,
        "writes_per_sec": round(n / wall_time, 3) if wall_time > 0 else 0,
        "avg_ms":         round(float(np.mean(latencies)), 1)           if latencies else 0,
        "p50_ms":         round(float(np.percentile(latencies, 50)), 1) if latencies else 0,
        "p95_ms":         round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
        "p99_ms":         round(float(np.percentile(latencies, 99)), 1) if latencies else 0,
        "latencies_ms":   latencies,
    }
    print(
        f"  delay={delay_ms:>5.0f}ms  c={concurrency:<4}"
        f"  {stats['writes_per_sec']:5.2f} w/s"
        f"  p95={stats['p95_ms']:6.0f}ms"
        f"  p99={stats['p99_ms']:6.0f}ms"
    )
    return stats


def _add_one(mem, idx: int) -> float:
    t0 = time.perf_counter()
    try:
        mem.add(make_messages(idx), user_id=f"bench_user_{idx % 5}")
        return (time.perf_counter() - t0) * 1000
    except Exception as e:
        logging.error(f"req {idx} failed: {e}")
        return -1.0


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot(all_stats: List[Dict], output: str):
    delays     = [s["delay_ms"]       for s in all_stats]
    p95_ms     = [s["p95_ms"]         for s in all_stats]
    p99_ms     = [s["p99_ms"]         for s in all_stats]
    concurrency = all_stats[0]["concurrency"] if all_stats else "?"

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Mem0 — Tail Latency vs Injected Remote Delay  (c={concurrency})\n"
        "H4: small increases in remote latency disproportionately hurt write p95/p99",
        fontsize=13, fontweight="bold",
    )

    ax.plot(delays, p95_ms, "s-",  color="#e15759", linewidth=2.5, markersize=9, label="p95")
    ax.plot(delays, p99_ms, "^--", color="#f28e2b", linewidth=2.5, markersize=9, label="p99")

    if len(delays) > 1:
        coeffs = np.polyfit(delays, p95_ms, 1)
        x_fit  = np.linspace(min(delays), max(delays), 100)
        y_fit  = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, ":", color="#e15759", linewidth=1.2, alpha=0.6,
                label=f"p95 trend  slope={coeffs[0]:.1f}×")

    ax.set_xlabel("Injected delay per remote call (ms)", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Exp F: sensitivity to injected remote latency")
    parser.add_argument("--requests",    type=int, default=15)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--delays", type=float, nargs="+", default=[0, 50, 100, 200, 500],
                        help="Extra ms injected per LLM/embed call (default: 0 50 100 200 500)")
    parser.add_argument("--output", type=str, default=f"{GRAPHS_DIR}/exp_inject_results.json")
    parser.add_argument("--plot",   type=str, default=f"{GRAPHS_DIR}/exp_inject.png")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment F: Sensitivity to Injected Remote Latency")
    print(f"  Delays (ms)        : {args.delays}")
    print(f"  Concurrency        : {args.concurrency}")
    print(f"  Requests per delay : {args.requests}")
    print("=" * 60)

    all_stats: List[Dict] = []
    for delay in args.delays:
        stats = run_scenario(delay_ms=delay, concurrency=args.concurrency, n_requests=args.requests)
        all_stats.append(stats)

    with open(args.output, "w") as f:
        json.dump([{k: v for k, v in s.items() if k != "latencies_ms"} for s in all_stats], f, indent=2)
    print(f"\n  Results saved → {args.output}")

    plot(all_stats, args.plot)


if __name__ == "__main__":
    main()
