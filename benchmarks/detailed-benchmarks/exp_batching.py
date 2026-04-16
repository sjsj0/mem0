"""
Experiment C: Batching Impact (Single vs Batch Requests)
==========================================================
Hypothesis:
  If batching were implemented, throughput would improve significantly.
  We measure the gap by comparing:
  - Individual requests (one at a time)
  - Pseudo-batched requests (fire multiple concurrently with ThreadPoolExecutor)

How it works:
  - Runs N requests serially (concurrency=1)
  - Runs N requests with high concurrency (concurrency=MIN(N, 16))
  - Computes throughput and average latency for both
  - Gap shows potential batching wins

Plots (saved to benchmarks/detailed-benchmarks/graphs/):
  1. Throughput: individual vs batched
  2. Latency: p50/p95/p99 comparison
  3. Efficiency ratio (batch throughput / individual throughput)

Usage:
    cd d:\\Masters\\Spring 26\\Storage Systems\\Project\\mem0
    python benchmarks/detailed-benchmarks/exp_batching.py
    python benchmarks/detailed-benchmarks/exp_batching.py --requests 30
"""

import os
import sys
import logging
import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ProgressPrinter, build_memory, make_messages

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)

GRAPHS_DIR = "benchmarks/detailed-benchmarks/graphs"


def run_serial(n_requests: int) -> Dict:
    """Runs requests one at a time (no concurrency)."""
    mem = build_memory()
    latencies = []

    print(f"\n  Running {n_requests} requests serially...")
    t_wall = time.perf_counter()
    progress = ProgressPrinter(label="serial", total=n_requests)

    for i in range(n_requests):
        t0 = time.perf_counter()
        try:
            mem.add(make_messages(i), user_id=f"batch_user_serial")
        except Exception as e:
            logging.error(f"req {i} failed: {e}")
        else:
            latencies.append((time.perf_counter() - t0) * 1000)
        finally:
            progress.tick()

    wall_time = time.perf_counter() - t_wall

    return {
        "mode": "serial",
        "n_requests": len(latencies),
        "wall_time_s": round(wall_time, 2),
        "throughput": round(len(latencies) / wall_time, 3) if wall_time > 0 else 0,
        "avg_ms": round(np.mean(latencies), 1) if latencies else 0,
        "p50_ms": round(float(np.percentile(latencies, 50)), 1) if latencies else 0,
        "p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
        "p99_ms": round(float(np.percentile(latencies, 99)), 1) if latencies else 0,
        "latencies": latencies,
    }


def run_concurrent(n_requests: int, concurrency: int = None) -> Dict:
    """Runs requests concurrently (pseudo-batching)."""
    if concurrency is None:
        concurrency = min(n_requests, 16)

    mem = build_memory()
    # Avoid a thread-race in lazy entity store initialization with local Qdrant.
    _ = mem.entity_store
    latencies = []
    lock = threading.Lock()

    print(f"\n  Running {n_requests} requests with concurrency={concurrency}...")
    t_wall = time.perf_counter()
    progress = ProgressPrinter(label="concurrent", total=n_requests)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(lambda i=i: _add_one(mem, i, latencies, lock))
                   for i in range(n_requests)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logging.error(e)
            finally:
                progress.tick()

    wall_time = time.perf_counter() - t_wall

    return {
        "mode": "concurrent",
        "concurrency": concurrency,
        "n_requests": len(latencies),
        "wall_time_s": round(wall_time, 2),
        "throughput": round(len(latencies) / wall_time, 3) if wall_time > 0 else 0,
        "avg_ms": round(np.mean(latencies), 1) if latencies else 0,
        "p50_ms": round(float(np.percentile(latencies, 50)), 1) if latencies else 0,
        "p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
        "p99_ms": round(float(np.percentile(latencies, 99)), 1) if latencies else 0,
        "latencies": latencies,
    }


def _add_one(mem, idx: int, latencies: List, lock: threading.Lock):
    t0 = time.perf_counter()
    mem.add(make_messages(idx), user_id=f"batch_user_concurrent")
    lat = (time.perf_counter() - t0) * 1000
    with lock:
        latencies.append(lat)


def plot_batching(serial: Dict, concurrent: Dict, output: str):
    """Plots serial vs concurrent performance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Mem0 — Batching Impact (Serial vs Concurrent)\n"
        "Gap shows potential speedup from request batching",
        fontsize=13, fontweight="bold",
    )

    # ── 1. Throughput comparison ───────────────────────────────────────────
    ax = axes[0]
    modes = ["Serial", "Concurrent"]
    throughputs = [serial["throughput"], concurrent["throughput"]]
    colors = ["#e15759", "#4e79a7"]
    bars = ax.bar(modes, throughputs, color=colors, alpha=0.8, width=0.5)
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f}\nreq/s", ha="center", va="bottom", fontsize=11, fontweight="bold")
    gain = ((concurrent["throughput"] - serial["throughput"]) / serial["throughput"] * 100) if serial["throughput"] else 0
    ax.text(0.5, max(throughputs) * 0.5, f"Batching gain:\n{gain:+.1f}%",
            ha="center", fontsize=12, fontweight="bold", 
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.set_ylabel("Throughput (req/s)", fontsize=11)
    ax.set_title("Throughput Improvement", fontsize=12)
    ax.set_ylim(0, max(throughputs) * 1.3)
    ax.grid(axis="y", alpha=0.3)

    # ── 2. Latency comparison ──────────────────────────────────────────────
    ax = axes[1]
    x = np.arange(3)
    width = 0.35
    metrics = ["p50", "p95", "p99"]
    serial_vals = [serial["p50_ms"], serial["p95_ms"], serial["p99_ms"]]
    concurrent_vals = [concurrent["p50_ms"], concurrent["p95_ms"], concurrent["p99_ms"]]

    ax.bar(x - width/2, serial_vals, width, label="Serial", color="#e15759", alpha=0.8)
    ax.bar(x + width/2, concurrent_vals, width, label="Concurrent", color="#4e79a7", alpha=0.8)

    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Latency Percentiles", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output}")


def main():
    parser = argparse.ArgumentParser(description="Exp C: Batching Impact")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--output", type=str, default=f"{GRAPHS_DIR}/exp_batching_results.json")
    parser.add_argument("--plot", type=str, default=f"{GRAPHS_DIR}/exp_batching.png")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment C: Batching Impact (Serial vs Concurrent)")
    print(f"  Total requests: {args.requests}")
    print("=" * 60)

    serial = run_serial(n_requests=args.requests)
    concurrent = run_concurrent(n_requests=args.requests)

    data = {
        "serial": {k: v for k, v in serial.items() if k != "latencies"},
        "concurrent": {k: v for k, v in concurrent.items() if k != "latencies"},
    }

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved → {args.output}")

    plot_batching(serial, concurrent, args.plot)

    print(f"\n{'='*60}")
    print(f"  Serial:     {serial['throughput']:.2f} req/s  p95={serial['p95_ms']:.0f}ms  avg={serial['avg_ms']:.0f}ms")
    print(f"  Concurrent: {concurrent['throughput']:.2f} req/s  p95={concurrent['p95_ms']:.0f}ms  avg={concurrent['avg_ms']:.0f}ms")
    gain = ((concurrent["throughput"] - serial["throughput"]) / serial["throughput"] * 100) if serial["throughput"] else 0
    print(f"  Batching gain: {gain:+.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
