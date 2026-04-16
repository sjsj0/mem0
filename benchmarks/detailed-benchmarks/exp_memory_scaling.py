"""
Experiment D: Memory Scaling (Performance vs Pre-Existing Memory Size)
======================================================================
Hypothesis:
  As pre-existing memories grow, retrieval latency increases (more vectors to search).
  Insert latency increases (larger index to update).
  This experiment measures the scaling behavior.

How it works:
  - Pre-populate memory with N memories (warm-up: 100, 500, 1000, 5000)
  - For each level, run M new add() calls and measure latency
  - Plot: latency vs pre-existing memory size
  - Shows if there's a linear, logarithmic, or worse scaling

Plots (saved to benchmarks/detailed-benchmarks/graphs/):
  1. Avg latency vs pre-existing memory size
  2. p95/p99 latency vs pre-existing memory size
  3. Throughput degradation as memory grows

Usage:
    cd d:\\Masters\\Spring 26\\Storage Systems\\Project\\mem0
    python benchmarks/detailed-benchmarks/exp_memory_scaling.py
    python benchmarks/detailed-benchmarks/exp_memory_scaling.py --memory-sizes 100 500 1000
"""

import os
import sys
import logging
import argparse
import json
import time
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


def run_level(pre_existing_count: int, new_requests: int) -> Dict:
    """
    Pre-populates memory with pre_existing_count memories,
    then runs new_requests new Memory.add() calls and measures latency.
    """
    mem = build_memory()
    latencies = []

    # Warm-up: pre-populate with memories
    print(f"\n  Pre-populating with {pre_existing_count} memories...")
    warmup_progress = ProgressPrinter(label="warmup", total=pre_existing_count)
    for i in range(pre_existing_count):
        try:
            mem.add(make_messages(i), user_id="scaling_warmup")
        except Exception as e:
            logging.debug(f"warmup {i} failed: {e}")
        finally:
            warmup_progress.tick()

    # Measure: run new requests with this memory size
    print(f"  Running {new_requests} new requests...")
    measure_progress = ProgressPrinter(label="measure", total=new_requests)

    t_wall = time.perf_counter()
    for i in range(new_requests):
        t0 = time.perf_counter()
        try:
            mem.add(make_messages(pre_existing_count + i), user_id="scaling_measurement")
        except Exception as e:
            logging.error(f"req {i} failed: {e}")
        else:
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
        finally:
            measure_progress.tick()

    wall_time = time.perf_counter() - t_wall

    result = {
        "pre_existing_count": pre_existing_count,
        "new_requests": len(latencies),
        "wall_time_s": round(wall_time, 2),
        "throughput": round(len(latencies) / wall_time, 3) if wall_time > 0 else 0,
        "avg_ms": round(np.mean(latencies), 1) if latencies else 0,
        "p50_ms": round(float(np.percentile(latencies, 50)), 1) if latencies else 0,
        "p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
        "p99_ms": round(float(np.percentile(latencies, 99)), 1) if latencies else 0,
        "min_ms": round(min(latencies), 1) if latencies else 0,
        "max_ms": round(max(latencies), 1) if latencies else 0,
    }
    print(f"    Pre-existing={pre_existing_count:>5}  avg={result['avg_ms']:>6.0f}ms  "
          f"p95={result['p95_ms']:>6.0f}ms  tps={result['throughput']:.2f}")
    return result


def plot_scaling(all_data: List[Dict], output: str):
    """Plots latency and throughput degradation as memory size grows."""
    sizes = [d["pre_existing_count"] for d in all_data]
    avg_ms = [d["avg_ms"] for d in all_data]
    p95_ms = [d["p95_ms"] for d in all_data]
    p99_ms = [d["p99_ms"] for d in all_data]
    throughput = [d["throughput"] for d in all_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Mem0 — Memory Scaling (Latency vs Pre-Existing Memory Size)\n"
        "Shows whether vector search degrades linearly or logarithmically",
        fontsize=13, fontweight="bold",
    )

    # ── 1. Latency scaling ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(sizes, avg_ms, "o-", color="#4e79a7", linewidth=2.5, markersize=8, label="Avg latency")
    ax.plot(sizes, p95_ms, "s--", color="#e15759", linewidth=2, markersize=7, label="p95 latency")
    ax.plot(sizes, p99_ms, "^:", color="#f28e2b", linewidth=2, markersize=7, label="p99 latency")

    ax.set_xlabel("Pre-Existing Memory Count", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Latency Scaling", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # ── 2. Throughput degradation ──────────────────────────────────────────
    ax = axes[1]
    baseline = throughput[0] if throughput else 1
    degradation = [(1 - t / baseline) * 100 for t in throughput]
    ax.plot(sizes, degradation, "d-", color="#59a14f", linewidth=2.5, markersize=8)
    ax.fill_between(sizes, 0, degradation, alpha=0.2, color="#59a14f")

    ax.set_xlabel("Pre-Existing Memory Count", fontsize=11)
    ax.set_ylabel("Throughput Degradation (%)", fontsize=11)
    ax.set_title("Throughput Loss as Memory Grows", fontsize=12)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output}")


def main():
    parser = argparse.ArgumentParser(description="Exp D: Memory Scaling")
    parser.add_argument("--memory-sizes", type=int, nargs="+", default=[100, 500, 1000, 2000],
                        help="Pre-existing memory sizes to test")
    parser.add_argument("--requests-per-level", type=int, default=10,
                        help="New requests to measure at each level")
    parser.add_argument("--output", type=str, default=f"{GRAPHS_DIR}/exp_memory_scaling_results.json")
    parser.add_argument("--plot", type=str, default=f"{GRAPHS_DIR}/exp_memory_scaling.png")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment D: Memory Scaling (Performance vs Pre-Existing Size)")
    print(f"  Memory sizes to test: {args.memory_sizes}")
    print(f"  New requests per level: {args.requests_per_level}")
    print("=" * 60)

    all_data = []
    for size in sorted(args.memory_sizes):
        data = run_level(pre_existing_count=size, new_requests=args.requests_per_level)
        all_data.append(data)

    with open(args.output, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nResults saved → {args.output}")

    plot_scaling(all_data, args.plot)

    print(f"\n{'='*60}")
    print(f"  {'Pre-Existing':<15}  {'Avg (ms)':<12}  {'p95 (ms)':<12}  {'Throughput':<12}")
    print(f"{'='*60}")
    for data in all_data:
        print(f"  {data['pre_existing_count']:<15}  {data['avg_ms']:<12.0f}  "
              f"{data['p95_ms']:<12.0f}  {data['throughput']:<12.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
