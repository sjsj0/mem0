"""
Experiment B+G: Write Throughput vs Concurrent Writers
=======================================================
Hypothesis H2:
  Throughput plateaus early because each write fires several sequential
  remote RPCs. More writers just pile up — they don't reduce per-request work.

Measures per concurrency level:
  writes/sec, avg, p50, p95, p99 latency

Plots (saved to benchmarks/graphs/):
  1. Throughput (writes/s) vs concurrency  — with ideal-linear reference
  2. Tail latency (p95, p99) vs concurrency
  3. Full latency fan (avg / p50 / p95 / p99)

Usage:
    cd /Users/saraagarwal/mem0
    python benchmarks/exp_throughput.py
    python benchmarks/exp_throughput.py --requests 40 --concurrency 1 2 4 8 16 32 64
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

from benchmarks.common import build_memory, make_messages

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)

GRAPHS_DIR = "benchmarks/graphs"


def run_one(mem, idx: int) -> float:
    """Returns wall-clock latency in ms, or -1 on failure."""
    t0 = time.perf_counter()
    try:
        mem.add(make_messages(idx), user_id=f"bench_user_{idx % 5}")
        return (time.perf_counter() - t0) * 1000
    except Exception as e:
        logging.error(f"req {idx} failed: {e}")
        return -1.0


def run_level(concurrency: int, n_requests: int) -> Dict:
    mem = build_memory()
    latencies: List[float] = []
    lock = threading.Lock()

    t_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_one, mem, i) for i in range(n_requests)]
        for fut in as_completed(futures):
            lat = fut.result()
            if lat >= 0:
                with lock:
                    latencies.append(lat)
    wall_time = time.perf_counter() - t_wall

    n = len(latencies)
    stats = {
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
        f"  c={concurrency:<4}  {stats['writes_per_sec']:6.2f} w/s"
        f"  avg={stats['avg_ms']:6.0f}ms"
        f"  p50={stats['p50_ms']:6.0f}ms"
        f"  p95={stats['p95_ms']:6.0f}ms"
        f"  p99={stats['p99_ms']:6.0f}ms"
    )
    return stats


def plot(all_stats: List[Dict], output: str):
    levels     = [s["concurrency"]    for s in all_stats]
    throughput = [s["writes_per_sec"] for s in all_stats]
    avg_ms     = [s["avg_ms"]         for s in all_stats]
    p50_ms     = [s["p50_ms"]         for s in all_stats]
    p95_ms     = [s["p95_ms"]         for s in all_stats]
    p99_ms     = [s["p99_ms"]         for s in all_stats]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Mem0 — Write Throughput vs Concurrency\n"
        "H2: throughput plateaus early while tail latency blows up",
        fontsize=13, fontweight="bold",
    )

    # ── 1. Throughput ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(levels, throughput, "o-", color="#4e79a7", linewidth=2.5, markersize=8, zorder=3, label="observed")
    ax.fill_between(levels, throughput, alpha=0.1, color="#4e79a7")
    ideal = [throughput[0] * c for c in levels]
    ax.plot(levels, ideal, "--", color="#bbb", linewidth=1.5, label="ideal linear scaling")

    for i in range(1, len(throughput)):
        gain = (throughput[i] - throughput[i - 1]) / (throughput[i - 1] + 1e-9)
        if gain < 0.10:
            ax.axvline(levels[i], color="#e15759", linestyle=":", linewidth=1.5, alpha=0.7)
            ax.text(levels[i] * 1.05, max(throughput) * 0.08,
                    f"plateau\nc={levels[i]}", color="#e15759", fontsize=8)
            break

    ax.set_xscale("log", base=2)
    ax.set_xticks(levels)
    ax.set_xticklabels([str(c) for c in levels])
    ax.set_xlabel("Concurrent writers", fontsize=11)
    ax.set_ylabel("Writes / second", fontsize=11)
    ax.set_title("Throughput vs Concurrency", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── 2. Tail latency ───────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(levels, p95_ms, "s--", color="#e15759", linewidth=2.5, markersize=8, label="p95")
    ax.plot(levels, p99_ms, "^-",  color="#f28e2b", linewidth=2.5, markersize=8, label="p99")
    ax.fill_between(levels, p95_ms, p99_ms, alpha=0.12, color="#e15759")

    for mult, label in [(2, "2×"), (5, "5×")]:
        threshold = p95_ms[0] * mult
        if any(v >= threshold for v in p95_ms):
            ax.axhline(threshold, color="#ccc", linestyle=":", linewidth=1)
            ax.text(levels[-1], threshold * 1.02, f"  {label} baseline p95",
                    va="bottom", fontsize=8, color="#999")

    ax.set_xscale("log", base=2)
    ax.set_xticks(levels)
    ax.set_xticklabels([str(c) for c in levels])
    ax.set_xlabel("Concurrent writers", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Tail Latency vs Concurrency", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output}")


def print_summary(all_stats: List[Dict]):
    print("\n" + "=" * 68)
    print(f"  {'c':>5}  {'w/s':>8}  {'avg ms':>8}  {'p50 ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}")
    print("-" * 68)
    for s in all_stats:
        print(f"  {s['concurrency']:>5}  {s['writes_per_sec']:>8.2f}"
              f"  {s['avg_ms']:>8.0f}  {s['p50_ms']:>8.0f}"
              f"  {s['p95_ms']:>8.0f}  {s['p99_ms']:>8.0f}")
    print("=" * 68)

    tps = [s["writes_per_sec"] for s in all_stats]
    for i in range(1, len(tps)):
        gain = (tps[i] - tps[i - 1]) / (tps[i - 1] + 1e-9)
        if gain < 0.10:
            print(f"\n  Throughput saturates at c={all_stats[i]['concurrency']}"
                  f"  ({gain*100:.1f}% gain over c={all_stats[i-1]['concurrency']})")
            break

    if all_stats[0]["p99_ms"] > 0:
        blowup = all_stats[-1]["p99_ms"] / all_stats[0]["p99_ms"]
        print(f"  p99 blowup  c=1 → c={all_stats[-1]['concurrency']}: {blowup:.1f}×")


def main():
    parser = argparse.ArgumentParser(description="Exp B: throughput vs concurrency")
    parser.add_argument("--requests",    type=int, default=30)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--output", type=str, default=f"{GRAPHS_DIR}/exp_throughput_results.json")
    parser.add_argument("--plot",   type=str, default=f"{GRAPHS_DIR}/exp_throughput.png")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment B: Write Throughput vs Concurrency")
    print(f"  Concurrency levels : {args.concurrency}")
    print(f"  Requests per level : {args.requests}")
    print("=" * 60)

    all_stats: List[Dict] = []
    for c in sorted(set(args.concurrency)):
        stats = run_level(concurrency=c, n_requests=args.requests)
        all_stats.append(stats)

    with open(args.output, "w") as f:
        json.dump([{k: v for k, v in s.items() if k != "latencies_ms"} for s in all_stats], f, indent=2)
    print(f"\n  Results saved → {args.output}")

    print_summary(all_stats)
    plot(all_stats, args.plot)


if __name__ == "__main__":
    main()
