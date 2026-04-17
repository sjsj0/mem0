"""
Experiment D: CPU & Memory Utilization vs Concurrency
======================================================
Hypothesis H3:
  Local resources remain underutilized even as latency rises sharply.
  The bottleneck is I/O wait (remote LLM/embed calls), not compute.

How it works:
  - Runs write benchmark at each concurrency level
  - A background thread samples CPU% and RSS memory every 500ms via psutil
  - Records avg/peak CPU and avg/peak memory per level

Plots (saved to benchmarks/graphs/):
  1. Throughput + CPU% vs concurrency (dual-axis line)
  2. p95 latency + memory usage vs concurrency (dual-axis line)
  3. CPU% and memory grouped bars across concurrency levels

Usage:
    cd /Users/saraagarwal/mem0
    python benchmarks/exp_resources.py
    python benchmarks/exp_resources.py --requests 20 --concurrency 1 2 4 8 16 32
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

try:
    import psutil
except ImportError:
    raise SystemExit("psutil required: pip install psutil")

from benchmarks.common import build_seeded_memory as build_memory, make_messages

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)

GRAPHS_DIR = "benchmarks/graphs"
SAMPLE_INTERVAL = 0.5   # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Background resource sampler
# ─────────────────────────────────────────────────────────────────────────────
class ResourceSampler:
    def __init__(self):
        self._proc = psutil.Process()
        self._proc.cpu_percent(interval=None)   # warm up
        self.samples: List[Dict] = []
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            cpu_sys  = psutil.cpu_percent(interval=None)        # system-wide CPU %
            cpu_proc = self._proc.cpu_percent(interval=None)    # this process CPU %
            mem_mb   = self._proc.memory_info().rss / 1024 / 1024
            self.samples.append({
                "t":        round(time.perf_counter(), 3),
                "cpu_sys":  cpu_sys,
                "cpu_proc": cpu_proc,
                "mem_mb":   round(mem_mb, 1),
            })
            time.sleep(SAMPLE_INTERVAL)

    def summary(self) -> Dict:
        if not self.samples:
            return {"avg_cpu_sys": 0, "peak_cpu_sys": 0, "avg_mem_mb": 0, "peak_mem_mb": 0}
        return {
            "avg_cpu_sys":  round(float(np.mean([s["cpu_sys"]  for s in self.samples])), 1),
            "peak_cpu_sys": round(float(np.max( [s["cpu_sys"]  for s in self.samples])), 1),
            "avg_mem_mb":   round(float(np.mean([s["mem_mb"]   for s in self.samples])), 1),
            "peak_mem_mb":  round(float(np.max( [s["mem_mb"]   for s in self.samples])), 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Run one concurrency level
# ─────────────────────────────────────────────────────────────────────────────
def run_level(concurrency: int, n_requests: int) -> Dict:
    mem = build_memory()
    latencies: List[float] = []
    lock = threading.Lock()
    sampler = ResourceSampler()

    sampler.start()
    t_wall = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_add_one, mem, i) for i in range(n_requests)]
        for fut in as_completed(futures):
            lat = fut.result()
            if lat >= 0:
                with lock:
                    latencies.append(lat)

    wall_time = time.perf_counter() - t_wall
    sampler.stop()

    n = len(latencies)
    res = sampler.summary()
    stats = {
        "concurrency":    concurrency,
        "n":              n,
        "writes_per_sec": round(n / wall_time, 3) if wall_time > 0 else 0,
        "p95_ms":         round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
        "p99_ms":         round(float(np.percentile(latencies, 99)), 1) if latencies else 0,
        **res,
    }
    print(
        f"  c={concurrency:<4}  {stats['writes_per_sec']:5.2f} w/s"
        f"  p95={stats['p95_ms']:6.0f}ms"
        f"  cpu_avg={stats['avg_cpu_sys']:5.1f}%"
        f"  cpu_peak={stats['peak_cpu_sys']:5.1f}%"
        f"  mem_avg={stats['avg_mem_mb']:6.0f}MB"
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
    levels      = [s["concurrency"]    for s in all_stats]
    throughput  = [s["writes_per_sec"] for s in all_stats]
    avg_cpu     = [s["avg_cpu_sys"]    for s in all_stats]
    peak_cpu    = [s["peak_cpu_sys"]   for s in all_stats]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()
    fig.suptitle(
        "Mem0 — Throughput vs CPU Utilization\n"
        "H3: CPU stays low while throughput plateaus — bottleneck is I/O wait, not compute",
        fontsize=13, fontweight="bold",
    )

    l1, = ax.plot(levels,  throughput, "o-",  color="#4e79a7", linewidth=2.5, markersize=8, label="Throughput (w/s)")
    l2, = ax2.plot(levels, avg_cpu,    "s--", color="#e15759", linewidth=2,   markersize=7, label="Avg CPU %")
    l3, = ax2.plot(levels, peak_cpu,   "^:",  color="#f28e2b", linewidth=1.5, markersize=6, label="Peak CPU %", alpha=0.7)

    ax.set_xscale("log", base=2)
    ax.set_xticks(levels)
    ax.set_xticklabels([str(c) for c in levels])
    ax.set_xlabel("Concurrent writers", fontsize=11)
    ax.set_ylabel("Writes / second",  color="#4e79a7", fontsize=11)
    ax2.set_ylabel("CPU utilization %", color="#e15759", fontsize=11)
    ax.tick_params(axis="y", labelcolor="#4e79a7")
    ax2.tick_params(axis="y", labelcolor="#e15759")
    ax.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]], fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Exp D: CPU/memory utilization vs concurrency")
    parser.add_argument("--requests",    type=int, default=20)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--output", type=str, default=f"{GRAPHS_DIR}/exp_resources_results.json")
    parser.add_argument("--plot",   type=str, default=f"{GRAPHS_DIR}/exp_resources.png")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment D: CPU & Memory Utilization vs Concurrency")
    print(f"  Concurrency levels : {args.concurrency}")
    print(f"  Requests per level : {args.requests}")
    print("=" * 60)

    all_stats: List[Dict] = []
    for c in sorted(set(args.concurrency)):
        stats = run_level(concurrency=c, n_requests=args.requests)
        all_stats.append(stats)

    with open(args.output, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Results saved → {args.output}")

    plot(all_stats, args.plot)


if __name__ == "__main__":
    main()
