"""
Experiment B: Memory Patterns During Execution
===============================================
Hypothesis:
  Memory usage spikes during LLM calls and vector operations.
  We track RSS (resident set size) and heap allocation over time.

How it works:
  - Background thread samples RSS and heap size every 100ms
  - Runs N concurrent requests with ThreadPoolExecutor
  - Records peak memory, average memory, and GC frequency
  - Time-series plot shows memory trajectory during the benchmark

Plots (saved to benchmarks/detailed-benchmarks/graphs/):
  1. Memory (RSS) over time during concurrent requests
  2. Memory usage distribution (histogram)
  3. Peak memory vs concurrency level

Usage:
    cd d:\\Masters\\Spring 26\\Storage Systems\\Project\\mem0
    python benchmarks/detailed-benchmarks/exp_memory_patterns.py
    python benchmarks/detailed-benchmarks/exp_memory_patterns.py --concurrency 1 4 8 --requests 20
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

try:
    import psutil
except ImportError:
    raise SystemExit("psutil required: pip install psutil")

from common import ProgressPrinter, build_memory, make_messages

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)

GRAPHS_DIR = "benchmarks/detailed-benchmarks/graphs"
SAMPLE_INTERVAL = 0.1  # seconds


class MemorySampler:
    """Samples RSS memory usage over time."""
    def __init__(self):
        self._proc = psutil.Process()
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
            mem_mb = self._proc.memory_info().rss / 1024 / 1024
            self.samples.append({
                "t": round(time.perf_counter() - self._t0, 2),
                "mem_mb": round(mem_mb, 1),
            })
            time.sleep(SAMPLE_INTERVAL)

    def summary(self) -> Dict:
        if not self.samples:
            return {"avg_mem_mb": 0, "peak_mem_mb": 0, "samples_count": 0}
        mems = [s["mem_mb"] for s in self.samples]
        return {
            "avg_mem_mb": round(np.mean(mems), 1),
            "peak_mem_mb": round(np.max(mems), 1),
            "min_mem_mb": round(np.min(mems), 1),
            "samples_count": len(self.samples),
        }


def run_level(concurrency: int, n_requests: int) -> Dict:
    """Runs Memory.add() at a given concurrency and samples memory."""
    mem = build_memory()
    # Avoid a thread-race in lazy entity store initialization with local Qdrant.
    _ = mem.entity_store
    sampler = MemorySampler()

    sampler.start()
    t_wall = time.perf_counter()
    progress = ProgressPrinter(label=f"c={concurrency}", total=n_requests)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(lambda i=i: mem.add(make_messages(i), user_id=f"mem_user_{i % 5}"))
                   for i in range(n_requests)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logging.error(e)
            finally:
                progress.tick()

    wall_time = time.perf_counter() - t_wall
    sampler.stop()

    summary = sampler.summary()
    result = {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "wall_time_s": round(wall_time, 2),
        **summary,
        "samples": sampler.samples,
    }
    print(f"  c={concurrency:<2}  wall={wall_time:.1f}s  "
          f"peak={summary['peak_mem_mb']:.0f}MB  avg={summary['avg_mem_mb']:.0f}MB")
    return result


def plot_memory_patterns(all_data: List[Dict], output: str):
    """Plots memory usage over time for each concurrency level."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "Mem0 — Memory Usage Patterns During Concurrent Requests\n"
        "Background: RSS sampling every 100ms during Memory.add() calls",
        fontsize=13, fontweight="bold",
    )

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    for i, data in enumerate(all_data):
        samples = data["samples"]
        ts = [s["t"] for s in samples]
        mems = [s["mem_mb"] for s in samples]
        ax.plot(ts, mems, linewidth=2, marker="o", markersize=3, alpha=0.75,
                color=colors[i % len(colors)], label=f"c={data['concurrency']}")

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Memory (MB)", fontsize=11)
    ax.set_title("Memory Usage Over Time", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output}")


def main():
    parser = argparse.ArgumentParser(description="Exp B: Memory Patterns During Execution")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--output", type=str, default=f"{GRAPHS_DIR}/exp_memory_patterns_results.json")
    parser.add_argument("--plot", type=str, default=f"{GRAPHS_DIR}/exp_memory_patterns.png")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment B: Memory Patterns During Execution")
    print(f"  Concurrency levels: {args.concurrency}")
    print(f"  Requests per level: {args.requests}")
    print("=" * 60)

    all_data = []
    for c in sorted(set(args.concurrency)):
        data = run_level(concurrency=c, n_requests=args.requests)
        all_data.append(data)

    # Save results (excluding full samples to keep JSON manageable)
    summary_data = [
        {k: v for k, v in d.items() if k != "samples"}
        for d in all_data
    ]
    with open(args.output, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nResults saved → {args.output}")

    plot_memory_patterns(all_data, args.plot)

    print(f"\n{'='*60}")
    for data in all_data:
        print(f"  c={data['concurrency']}  peak={data['peak_mem_mb']:.0f}MB  avg={data['avg_mem_mb']:.0f}MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
