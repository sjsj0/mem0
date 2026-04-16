import argparse
import logging
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from benchmarks.batching.common import (
    ProgressPrinter,
    build_memory,
    dump_json,
    ensure_graphs_dir,
    make_messages,
)

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)


@dataclass
class Job:
    idx: int
    submitted_at: float
    done_event: threading.Event
    ack_ms: float = 0.0
    e2e_ms: float = 0.0


class AsyncAckProcessor:
    def __init__(self, workers: int):
        self.mem = build_memory()
        self.workers = max(1, workers)
        self.q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.threads: List[threading.Thread] = []

    def start(self):
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, name=f"ack-worker-{i}", daemon=True)
            t.start()
            self.threads.append(t)

    def _worker(self):
        while not self.stop.is_set():
            try:
                job = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.mem.add(make_messages(job.idx), user_id=f"ack_user_{job.idx % 5}")
            except Exception as e:
                logging.error(f"background req {job.idx} failed: {e}")
            finally:
                job.e2e_ms = (time.perf_counter() - job.submitted_at) * 1000.0
                job.done_event.set()
                self.q.task_done()

    def submit(self, idx: int) -> Job:
        t0 = time.perf_counter()
        job = Job(idx=idx, submitted_at=t0, done_event=threading.Event())
        self.q.put(job)
        job.ack_ms = (time.perf_counter() - t0) * 1000.0
        return job

    def shutdown(self):
        self.q.join()
        self.stop.set()
        for t in self.threads:
            t.join(timeout=1.0)


def run_sync(n_requests: int, concurrency: int) -> Dict:
    mem = build_memory()

    latencies = []
    progress = ProgressPrinter(label="sync", total=n_requests)
    t0 = time.perf_counter()

    def do_one(i: int):
        s = time.perf_counter()
        try:
            mem.add(make_messages(i), user_id=f"sync_user_{i % 5}")
            return (time.perf_counter() - s) * 1000.0
        except Exception as e:
            logging.error(f"sync req {i} failed: {e}")
            return -1.0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(do_one, i) for i in range(n_requests)]
        for fut in as_completed(futures):
            v = fut.result()
            if v >= 0:
                latencies.append(v)
            progress.tick()

    wall = time.perf_counter() - t0
    return {
        "mode": "sync",
        "n": len(latencies),
        "wall_s": round(wall, 3),
        "throughput": round(len(latencies) / wall, 3) if wall > 0 else 0,
        "p50_ms": round(float(np.percentile(latencies, 50)), 1) if latencies else 0,
        "p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
        "p99_ms": round(float(np.percentile(latencies, 99)), 1) if latencies else 0,
        "latencies_ms": latencies,
    }


def run_async_ack(n_requests: int, ingress_concurrency: int, workers: int) -> Dict:
    proc = AsyncAckProcessor(workers=workers)
    proc.start()

    jobs: List[Job] = []
    progress_submit = ProgressPrinter(label="enqueue", total=n_requests)

    t_submit0 = time.perf_counter()

    def submit_one(i: int):
        return proc.submit(i)

    with ThreadPoolExecutor(max_workers=ingress_concurrency) as pool:
        futures = [pool.submit(submit_one, i) for i in range(n_requests)]
        for fut in as_completed(futures):
            jobs.append(fut.result())
            progress_submit.tick()

    submit_wall = time.perf_counter() - t_submit0

    progress_done = ProgressPrinter(label="background", total=n_requests)
    for j in jobs:
        j.done_event.wait()
        progress_done.tick()

    end_wall = max((j.e2e_ms for j in jobs), default=0.0) / 1000.0
    proc.shutdown()

    ack = [j.ack_ms for j in jobs]
    e2e = [j.e2e_ms for j in jobs]

    return {
        "mode": "async_ack",
        "n": len(jobs),
        "ingress_wall_s": round(submit_wall, 3),
        "completion_wall_s": round(end_wall, 3),
        "ingress_throughput": round(len(jobs) / submit_wall, 3) if submit_wall > 0 else 0,
        "completion_throughput": round(len(jobs) / end_wall, 3) if end_wall > 0 else 0,
        "ack_p50_ms": round(float(np.percentile(ack, 50)), 3) if ack else 0,
        "ack_p95_ms": round(float(np.percentile(ack, 95)), 3) if ack else 0,
        "ack_p99_ms": round(float(np.percentile(ack, 99)), 3) if ack else 0,
        "e2e_p50_ms": round(float(np.percentile(e2e, 50)), 1) if e2e else 0,
        "e2e_p95_ms": round(float(np.percentile(e2e, 95)), 1) if e2e else 0,
        "e2e_p99_ms": round(float(np.percentile(e2e, 99)), 1) if e2e else 0,
        "ack_ms": ack,
        "e2e_ms": e2e,
    }


def plot(sync_stats: Dict, async_stats: Dict, out_plot: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Batching Experiment: ACK-first Queue vs Sync Writes", fontsize=13, fontweight="bold")

    labels = ["sync p95", "async ACK p95", "async e2e p95"]
    values = [
        sync_stats["p95_ms"],
        async_stats["ack_p95_ms"],
        async_stats["e2e_p95_ms"],
    ]
    colors = ["#e15759", "#59a14f", "#4e79a7"]

    ax = axes[0]
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Client ACK Latency vs End-to-End")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    t_labels = ["sync completion", "async ingress", "async completion"]
    t_values = [
        sync_stats["throughput"],
        async_stats["ingress_throughput"],
        async_stats["completion_throughput"],
    ]
    bars = ax.bar(t_labels, t_values, color=["#f28e2b", "#76b7b2", "#4e79a7"], alpha=0.9)
    ax.set_ylabel("Requests / second")
    ax.set_title("Throughput")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, t_values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {out_plot}")


def main():
    parser = argparse.ArgumentParser(description="ACK-first queue experiment")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--sync-concurrency", type=int, default=8)
    parser.add_argument("--ingress-concurrency", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="benchmarks/batching/graphs/exp_async_ack_queue_results.json")
    parser.add_argument("--plot", type=str, default="benchmarks/batching/graphs/exp_async_ack_queue.png")
    args = parser.parse_args()

    ensure_graphs_dir()

    print("=" * 64)
    print("Experiment: ACK-first queueing (benchmark-only)")
    print(f"  requests={args.requests} sync_concurrency={args.sync_concurrency}")
    print(f"  ingress_concurrency={args.ingress_concurrency} workers={args.workers}")
    print("=" * 64)

    sync_stats = run_sync(args.requests, args.sync_concurrency)
    async_stats = run_async_ack(args.requests, args.ingress_concurrency, args.workers)

    print("\nSummary")
    print(f"  sync p95={sync_stats['p95_ms']:.1f}ms")
    print(f"  async ACK p95={async_stats['ack_p95_ms']:.3f}ms")
    print(f"  async e2e p95={async_stats['e2e_p95_ms']:.1f}ms")

    data = {
        "config": {
            "requests": args.requests,
            "sync_concurrency": args.sync_concurrency,
            "ingress_concurrency": args.ingress_concurrency,
            "workers": args.workers,
        },
        "sync": {k: v for k, v in sync_stats.items() if k != "latencies_ms"},
        "async_ack": {k: v for k, v in async_stats.items() if k not in ("ack_ms", "e2e_ms")},
    }
    dump_json(args.output, data)
    print(f"  Results saved -> {args.output}")

    plot(sync_stats, async_stats, args.plot)


if __name__ == "__main__":
    main()
