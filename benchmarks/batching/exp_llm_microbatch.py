import argparse
import logging
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

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


class FakeLLMNoBatch:
    """Simple per-request LLM simulator returning valid extraction JSON."""

    def __init__(self, base_ms: float, per_item_ms: float):
        self.base_ms = base_ms
        self.per_item_ms = per_item_ms

    def generate_response(self, messages, response_format=None):
        time.sleep((self.base_ms + self.per_item_ms) / 1000.0)
        return '{"memory": []}'


@dataclass
class BatchReq:
    done: threading.Event
    result: Optional[str] = None
    error: Optional[Exception] = None


class FakeLLMBatched:
    """Micro-batching LLM simulator used only in benchmark experiments."""

    def __init__(self, base_ms: float, per_item_ms: float, batch_window_ms: float, max_batch_size: int):
        self.base_ms = base_ms
        self.per_item_ms = per_item_ms
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size

        self.q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

        self.batch_sizes: List[int] = []
        self.lock = threading.Lock()
        self.t.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                first = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [first]
            deadline = time.perf_counter() + (self.batch_window_ms / 1000.0)
            while len(batch) < self.max_batch_size:
                remain = deadline - time.perf_counter()
                if remain <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=remain))
                except queue.Empty:
                    break

            with self.lock:
                self.batch_sizes.append(len(batch))

            # One batched model execution.
            total_ms = self.base_ms + (self.per_item_ms * len(batch))
            time.sleep(total_ms / 1000.0)

            for req in batch:
                req.result = '{"memory": []}'
                req.done.set()
                self.q.task_done()

    def generate_response(self, messages, response_format=None):
        req = BatchReq(done=threading.Event())
        self.q.put(req)
        req.done.wait()
        if req.error:
            raise req.error
        return req.result if req.result is not None else '{"memory": []}'

    def stats(self) -> Dict:
        with self.lock:
            if not self.batch_sizes:
                return {"avg_batch_size": 0.0, "max_batch_size_seen": 0}
            return {
                "avg_batch_size": round(float(np.mean(self.batch_sizes)), 3),
                "max_batch_size_seen": int(np.max(self.batch_sizes)),
            }

    def shutdown(self):
        self.q.join()
        self.stop.set()
        self.t.join(timeout=1.0)


def run_level(
    concurrency: int,
    n_requests: int,
    batched: bool,
    base_ms: float,
    per_item_ms: float,
    batch_window_ms: float,
    max_batch_size: int,
) -> Dict:
    mem = build_memory()

    if batched:
        llm = FakeLLMBatched(
            base_ms=base_ms,
            per_item_ms=per_item_ms,
            batch_window_ms=batch_window_ms,
            max_batch_size=max_batch_size,
        )
        mode = "batched"
    else:
        llm = FakeLLMNoBatch(base_ms=base_ms, per_item_ms=per_item_ms)
        mode = "unbatched"

    mem.llm.generate_response = llm.generate_response

    latencies = []
    progress = ProgressPrinter(label=f"{mode}-c={concurrency}", total=n_requests)

    def do_one(i: int):
        s = time.perf_counter()
        try:
            mem.add(make_messages(i), user_id=f"llm_batch_user_{i % 5}")
            return (time.perf_counter() - s) * 1000.0
        except Exception as e:
            logging.error(f"{mode} req {i} failed: {e}")
            return -1.0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(do_one, i) for i in range(n_requests)]
        for fut in as_completed(futures):
            v = fut.result()
            if v >= 0:
                latencies.append(v)
            progress.tick()

    wall = time.perf_counter() - t0

    stats = {
        "mode": mode,
        "concurrency": concurrency,
        "n": len(latencies),
        "throughput": round(len(latencies) / wall, 3) if wall > 0 else 0,
        "p50_ms": round(float(np.percentile(latencies, 50)), 1) if latencies else 0,
        "p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
        "p99_ms": round(float(np.percentile(latencies, 99)), 1) if latencies else 0,
    }

    if batched:
        stats.update(llm.stats())
        llm.shutdown()
    else:
        stats.update({"avg_batch_size": 1.0, "max_batch_size_seen": 1})

    print(
        f"  {mode:9s} c={concurrency:<3d}"
        f"  tps={stats['throughput']:.2f}"
        f"  p95={stats['p95_ms']:.0f}ms"
        f"  avg_batch={stats['avg_batch_size']:.2f}"
    )
    return stats


def plot(results: List[Dict], out_plot: str):
    levels = sorted({r["concurrency"] for r in results})
    by_mode = {
        "unbatched": {r["concurrency"]: r for r in results if r["mode"] == "unbatched"},
        "batched": {r["concurrency"]: r for r in results if r["mode"] == "batched"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Batching Experiment: LLM Micro-batching (benchmark simulator)", fontsize=13, fontweight="bold")

    ax = axes[0]
    un_tps = [by_mode["unbatched"][c]["throughput"] for c in levels]
    ba_tps = [by_mode["batched"][c]["throughput"] for c in levels]
    ax.plot(levels, un_tps, "o-", color="#e15759", label="unbatched")
    ax.plot(levels, ba_tps, "s-", color="#4e79a7", label="batched")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Requests / second")
    ax.set_title("Throughput")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    un_p95 = [by_mode["unbatched"][c]["p95_ms"] for c in levels]
    ba_p95 = [by_mode["batched"][c]["p95_ms"] for c in levels]
    ax.plot(levels, un_p95, "o--", color="#e15759", label="unbatched p95")
    ax.plot(levels, ba_p95, "s--", color="#4e79a7", label="batched p95")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("p95 Latency")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2]
    avg_batch = [by_mode["batched"][c]["avg_batch_size"] for c in levels]
    bars = ax.bar([str(c) for c in levels], avg_batch, color="#59a14f", alpha=0.9)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Average batch size")
    ax.set_title("Observed Batch Size")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, avg_batch):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {out_plot}")


def main():
    parser = argparse.ArgumentParser(description="LLM micro-batching benchmark experiment")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--batch-window-ms", type=float, default=10.0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--base-ms", type=float, default=120.0,
                        help="Per-batch fixed latency component (simulated)")
    parser.add_argument("--per-item-ms", type=float, default=20.0,
                        help="Per-item incremental latency component (simulated)")
    parser.add_argument("--output", type=str, default="benchmarks/batching/graphs/exp_llm_microbatch_results.json")
    parser.add_argument("--plot", type=str, default="benchmarks/batching/graphs/exp_llm_microbatch.png")
    args = parser.parse_args()

    ensure_graphs_dir()

    print("=" * 72)
    print("Experiment: LLM micro-batching (benchmark-only simulator)")
    print(f"  requests={args.requests} concurrency={args.concurrency}")
    print(f"  batch_window_ms={args.batch_window_ms} max_batch_size={args.max_batch_size}")
    print(f"  base_ms={args.base_ms} per_item_ms={args.per_item_ms}")
    print("=" * 72)

    results: List[Dict] = []
    for c in sorted(set(args.concurrency)):
        results.append(
            run_level(
                concurrency=c,
                n_requests=args.requests,
                batched=False,
                base_ms=args.base_ms,
                per_item_ms=args.per_item_ms,
                batch_window_ms=args.batch_window_ms,
                max_batch_size=args.max_batch_size,
            )
        )
        results.append(
            run_level(
                concurrency=c,
                n_requests=args.requests,
                batched=True,
                base_ms=args.base_ms,
                per_item_ms=args.per_item_ms,
                batch_window_ms=args.batch_window_ms,
                max_batch_size=args.max_batch_size,
            )
        )

    dump_json(args.output, results)
    print(f"  Results saved -> {args.output}")

    plot(results, args.plot)


if __name__ == "__main__":
    main()
