"""
Updated Experiment: Throughput vs Concurrency — Batched vs Unbatched LLM
=========================================================================
Sweeps concurrency levels and compares writes/sec and p95 latency between
the standard pipeline (unbatched) and the LLMBatchBundler pipeline (batched).

Batching model (real Azure OpenAI calls):
  Unbatched: each write fires its own Azure LLM call independently.
  Batched:   LLMBatchBundler collects concurrent writes within the batch window,
             then calls generate_response sequentially for each item in the batch
             (Azure has no native vLLM-style batch endpoint).

Expected result:
  Unbatched throughput plateaus quickly (every write blocks on its own LLM call).
  Batched throughput keeps scaling because the batch call cost is shared.

Runs against the pre-seeded 10k-memory Qdrant collection for realistic
VS Search and embedding latencies.

Pre-requisite:
    python benchmarks/seed.py   (run once — ~30s for 50k vectors)

Usage:
    cd /Users/saraagarwal/mem0
    python updated_experiments/exp_throughput.py
    python updated_experiments/exp_throughput.py --requests 20 --concurrency 1 2 4 8 16 32
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.common import build_seeded_memory, make_messages

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)

GRAPHS_DIR = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Azure batch endpoint adapter
#
# Azure OpenAI has no vLLM-style batch endpoint, so we simulate one by calling
# generate_response sequentially for each item in the batch. The LLMBatchBundler
# still runs (collecting concurrent requests in the window), but each item makes
# a real Azure API call.
# ─────────────────────────────────────────────────────────────────────────────
def install_azure_batch_endpoint(mem):
    real_generate = mem.llm.generate_response

    def azure_batch_endpoint(messages_list, batch_url, **kwargs):  # noqa: ARG001
        return [real_generate(msgs, **kwargs) for msgs in messages_list]

    mem.llm.generate_batch_endpoint_response = azure_batch_endpoint


# ─────────────────────────────────────────────────────────────────────────────
# Memory factories
# ─────────────────────────────────────────────────────────────────────────────
def build_unbatched_mem():
    os.environ.pop("MEM0_BATCH_INFERENCE_URL", None)
    return build_seeded_memory()


def build_batched_mem(batch_size: int = 10, batch_timeout: float = 0.05):
    os.environ["MEM0_BATCH_INFERENCE_URL"] = "http://localhost:8000/v1/batch"
    os.environ["MEM0_BATCH_SIZE"]          = str(batch_size)
    os.environ["MEM0_BATCH_TIMEOUT"]       = str(batch_timeout)
    try:
        mem = build_seeded_memory()
    finally:
        os.environ.pop("MEM0_BATCH_INFERENCE_URL", None)
        os.environ.pop("MEM0_BATCH_SIZE", None)
        os.environ.pop("MEM0_BATCH_TIMEOUT", None)

    assert mem.batch_bundler is not None, "LLMBatchBundler not initialized"
    install_azure_batch_endpoint(mem)
    return mem


# ─────────────────────────────────────────────────────────────────────────────
# Run one concurrency level for one mode
# ─────────────────────────────────────────────────────────────────────────────
def run_level(
    mode: str,
    concurrency: int,
    n_requests: int,
    batch_size: int,
    batch_timeout: float,
) -> Dict:
    mem = build_batched_mem(batch_size, batch_timeout) if mode == "batched" else build_unbatched_mem()

    latencies: List[float] = []

    def do_one(idx: int) -> float:
        t0 = time.perf_counter()
        try:
            mem.add(make_messages(idx), user_id=f"bench_user_{idx % 5}")
            return (time.perf_counter() - t0) * 1000
        except Exception as e:
            logging.error(f"[{mode} c={concurrency}] req {idx} failed: {e}")
            return -1.0

    t_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(do_one, i) for i in range(n_requests)]
        for f in as_completed(futs):
            v = f.result()
            if v >= 0:
                latencies.append(v)
    wall = time.perf_counter() - t_wall

    result = {
        "mode":        mode,
        "concurrency": concurrency,
        "n":           len(latencies),
        "throughput":  round(len(latencies) / wall, 3) if wall > 0 else 0,
        "p50_ms":      round(float(np.percentile(latencies, 50)),  1) if latencies else 0,
        "p95_ms":      round(float(np.percentile(latencies, 95)),  1) if latencies else 0,
        "p99_ms":      round(float(np.percentile(latencies, 99)),  1) if latencies else 0,
        "wall_s":      round(wall, 2),
    }

    print(
        f"  {mode:9s}  c={concurrency:<3d}"
        f"  tps={result['throughput']:.2f}"
        f"  p50={result['p50_ms']:.0f}ms"
        f"  p95={result['p95_ms']:.0f}ms"
        f"  p99={result['p99_ms']:.0f}ms"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot(all_results: List[Dict], output: str):
    levels = sorted({r["concurrency"] for r in all_results})
    by_mode = {
        "unbatched": {r["concurrency"]: r for r in all_results if r["mode"] == "unbatched"},
        "batched":   {r["concurrency"]: r for r in all_results if r["mode"] == "batched"},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Mem0 — Throughput & Latency: LLMBatchBundler vs Unbatched\n"
        "Batching amortizes LLM HTTP overhead across concurrent writers",
        fontsize=13, fontweight="bold",
    )

    # --- Throughput ---
    un_tps = [by_mode["unbatched"][c]["throughput"] for c in levels]
    ba_tps = [by_mode["batched"][c]["throughput"]   for c in levels]
    ax1.plot(levels, un_tps, "o-", color="#e15759", linewidth=2, markersize=7, label="Unbatched")
    ax1.plot(levels, ba_tps, "s-", color="#4e79a7", linewidth=2, markersize=7, label="Batched (LLMBatchBundler)")
    ax1.set_xlabel("Concurrency (writers)", fontsize=11)
    ax1.set_ylabel("Writes / second", fontsize=11)
    ax1.set_title("Throughput", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(levels)

    for c, un, ba in zip(levels, un_tps, ba_tps):
        if un > 0:
            speedup = ba / un
            ax1.annotate(f"{speedup:.1f}×", xy=(c, ba), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=8,
                         color="#4e79a7", fontweight="bold")

    # --- p95 Latency ---
    un_p95 = [by_mode["unbatched"][c]["p95_ms"] for c in levels]
    ba_p95 = [by_mode["batched"][c]["p95_ms"]   for c in levels]
    ax2.plot(levels, un_p95, "o--", color="#e15759", linewidth=2, markersize=7, label="Unbatched p95")
    ax2.plot(levels, ba_p95, "s--", color="#4e79a7", linewidth=2, markersize=7, label="Batched p95")
    ax2.set_xlabel("Concurrency (writers)", fontsize=11)
    ax2.set_ylabel("Latency (ms)", fontsize=11)
    ax2.set_title("p95 Latency", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(levels)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Updated throughput: LLMBatchBundler vs unbatched concurrency sweep"
    )
    parser.add_argument("--requests",     type=int,         default=20)
    parser.add_argument("--concurrency",  type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--batch-size",   type=int,         default=10)
    parser.add_argument("--batch-timeout",type=float,       default=0.05)
    parser.add_argument("--output", type=str,
                        default=os.path.join(GRAPHS_DIR, "exp_throughput_results.json"))
    parser.add_argument("--plot", type=str,
                        default=os.path.join(GRAPHS_DIR, "exp_throughput.png"))
    args = parser.parse_args()

    print("=" * 66)
    print("Updated Throughput: Batched vs Unbatched LLM — Concurrency Sweep")
    print(f"  Concurrency levels : {args.concurrency}")
    print(f"  Requests per level : {args.requests}")
    print(f"  Batch size         : {args.batch_size}")
    print(f"  Batch timeout      : {args.batch_timeout}s")
    print(f"  LLM model          : Azure OpenAI (real API calls)")
    print(f"  VS collection      : pre-seeded 10k memories/user")
    print("=" * 66)

    all_results: List[Dict] = []
    for c in sorted(set(args.concurrency)):
        print(f"\n  Concurrency = {c}")
        all_results.append(
            run_level("unbatched", c, args.requests, args.batch_size, args.batch_timeout)
        )
        all_results.append(
            run_level("batched",   c, args.requests, args.batch_size, args.batch_timeout)
        )

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {args.output}")

    plot(all_results, args.plot)


if __name__ == "__main__":
    main()
