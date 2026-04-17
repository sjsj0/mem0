"""
Updated Experiment: Concurrent Write Trace — Batched vs Unbatched LLM
=======================================================================
Uses LLMBatchBundler from mem0/memory/main.py to demonstrate how
concurrent Memory.add() calls benefit from batched LLM inference.

Batching model (real Azure OpenAI calls):
  Unbatched: each writer fires its own Azure LLM call independently.
  Batched:   LLMBatchBundler collects concurrent writes within the batch window,
             then calls generate_response sequentially for each item in the batch
             (Azure has no native vLLM-style batch endpoint).

Runs against the pre-seeded 10k-memory Qdrant collection so VS Search
and embedding calls are fully realistic.

Pre-requisite:
    python benchmarks/seed.py   (run once — ~30s for 50k vectors)

Usage:
    cd /Users/saraagarwal/mem0
    python updated_experiments/exp_trace_realistic.py
    python updated_experiments/exp_trace_realistic.py --writers 6 --batch-timeout 0.05
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from benchmarks.common import build_seeded_memory, make_messages

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)

GRAPHS_DIR = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)


PHASE_COLORS = {
    "embed_query": "#4e79a7",
    "vs_search":   "#f28e2b",
    "llm_wait":    "#e15759",
    "embed_batch": "#76b7b2",
    "vs_insert":   "#59a14f",
}
PHASE_LABELS = {
    "embed_query": "Embed (query)",
    "vs_search":   "VS Search",
    "llm_wait":    "LLM (extract/wait)",
    "embed_batch": "Embed (batch)",
    "vs_insert":   "VS Insert",
}
PHASE_ORDER = ["embed_query", "vs_search", "llm_wait", "embed_batch", "vs_insert"]


# ─────────────────────────────────────────────────────────────────────────────
# Azure batch endpoint adapter
# ─────────────────────────────────────────────────────────────────────────────
# Azure batch endpoint adapter
#
# Azure OpenAI has no vLLM-style batch endpoint, so we simulate one by calling
# generate_response sequentially for each item in the batch. The LLMBatchBundler
# still runs (collecting concurrent requests in the window), but each item in
# the batch makes a real Azure API call.
# ─────────────────────────────────────────────────────────────────────────────
def install_azure_batch_endpoint(mem):
    real_generate = mem.llm.generate_response

    def azure_batch_endpoint(messages_list, batch_url, **kwargs):  # noqa: ARG001
        return [real_generate(msgs, **kwargs) for msgs in messages_list]

    mem.llm.generate_batch_endpoint_response = azure_batch_endpoint


# ─────────────────────────────────────────────────────────────────────────────
# Per-thread event tracer
# ─────────────────────────────────────────────────────────────────────────────
class PerThreadTracer:
    """Records (phase, start_ms, end_ms) per caller thread."""

    def __init__(self):
        self.events: Dict[int, List[Tuple[str, float, float]]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, phase: str, t_start: float, t_end: float):
        tid = threading.get_ident()
        with self._lock:
            self.events[tid].append((phase, t_start * 1000, t_end * 1000))

    def wrap(self, method, phase: str):
        tracer = self
        @wraps(method)
        def wrapper(*args, **kwargs):
            t_s = time.perf_counter()
            result = method(*args, **kwargs)
            tracer.record(phase, t_s, time.perf_counter())
            return result
        return wrapper

    def install_unbatched(self, mem):
        mem.llm.generate_response       = self.wrap(mem.llm.generate_response,      "llm_wait")
        mem.embedding_model.embed       = self.wrap(mem.embedding_model.embed,       "embed_query")
        mem.embedding_model.embed_batch = self.wrap(mem.embedding_model.embed_batch, "embed_batch")
        mem.vector_store.search         = self.wrap(mem.vector_store.search,         "vs_search")
        mem.vector_store.insert         = self.wrap(mem.vector_store.insert,         "vs_insert")

    def install_batched(self, mem):
        # LLM phase: wrap batch_bundler.add_request (caller-side wait time)
        orig    = mem.batch_bundler.add_request
        tracer  = self
        def traced_add_request(messages, **kwargs):
            t_s = time.perf_counter()
            result = orig(messages, **kwargs)
            tracer.record("llm_wait", t_s, time.perf_counter())
            return result
        mem.batch_bundler.add_request   = traced_add_request
        mem.embedding_model.embed       = self.wrap(mem.embedding_model.embed,       "embed_query")
        mem.embedding_model.embed_batch = self.wrap(mem.embedding_model.embed_batch, "embed_batch")
        mem.vector_store.search         = self.wrap(mem.vector_store.search,         "vs_search")
        mem.vector_store.insert         = self.wrap(mem.vector_store.insert,         "vs_insert")


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
# Run one concurrent burst, return per-writer timeline
# ─────────────────────────────────────────────────────────────────────────────
def run_burst(mode: str, n_writers: int, batch_size: int, batch_timeout: float) -> Dict:
    mem = build_batched_mem(batch_size, batch_timeout) if mode == "batched" else build_unbatched_mem()

    tracer = PerThreadTracer()
    if mode == "batched":
        tracer.install_batched(mem)
    else:
        tracer.install_unbatched(mem)

    # All writers start at the same wall-clock origin
    t_origin = [None]
    t_origin_lock = threading.Lock()
    barrier = threading.Barrier(n_writers)

    def do_write(idx: int):
        barrier.wait()
        with t_origin_lock:
            if t_origin[0] is None:
                t_origin[0] = time.perf_counter()
        try:
            mem.add(make_messages(idx), user_id=f"bench_user_{idx % 5}")
        except Exception as e:
            logging.error(f"[{mode}] writer {idx} failed: {e}")

    with ThreadPoolExecutor(max_workers=n_writers) as pool:
        futs = [pool.submit(do_write, i) for i in range(n_writers)]
        for f in as_completed(futs):
            f.result()

    # Re-anchor all timestamps relative to burst start
    origin_ms = (t_origin[0] or 0.0) * 1000
    writers = []
    for tid, evs in tracer.events.items():
        anchored = [(ph, s - origin_ms, e - origin_ms) for ph, s, e in evs]
        writers.append((tid, anchored))
    writers.sort(key=lambda kv: min((e[1] for e in kv[1]), default=0))

    return {"mode": mode, "writers": writers}


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot(unbatched_result: Dict, batched_result: Dict, output: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(
        "Mem0 — Concurrent Write Critical Path: Unbatched vs Batched LLM\n"
        "Each row = one concurrent writer  |  Red bars = LLM extract phase\n"
        "Batched: writers share one LLM call → red bars end at the same time",
        fontsize=12, fontweight="bold",
    )

    for ax, result in zip(axes, [unbatched_result, batched_result]):
        mode    = result["mode"]
        writers = result["writers"]
        n       = len(writers)
        max_t   = 0.0

        for w_idx, (tid, events) in enumerate(writers):
            y = n - 1 - w_idx
            for phase, t_s, t_e in events:
                color    = PHASE_COLORS.get(phase, "#aaa")
                duration = t_e - t_s
                ax.barh(y, duration, left=t_s, height=0.55,
                        color=color, alpha=0.88, edgecolor="white", linewidth=0.4)
                if duration > 40:
                    ax.text(t_s + duration / 2, y, f"{duration:.0f}ms",
                            ha="center", va="center", fontsize=7.5,
                            color="white", fontweight="bold")
                max_t = max(max_t, t_e)

        ax.set_yticks(list(range(n)))
        ax.set_yticklabels([f"Writer {n - i}" for i in range(n)], fontsize=9)
        ax.set_xlabel("Wall time from burst start (ms)", fontsize=10)
        suffix = "(LLMBatchBundler active)" if mode == "batched" else "(individual LLM calls)"
        ax.set_title(
            f"{mode.title()} {suffix}   —   {n} concurrent writers   —   "
            f"total ≈ {max_t:.0f} ms",
            fontsize=11, loc="left",
        )
        ax.set_xlim(0, max_t * 1.08)
        ax.grid(axis="x", alpha=0.25)

        patches = [
            mpatches.Patch(color=PHASE_COLORS[ph], label=PHASE_LABELS[ph])
            for ph in PHASE_ORDER
        ]
        ax.legend(handles=patches, fontsize=8, loc="upper right", ncol=3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Updated trace: LLMBatchBundler vs unbatched concurrent writes"
    )
    parser.add_argument("--writers",       type=int,   default=4)
    parser.add_argument("--batch-size",    type=int,   default=10)
    parser.add_argument("--batch-timeout", type=float, default=0.05)
    parser.add_argument("--plot", type=str,
                        default=os.path.join(GRAPHS_DIR, "exp_trace_realistic.png"))
    args = parser.parse_args()

    print("=" * 66)
    print("Updated Trace: Concurrent Writes — Batched vs Unbatched LLM")
    print(f"  Writers       : {args.writers}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Batch timeout : {args.batch_timeout}s")
    print(f"  LLM model     : Azure OpenAI (real API calls)")
    print(f"  VS collection : pre-seeded 10k memories/user")
    print("=" * 66)

    print("\n  Running UNBATCHED burst ...")
    unbatched = run_burst("unbatched", args.writers, args.batch_size, args.batch_timeout)
    for i, (_, evs) in enumerate(unbatched["writers"]):
        total = max((e for _, _, e in evs), default=0)
        llm   = sum(e - s for ph, s, e in evs if ph == "llm_wait")
        print(f"    Writer {i+1}: total={total:.0f}ms  llm={llm:.0f}ms")

    print("\n  Running BATCHED burst ...")
    batched = run_burst("batched", args.writers, args.batch_size, args.batch_timeout)
    for i, (_, evs) in enumerate(batched["writers"]):
        total = max((e for _, _, e in evs), default=0)
        llm   = sum(e - s for ph, s, e in evs if ph == "llm_wait")
        print(f"    Writer {i+1}: total={total:.0f}ms  llm={llm:.0f}ms")

    un_wall = max((max((e for _, _, e in evs), default=0) for _, evs in unbatched["writers"]), default=0)
    ba_wall = max((max((e for _, _, e in evs), default=0) for _, evs in batched["writers"]), default=0)
    speedup = un_wall / ba_wall if ba_wall > 0 else 0
    print(f"\n  Unbatched wall time : {un_wall:.0f}ms")
    print(f"  Batched   wall time : {ba_wall:.0f}ms")
    print(f"  Speedup             : {speedup:.2f}×")

    plot(unbatched, batched, args.plot)


if __name__ == "__main__":
    main()
