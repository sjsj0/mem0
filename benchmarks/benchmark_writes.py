"""
Mem0 Write Latency Benchmark
=============================
Throttles Memory.add() with concurrent write requests and measures per-phase
latency to find the bottleneck.

How it works
------------
Instead of copying production code, we wrap the individual service methods
(llm.generate_response, embedding_model.embed/embed_batch, vector_store.search,
vector_store.insert, db.get_last_messages, db.save_messages, db.batch_add_history,
extract_entities_batch) with thin timing decorators at Memory init time.

This means we always measure the real production path — no code duplication,
no drift risk.

Phases measured:
  llm_extract     — LLM call to extract facts
  embed_search    — embedding the query (before VS search)
  embed_batch     — batch embedding extracted memory texts
  embed_entity    — batch embedding entity texts
  vs_search       — vector store search (find existing memories)
  vs_insert       — vector store insert (persist new memories)
  entity_search   — entity store search_batch
  entity_insert   — entity store insert / update
  db_get_messages — SQLite get_last_messages
  db_save         — SQLite save_messages + batch_add_history

Usage:
    cd /Users/saraagarwal/mem0
    python benchmarks/benchmark_writes.py --requests 5 --concurrency 1
    python benchmarks/benchmark_writes.py --requests 20 --concurrency 1 2 4
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# TimedMemory
# ─────────────────────────────────────────────────────────────────────────────
class TimedMemory(Memory):
    """
    Wraps individual service methods with timing decorators at __init__.
    No production code is copied — we always measure the real pipeline.

    Per-request phase timings are accumulated in self.call_log (thread-safe).
    After each add() call completes, collect_timings() drains call_log and
    returns a dict of {phase: total_elapsed_seconds} for that request.
    """

    def __init__(self, config: MemoryConfig = MemoryConfig()):
        super().__init__(config)
        self._call_log: List[Dict[str, float]] = []
        self._bg_log: List[Dict[str, float]] = []
        self._log_lock = threading.Lock()
        self._last_bg_future = None
        self._patch_methods()

    def _patch_methods(self):
        log = self._call_log
        lock = self._log_lock

        def timed_append(method, bucket, is_bg=False):
            @wraps(method)
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                result = method(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                with lock:
                    if is_bg:
                        self._bg_log.append({bucket: elapsed})
                    else:
                        log.append({bucket: elapsed})
                return result
            return wrapper

        # LLM
        self.llm.generate_response = timed_append(
            self.llm.generate_response, "llm_extract"
        )

        # Validation LLM (Speculative)
        if self.validation_llm:
            self.validation_llm.generate_response = timed_append(
                self.validation_llm.generate_response, "v_llm_validate", is_bg=True
            )
            
            # Patch executor to capture the future
            original_submit = self._validation_executor.submit
            def timed_submit(fn, *args, **kwargs):
                future = original_submit(fn, *args, **kwargs)
                self._last_bg_future = future
                return future
            self._validation_executor.submit = timed_submit
            
        # Batch Bundler
        if hasattr(self, "validation_batch_bundler") and self.validation_batch_bundler:
            original_add = self.validation_batch_bundler.add_request
            self.validation_batch_bundler.add_request = timed_append(
                original_add, "v_llm_batch_wait", is_bg=True
            )

        # Embedder
        original_embed = self.embedding_model.embed
        original_embed_batch = self.embedding_model.embed_batch

        self.embedding_model.embed = timed_append(original_embed, "embed_single")
        self.embedding_model.embed_batch = timed_append(original_embed_batch, "embed_batch")

        # Vector store — main memory store
        self.vector_store.search = timed_append(self.vector_store.search, "vs_search")
        self.vector_store.insert = timed_append(self.vector_store.insert, "vs_insert")

        # SQLite history DB
        self.db.get_last_messages = timed_append(self.db.get_last_messages, "db_get_messages")
        self.db.save_messages = timed_append(self.db.save_messages, "db_save_messages")
        self.db.batch_add_history = timed_append(self.db.batch_add_history, "db_history")

        # Entity store (lazily initialised — patch after first access)
        self._entity_patched = False

    def _ensure_entity_patched(self):
        if self._entity_patched:
            return
        try:
            es = self.entity_store  # triggers lazy init
            lock = self._log_lock

            def timed_append(method, bucket):
                @wraps(method)
                def wrapper(*args, **kwargs):
                    t0 = time.perf_counter()
                    result = method(*args, **kwargs)
                    with lock:
                        self._call_log.append({bucket: time.perf_counter() - t0})
                    return result
                return wrapper

            es.search_batch = timed_append(es.search_batch, "entity_search")
            es.insert = timed_append(es.insert, "entity_insert")
            es.update = timed_append(es.update, "entity_update")
            self._entity_patched = True
        except Exception:
            pass

    def add(self, messages, **kwargs):
        self._ensure_entity_patched()
        t_start = time.perf_counter()
        result = super().add(messages, **kwargs)
        t_e2e = time.perf_counter() - t_start

        # If speculative was active, wait for the bg task to get "e2e + bg"
        t_total_inc_bg = t_e2e
        if self._last_bg_future:
            try:
                self._last_bg_future.result()
                t_total_inc_bg = time.perf_counter() - t_start
            except Exception:
                pass
            self._last_bg_future = None

        # drain the call log accumulated during this add() call
        with self._log_lock:
            entries = list(self._call_log)
            self._call_log.clear()
            bg_entries = list(self._bg_log)
            self._bg_log.clear()

        # merge into a single dict, summing repeated buckets
        phases: Dict[str, float] = {
            "e2e": t_e2e,
            "e2e_inc_bg": t_total_inc_bg,
            "total": t_e2e # for backward compatibility with plotting logic
        }
        for entry in entries + bg_entries:
            for bucket, elapsed in entry.items():
                phases[bucket] = phases.get(bucket, 0.0) + elapsed

        with self._log_lock:
            if not hasattr(self, "_timings"):
                self._timings = []
            self._timings.append(phases)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Build a fresh isolated Memory instance
# ─────────────────────────────────────────────────────────────────────────────
def build_memory(args) -> TimedMemory:
    tmp = tempfile.mkdtemp(prefix="mem0_bench_")
    ollama_url = os.getenv("OLLAMA_BASE_URL", args.ollama_url)

    vs_config = {
        "collection_name": f"bench_{uuid.uuid4().hex[:8]}",
        "embedding_model_dims": 768,
    }
    if args.qdrant_url:
        vs_config["url"] = args.qdrant_url
        if args.qdrant_api_key:
            vs_config["api_key"] = args.qdrant_api_key
    else:
        vs_config["path"] = tmp

    validation_llm = None
    if args.speculative:
        validation_llm = LlmConfig(
            provider=args.v_llm_provider,
            config={
                "model": args.v_llm_model,
                "ollama_base_url": ollama_url if args.v_llm_provider == "ollama" else None,
                "temperature": 0,
            },
        )
        if args.batch_url:
            validation_llm.config["batch_url"] = args.batch_url

    config = MemoryConfig(
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config=vs_config,
        ),
        llm=LlmConfig(
            provider="ollama",
            config={
                "model": args.model,
                "ollama_base_url": ollama_url,
                "temperature": 0,
            },
        ),
        embedder=EmbedderConfig(
            provider="ollama",
            config={
                "model": "nomic-embed-text",
                "ollama_base_url": ollama_url,
                "embedding_dims": 768,
            },
        ),
        history_db_path=os.path.join(tmp, "history.db"),
        validation_llm=validation_llm,
    )
    
    # Batching can also be enabled via ENV
    if args.batch_url:
        os.environ["MEM0_VALIDATION_BATCH_URL"] = args.batch_url
        
    return TimedMemory(config)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic messages
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_MESSAGES = [
    "I just started a new job as a software engineer at a fintech startup.",
    "My daughter had her first birthday last Saturday. We threw a small party.",
    "I've been training for a half-marathon for the past three months.",
    "We adopted a golden retriever puppy named Biscuit last week.",
    "I'm learning to play the guitar. Just finished my fifth lesson.",
    "Had a rough week — my car broke down twice and wiped out my savings buffer.",
    "I got promoted to senior engineer! Completely unexpected.",
    "We're thinking of buying a house in the suburbs.",
    "I signed up for an online MBA program starting in September.",
    "Just booked flights to Japan for cherry blossom season next April.",
    "My therapist suggested journaling daily. Kept it up for two weeks.",
    "Started intermittent fasting last month. Lost 8 pounds so far.",
    "The project I've been leading for six months finally shipped to production.",
    "My grandfather was diagnosed with early-stage Alzheimer's.",
    "I've been reading one book a week this year.",
    "Finished my first open-source contribution! A small bug fix in a Python library.",
    "I started volunteering at the local food bank every other Saturday.",
    "Moved to a new apartment downtown — smaller but closer to the office.",
    "My team is switching from daily standups to async Slack updates.",
    "I've been meditating for 10 minutes every morning for 30 days straight.",
]


def make_messages(idx: int) -> List[Dict[str, str]]:
    msg = SAMPLE_MESSAGES[idx % len(SAMPLE_MESSAGES)]
    return [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": "Thanks for sharing that."},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Run one add() and return its timing record
# ─────────────────────────────────────────────────────────────────────────────
def run_one(mem: TimedMemory, request_idx: int) -> Dict[str, float]:
    user_id = f"bench_user_{request_idx % 5}"
    try:
        mem.add(make_messages(request_idx), user_id=user_id)
    except Exception as e:
        logging.error(f"req {request_idx} failed: {e}")
        return {}
    
    # Small wait to allow background tasks to at least start/queue
    # if we want to capture their start. But speculative decoding is background,
    # so we might not see the full v_llm_validate in the same 'total' bucket.
    # However, the TimedMemory.add implementation clears BG log too.
    
    with mem._log_lock:
        return mem._timings[-1].copy() if hasattr(mem, "_timings") and mem._timings else {}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner for one concurrency level
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(n_requests: int, concurrency: int, args) -> List[Dict[str, float]]:
    print(f"\n{'─'*60}")
    print(f"  concurrency={concurrency}   total_requests={n_requests}")
    print(f"{'─'*60}")

    mem = build_memory(args)
    results: List[Dict[str, float]] = []
    results_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_one, mem, i): i for i in range(n_requests)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                r = fut.result()
                if r:
                    with results_lock:
                        results.append(r)
                    print(
                        f"  req {idx:3d} | total {r.get('total', 0)*1000:6.0f}ms"
                        f" | llm {r.get('llm_extract', 0)*1000:5.0f}ms"
                        f" | v_llm {r.get('v_llm_validate', r.get('v_llm_batch_wait', 0))*1000:5.0f}ms"
                        f" | vs_search {r.get('vs_search', 0)*1000:5.0f}ms"
                    )
            except Exception as e:
                print(f"  req {idx}: ERROR {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
PHASES = [
    "e2e",
    "e2e_inc_bg",
    "llm_extract",
    "v_llm_validate",
    "v_llm_batch_wait",
    "embed_single",
    "embed_batch",
    "vs_search",
    "vs_insert",
    "entity_search",
    "entity_insert",
    "entity_update",
    "db_get_messages",
    "db_save_messages",
    "db_history",
]

COLORS = [
    "#e15759", "#4e79a7", "#76b7b2", "#f28e2b",
    "#59a14f", "#b07aa1", "#ff9da7", "#edc948",
    "#9c755f", "#bab0ac", "#499894", "#86bc25", "#00adff"
]


def plot_results(all_data: Dict[int, List[Dict[str, float]]], output_path: str):
    concurrency_levels = sorted(all_data.keys())

    # only plot phases that actually appear in results
    active_phases = [
        p for p in PHASES
        if any(r.get(p, 0) > 0 for runs in all_data.values() for r in runs)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Mem0 — Write Latency Benchmark", fontsize=15, fontweight="bold", y=0.99)

    # ── 1. Stacked bar: mean phase latency per concurrency level ──────────
    ax = axes[0, 0]
    x = np.arange(len(concurrency_levels))
    bottoms = np.zeros(len(concurrency_levels))
    for i, phase in enumerate(active_phases):
        means = [
            np.mean([r.get(phase, 0) * 1000 for r in all_data[c] if r]) or 0
            for c in concurrency_levels
        ]
        ax.bar(x, means, 0.5, bottom=bottoms,
               color=COLORS[i % len(COLORS)], label=phase, alpha=0.9)
        bottoms += np.array(means)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in concurrency_levels])
    ax.set_xlabel("Concurrency (parallel writers)")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Mean Latency Stack by Phase")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # ── 2. Box plot: total latency distribution per concurrency ───────────
    ax = axes[0, 1]
    box_data = [
        [r.get("total", 0) * 1000 for r in all_data[c] if r]
        for c in concurrency_levels
    ]
    bp = ax.boxplot(box_data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    cmap = plt.cm.Set2(np.linspace(0, 0.8, len(concurrency_levels)))
    for patch, color in zip(bp["boxes"], cmap):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels([f"c={c}" for c in concurrency_levels])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Total latency (ms)")
    ax.set_title("Total Latency Distribution")
    ax.grid(axis="y", alpha=0.3)

    # ── 3. Horizontal bar: phase breakdown at lowest concurrency ──────────
    ax = axes[1, 0]
    c1 = concurrency_levels[0]
    c1_data = all_data[c1]
    phase_means = {
        p: np.mean([r.get(p, 0) * 1000 for r in c1_data if r])
        for p in active_phases
    }
    sorted_phases = sorted(active_phases, key=lambda p: phase_means[p], reverse=True)
    values = [phase_means[p] for p in sorted_phases]
    bar_colors = [COLORS[active_phases.index(p) % len(COLORS)] for p in sorted_phases]
    bars = ax.barh(sorted_phases, values, color=bar_colors, alpha=0.88)
    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} ms", va="center", fontsize=9)
    ax.set_xlabel("Mean latency (ms)")
    ax.set_title(f"Phase Breakdown at c={c1}  —  sorted by cost (bottleneck at top)")
    ax.grid(axis="x", alpha=0.3)

    # ── 4. Throughput + p95 vs concurrency ────────────────────────────────
    ax = axes[1, 1]
    ax2 = ax.twinx()
    throughputs, p95s, medians = [], [], []
    for c in concurrency_levels:
        vals = [r.get("total", 0) for r in all_data[c] if r]
        if vals:
            wall = sum(vals) / c
            throughputs.append(len(vals) / wall if wall else 0)
            p95s.append(np.percentile(vals, 95) * 1000)
            medians.append(np.median(vals) * 1000)
        else:
            throughputs.append(0); p95s.append(0); medians.append(0)

    l1, = ax.plot(concurrency_levels, throughputs, "o-", color="#4e79a7",
                  linewidth=2, markersize=7, label="Throughput (req/s)")
    l2, = ax2.plot(concurrency_levels, p95s, "s--", color="#e15759",
                   linewidth=2, markersize=7, label="p95 latency (ms)")
    l3, = ax2.plot(concurrency_levels, medians, "^-.", color="#59a14f",
                   linewidth=2, markersize=7, label="Median latency (ms)")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Throughput (req/s)", color="#4e79a7")
    ax2.set_ylabel("Latency (ms)", color="#e15759")
    ax.set_title("Throughput & Latency vs Concurrency")
    ax.tick_params(axis="y", labelcolor="#4e79a7")
    ax2.tick_params(axis="y", labelcolor="#e15759")
    ax.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]],
              fontsize=8, loc="upper left")
    ax.set_xticks(concurrency_levels)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved  →  {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary table
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(all_data: Dict[int, List[Dict[str, float]]]):
    concurrency_levels = sorted(all_data.keys())
    active_phases = [
        p for p in PHASES
        if any(r.get(p, 0) > 0 for runs in all_data.values() for r in runs)
    ]

    print("\n" + "=" * 80)
    print("LATENCY SUMMARY (ms)  —  mean / p95")
    print("=" * 80)
    print(f"{'phase':<22}" + "".join(f"  c={c} mean/p95" for c in concurrency_levels))
    print("-" * 80)
    for phase in ["total"] + active_phases:
        row = f"{phase:<22}"
        for c in concurrency_levels:
            vals = [r.get(phase, 0) * 1000 for r in all_data[c] if r]
            row += f"  {np.mean(vals):6.0f} / {np.percentile(vals,95):5.0f}  " if vals else "        —  "
        print(row)

    print(f"\nBOTTLENECK  (% of total at c={concurrency_levels[0]}):")
    c1_data = all_data[concurrency_levels[0]]
    avg_total = np.mean([r.get("total", 1e-9) for r in c1_data if r])
    for phase in sorted(active_phases, key=lambda p: -np.mean([r.get(p, 0) for r in c1_data if r])):
        avg = np.mean([r.get(phase, 0) for r in c1_data if r])
        pct = 100 * avg / avg_total if avg_total else 0
        bar = "█" * int(pct / 2)
        print(f"  {phase:<22}  {pct:5.1f}%  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Mem0 write latency benchmark")
    parser.add_argument("--requests", type=int, default=20,
                        help="Write requests per concurrency level (default: 20)")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4],
                        help="Concurrency levels to test (default: 1 2 4)")
    parser.add_argument("--model", type=str, default="llama3.2:1b",
                        help="Primary LLM model (default: llama3.2:1b)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama base URL")
    
    # Speculative Decoding & Batching
    parser.add_argument("--speculative", action="store_true",
                        help="Activate speculative decoding (background reconciliation)")
    parser.add_argument("--v-llm-model", type=str, default="llama3:8b",
                        help="Validation LLM model (default: llama3:8b)")
    parser.add_argument("--v-llm-provider", type=str, default="ollama",
                        help="Validation LLM provider (default: ollama)")
    parser.add_argument("--batch-url", type=str, default=None,
                        help="URL for batch LLM inference (activates batching if set)")
    
    # Qdrant Server
    parser.add_argument("--qdrant-url", type=str, default=None,
                        help="Qdrant server URL (e.g. http://localhost:6333)")
    parser.add_argument("--qdrant-api-key", type=str, default=None,
                        help="Qdrant API key")

    parser.add_argument("--output", type=str, default="benchmarks/benchmark_results.json",
                        help="File to save raw timings JSON")
    parser.add_argument("--plot", type=str, default="benchmarks/benchmark_latency.png",
                        help="File to save the latency breakdown plot")
    args = parser.parse_args()

    print(f"Primary Model: {args.model}")
    if args.speculative:
        print(f"Speculative Decoding: ON (Validation Model: {args.v_llm_model})")
        if args.batch_url:
            print(f"Batching: ON (URL: {args.batch_url})")
    if args.qdrant_url:
        print(f"Vector Store: Qdrant Server at {args.qdrant_url}")

    all_data: Dict[int, List[Dict[str, float]]] = {}
    for c in sorted(set(args.concurrency)):
        all_data[c] = run_benchmark(n_requests=args.requests, concurrency=c, args=args)

    with open(args.output, "w") as f:
        json.dump({str(k): v for k, v in all_data.items()}, f, indent=2)
    print(f"\nRaw timings saved  →  {args.output}")

    print_summary(all_data)
    plot_results(all_data, args.plot)


if __name__ == "__main__":
    main()
