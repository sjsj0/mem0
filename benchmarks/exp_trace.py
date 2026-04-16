"""
Experiment H: Single Write Request — Gantt Trace
=================================================
Goal:
  Show that one write has a mostly sequential chain of remote calls.
  Visualise the critical path so the "mostly waiting" story is obvious.

How it works:
  - Wraps each service method (LLM, embedder, vector store, SQLite)
    with (start_time, end_time) recording
  - Runs a single Memory.add() call
  - Plots a Gantt chart: x-axis = wall time, coloured bars = each call

Phases captured:
  llm_extract, embed_single, embed_batch,
  vs_search, vs_insert,
  entity_search, entity_insert,
  db_get_messages, db_save_messages, db_history

Plot saved to benchmarks/graphs/exp_trace.png

Usage:
    cd /Users/saraagarwal/mem0
    python benchmarks/exp_trace.py
    python benchmarks/exp_trace.py --runs 3   # overlay multiple traces
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import logging
import threading
import time
from functools import wraps
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from benchmarks.common import build_memory, make_messages

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)

GRAPHS_DIR = "benchmarks/graphs"

# phase → display label + color
PHASE_META = {
    "llm_extract":    ("LLM Extract",       "#e15759"),
    "embed_single":   ("Embed (query)",      "#4e79a7"),
    "embed_batch":    ("Embed (batch)",      "#76b7b2"),
    "vs_search":      ("VS Search",          "#f28e2b"),
    "vs_insert":      ("VS Insert",          "#59a14f"),
    "entity_search":  ("Entity Search",      "#b07aa1"),
    "entity_insert":  ("Entity Insert",      "#ff9da7"),
    "entity_update":  ("Entity Update",      "#9c755f"),
    "db_get_messages":("DB Get Messages",    "#bab0ac"),
    "db_save_messages":("DB Save Messages",  "#edc948"),
    "db_history":     ("DB History",         "#499894"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Trace recorder
# ─────────────────────────────────────────────────────────────────────────────
class TraceRecorder:
    """Records (phase, t_start, t_end) for every service call."""

    def __init__(self):
        self.events: List[Tuple[str, float, float]] = []
        self._lock = threading.Lock()
        self._t0: float = 0.0

    def set_origin(self, t0: float):
        self._t0 = t0

    def record(self, phase: str, t_start: float, t_end: float):
        with self._lock:
            self.events.append((phase, t_start - self._t0, t_end - self._t0))

    def wrap(self, method, phase: str):
        recorder = self
        @wraps(method)
        def wrapper(*args, **kwargs):
            t_start = time.perf_counter()
            result  = method(*args, **kwargs)
            t_end   = time.perf_counter()
            recorder.record(phase, t_start, t_end)
            return result
        return wrapper

    def install(self, mem):
        mem.llm.generate_response        = self.wrap(mem.llm.generate_response,        "llm_extract")
        mem.embedding_model.embed        = self.wrap(mem.embedding_model.embed,         "embed_single")
        mem.embedding_model.embed_batch  = self.wrap(mem.embedding_model.embed_batch,   "embed_batch")
        mem.vector_store.search          = self.wrap(mem.vector_store.search,           "vs_search")
        mem.vector_store.insert          = self.wrap(mem.vector_store.insert,           "vs_insert")
        mem.db.get_last_messages         = self.wrap(mem.db.get_last_messages,          "db_get_messages")
        mem.db.save_messages             = self.wrap(mem.db.save_messages,              "db_save_messages")
        mem.db.batch_add_history         = self.wrap(mem.db.batch_add_history,          "db_history")
        # entity store patched after lazy init — done in run_trace()


# ─────────────────────────────────────────────────────────────────────────────
# Run a single trace
# ─────────────────────────────────────────────────────────────────────────────
def run_trace(idx: int = 0) -> Tuple[List, float]:
    """
    Returns (events, total_ms).
    events = list of (phase, t_start_ms, t_end_ms) relative to add() start.
    """
    mem = build_memory()
    recorder = TraceRecorder()
    recorder.install(mem)

    # patch entity store after lazy init by doing a warm-up search
    try:
        _ = mem.entity_store
        mem.entity_store.search_batch = recorder.wrap(mem.entity_store.search_batch, "entity_search")
        mem.entity_store.insert       = recorder.wrap(mem.entity_store.insert,       "entity_insert")
        mem.entity_store.update       = recorder.wrap(mem.entity_store.update,       "entity_update")
    except Exception:
        pass

    t0 = time.perf_counter()
    recorder.set_origin(t0)

    try:
        mem.add(make_messages(idx), user_id="trace_user")
    except Exception as e:
        logging.error(f"trace add failed: {e}")

    total_ms = (time.perf_counter() - t0) * 1000

    # convert to ms
    events_ms = [(phase, t_s * 1000, t_e * 1000) for phase, t_s, t_e in recorder.events]
    return events_ms, total_ms


# ─────────────────────────────────────────────────────────────────────────────
# Gantt plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_gantt(all_events: List[List[Tuple]], all_totals: List[float], output: str):
    n_runs = len(all_events)

    fig, axes = plt.subplots(n_runs, 1,
                             figsize=(14, 3.5 * n_runs),
                             squeeze=False)
    fig.suptitle(
        "Mem0 — Single Write Request Critical Path (Gantt)\n"
        "Each bar = one remote/local call, x-axis = wall time",
        fontsize=13, fontweight="bold",
    )

    for run_idx, (events, total_ms) in enumerate(zip(all_events, all_totals)):
        ax = axes[run_idx][0]

        # assign a y-position to each phase (group same phase together)
        seen_phases = []
        phase_y: Dict[str, int] = {}
        for phase, _, _ in events:
            if phase not in phase_y:
                phase_y[phase] = len(seen_phases)
                seen_phases.append(phase)

        for phase, t_s, t_e in events:
            y     = phase_y[phase]
            color = PHASE_META.get(phase, ("", "#aaa"))[1]
            label = PHASE_META.get(phase, (phase, ""))[0]
            duration = t_e - t_s
            ax.barh(y, duration, left=t_s, height=0.5,
                    color=color, alpha=0.88, edgecolor="white", linewidth=0.5)
            if duration > total_ms * 0.02:   # only label if wide enough
                ax.text(t_s + duration / 2, y, f"{duration:.0f}ms",
                        ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

        # y-axis labels
        ax.set_yticks(list(phase_y.values()))
        ax.set_yticklabels(
            [PHASE_META.get(p, (p, ""))[0] for p in phase_y],
            fontsize=9,
        )
        ax.set_xlabel("Wall time since add() start (ms)", fontsize=10)
        ax.set_title(
            f"Run {run_idx + 1}   total={total_ms:.0f}ms",
            fontsize=11, loc="left",
        )
        ax.set_xlim(0, total_ms * 1.05)
        ax.grid(axis="x", alpha=0.3)

        # legend patches
        patches = [
            mpatches.Patch(color=PHASE_META.get(p, ("", "#aaa"))[1],
                           label=PHASE_META.get(p, (p, ""))[0])
            for p in phase_y
        ]
        ax.legend(handles=patches, fontsize=8, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_trace_summary(events: List[Tuple], total_ms: float):
    # aggregate duration per phase
    phase_time: Dict[str, float] = {}
    for phase, t_s, t_e in events:
        phase_time[phase] = phase_time.get(phase, 0) + (t_e - t_s)

    print(f"\n  Total add() time: {total_ms:.0f}ms")
    print(f"  {'Phase':<22}  {'ms':>8}  {'% of total':>10}  bar")
    print("  " + "-" * 58)
    for phase, dur in sorted(phase_time.items(), key=lambda x: -x[1]):
        pct = 100 * dur / total_ms if total_ms else 0
        bar = "█" * int(pct / 2)
        label = PHASE_META.get(phase, (phase, ""))[0]
        print(f"  {label:<22}  {dur:>8.0f}  {pct:>9.1f}%  {bar}")

    # identify critical path (sequential wait time)
    accounted = sum(phase_time.values())
    unaccounted = max(0, total_ms - accounted)
    print(f"\n  Sum of phase times : {accounted:.0f}ms")
    print(f"  Unaccounted (overhead/overlap) : {unaccounted:.0f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Exp H: Gantt trace of single write")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of individual write traces to collect and overlay (default: 1)")
    parser.add_argument("--plot", type=str, default=f"{GRAPHS_DIR}/exp_trace.png")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment H: Single Write Request Gantt Trace")
    print(f"  Traces to collect : {args.runs}")
    print("=" * 60)

    all_events: List[List[Tuple]] = []
    all_totals: List[float] = []

    for i in range(args.runs):
        print(f"\n  Running trace {i + 1}/{args.runs} ...")
        events, total_ms = run_trace(idx=i)
        all_events.append(events)
        all_totals.append(total_ms)
        print_trace_summary(events, total_ms)

    plot_gantt(all_events, all_totals, args.plot)


if __name__ == "__main__":
    main()
