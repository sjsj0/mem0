"""
Shared utilities for detailed benchmark experiments.
Measures: LLM breakdown, memory patterns, batching, and memory scaling.
"""

import json
import os
import sys
import tempfile
import threading
import time
import uuid
from urllib.request import urlopen
from functools import wraps
from typing import Callable, Dict, List

# os.environ["MEM0_TELEMETRY"] = "False"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig


class ProgressPrinter:
    """Thread-safe progress printer for long request loops."""

    def __init__(
        self,
        label: str,
        total: int,
        percent_step: int = 10,
        min_interval_s: float = 1.5,
    ):
        self.label = label
        self.total = max(total, 0)
        self.percent_step = max(percent_step, 1)
        self.min_interval_s = max(min_interval_s, 0.0)

        self.done = 0
        self._next_percent = self.percent_step
        self._last_print = time.perf_counter()
        self._lock = threading.Lock()

    def _percent(self) -> float:
        if self.total == 0:
            return 100.0
        return min(100.0, (self.done / self.total) * 100.0)

    def tick(self, count: int = 1):
        with self._lock:
            self.done += count
            if self.total:
                self.done = min(self.done, self.total)

            now = time.perf_counter()
            pct = self._percent()

            should_print = False
            if self.total and self.done >= self.total:
                should_print = True
            elif pct >= self._next_percent:
                should_print = True
                while pct >= self._next_percent:
                    self._next_percent += self.percent_step
            elif self.done > 0 and (now - self._last_print) >= self.min_interval_s:
                should_print = True

            if should_print:
                print(f"    [{self.label}] {self.done}/{self.total} ({pct:.0f}%)", flush=True)
                self._last_print = now


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic messages
# ─────────────────────────────────────────────────────────────────────────────
MESSAGES = [
    "I just started a new job as a software engineer at a fintech startup.",
    "My daughter had her first birthday last Saturday.",
    "I've been training for a half-marathon for the past three months.",
    "We adopted a golden retriever puppy named Biscuit last week.",
    "I'm learning to play the guitar. Just finished my fifth lesson.",
    "Had a rough week — my car broke down twice.",
    "I got promoted to senior engineer! Completely unexpected.",
    "We're thinking of buying a house in the suburbs.",
    "I signed up for an online MBA program starting in September.",
    "Just booked flights to Japan for cherry blossom season.",
    "My therapist suggested journaling daily. Kept it up for two weeks.",
    "Started intermittent fasting last month. Lost 8 pounds.",
    "The project I led for six months finally shipped to production.",
    "My grandfather was diagnosed with early-stage Alzheimer's.",
    "I've been reading one book a week this year.",
    "Finished my first open-source contribution!",
    "I started volunteering at the local food bank every other Saturday.",
    "Moved to a new apartment downtown — closer to the office.",
    "My team switched from daily standups to async Slack updates.",
    "I've been meditating 10 minutes every morning for 30 days.",
]


def make_messages(idx: int) -> List[Dict]:
    return [
        {"role": "user", "content": MESSAGES[idx % len(MESSAGES)]},
        {"role": "assistant", "content": "Thanks for sharing that."},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Memory factory — local Qdrant + Ollama
# ─────────────────────────────────────────────────────────────────────────────
def _list_ollama_models(ollama_url: str) -> List[str]:
    """Returns installed Ollama model names from /api/tags, or [] on failure."""
    try:
        with urlopen(f"{ollama_url.rstrip('/')}/api/tags", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    models = payload.get("models", [])
    seen = set()
    names = []
    for m in models:
        if not isinstance(m, dict):
            continue
        for key in ("name", "model"):
            value = m.get(key)
            if isinstance(value, str) and value and value not in seen:
                seen.add(value)
                names.append(value)
    return names


def _resolve_llm_model(ollama_url: str) -> str:
    """
    Uses a single required LLM model for benchmark consistency.
    Fails fast with an actionable error if that model is unavailable.
    """
    requested = "llama3.2:1b"
    installed = _list_ollama_models(ollama_url)

    if not installed:
        raise RuntimeError(
            "Could not fetch installed Ollama models. "
            "Make sure Ollama is running with 'ollama serve', then install the required model with "
            "'ollama pull llama3.2:1b'."
        )

    if requested in installed:
        return requested

    raise RuntimeError(
        "Required Ollama model 'llama3.2:1b' is not installed. "
        f"Models visible at {ollama_url}: {installed}. "
        "Install it with: ollama pull llama3.2:1b"
    )


def build_memory() -> Memory:
    tmp = tempfile.mkdtemp(prefix="mem0_detailed_bench_")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model = _resolve_llm_model(ollama_url)
    embed_model = os.getenv("MEM0_BENCH_EMBED_MODEL", "nomic-embed-text")

    config = MemoryConfig(
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config={
                "collection_name": f"detailed_{uuid.uuid4().hex[:8]}",
                "embedding_model_dims": 768,
                "path": tmp,
            },
        ),
        llm=LlmConfig(
            provider="ollama",
            config={"model": llm_model, "ollama_base_url": ollama_url, "temperature": 0},
        ),
        embedder=EmbedderConfig(
            provider="ollama",
            config={"model": embed_model, "ollama_base_url": ollama_url, "embedding_dims": 768},
        ),
        history_db_path=os.path.join(tmp, "history.db"),
    )
    return Memory(config)


# ─────────────────────────────────────────────────────────────────────────────
# Generic method wrapper
# ─────────────────────────────────────────────────────────────────────────────
def wrap(method: Callable, before: Callable = None, after: Callable = None) -> Callable:
    """
    Wraps a bound method: calls before() on entry, after(elapsed) on exit.
    Both are optional.
    """
    @wraps(method)
    def wrapper(*args, **kwargs):
        if before:
            before()
        t0 = time.perf_counter()
        result = method(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if after:
            after(elapsed)
        return result
    return wrapper
