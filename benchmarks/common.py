"""
Shared utilities for all mem0 benchmark experiments.
"""

import os
import sys
import tempfile
import threading
import time
import uuid
from functools import wraps
from typing import Callable, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig


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
        {"role": "user",      "content": MESSAGES[idx % len(MESSAGES)]},
        {"role": "assistant", "content": "Thanks for sharing that."},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Memory factory — local Qdrant + Ollama
# ─────────────────────────────────────────────────────────────────────────────
def build_memory() -> Memory:
    tmp = tempfile.mkdtemp(prefix="mem0_bench_")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    config = MemoryConfig(
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config={
                "collection_name": f"bench_{uuid.uuid4().hex[:8]}",
                "embedding_model_dims": 768,
                "path": tmp,
            },
        ),
        llm=LlmConfig(
            provider="ollama",
            config={"model": "llama3.2:1b", "ollama_base_url": ollama_url, "temperature": 0},
        ),
        embedder=EmbedderConfig(
            provider="ollama",
            config={"model": "nomic-embed-text", "ollama_base_url": ollama_url, "embedding_dims": 768},
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
