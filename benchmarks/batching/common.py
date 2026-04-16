import json
import os
import sys
import tempfile
import threading
import time
import uuid
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.memory import main as memory_main
from mem0.vector_stores.configs import VectorStoreConfig


MESSAGES = [
    "I just started a new job as a software engineer at a fintech startup.",
    "My daughter had her first birthday last Saturday.",
    "I've been training for a half-marathon for the past three months.",
    "We adopted a golden retriever puppy named Biscuit last week.",
    "I'm learning to play the guitar. Just finished my fifth lesson.",
    "Had a rough week; my car broke down twice.",
    "I got promoted to senior engineer! Completely unexpected.",
    "We're thinking of buying a house in the suburbs.",
    "I signed up for an online MBA program starting in September.",
    "Just booked flights to Japan for cherry blossom season.",
]


def make_messages(idx: int) -> List[Dict]:
    return [
        {"role": "user", "content": MESSAGES[idx % len(MESSAGES)]},
        {"role": "assistant", "content": "Thanks for sharing that."},
    ]


def build_memory() -> Memory:
    tmp = tempfile.mkdtemp(prefix="mem0_batching_bench_")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    config = MemoryConfig(
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config={
                "collection_name": f"batching_{uuid.uuid4().hex[:8]}",
                "embedding_model_dims": 768,
                "path": tmp,
            },
        ),
        llm=LlmConfig(
            provider="ollama",
            config={
                "model": "llama3.2:1b",
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
    )

    # These experiments focus on ACK queueing / LLM batching behavior.
    # Disable entity extraction to avoid local Qdrant entity-store lock conflicts
    # (entity store opens a second local client on the same storage path).
    memory_main.extract_entities_batch = lambda texts: [[] for _ in texts]

    return Memory(config)


class ProgressPrinter:
    def __init__(self, label: str, total: int, step_pct: int = 10):
        self.label = label
        self.total = max(total, 0)
        self.step_pct = max(step_pct, 1)
        self.done = 0
        self.next_pct = self.step_pct
        self.lock = threading.Lock()

    def tick(self):
        with self.lock:
            self.done += 1
            if self.total > 0 and self.done > self.total:
                self.done = self.total
            pct = 100.0 if self.total == 0 else (self.done / self.total) * 100.0
            if self.done == self.total or pct >= self.next_pct:
                print(f"    [{self.label}] {self.done}/{self.total} ({pct:.0f}%)", flush=True)
                while pct >= self.next_pct:
                    self.next_pct += self.step_pct


def ensure_graphs_dir() -> str:
    path = "benchmarks/batching/graphs"
    os.makedirs(path, exist_ok=True)
    return path


def dump_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
