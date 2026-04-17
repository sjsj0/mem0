"""
Shared utilities for all mem0 benchmark experiments.

LLM provider is controlled by the LLM_PROVIDER environment variable.
Supported values:

  azure_openai  (default)
    LLM_AZURE_ENDPOINT, LLM_AZURE_OPENAI_API_KEY,
    LLM_AZURE_DEPLOYMENT, LLM_AZURE_API_VERSION

  ollama
    OLLAMA_BASE_URL   (default: http://localhost:11434)
    LLM_MODEL         (default: llama3.2:1b)

  vllm
    VLLM_BASE_URL     (default: http://localhost:8000/v1)
    VLLM_API_KEY      (default: vllm-api-key)
    LLM_MODEL         (optional; auto-resolved from /v1/models if missing/invalid)

  openai            (also works for llama.cpp OpenAI-compatible server)
    OPENAI_API_KEY
    LLM_BASE_URL      (override for llama.cpp: http://localhost:8080/v1)
    LLM_MODEL         (required)

Embedder is always Ollama nomic-embed-text (local):
  OLLAMA_BASE_URL   (default: http://localhost:11434)

Rate limiting (Azure only — 280 calls/min by default):
  LLM_RATE_LIMIT    (calls/min, default: 280 for azure_openai, 0 = disabled)

Seed controls:
    MEM0_BENCH_SKIP_SEED   (1/true/yes: run seeded experiments without seed data)

Entity-linking control:
    MEM0_BENCH_ENABLE_ENTITY_LINKING (0/false/no/off default; 1=true to enable)
"""

import os
import sys
import tempfile
import threading
import time
import uuid
from functools import wraps
from typing import Callable, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Token-bucket rate limiter (used for Azure to stay under API limits)
# ─────────────────────────────────────────────────────────────────────────────
def _load_local_env_file() -> None:
    """Load benchmarks/.env into process env if present (without overriding existing vars)."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(env_path):
        return

    preexisting = set(os.environ.keys())
    loaded_here = set()

    with open(env_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (key in loaded_here or key not in preexisting):
                os.environ[key] = value
                loaded_here.add(key)


_load_local_env_file()


class RateLimiter:
    """
    Token bucket: allows at most `rate_per_min` calls per minute.
    Threads block until a token is available.
    """
    def __init__(self, rate_per_min: int):
        self._rate   = rate_per_min / 60.0
        self._tokens = self._rate            # start with 1 second worth only
        self._max    = float(rate_per_min)
        self._last   = time.monotonic()
        self._lock   = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._max,
                    self._tokens + (now - self._last) * self._rate,
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            time.sleep(0.05)

    def wrap(self, method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            self.acquire()
            return method(*args, **kwargs)
        return wrapper


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.memory import main as memory_main


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
# LLM config builder — switches on LLM_PROVIDER env var
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_DIMS = 768   # nomic-embed-text (Ollama)


def _build_llm_config() -> LlmConfig:
    provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()

    if provider == "azure_openai":
        return LlmConfig(provider="azure_openai", config={"temperature": 0})

    elif provider == "ollama":
        return LlmConfig(provider="ollama", config={
            "model":            os.getenv("LLM_MODEL", "llama3.2:1b"),
            "ollama_base_url":  os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "temperature":      0,
        })

    elif provider == "vllm":
        return LlmConfig(provider="vllm", config={
            "model":           os.getenv("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
            "vllm_base_url":   os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            "api_key":         os.getenv("VLLM_API_KEY", "vllm-api-key"),
            "temperature":     0,
        })

    elif provider == "openai":
        # Also works for llama.cpp (set LLM_BASE_URL=http://localhost:8080/v1)
        cfg = {
            "model":       os.getenv("LLM_MODEL", "gpt-4o"),
            "temperature": 0,
        }
        if os.getenv("LLM_BASE_URL"):
            cfg["openai_base_url"] = os.getenv("LLM_BASE_URL")
        return LlmConfig(provider="openai", config=cfg)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            "Supported: azure_openai, ollama, vllm, openai"
        )


def _apply_rate_limiter(mem: Memory) -> Memory:
    """Apply rate limiter for providers that have API rate limits."""
    provider  = os.getenv("LLM_PROVIDER", "azure_openai").lower()
    # Default: 280/min for azure, 0 (disabled) for everything else
    default   = 280 if provider == "azure_openai" else 0
    rate      = int(os.getenv("LLM_RATE_LIMIT", str(default)))
    if rate > 0:
        limiter = RateLimiter(rate_per_min=rate)
        mem.llm.generate_response = limiter.wrap(mem.llm.generate_response)
    return mem


def _configure_entity_linking_for_benchmarks() -> None:
    """Disable entity-linking by default to avoid embedded Qdrant path lock conflicts."""
    enabled = os.getenv("MEM0_BENCH_ENABLE_ENTITY_LINKING", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        memory_main.extract_entities_batch = lambda texts: [[] for _ in texts]


# ─────────────────────────────────────────────────────────────────────────────
# Memory factories
# ─────────────────────────────────────────────────────────────────────────────
def build_memory() -> Memory:
    tmp        = tempfile.mkdtemp(prefix="mem0_bench_")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    _configure_entity_linking_for_benchmarks()
    config = MemoryConfig(
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config={
                "collection_name":     f"bench_{uuid.uuid4().hex[:8]}",
                "embedding_model_dims": EMBEDDING_DIMS,
                "path":                tmp,
            },
        ),
        llm=_build_llm_config(),
        embedder=EmbedderConfig(
            provider="ollama",
            config={"model": "nomic-embed-text", "ollama_base_url": ollama_url, "embedding_dims": EMBEDDING_DIMS},
        ),
        history_db_path=os.path.join(tmp, "history.db"),
    )
    return _apply_rate_limiter(Memory(config))


# ─────────────────────────────────────────────────────────────────────────────
# Seeded memory factory — points at pre-populated persistent collection
# ─────────────────────────────────────────────────────────────────────────────
SEED_DIR         = os.path.join(os.path.dirname(__file__), "data")
SEED_QDRANT_PATH = os.path.join(SEED_DIR, "seeded_qdrant")
SEED_COLLECTION  = "bench_seeded"
SEED_HISTORY_DB  = os.path.join(SEED_DIR, "history_seeded.db")


def build_seeded_memory() -> Memory:
    """
    Connect to the pre-seeded Qdrant collection created by seed.py.
    Run `python benchmarks/seed.py` once before using this.
    """
    skip_seed = os.getenv("MEM0_BENCH_SKIP_SEED", "0").strip().lower() in {"1", "true", "yes", "on"}
    if skip_seed:
        # Fallback mode for quick local iteration where realistic seeded data is not required.
        return build_memory()

    if not os.path.isdir(SEED_QDRANT_PATH):
        raise RuntimeError(
            f"Seeded collection not found at {SEED_QDRANT_PATH}.\n"
            "Run:  python benchmarks/seed.py"
        )
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    _configure_entity_linking_for_benchmarks()
    config = MemoryConfig(
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config={
                "collection_name":     SEED_COLLECTION,
                "embedding_model_dims": EMBEDDING_DIMS,
                "path":                SEED_QDRANT_PATH,
            },
        ),
        llm=_build_llm_config(),
        embedder=EmbedderConfig(
            provider="ollama",
            config={"model": "nomic-embed-text", "ollama_base_url": ollama_url, "embedding_dims": EMBEDDING_DIMS},
        ),
        history_db_path=SEED_HISTORY_DB,
    )
    return _apply_rate_limiter(Memory(config))


# ─────────────────────────────────────────────────────────────────────────────
# Generic method wrapper
# ─────────────────────────────────────────────────────────────────────────────
def wrap(method: Callable, before: Callable = None, after: Callable = None) -> Callable:
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
