# Mem0 Write Pipeline Benchmarks

Benchmarks for the Mem0 OSS `Memory.add()` write pipeline. Each experiment targets a specific hypothesis about where the bottleneck is and how the system behaves under load.

## Setup

### Requirements

```bash
pip install mem0ai matplotlib numpy psutil azure-identity
```

**Ollama** (for embeddings):
- Install the native Mac app from [ollama.com](https://ollama.com) (arm64 for M-series)
- Pull the embedding model: `ollama pull nomic-embed-text`

**Azure OpenAI** (for LLM):
- Fill in `benchmarks/.env` with your Azure credentials
- Load before running: `export $(cat benchmarks/.env | grep -v '^#' | xargs)`

### Seed the collection (run once)

All experiments run against a pre-populated Qdrant collection (10k memories per user, 5 users = 50k vectors). Seeding uses random vectors so no API calls are needed — takes ~30 seconds.

```bash
cd /Users/saraagarwal/mem0
python benchmarks/seed.py --memories-per-user 10000 --users 5
```

If you want to skip seeding for quick local runs, set:

```bash
export MEM0_BENCH_SKIP_SEED=1
```

This falls back to a fresh local collection for experiments that normally require seeded data.

---

## Experiments

### exp_trace — Single Write Gantt Chart

**File:** `benchmarks/exp_trace.py`  
**Hypothesis:** A single write is a mostly sequential chain of remote calls dominated by the LLM.

Wraps every service call (LLM, embedder, vector store, SQLite) with timers and plots a Gantt chart showing exactly where time goes in one `Memory.add()`. X-axis is wall time, each bar is one phase.

```bash
python benchmarks/exp_trace.py --runs 1
```

**Output:** `benchmarks/graphs/exp_trace.png`

---

### exp_trace_realistic — Gantt Chart (10k memories/user)

**File:** `benchmarks/exp_trace_realistic.py`  
**Same as exp_trace but against the pre-seeded 10k collection.** VS Search now scans real vectors, and the LLM prompt includes top-10 existing memories — a realistic production scenario.

```bash
python benchmarks/exp_trace_realistic.py --runs 1
```

**Output:** `benchmarks/graphs/exp_trace_realistic.png`

---

### exp_throughput — Writes/sec vs Concurrency

**File:** `benchmarks/exp_throughput.py`  
**Hypothesis:** Throughput plateaus early because each write fires several sequential remote RPCs. More writers just pile up — they don't reduce per-request work.

Sweeps concurrency from 1 → 64 writers and records writes/sec and tail latency (p50/p95/p99) at each level.

```bash
python benchmarks/exp_throughput.py --requests 20 --concurrency 1 2 4 8 16 32 64
```

**Output:** `benchmarks/graphs/exp_throughput.png`, `exp_throughput_results.json`

---

### exp_resources — CPU & Memory vs Concurrency

**File:** `benchmarks/exp_resources.py`  
**Hypothesis:** Local resources stay underutilized even as latency rises — the bottleneck is I/O wait (remote LLM calls), not compute.

Same concurrency sweep as exp_throughput but also samples system CPU% and process memory via psutil every 500ms. Plots throughput and CPU on dual axes — if CPU stays flat while throughput plateaus, the bottleneck is confirmed as I/O.

```bash
python benchmarks/exp_resources.py --requests 20 --concurrency 1 2 4 8 16 32
```

**Output:** `benchmarks/graphs/exp_resources.png`, `exp_resources_results.json`

---

### exp_inject — Sensitivity to Remote API Latency

**File:** `benchmarks/exp_inject.py`  
**Hypothesis:** Small increases in remote inference latency disproportionately hurt write tail latency — the architecture is network-bound.

Injects an artificial sleep before every LLM and embedding call at increasing delays (0, 50, 100, 200, 500ms), simulating a degraded or overloaded API. Measures how steeply p95/p99 rises as a function of injected delay.

```bash
python benchmarks/exp_inject.py --requests 10 --delays 0 50 100 200 500
```

**Output:** `benchmarks/graphs/exp_inject.png`, `exp_inject_results.json`

---

### exp_queue — In-Flight RPC Queue Buildup

**File:** `benchmarks/exp_queue.py`  
**Hypothesis:** As write concurrency increases, LLM calls pile up as independent in-flight requests — no batching or queuing is happening.

Instruments LLM and embedding calls with atomic in-flight counters, sampled every 50ms by a background thread. Shows how many simultaneous LLM calls are in-flight during a concurrent write burst.

```bash
python benchmarks/exp_queue.py --requests 20 --concurrency 1 4 16 32
```

**Output:** `benchmarks/graphs/exp_queue.png`, `exp_queue_results.json`

---

## Run All

```bash
cd /Users/saraagarwal/mem0
export $(cat benchmarks/.env | grep -v '^#' | xargs)

python benchmarks/exp_trace_realistic.py --runs 1
python benchmarks/exp_throughput.py --requests 20 --concurrency 1 2 4 8 16 32 64
python benchmarks/exp_resources.py --requests 20 --concurrency 1 2 4 8 16 32
python benchmarks/exp_inject.py --requests 10 --delays 0 50 100 200 500
python benchmarks/exp_queue.py --requests 20 --concurrency 1 4 16 32
```

All plots are saved to `benchmarks/graphs/`. JSON result files are saved alongside the plots.

---

## Architecture

```
common.py          — shared Memory factory (Azure LLM + Ollama embedder + Qdrant embedded)
seed.py            — one-time seeding of 10k memories/user into persistent Qdrant collection
benchmarks/data/   — seeded Qdrant collection (gitignored, generated by seed.py)
benchmarks/graphs/ — all PNG plots and JSON result files
```

**LLM:** Azure OpenAI (`gpt-4.1-mini`)  
**Embedder:** Ollama (`nomic-embed-text`, 768 dims, local)  
**Vector store:** Qdrant embedded (local file, no server)  
**History DB:** SQLite (local)
