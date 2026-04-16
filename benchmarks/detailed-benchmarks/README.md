"""
Mem0 Detailed Benchmarks (Deep Profiling)
==========================================
This folder contains 4 deep-profiling experiments that measure aspects of Mem0 performance
that the standard benchmarks folder does NOT reveal.

✅ Folder: benchmarks/detailed-benchmarks/

1. **exp_llm_breakdown.py** - LLM Network vs Inference Time
   Measures: what fraction of LLM latency is network roundtrip vs actual model inference?
   Output: graphs/exp_llm_breakdown.png, graphs/exp_llm_breakdown_results.json

2. **exp_memory_patterns.py** - Memory Usage Patterns
   Measures: RSS and heap memory during concurrent requests, peak memory, trends
   Output: graphs/exp_memory_patterns.png, graphs/exp_memory_patterns_results.json

3. **exp_batching.py** - Batching Impact
   Measures: throughput and latency for serial vs pseudo-batched (concurrent) requests
   Output: graphs/exp_batching.png, graphs/exp_batching_results.json

4. **exp_memory_scaling.py** - Performance Scaling with Memory Size
   Measures: how latency and throughput degrade as pre-existing memories grow
   Output: graphs/exp_memory_scaling.png, graphs/exp_memory_scaling_results.json

---

Prerequisites:
===============
1. Install Mem0 in development mode:
   $ pip install -e .

2. Install optional dependencies:
   $ pip install psutil ollama

3. Start Ollama server (in a separate terminal):
   $ ollama serve

4. Pull the required models:
   $ ollama pull llama3.2:1b
   $ ollama pull nomic-embed-text

---

Quick Start - Run All Experiments:
===================================

# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run experiments
cd d:\Masters\Spring 26\Storage Systems\Project\mem0

# Experiment A: LLM Breakdown
python benchmarks/detailed-benchmarks/exp_llm_breakdown.py --requests 15

# Experiment B: Memory Patterns
python benchmarks/detailed-benchmarks/exp_memory_patterns.py --concurrency 1 4 8 --requests 10

# Experiment C: Batching Impact
python benchmarks/detailed-benchmarks/exp_batching.py --requests 30

# Experiment D: Memory Scaling
python benchmarks/detailed-benchmarks/exp_memory_scaling.py --memory-sizes 100 500 1000 --requests-per-level 10

# Run all experiments at once (sequential):
python benchmarks/detailed-benchmarks/exp_llm_breakdown.py --requests 10 && \
python benchmarks/detailed-benchmarks/exp_memory_patterns.py --concurrency 1 4 --requests 10 && \
python benchmarks/detailed-benchmarks/exp_batching.py --requests 20 && \
python benchmarks/detailed-benchmarks/exp_memory_scaling.py --memory-sizes 100 500 1000 --requests-per-level 10

---

View Results:
==============
All graphs and JSON results are saved to benchmarks/detailed-benchmarks/graphs/:
  - exp_llm_breakdown.png
  - exp_llm_breakdown_results.json
  - exp_memory_patterns.png
  - exp_memory_patterns_results.json
  - exp_batching.png
  - exp_batching_results.json
  - exp_memory_scaling.png
  - exp_memory_scaling_results.json

---

Detailed Usage by Experiment:
=============================

Experiment A: LLM Breakdown (Network vs Inference)
---------------------------------------------------
Description:
  Separates LLM latency into network overhead and model inference time.
  Helps identify if optimization should focus on network efficiency or model speed.

Commands:
  # Measure 10 requests
  python benchmarks/detailed-benchmarks/exp_llm_breakdown.py --requests 10

  # Fast sanity check
  python benchmarks/detailed-benchmarks/exp_llm_breakdown.py --quick

  # Measure 20 requests with detailed output
  python benchmarks/detailed-benchmarks/exp_llm_breakdown.py --requests 20

  # Custom output paths
  python benchmarks/detailed-benchmarks/exp_llm_breakdown.py \
    --requests 15 \
    --output benchmarks/detailed-benchmarks/graphs/my_llm_results.json \
    --plot benchmarks/detailed-benchmarks/graphs/my_llm_plot.png

Expected output:
  - benchmarks/detailed-benchmarks/graphs/exp_llm_breakdown.png
  - benchmarks/detailed-benchmarks/graphs/exp_llm_breakdown_results.json

Interpretation:
  - Network overhead > 25%? → Consider request batching or persistent connections
  - Network overhead < 10%? → LLM inference is the main bottleneck

---

Experiment B: Memory Patterns (RSS & Heap Tracking)
---------------------------------------------------
Description:
  Tracks resident set size (RSS) and heap memory during concurrent requests.
  Detects memory leaks, spikes, and sustained high usage.

Commands:
  # Test with concurrency 1 and 4, 10 requests each
  python benchmarks/detailed-benchmarks/exp_memory_patterns.py --concurrency 1 4 --requests 10

  # Test with higher concurrency
  python benchmarks/detailed-benchmarks/exp_memory_patterns.py --concurrency 1 2 4 8 16 --requests 20

  # Custom output
  python benchmarks/detailed-benchmarks/exp_memory_patterns.py \
    --concurrency 1 4 8 \
    --requests 15 \
    --output benchmarks/detailed-benchmarks/graphs/mem_results.json \
    --plot benchmarks/detailed-benchmarks/graphs/mem_plot.png

Expected output:
  - benchmarks/detailed-benchmarks/graphs/exp_memory_patterns.png  (time-series plot)
  - benchmarks/detailed-benchmarks/graphs/exp_memory_patterns_results.json

Interpretation:
  - Peak memory >> average memory? → Spiky allocations (normal for LLM)
  - Peak memory grows with concurrency? → Parallel allocations accumulate
  - Sustained high memory at end? → Possible memory leak

---

Experiment C: Batching Impact (Serial vs Concurrent)
---------------------------------------------------
Description:
  Compares single-threaded requests vs high-concurrency pseudo-batching.
  Shows potential throughput gains from true batching implementation.

Commands:
  # Run 20 total requests (serial vs concurrent)
  python benchmarks/detailed-benchmarks/exp_batching.py --requests 20

  # Run 50 requests
  python benchmarks/detailed-benchmarks/exp_batching.py --requests 50

  # Custom output
  python benchmarks/detailed-benchmarks/exp_batching.py \
    --requests 30 \
    --output benchmarks/detailed-benchmarks/graphs/batch_results.json \
    --plot benchmarks/detailed-benchmarks/graphs/batch_plot.png

Expected output:
  - benchmarks/detailed-benchmarks/graphs/exp_batching.png  (throughput + latency comparison)
  - benchmarks/detailed-benchmarks/graphs/exp_batching_results.json

Interpretation:
  - Concurrent throughput > 2× serial? → Significant batching opportunity
  - Concurrent throughput ≈ serial? → Already near optimal parallelism
  - Concurrent p99 >> serial p99? → Tail latency may suffer under load

---

Experiment D: Memory Scaling (Latency vs Data Size)
---------------------------------------------------
Description:
  Pre-populates memory with various amounts of data, then measures latency
  for new requests. Shows how performance degrades as the database grows.

Commands:
  # Test with 100, 500, 1000 pre-existing memories
  python benchmarks/detailed-benchmarks/exp_memory_scaling.py \
    --memory-sizes 100 500 1000 \
    --requests-per-level 10

  # Larger scale test
  python benchmarks/detailed-benchmarks/exp_memory_scaling.py \
    --memory-sizes 100 500 1000 2000 5000 \
    --requests-per-level 15

  # Quick test (smaller datasets)
  python benchmarks/detailed-benchmarks/exp_memory_scaling.py \
    --memory-sizes 50 100 200 \
    --requests-per-level 5

  # Custom output
  python benchmarks/detailed-benchmarks/exp_memory_scaling.py \
    --memory-sizes 100 500 1000 \
    --requests-per-level 10 \
    --output benchmarks/detailed-benchmarks/graphs/scaling_results.json \
    --plot benchmarks/detailed-benchmarks/graphs/scaling_plot.png

Expected output:
  - benchmarks/detailed-benchmarks/graphs/exp_memory_scaling.png  (latency + throughput degradation)
  - benchmarks/detailed-benchmarks/graphs/exp_memory_scaling_results.json

Interpretation:
  - Latency grows 10% per 2× memory?     → Logarithmic scaling (good)
  - Latency grows 50% per 2× memory?     → Linear scaling (acceptable)
  - Latency grows 100%+ per 2× memory?   → Quadratic or worse (concerning)

---

Interpreting Results:
=====================

LLM Breakdown:
  - Network overhead >25%? → Consider request batching or persistent connections
  - <10%?                → LLM inference is the main bottleneck, focus on model optimization

Memory Patterns:
  - Peak memory >> average?  → Spiky allocations during LLM/embedding calls
  - Peak > baseline by >100%? → Possible memory leak or unbounded cache
  - Stable over time?        → Good, no memory leaks

Batching:
  - Throughput gain >50%? → Significant potential for optimization
  - Gain <10%?           → Already near optimal parallelism
  - Latency increase?    → Tail latency (p99) may suffer under high concurrency

Memory Scaling:
  - Latency grows 10% per doubling of memory? → Logarithmic (good)
  - Latency grows 50% per doubling?          → Linear (might be concerning at large scale)
  - Latency grows 2× per doubling?           → Potentially problematic (quadratic or worse)

---

Common Issues:
==============
- "No module named 'psutil'": install with `pip install psutil`
- "Connection refused" to Ollama: ensure `ollama serve` is running
- Benchmarks too slow: reduce --requests or --memory-sizes
- "ImportError: No package metadata": run `pip install -e .` in the repo root

---

Contributing:
==============
To add a new detailed benchmark experiment:
1. Copy one of the exp_*.py templates
2. Modify the hypothesis, measurement logic, and plots
3. Add corresponding section in this README
4. Ensure it saves JSON results and PNG graphs to the graphs/ folder
