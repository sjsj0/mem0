"""
Mem0 Batching Experiments
=========================

Goal:
  Evaluate whether two architecture changes help under load, without changing
  production code outside benchmarks:

  1) ACK-first ingestion:
     - Client gets an immediate ACK.
     - Request is processed in a background queue.

  2) LLM micro-batching:
     - Group nearby LLM calls and process as a batch.

These experiments are benchmark-only prototypes to estimate impact.

Outputs:
  Saved under benchmarks/batching/graphs/ as PNG and JSON.

Scripts:
  - exp_async_ack_queue.py
    Compares synchronous write latency vs ACK latency and end-to-end latency
    with background processing.

  - exp_llm_microbatch.py
    Compares unbatched vs micro-batched LLM call handling using a benchmark
    LLM simulator (valid JSON response), to estimate throughput/latency impact.

Quick start:
  cd d:/Masters/Spring 26/Storage Systems/Project/mem0

  python benchmarks/batching/exp_async_ack_queue.py --requests 40 --ingress-concurrency 8 --workers 4

  python benchmarks/batching/exp_llm_microbatch.py --requests 60 --concurrency 1 4 8 16 --batch-window-ms 10 --max-batch-size 8
"""
