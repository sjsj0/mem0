# Update: Cross-Request LLM Batching Support

This update introduces high-throughput batching for LLM inference calls across concurrent memory operations. It is specifically designed to optimize performance when multiple agents or users are interacting with `mem0` simultaneously.

## New Features

### 1. Global Request Bundling
A new `LLMBatchBundler` has been added to the `Memory` class. It intercepts LLM extraction requests from concurrent `add()` calls and bundles them into a single request. 
- **Latency Optimization**: Collects requests within a configurable time window (default 100ms).
- **Throughput Gains**: Reduces total HTTP requests to the LLM backend by up to your `batch_size`.

### 2. Specialized Batch Endpoint Support
Added `generate_batch_endpoint_response` to the LLM provider layer, with native support for:
- **vLLM**: Optimized for vLLM's `batched_chat_completions` format (list of message sets).
- **OpenAI-Compatible Servers**: Support for hitting specialized `/batch` endpoints using a single POST request.

### 3. Automatic Synchronization
The `add()` method remains synchronous for the end-user. The bundling layer handles thread synchronization so that each caller receives their specific portion of the batched result once the global batch completes.

## Configuration

You can enable batching by setting environment variables or updating your `LlmConfig`.

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `MEM0_BATCH_INFERENCE_URL` | The URL of the specialized batch endpoint | None (Disabled) |
| `MEM0_BATCH_SIZE` | Maximum number of requests to bundle | 10 |
| `MEM0_BATCH_TIMEOUT` | Time to wait before flushing a partial batch | 0.1s |

### Example Usage (vLLM)

```python
from mem0 import Memory
import os

os.environ["MEM0_BATCH_INFERENCE_URL"] = "http://localhost:8000/v1/batched_chat/completions"

memory = Memory()

# If multiple threads call this concurrently, they will be batched
memory.add("First secret fact", user_id="user_1") 
```

## Architectural Changes
- **`mem0.memory.main.py`**: Added `LLMBatchBundler` and updated `Memory` initialization.
- **`mem0.llms.base.py`**: Added new batching interfaces to `LLMBase`.
- **`mem0.llms.openai.py` / `mem0.llms.vllm.py`**: Implemented provider-specific batch HTTP logic.
