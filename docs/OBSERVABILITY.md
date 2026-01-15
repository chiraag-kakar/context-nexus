# Observability Guide: Understanding Your Queries

> **See inside every query: timing, tokens, retrieval, and confidence scores**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Trace Data Reference](#trace-data-reference)
3. [Debugging with Traces](#debugging-with-traces)
4. [Performance Optimization](#performance-optimization)
5. [Monitoring in Production](#monitoring-in-production)
6. [Common Patterns](#common-patterns)

---

## Quick Start

Enable tracing by passing `trace=True` to any query:

```python
from context_nexus import ContextNexus, Agent

nexus = ContextNexus()
await nexus.ingest(["./docs/"])

agent = Agent(nexus, token_budget=8000)

# Run query WITH tracing
answer = await agent.query("What is RAG?", trace=True)

# Access trace information
print(answer.trace)
# Output:
# Trace(
#   steps=[...],
#   tokens_used=1523,
#   latency_ms=1847.3,
#   chunks_retrieved=5
# )
```

---

## Trace Data Reference

### The Trace Object

```python
@dataclass
class Trace:
    steps: list[dict]          # Execution steps with timing
    tokens_used: int           # Total tokens consumed
    latency_ms: float          # Total execution time
    chunks_retrieved: int      # Number of documents found
```

### Accessing Trace Information

```python
answer = await agent.query("Your question", trace=True)

if answer.trace:
    # Total metrics
    print(f"Total latency: {answer.trace.latency_ms}ms")
    print(f"Tokens used: {answer.trace.tokens_used}")
    print(f"Chunks found: {answer.trace.chunks_retrieved}")
    
    # Step-by-step breakdown
    print("\nExecution steps:")
    for step in answer.trace.steps:
        print(f"  {step['name']}: {step['duration_ms']:.1f}ms")
        if 'chunks_found' in step:
            print(f"    → Found {step['chunks_found']} chunks")
```

### Trace Steps Explained

Every query goes through these steps:

| Step | Duration | Meaning |
|------|----------|---------|
| **retrieve** | 50-500ms | Time to search documents |
| **generate** | 500-5000ms | Time for LLM to respond |

**Example output:**
```python
[
    {
        'name': 'retrieve',
        'duration_ms': 234.5,
        'chunks_found': 5
    },
    {
        'name': 'generate',
        'duration_ms': 1205.3
    }
]
```

---

## Debugging with Traces

### Problem: Query is Too Slow

```python
answer = await agent.query(question, trace=True)

# Check timing breakdown
total = answer.trace.latency_ms
retrieve_time = answer.trace.steps[0]['duration_ms']
generate_time = answer.trace.steps[1]['duration_ms']

print(f"Total: {total}ms")
print(f"  Retrieve: {retrieve_time}ms ({retrieve_time/total:.0%})")
print(f"  Generate: {generate_time}ms ({generate_time/total:.0%})")

# If retrieve > 500ms:
#   ❌ Too much data to search
#   Solutions:
#   1. Reduce number of chunks: nexus.config.chunk_size = 256
#   2. Use Qdrant instead of FAISS for large datasets

# If generate > 3000ms:
#   ❌ LLM is slow (usually not our problem)
#   Solutions:
#   1. Use faster LLM: gpt-4o-mini instead of gpt-4
#   2. Reduce context size: answer.trace.tokens_used
```

### Problem: Wrong Documents Retrieved

```python
answer = await agent.query(question, trace=True)

chunks = answer.trace.chunks_retrieved
print(f"Retrieved {chunks} chunks")

# If chunks < 3:
#   ❌ Not enough documents found
#   Solutions:
#   1. Check embedding quality: nexus.config.embedding.provider
#   2. Try hybrid retrieval (default)
#   3. Increase limit: nexus.retrieve(question, limit=20)

# If retrieved wrong documents:
#   ❌ Quality issue
#   Solutions:
#   1. Use better embedding model: OpenAI's text-embedding-3-small
#   2. Adjust chunking: smaller chunks = more precise
```

### Problem: Too Many Tokens Used

```python
answer = await agent.query(question, trace=True)

tokens = answer.trace.tokens_used
budget = agent.token_budget

print(f"Tokens: {tokens} / {budget}")

if tokens > budget * 0.8:
    print("⚠️ Approaching token budget limit")
    
    # Solutions:
    # 1. Reduce chunks: limit=5 instead of limit=10
    # 2. Use smaller model: gpt-4o-mini instead of gpt-4
    # 3. Compress context: enable compression mode
```

---

## Performance Optimization

### Baseline Performance

Typical latency breakdown (depends on your setup):

```
Query latency: ~1-2 seconds

┌─ Retrieval phase: 300-500ms
│  ├─ Embedding query: 100ms (local model)
│  ├─ Vector search: 150ms (FAISS)
│  └─ Graph search: 100ms
│
└─ Generation phase: 500-1500ms
   └─ LLM API call (slowest part)
```

### Optimization Checklist

#### 1. Embedding Model

```python
# Option A: Fast local (development)
config = Config(
    embedding=EmbeddingConfig(
        provider="local",
        model="all-MiniLM-L6-v2",
        dimensions=384
    )
)
# Latency: ~100ms per query

# Option B: High quality remote (production)
config = Config(
    embedding=EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536
    )
)
# Latency: ~100ms per query
# Quality: Better relevance
```

#### 2. Vector Store Configuration

```python
# Development: In-memory FAISS
nexus = ContextNexus(vector_store="faiss")
# Latency: 50-100ms
# Limit: ~1M embeddings per machine

# Production: Qdrant (distributed)
nexus = ContextNexus(vector_store="qdrant")
# Latency: 100-200ms (network overhead)
# Limit: Unlimited (distributed)
```

#### 3. Chunk Size Tuning

```python
# Small chunks: More precision, slower retrieval
nexus.config.chunk_size = 256
# Pros: More specific results
# Cons: More chunks to search

# Large chunks: Faster retrieval, less precision
nexus.config.chunk_size = 1024
# Pros: Fewer chunks, faster
# Cons: Less targeted results
```

#### 4. Retrieval Limit

```python
# Get only top results
results = await nexus.retrieve(question, limit=5)
# Much faster than limit=20

# But verify quality:
if len(results) < 3:
    # Might be missing relevant info
    # Increase limit to 10
```

### Benchmarking

Create a benchmark script:

```python
import asyncio
import time
from context_nexus import ContextNexus, Agent

async def benchmark():
    nexus = ContextNexus()
    await nexus.ingest(["./docs/"])
    
    agent = Agent(nexus)
    
    questions = [
        "What is RAG?",
        "How do embeddings work?",
        "What's a knowledge graph?",
    ]
    
    latencies = []
    token_counts = []
    
    for question in questions:
        start = time.time()
        answer = await agent.query(question, trace=True)
        elapsed = time.time() - start
        
        latencies.append(answer.trace.latency_ms)
        token_counts.append(answer.trace.tokens_used)
        
        print(f"Q: {question[:40]}...")
        print(f"  Latency: {answer.trace.latency_ms:.0f}ms")
        print(f"  Tokens: {answer.trace.tokens_used}")
        print()
    
    # Summary
    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(token_counts) / len(token_counts)
    
    print(f"Average latency: {avg_latency:.0f}ms")
    print(f"Average tokens: {avg_tokens:.0f}")

asyncio.run(benchmark())
```

---

## Monitoring in Production

### Logging Queries

```python
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

async def logged_query(question: str, user_id: str = None):
    """Query with comprehensive logging"""
    start = time.time()
    
    answer = await agent.query(question, trace=True)
    elapsed = time.time() - start
    
    # Log structured data
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "question": question,
        "answer_length": len(answer.text),
        "confidence": answer.confidence,
        "latency_ms": answer.trace.latency_ms,
        "tokens_used": answer.trace.tokens_used,
        "chunks_retrieved": answer.trace.chunks_retrieved,
        "sources_count": len(answer.sources),
    }
    
    logger.info("query_completed", extra=log_entry)
    
    return answer
```

### Health Checks

```python
async def health_check():
    """Verify system is working"""
    
    # Test 1: Can we retrieve?
    try:
        results = await nexus.retrieve("test query", limit=1)
        retrieval_ok = len(results) > 0
    except Exception as e:
        retrieval_ok = False
        logger.error(f"Retrieval failed: {e}")
    
    # Test 2: Can we generate?
    try:
        answer = await agent.query("What is 2+2?", trace=True)
        generation_ok = len(answer.text) > 0
    except Exception as e:
        generation_ok = False
        logger.error(f"Generation failed: {e}")
    
    # Test 3: Are we within budgets?
    last_answer = await agent.query("test")
    tokens_ok = last_answer.trace.tokens_used < agent.token_budget
    latency_ok = last_answer.trace.latency_ms < 5000
    
    health = {
        "retrieval": retrieval_ok,
        "generation": generation_ok,
        "tokens": tokens_ok,
        "latency": latency_ok,
        "overall": all([retrieval_ok, generation_ok, tokens_ok, latency_ok])
    }
    
    return health
```

### Alerting Rules

```python
# Alert if these metrics exceed thresholds:

ALERTS = {
    "high_latency": {
        "threshold_ms": 5000,
        "condition": "answer.trace.latency_ms > threshold"
    },
    "poor_quality": {
        "threshold": 0.5,
        "condition": "answer.confidence < threshold"
    },
    "token_overflow": {
        "threshold_pct": 0.9,
        "condition": "answer.trace.tokens_used > agent.token_budget * 0.9"
    },
    "no_results": {
        "threshold": 1,
        "condition": "answer.trace.chunks_retrieved < threshold"
    },
}
```

---

## Common Patterns

### Pattern 1: Confidence-Based Refinement

```python
async def smart_query(question: str):
    """Retry if confidence is low"""
    
    answer = await agent.query(question, trace=True)
    
    if answer.confidence < 0.7:
        print(f"Low confidence: {answer.confidence:.0%}")
        print("Retrying with more context...")
        
        # Ask with more retrieved documents
        results = await nexus.retrieve(question, limit=20)
        answer = await agent.query(question, trace=True)
    
    return answer
```

### Pattern 2: Performance Tracking

```python
import statistics

latencies = []
token_usage = []

for i in range(10):
    answer = await agent.query(test_question, trace=True)
    latencies.append(answer.trace.latency_ms)
    token_usage.append(answer.trace.tokens_used)

# Statistics
print(f"Latency - mean: {statistics.mean(latencies):.0f}ms, "
      f"p95: {statistics.quantiles(latencies, n=20)[18]:.0f}ms")
print(f"Tokens - mean: {statistics.mean(token_usage):.0f}, "
      f"max: {max(token_usage)}")
```

### Pattern 3: Cost Estimation

```python
async def estimate_costs(num_queries: int):
    """Estimate monthly costs"""
    
    # Run sample queries
    token_totals = []
    for _ in range(10):
        answer = await agent.query("test", trace=True)
        token_totals.append(answer.trace.tokens_used)
    
    avg_tokens = sum(token_totals) / len(token_totals)
    monthly_tokens = avg_tokens * num_queries
    
    # OpenAI pricing
    embedding_cost = (monthly_tokens * 0.02) / 1_000_000  # text-embedding-3-small
    generation_tokens = monthly_tokens * 3  # Rough estimate
    llm_cost = (generation_tokens * 0.003) / 1_000_000    # gpt-4o-mini
    
    total_cost = embedding_cost + llm_cost
    
    print(f"Monthly: {num_queries:,} queries")
    print(f"Average tokens per query: {avg_tokens:.0f}")
    print(f"Total tokens: {monthly_tokens:,}")
    print(f"Estimated cost: ${total_cost:.2f}/month")
```

### Pattern 4: Comparing Retrieval Methods

```python
async def compare_retrievals(question: str):
    """Compare vector vs graph vs hybrid"""
    
    methods = ["vector", "graph", "hybrid"]
    results_by_method = {}
    
    for method in methods:
        results = await nexus.retrieve(question, mode=method, limit=5)
        results_by_method[method] = {
            "count": len(results),
            "top_scores": [r.score for r in results[:3]]
        }
    
    # Compare
    for method, data in results_by_method.items():
        print(f"{method}: {data['count']} results, "
              f"top scores: {[f'{s:.3f}' for s in data['top_scores']]}")
```

---

## Trace Data Dictionary

### trace.steps

Each step in `trace.steps` contains:

```python
{
    "name": str,           # "retrieve" or "generate"
    "duration_ms": float,  # How long this step took
    "chunks_found": int,   # (retrieve only) Documents retrieved
}
```

### trace.tokens_used

Estimated tokens consumed in the query:

```
= tokens in question
+ tokens in retrieved context
+ tokens in generated answer
```

**Note:** This is an estimate. Actual tokens may vary slightly depending on tokenizer.

### trace.latency_ms

Total wall-clock time from start of query to end:

```
= retrieve time
+ generate time
+ overhead (<10ms)
```

### trace.chunks_retrieved

Number of document chunks retrieved for this query.

```
Higher = more context
Lower = more focused
```

---

## FAQ

**Q: Why doesn't my trace show individual token counts per chunk?**
A: We show aggregate tokens to keep the trace simple. Use `len(chunk.content) / 4` for rough estimates.

**Q: Can I export traces to external systems?**
A: Yes! The trace is a dataclass. Serialize to JSON and send to your monitoring system.

**Q: What if I don't want to trace (privacy concern)?**
A: Pass `trace=False` (default). No trace data is collected.

**Q: Should I trace every query in production?**
A: No, it adds ~5-10% overhead. Sample 10-20% of queries or trace only slow/failed queries.

**Q: How do I compare performance across configurations?**
A: Run the same 10 questions with each config, compare `latency_ms` and `tokens_used` distributions.

---

## Next Steps

- [Blog: Performance Optimization](./blog.md#performance-optimization)
- [Use Cases: Production Patterns](./use_cases.md)
- [Examples: Benchmark Script](../examples/05_benchmark.py)
