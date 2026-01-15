# Context Nexus - Architecture Document

> **Version**: 1.0 | **Status**: Draft | **Last Updated**: 2026-01-15

---

## 1. System Overview

### 1.1 High-Level Architecture

![Context Nexus System Architecture](./images/system_architecture.png)

### 1.2 Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Control Plane / Data Plane Separation** | Python for flexibility & ecosystem; Rust for performance |
| **Async-First** | GenAI is I/O-bound; async enables concurrency |
| **Fail-Fast with Recovery** | Explicit errors, circuit breakers, graceful degradation |
| **Observable by Default** | Tracing, metrics, and evaluation built-in |
| **Pluggable Backends** | No vendor lock-in; abstract interfaces |

---

## 2. Component Architecture

### 2.1 Python Control Plane

```
context_nexus/
├── __init__.py           # Public API exports
├── core/
│   ├── config.py         # Configuration management
│   ├── exceptions.py     # Exception hierarchy
│   └── types.py          # Core type definitions
├── ingestion/
│   ├── ingester.py       # Main ingestion orchestrator
│   ├── loaders/          # Document loaders (PDF, HTML, etc.)
│   ├── chunkers/         # Chunking strategies
│   └── extractors/       # Metadata extractors
├── retrieval/
│   ├── base.py           # Abstract retriever interface
│   ├── vector.py         # Vector retrieval
│   ├── graph.py          # Graph traversal
│   ├── hybrid.py         # Hybrid fusion
│   └── reranker.py       # Cross-encoder reranking
├── context/
│   ├── window.py         # Context window management
│   ├── budget.py         # Token budgeting
│   └── compressor.py     # Context compression
├── agents/
│   ├── agent.py          # Agent abstraction
│   ├── workflow.py       # LangGraph-compatible workflows
│   └── tools.py          # Tool definitions
├── observability/
│   ├── tracer.py         # Tracing (LangSmith-compatible)
│   ├── metrics.py        # Metrics collection
│   └── evaluator.py      # Evaluation framework
└── integrations/
    ├── langchain.py      # LangChain adapters
    ├── langgraph.py      # LangGraph integration
    └── mcp.py            # MCP server (future)
```

### 2.2 Rust Data Plane

```
aegis_core/
├── Cargo.toml
├── src/
│   ├── lib.rs            # PyO3 module exports
│   ├── token/
│   │   ├── mod.rs
│   │   ├── counter.rs    # Fast token counting (tiktoken-compatible)
│   │   ├── budget.rs     # Token budget enforcement
│   │   └── truncator.rs  # Smart truncation
│   ├── vector/
│   │   ├── mod.rs
│   │   ├── scorer.rs     # Similarity scoring (SIMD-optimized)
│   │   ├── fusion.rs     # Score fusion (RRF, weighted)
│   │   └── filter.rs     # Metadata filtering
│   ├── graph/
│   │   ├── mod.rs
│   │   ├── traversal.rs  # BFS/DFS with cutoffs
│   │   ├── expander.rs   # Multi-hop expansion
│   │   └── scorer.rs     # Graph relevance scoring
│   ├── chunk/
│   │   ├── mod.rs
│   │   ├── splitter.rs   # Text splitting algorithms
│   │   ├── merger.rs     # Chunk merging
│   │   └── dedup.rs      # Deduplication (SimHash)
│   └── compress/
│       ├── mod.rs
│       ├── ranker.rs     # Relevance ranking
│       └── pruner.rs     # Context pruning
```

---

## 3. Python ↔ Rust Boundary

### 3.1 Boundary Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Minimize crossings** | Batch operations, not per-item |
| **Zero-copy where possible** | Use PyO3's buffer protocol |
| **Release GIL aggressively** | All Rust computations |
| **Clear ownership** | Rust owns data during computation |

### 3.2 Python → Rust API

```python
# Python side (type-safe interface)
from context_nexus._core import (
    count_tokens,           # str -> int
    score_vectors,          # (query, docs) -> scores
    fuse_scores,            # (scores_a, scores_b, weights) -> fused
    traverse_graph,         # (edges, start, max_hops) -> nodes
    chunk_text,             # (text, strategy) -> chunks
    compress_context,       # (chunks, budget) -> compressed
)
```

```rust
// Rust side (PyO3 exports)
#[pyfunction]
fn count_tokens(text: &str, model: &str) -> PyResult<usize> {
    Ok(tokenizer::count(text, model)?)
}

#[pyfunction]
fn score_vectors<'py>(
    py: Python<'py>,
    query: PyReadonlyArrayDyn<f32>,
    documents: PyReadonlyArrayDyn<f32>,
) -> PyResult<&'py PyArray1<f32>> {
    py.allow_threads(|| {
        scorer::cosine_batch(query, documents)
    })
}
```

### 3.3 Performance Boundaries

| Operation | Language | Rationale |
|-----------|----------|-----------|
| LLM API calls | Python | I/O-bound, ecosystem support |
| Vector DB queries | Python | Client libraries |
| Score computation | **Rust** | CPU-bound, SIMD |
| Token counting | **Rust** | Called millions of times |
| Graph traversal | **Rust** | Memory-intensive, recursive |
| Chunking | **Rust** | Large text processing |
| Compression ranking | **Rust** | Sorting, filtering |
| Workflow logic | Python | Flexibility, LangGraph |
| Config parsing | Python | Simplicity |
| Tracing | Python | Ecosystem (OpenTelemetry) |

---

## 4. Data Flow

### 4.1 Ingestion Flow

![Document Ingestion Pipeline](./images/ingestion_flow.png)

### 4.2 Retrieval Flow

![Hybrid Retrieval Pipeline](./images/retrieval_flow.png)

### 4.3 Agent Execution Flow

![Agent Execution Workflow](./images/agent_execution_flow.png)

---

## 5. Technology Decisions & Tradeoffs

### 5.1 Vector Database Selection

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Qdrant** | Rust-native, filtering, gRPC | Newer, smaller community | **Primary** |
| FAISS | Fast, well-tested, local | No filtering, no persistence | Fallback |
| Pinecone | Managed, scalable | Vendor lock-in, cost | Optional |
| Milvus | Feature-rich | Complex deployment | Optional |

**Decision**: Qdrant as primary (performance + filtering), FAISS for local/testing.

### 5.2 Graph Database Selection

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Neo4j** | Mature, Cypher, community | Heavy, licensing | **Primary** |
| NetworkX | Simple, in-memory | Not scalable | Testing |
| Neptune | AWS-managed | Vendor lock-in | Enterprise |

**Decision**: Neo4j for production, NetworkX for local development.

### 5.3 Async vs Sync

**Decision**: **Async-first with sync wrappers**

```python
# Primary API (async)
result = await nexus.retrieve(query)

# Convenience wrapper (sync)
result = nexus.retrieve_sync(query)  # Uses asyncio.run()
```

### 5.4 Caching Strategy

| Cache Layer | TTL | Use Case |
|-------------|-----|----------|
| **Embedding cache** | 24h | Avoid re-embedding unchanged content |
| **Query cache** | 5min | Repeated similar queries |
| **LLM response cache** | 1h | Deterministic queries |

---

## 6. Failure Modes & Mitigations

### 6.1 Failure Taxonomy

![Failure Taxonomy](./images/failure_taxonomy.png)

### 6.2 Detailed Failure Analysis

#### Embedding API Rate Exhaustion

| Aspect | Detail |
|--------|--------|
| **Scenario** | Burst ingestion exceeds API rate limits |
| **Detection** | 429 responses, latency spike |
| **Impact** | Ingestion stalls, backlog grows |
| **Mitigation** | Token bucket rate limiter, exponential backoff, queue with backpressure |

```python
class RateLimitedEmbedder:
    def __init__(self, rate_limit: int = 1000):
        self.limiter = TokenBucket(rate_limit, per="minute")
        self.queue = BackpressureQueue(max_size=10000)
    
    async def embed(self, texts: list[str]) -> list[Embedding]:
        async with self.limiter:
            try:
                return await self._embed_batch(texts)
            except RateLimitError:
                await self.queue.backoff()
                raise RetryableError()
```

#### Vector DB Tail Latency Spikes

| Aspect | Detail |
|--------|--------|
| **Scenario** | P99 latency exceeds SLA (> 500ms) |
| **Mitigation** | Timeout with fallback, hedged requests, circuit breaker |

```python
class ResilientVectorStore:
    async def search(self, query: Embedding) -> list[Result]:
        async with self.circuit_breaker:
            try:
                return await asyncio.wait_for(self._search(query), timeout=0.3)
            except asyncio.TimeoutError:
                return await self._fallback_search(query)
```

#### Token Explosion (Prompt Chaining)

| Aspect | Detail |
|--------|--------|
| **Scenario** | Multi-step agent accumulates unbounded context |
| **Mitigation** | Hard token ceiling per step, compression checkpoints |

#### Graph Traversal Explosion

| Aspect | Detail |
|--------|--------|
| **Scenario** | Highly connected node causes O(n^k) expansion |
| **Mitigation** | Bounded BFS (max_depth, max_nodes), early termination |

```rust
pub fn traverse(
    graph: &Graph, start: NodeId, max_depth: usize, max_nodes: usize,
) -> TraversalResult {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((start, 0));
    
    while let Some((node, depth)) = queue.pop_front() {
        if depth > max_depth || visited.len() >= max_nodes {
            return TraversalResult::partial(visited);
        }
        // ... traversal logic
    }
    TraversalResult::complete(visited)
}
```

---

## 7. Scaling Strategies

### 7.1 Horizontal Scaling Architecture

![Horizontal Scaling Architecture](./images/horizontal_scaling.png)

### 7.2 Component Scaling Matrix

| Component | Scaling Method | Bottleneck | Mitigation |
|-----------|----------------|------------|------------|
| **Ingestion** | Worker pool + queue | Embedding API rate | Batching, rate limiting |
| **Vector DB** | Read replicas, sharding | Memory, IOPS | Tiered storage |
| **Graph DB** | Read replicas | CPU (traversal) | Query caching |
| **LLM Calls** | Concurrency limits | API rate limits | Multiple providers |
| **Context Processing** | Thread pool (Rust) | CPU | SIMD optimization |

### 7.3 Capacity Planning

```
Ingestion: 10,000 docs/hour → 10 workers (embedding rate limited to 1,000/min)
Retrieval: p50 < 100ms, p99 < 500ms → Vector(30ms) + Graph(50ms) + Fusion(5ms)
Concurrency: 1,000 concurrent users → 5 instances at 200 req/s each
```

---

## 8. Observability Architecture

### 8.1 Key Metrics

| Category | Metric | Alert Threshold |
|----------|--------|-----------------|
| **Latency** | retrieval_latency_p99 | > 500ms |
| **Throughput** | ingestion_docs_per_min | < 100 |
| **Errors** | llm_error_rate | > 5% |
| **Cost** | tokens_per_query | > 10,000 |
| **Quality** | retrieval_relevance_score | < 0.7 |

### 8.2 LangSmith Integration

```python
from context_nexus import Tracer

tracer = Tracer(backend="langsmith", project="context-nexus-prod")

with tracer.span("agent_run") as span:
    result = await agent.run(query)
    span.set_attribute("tokens", result.tokens_used)
```

---

## 9. Security Model

### 9.1 Multi-Tenant Isolation

| Layer | Isolation Method |
|-------|------------------|
| **Data** | Namespace prefixes, row-level security |
| **Compute** | Separate worker pools per tenant |
| **Observability** | Tenant-scoped traces and metrics |

### 9.2 Prompt Injection Mitigation

| Attack Vector | Mitigation |
|---------------|------------|
| Malicious documents | Input sanitization, content scanning |
| Query injection | Query parameterization, output validation |
| Tool abuse | Sandboxed execution, capability limits |

---

## 10. Build & Distribution

### 10.1 Project Structure

```
context-nexus/
├── python/
│   └── context_nexus/      # Python package
├── rust/
│   └── aegis_core/         # Rust crate
├── bindings/
│   └── pyo3/               # PyO3 bindings
├── tests/
├── docs/
├── pyproject.toml          # maturin build config
├── Cargo.toml
└── .github/workflows/      # CI/CD
```

### 10.2 Build System

**Tool**: `maturin` (Rust → Python wheels)

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "context-nexus"
requires-python = ">=3.10"
```

### 10.3 Distribution

| Platform | Method |
|----------|--------|
| **PyPI** | Pre-built wheels (manylinux, macOS, Windows) |
| **Docker** | Official images with all dependencies |

---

## Summary: Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Language split** | Python + Rust (PyO3) | Ecosystem + performance |
| **Vector DB** | Qdrant (primary) | Rust-native, filtering |
| **Graph DB** | Neo4j (primary) | Mature, Cypher |
| **Async model** | Async-first | I/O-bound workloads |
| **Ingestion** | Event-driven queues | Backpressure handling |
| **Token management** | Hard budgets | Prevent cost explosions |
| **Observability** | LangSmith-compatible | Ecosystem standard |
| **Build** | maturin | Best Rust→Python tooling |
