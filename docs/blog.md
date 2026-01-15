# Context Nexus: Building Production-Grade AI Agents in Python

> *A journey from "RAG demo" to "reliable AI system"*

---

## Introduction

*This blog documents the development of Context Nexus, an open-source SDK for building production-ready agentic AI systems. It will be updated as the project evolves.*

---

### The Problem We're Solving

Every developer building with LLMs hits the same walls:

1. **"My RAG demo worked great, but production is a nightmare"**
2. **"The AI keeps forgetting context or overflowing the token limit"**
3. **"I can't explain how the AI arrived at this answer"**
4. **"It works... until 100 users hit it at once"**

We've been there. So we built Context Nexus.

---

## What is Context Nexus?

Context Nexus is a Python SDK that helps you build AI applications that can **intelligently search, reason, and answer questions** over your documents.

Think of it as the infrastructure layer between your documents and your LLM.

```python
# The goal: This simple code should just work in production
from context_nexus import ContextNexus, Agent

nexus = ContextNexus()
await nexus.ingest(["./my-docs/"])

agent = Agent(nexus)
answer = await agent.query("What's our refund policy?")
print(answer.text, answer.sources)
```

Sounds simple? The complexity is hidden:
- Smart chunking that preserves context (Rust-optimized)
- Hybrid search (semantic + graph-based)
- Automatic token budget management
- Full traceability for every answer
- Graceful handling when things go wrong

---

## Why We Built It This Way

### Decision 1: Python + Rust

We wanted the **best of both worlds**:

| Python | Rust |
|--------|------|
| Easy to use | Blazing fast |
| Great ecosystem | Memory safe |
| Quick iteration | Predictable performance |

**Our split:**
- Python: Your application code, LLM calls, configuration
- Rust: Token counting, vector scoring, graph traversal

You write Python. The hot paths are Rust. You don't notice the boundary.

### Decision 2: Hybrid Retrieval

Vector search alone isn't enough.

```
User: "What services depend on payments?"

Vector search finds: Documents that mention "payments"
Graph search finds: Services connected to the payments node
Hybrid: Both, intelligently combined
```

We combine semantic similarity with relationship reasoning.

### Decision 3: Token Budgets, Not Token Prayers

Most RAG systems pray they don't overflow the context window.

We enforce budgets:

```python
agent = Agent(nexus, token_budget=8000)
# The agent will NEVER send more than 8000 tokens
# It automatically compresses and prioritizes
```

### Decision 4: Observable by Default

Every query produces a trace:

```python
answer = await agent.query("...")
print(answer.trace.steps)        # What happened
print(answer.trace.tokens_used)  # Token consumption
print(answer.trace.sources)      # What documents were used
```

No more "the AI said something weird and I don't know why."

---

## Building Your First Agent

### Step 1: Install

```bash
pip install context-nexus
```

### Step 2: Ingest Your Documents

```python
from context_nexus import ContextNexus

nexus = ContextNexus()

# Ingest local files
await nexus.ingest(["./docs/"])

# Check what was ingested
print(f"Indexed {nexus.stats.documents} documents")
print(f"Created {nexus.stats.chunks} chunks")
print(f"Built {nexus.stats.graph_nodes} graph nodes")
```

### Step 3: Ask Questions

```python
from context_nexus import Agent

agent = Agent(nexus)

answer = await agent.query("How do I deploy to production?")

print(answer.text)
for source in answer.sources:
    print(f"  - {source.title} ({source.relevance:.0%} relevant)")
```

### Step 4: Go Beyond Q&A

```python
# Multi-step reasoning
answer = await agent.query(
    "List all APIs that use authentication, then summarize their security models"
)

# Structured output
findings = await agent.analyze(
    "Find potential security issues in the codebase",
    output_schema=SecurityReport
)

# Workflows
workflow = Workflow([
    Step("research", retriever=deep_retriever),
    Step("validate", validator=fact_checker),
    Step("synthesize", generator=answer_generator),
])
```

---

## Architecture Deep Dive

Context Nexus uses a 5-stage ingestion pipeline:

### 1. Loading
**Module**: `context_nexus/ingestion/loader.py`
- Async file loading with error handling
- Supports 15+ file types (md, py, js, json, etc.)
- Auto-ignores common patterns (.git, node_modules)

### 2. Chunking
**Module**: `context_nexus/ingestion/chunker.py`
- Uses Rust `chunk_text` function (10-100x faster)
- Configurable size and overlap
- Preserves document metadata

### 3. Embedding
**Module**: `context_nexus/ingestion/embedder.py`
- OpenAI text-embedding-3-small (1536 dims)
- Batch processing (100 chunks at a time)
- Async HTTP with proper cleanup

### 4. Vector Indexing
**Module**: `context_nexus/ingestion/indexer.py`
- FAISS for development (local, fast)
- Qdrant for production (scalable, persistent)
- Cosine similarity with L2 normalization

### 5. Graph Building
**Module**: `context_nexus/ingestion/indexer.py`
- NetworkX for development
- Neo4j for production
- Links sequential chunks automatically

---

## Performance & Reliability

### Benchmarks
- **Chunking**: 50,000 tokens/sec (Rust implementation)
- **Vector search**: <50ms for 10k documents (FAISS)
- **End-to-end query**: 1-3 seconds (including LLM)

### Production Features
- **Token budgets**: Hard limits, never overflow
- **Error handling**: Async cleanup, graceful degradation
- **Observability**: Full trace for every query
- **Testing**: Integration test suite included

### Scaling
- **Local dev**: FAISS + NetworkX (no external deps)
- **Production**: Qdrant + Neo4j (horizontal scaling)
- **Hybrid**: Use FAISS for dev, swap to Qdrant for prod (zero code changes)

---

## Use Case Walkthroughs

### Use Case 1: Document Q&A
**Example**: [`examples/02_full_workflow.py`](../examples/02_full_workflow.py)

Demonstrates complete lifecycle:
- Document ingestion with stats
- Semantic search and retrieval
- Agent query with token management
- Source attribution and tracing

### Use Case 2: Code Analysis
**Example**: [`examples/03_code_analysis.py`](../examples/03_code_analysis.py)

Shows how to:
- Ingest codebases with larger chunk sizes
- Perform semantic code search
- Use graph traversal for impact analysis
- Multi-step reasoning workflows

### Use Case 3: Research & Synthesis
**Example**: [`examples/04_research_agent.py`](../examples/04_research_agent.py)

Illustrates:
- Iterative research with confidence tracking
- Rate limiting and batch processing
- Multi-source synthesis
- Production best practices

---

## Contributing

Context Nexus is open source. We welcome contributions!

**Getting started:**
```bash
git clone https://github.com/context-nexus/context-nexus
cd context-nexus
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

---

## Follow Along

- **GitHub**: [context-nexus/context-nexus](https://github.com/context-nexus/context-nexus)
- **PyPI**: [pypi.org/project/context-nexus](https://pypi.org/project/context-nexus)

*Star the repo to follow our progress!*
