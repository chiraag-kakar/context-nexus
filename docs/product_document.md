# Context Nexus - Product Document

> **Version**: 1.0 | **Last Updated**: 2026-01-15

---

## What is Context Nexus?

**Context Nexus** is an open-source Python SDK that helps developers build AI applications that can **search, reason, and answer questions** over large collections of documents—like a super-powered search engine that actually understands your content.

### The Simple Explanation

Imagine you have thousands of documents—code, wikis, PDFs, research papers. You want to ask questions like:

> *"What services depend on the payment module, and which ones changed last week?"*

A regular search gives you keyword matches. **Context Nexus gives you intelligent answers with sources.**

It does this by combining:
- **Semantic Search**: Understands *meaning*, not just keywords
- **Graph Reasoning**: Understands *relationships* between things  
- **Smart Context Management**: Never overwhelms the AI with too much information
- **Production Reliability**: Handles failures gracefully, scales under load

---

## Who Is This For?

| You Are... | Context Nexus Helps You... |
|------------|----------------------------|
| **Developer building AI features** | Add intelligent Q&A to your app in hours, not months |
| **ML Engineer productionizing RAG** | Get hybrid retrieval, context management, and observability out of the box |
| **Team building internal tools** | Create knowledge assistants, research tools, or incident analyzers |

### Not For You If...
- You want a no-code chatbot builder
- You need a simple keyword search
- You're looking for a hosted SaaS (we're an SDK)

---

## What Can You Build?

### Example 1: Engineering Knowledge Assistant

```python
from context_nexus import ContextNexus, Agent

# Point it at your docs
nexus = ContextNexus()
await nexus.ingest(["./docs/", "./wiki/", "./code/"])

# Ask complex questions
agent = Agent(nexus)
answer = await agent.query(
    "What services depend on payments and changed in the last sprint?"
)

print(answer.text)      # The intelligent answer
print(answer.sources)   # Links to source documents
print(answer.confidence) # How confident the answer is
```

**Real-world uses:**
- Internal documentation Q&A
- Codebase understanding
- Onboarding assistants

### Example 2: Research & Compliance Tool

```python
# Ingest contracts and policies
await nexus.ingest(["./contracts/", "./policies/"])

# Find conflicts
answer = await agent.query(
    "Which contracts have indemnification clauses that conflict with our 2026 policy?"
)

# Get structured output with citations
for finding in answer.findings:
    print(f"Contract: {finding.source}")
    print(f"Issue: {finding.issue}")
    print(f"Clause: {finding.excerpt}")
```

**Real-world uses:**
- Legal research
- Compliance auditing
- Due diligence

### Example 3: Incident Analysis Agent

```python
# Connect to your incident data
await nexus.ingest([
    LogSource("datadog"),
    TicketSource("jira"), 
    WikiSource("confluence")
])

# Analyze an incident
answer = await agent.query(
    "What caused the outage on Jan 10? Have we seen similar issues?"
)

# Get a structured incident report
print(answer.root_cause)
print(answer.similar_incidents)
print(answer.recommendations)
```

---

## How Does It Work?

### The 30-Second Version

1. **Ingest**: Feed it your documents. It chunks them smartly, creates embeddings, and builds a knowledge graph.

2. **Retrieve**: When you ask a question, it searches both by meaning (vectors) AND by relationships (graph).

3. **Reason**: It manages context intelligently—never overloading the LLM, always keeping the most relevant info.

4. **Answer**: Returns grounded answers with sources, confidence scores, and full traceability.

### Why Both Python AND Rust?

| Python Handles... | Rust Handles... |
|-------------------|-----------------|
| Your application logic | Token counting (millions/sec) |
| LLM API calls | Vector scoring (SIMD-optimized) |
| Workflow orchestration | Graph traversal (memory-efficient) |
| Easy configuration | Context compression (fast) |

**Result**: Developer-friendly API with performance where it matters.

---

## Quick Start (5 Minutes)

### Installation

```bash
pip install context-nexus
```

### Your First Agent

```python
from context_nexus import ContextNexus, Agent

# Initialize (uses sensible defaults)
nexus = ContextNexus()

# Ingest some documents
await nexus.ingest(["./my-docs/"])

# Create an agent
agent = Agent(nexus)

# Ask a question
answer = await agent.query("What is our refund policy?")
print(answer.text)
```

### With Custom Configuration

```python
from context_nexus import ContextNexus, HybridRetriever, Agent

nexus = ContextNexus(
    vector_store="qdrant",        # or "faiss" for local
    graph_store="networkx",       # lightweight, in-memory
    embeddings="openai",          # or "local" for offline
)

retriever = HybridRetriever(
    vector_weight=0.6,            # 60% semantic similarity
    graph_weight=0.4,             # 40% relationship-based
)

agent = Agent(
    nexus,
    retriever=retriever,
    token_budget=8000,            # Never exceed this per query
)
```

---

## Core Concepts

### 1. Ingestion
Transform raw documents into searchable knowledge.

```python
# Ingest from multiple sources
await nexus.ingest([
    "./local-docs/",                    # Local files
    "https://docs.example.com/",        # Websites
    DatabaseSource("postgresql://..."), # Databases
])
```

### 2. Hybrid Retrieval
Combine the best of semantic search and graph reasoning.

```python
# Vector: "Find documents about authentication"
# Graph: "Find all services connected to the auth service"
# Hybrid: "Find auth documentation for connected services"
```

### 3. Token Budgeting
Never overflow your LLM's context window.

```python
agent = Agent(nexus, token_budget=8000)
# Automatically compresses/prioritizes context to fit
```

### 4. Observability
See exactly what your agent is doing.

```python
answer = await agent.query("...", trace=True)
print(answer.trace.steps)       # Each reasoning step
print(answer.trace.tokens_used) # Token consumption
print(answer.trace.latency_ms)  # Performance breakdown
```

---

## Technology Decisions

| Component | Default | Why |
|-----------|---------|-----|
| **Vector Store** | FAISS (local) / Qdrant (production) | Fast, filtering support, no vendor lock-in |
| **Graph Store** | NetworkX (local) / Neo4j (production) | Simple to start, scales when needed |
| **Embeddings** | OpenAI / Local models | Flexible based on requirements |
| **LLM** | OpenAI / Anthropic / Local | Pluggable, use what you prefer |
| **Python Version** | 3.10+ | LTS stable, wide compatibility |
| **Tracing** | Pluggable (LangSmith, OpenTelemetry, custom) | No vendor lock-in |

---

## Comparison

| Feature | LangChain | LlamaIndex | **Context Nexus** |
|---------|-----------|------------|-------------------|
| Vector retrieval | ✅ | ✅ | ✅ |
| Graph reasoning | ❌ | Partial | ✅ **Native** |
| Hybrid fusion | Manual | Manual | ✅ **Built-in** |
| Token budgeting | ❌ | ❌ | ✅ **Automatic** |
| Rust performance | ❌ | ❌ | ✅ **Hot paths** |
| Production-ready | Partial | Partial | ✅ **First-class** |

---

## What's Included

```
context-nexus/
├── Ingestion        # Document loading, chunking, embedding
├── Retrieval        # Vector, graph, and hybrid search
├── Context          # Token budgeting, compression
├── Agents           # Workflow orchestration
├── Observability    # Tracing, metrics, evaluation
└── Integrations     # LangGraph, LangSmith, popular DBs
```

---

## Get Started

```bash
# Install
pip install context-nexus

# Run the quickstart
python -m context_nexus.quickstart

# Read the docs
open https://context-nexus.dev/docs
```

**GitHub**: [github.com/context-nexus/context-nexus](https://github.com/context-nexus/context-nexus)  
**PyPI**: [pypi.org/project/context-nexus](https://pypi.org/project/context-nexus)
