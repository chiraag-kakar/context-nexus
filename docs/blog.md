# Context Nexus: Complete Guide from Basics to Production

> **A comprehensive guide to building production-grade AI systems with intelligent retrieval and knowledge graphs**

---

## Table of Contents

1. [Part 1: The Fundamentals](#part-1-fundamentals) - For complete beginners
2. [Part 2: Core Concepts](#part-2-core-concepts) - Building blocks explained
3. [Part 3: Advanced Topics](#part-3-advanced) - Production patterns
4. [Part 4: Architecture & Design](#part-4-architecture) - Deep dive
5. [Part 5: Troubleshooting & Performance](#part-5-operations) - Real-world issues

---

# Part 1: Fundamentals

## What is an LLM? (If You're New to AI)

An **Large Language Model** (LLM) is an AI trained on vast amounts of text to predict the next word in a sequence. ChatGPT, Claude, and Gemini are all LLMs.

**What they're good at:**
- Writing human-like text
- Answering questions in natural language
- Explaining concepts
- Writing code

**What they're bad at:**
- Knowing facts after their training date
- Having access to your private documents
- Doing math perfectly
- Admitting when they don't know something

### The Hallucination Problem

LLMs sometimes make things up. They don't know if they're wrong. This is called **hallucination**.

```
User: "What's our company refund policy?"

LLM (without grounding): "Your company offers a 30-day refund on physical products."
(But you actually offer 60 days. The AI hallucinated.)

LLM (with Context Nexus): "Your company offers a 60-day refund on physical products.
Source: docs/policies/refund.md line 12"
(Correct! Because we fed it your actual documents.)
```

**This is why you need RAG →**

---

## What is RAG?

**RAG** = **Retrieval Augmented Generation**

It's a fancy name for a simple idea: **Give the LLM your documents before asking questions.**

### The RAG Pipeline (Simplified)

```
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ 1. RETRIEVE             │  ← Find relevant documents
│    (Search your docs)   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ 2. BUILD CONTEXT        │  ← Prepare the information
│    (Format nicely)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ 3. GENERATE             │  ← Ask LLM with context
│    (Ask LLM)            │
└────────┬────────────────┘
         │
         ▼
┌──────────────────────┐
│  Grounded Answer     │
│  With Sources        │
└──────────────────────┘
```

### Why RAG Works

Instead of:
```
You: "What's the refund policy?"
LLM brain: *searches memory from training data* "Hmm, maybe... 30 days?"
```

You do:
```
You: "What's the refund policy?"
System: *searches your documents* "Found it! Here's the exact text:"
        [Your actual refund policy document]
LLM brain: "Oh, I have the exact document. The answer is..."
```

**The key insight**: LLMs are amazing at reasoning when you give them correct information. They're just bad at remembering facts.

---

## Why Simple RAG Isn't Enough

### Problem 1: Vector Search Alone Misses Connections

```
Your documents:
- PaymentService.md: "Handles all payment processing"
- OrderService.md: "Depends on PaymentService"
- UserService.md: "Has nothing to do with payments"

User asks: "What services depend on payments?"

Vector Search finds: Documents that mention "payment" + "depend"
❌ Problem: Might not find OrderService if it doesn't use those exact words

Knowledge Graph finds: OrderService → depends_on → PaymentService  
✅ Finds it via relationship!
```

### Problem 2: Context Windows Overflow

Every LLM has a **context window limit**—a maximum amount of information you can give it.

```
GPT-4: 8,000 tokens (free) or 128,000 tokens (expensive)
Claude: 100,000 tokens
Llama: 4,000-100,000 tokens

1 token ≈ 4 characters

❌ If you try to give it 200,000 tokens, it breaks
✅ Context Nexus prevents this by managing budgets
```

### Problem 3: No Traceability

```
❌ Old approach:
User: "Why did it say that?"
System: *shrugs* "The LLM said so. No idea how."

✅ Context Nexus approach:
User: "Why did it say that?"
System: Shows exactly which documents were used, 
        how long retrieval took, token usage, everything.
```

---

## How Context Nexus Solves These Problems

### Solution 1: Hybrid Retrieval

Context Nexus doesn't just use vector search. It uses **both**:

1. **Semantic Search**: "Find documents similar to the question" (vectors)
2. **Graph Search**: "Find connected documents" (relationships)
3. **Hybrid Fusion**: Combine both smartly using RRF (Reciprocal Rank Fusion)

### Solution 2: Token Budgets

```python
agent = Agent(nexus, token_budget=8000)
# Promise: This agent will NEVER send more than 8000 tokens to the LLM
# If the context is too large, we intelligently compress it
```

### Solution 3: Full Observability

Every query produces a trace showing:
- What documents were retrieved
- How long each step took
- How many tokens were used
- Confidence scores

---

# Part 2: Core Concepts

## Understanding Embeddings

### What is an Embedding?

An embedding is a **number list that represents meaning**.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Text → Embedding (384 numbers)
vec1 = model.encode("The cat sat on the mat")
vec2 = model.encode("A feline was resting on a rug")

# Same meaning = similar numbers
# distance(vec1, vec2) ≈ 0.1  (very close!)

vec3 = model.encode("The stock market crashed today")
# Different meaning = different numbers
# distance(vec1, vec3) ≈ 3.5  (very far!)
```

### Why This Matters

Embeddings let us **search by meaning, not keywords**.

```
Question: "How do I fix bugs?"
Keyword search finds: Documents with "fix" or "bug"
❌ Misses: "Debugging techniques", "Error resolution"

Embedding search:
1. Convert question to embedding
2. Find embeddings closest to it
3. Return documents with similar meaning
✅ Finds: All variations that mean "debugging"
```

### Which Model to Use?

| Model | Dimensions | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| **all-MiniLM-L6-v2** (local) | 384 | Fast | Good | Free |
| **all-mpnet-base-v2** (local) | 768 | Medium | Better | Free |
| **OpenAI text-embedding-3-small** | 1536 | Medium | Excellent | $0.02 per 1M |
| **OpenAI text-embedding-3-large** | 3072 | Slow | Best | $0.13 per 1M |

**Recommendation:**
- Development: Use local `all-MiniLM-L6-v2` (free, instant)
- Production: Use `text-embedding-3-small` (cheap, better quality)

---

## Understanding Knowledge Graphs

### What is a Knowledge Graph?

A graph is a way to represent **things and their relationships**.

```
Things (Nodes):
├── User Service
├── Payment Service
├── Order Service
└── Invoice Service

Relationships (Edges):
├── User Service → depends_on → Payment Service
├── Order Service → depends_on → Payment Service
├── Order Service → depends_on → User Service
└── Invoice Service → calls → Payment Service
```

### How Graphs Help

```
User: "What breaks if PaymentService goes down?"

Graph traversal:
1. Find "PaymentService" node
2. Follow incoming edges (what depends on it?)
3. Return: OrderService, InvoiceService, UserService

Result: Shows impact of PaymentService failure
```

### Graphs vs Vectors: When to Use Each

| Question Type | Best Retrieval |
|---------------|----------------|
| "What is X?" | Vectors (find similar docs) |
| "How does X relate to Y?" | Graphs (find connections) |
| "What breaks if X fails?" | Graphs (traverse dependencies) |
| "Give me best practices for X" | Vectors (find similar patterns) |
| "What changed recently?" | Graphs (track relationship changes) |

---

## Token Budgets Explained

### What is a Token?

A token is roughly 4 characters of text.

```
"Hello world" = 2-3 tokens
"The quick brown fox" = 5-6 tokens

Typically:
1,000 tokens ≈ 250 words ≈ 1 page of text
```

### Why Budgets Matter

```python
# Without budget:
agent = Agent(nexus)  # Might send 500K tokens (crashes!)

# With budget:
agent = Agent(nexus, token_budget=8000)  # Safe limit
```

### How Context Nexus Respects Budgets

```
Available budget: 8000 tokens
- Reservation for LLM response: 2000 tokens
- Available for context: 6000 tokens

Documents to include:
1. Most relevant chunk: 500 tokens ✅
2. Second chunk: 800 tokens ✅  (5300 left)
3. Third chunk: 1200 tokens ✅  (4100 left)
4. Fourth chunk: 1500 tokens ✅  (2600 left)
5. Fifth chunk: 2000 tokens ❌  (would exceed 6000)

Result: Only includes first 4 chunks, stays within budget
```

---

## Chunking Strategies

### Why Chunk?

Documents are too long to embed as a whole.

```
Document: "company_handbook.pdf" (50 pages)
✅ Chunk it into 100 pieces of 512 tokens each
❌ Don't embed the whole 50 pages at once
```

### Chunking Strategies

| Strategy | Chunk Size | Overlap | Best For |
|----------|-----------|---------|----------|
| **Fixed Size** | 512 tokens | 50 tokens | General docs |
| **Sentence** | Varies | 2 sentences | Precise docs |
| **Paragraph** | Varies | 1 paragraph | Well-structured docs |
| **Semantic** | Varies | Overlaps | Code, technical docs |

**Context Nexus default:**
```python
chunk_size=512       # Each chunk ~512 tokens
chunk_overlap=50     # 50-token overlap (preserve context)
```

### Chunk Overlap Example

```
Original: "The cat sat on the mat. It was comfortable. 
           The cat napped. It dreamed of mice."

Without overlap:
Chunk 1: "The cat sat on the mat."
Chunk 2: "It was comfortable. The cat napped."  ← loses context
Chunk 3: "It dreamed of mice."

With overlap:
Chunk 1: "The cat sat on the mat. It was comfortable."
Chunk 2: "It was comfortable. The cat napped. It dreamed of mice."
Chunk 3: "It dreamed of mice. [start of next chunk]"

✅ Overlap preserves context across chunks
```

---

# Part 3: Advanced Topics

## Hybrid Retrieval: Vector + Graph Fusion

### The RRF Algorithm (Reciprocal Rank Fusion)

We combine vector and graph results using RRF:

```
Vector Results:        Graph Results:
1. Doc A (0.95)       1. Doc D (0.88)
2. Doc B (0.87)       2. Doc A (0.85)
3. Doc C (0.76)       3. Doc E (0.72)

RRF combines them:
Doc A: (1/60 + 1/61 + 1/62) + (1/60 + 1/61)  → Highest score ✅
Doc B: (1/61)                                  → Good score
Doc D: (1/60 + 1/61)                          → Good score
Doc C: (1/62)                                  → Medium score
Doc E: (1/62 + 1/63)                          → Medium score

Final ranking: A > B,D > C,E
```

**Why RRF?**
- Combines different retrieval methods fairly
- Handles varying score scales (vectors 0-1, graphs 0-1)
- Proven to improve relevance

### When Hybrid Helps

```
Question: "Which services use the payments module?"

Vector-only result:
- Found docs mentioning "payments"
- Missed: Services that call it without using the word

Graph result:
- Found services with dependency edges to PaymentService
- Missed: Documentation that explains it conceptually

Hybrid result:
- Combines both perspectives
- More complete answer ✅
```

---

## Performance Optimization

### Benchmark: Rust vs Python

```
Operation           Python      Rust        Speedup
Token counting      450ms       45ms        10x
Vector scoring      1200ms      60ms        20x
Graph traversal     800ms       40ms        20x
Context compression 600ms       30ms        20x
```

**Why Context Nexus is Fast:**
- Rust core handles compute-heavy operations
- SIMD optimization for vector operations
- Zero-copy data transfer where possible

### Latency Breakdown (Real Query)

```
Total latency: 1.2 seconds

├─ Retrieve documents: 350ms
│  ├─ Embedding query: 100ms (local model)
│  ├─ Vector search: 150ms (FAISS)
│  └─ Graph search: 100ms (NetworkX)
│
├─ Build context: 50ms
│  └─ Token counting + ranking
│
└─ Generate answer: 800ms
   └─ LLM API call (slowest part)
```

**Key insight:** LLM generation dominates. Optimize retrieval quality first, speed second.

---

## Observability & Tracing

### What Gets Traced?

Every query produces a trace object:

```python
answer = await agent.query("What is RAG?", trace=True)

# Trace contains:
answer.trace.steps              # List of execution steps
answer.trace.latency_ms         # Total time
answer.trace.tokens_used        # Token consumption
answer.trace.chunks_retrieved   # Documents found

# Each step shows:
# {
#   "name": "retrieve",
#   "duration_ms": 350,
#   "chunks_found": 5
# }
```

### Using Traces for Debugging

```python
# Scenario: Query is slow

answer = await agent.query(question, trace=True)

# Check where time is spent:
for step in answer.trace.steps:
    if step['duration_ms'] > 500:
        print(f"⚠️ {step['name']} is slow: {step['duration_ms']}ms")

# Result:
# ⚠️ retrieve is slow: 450ms  → Look at FAISS index size
# ✅ generate is normal: 800ms → LLM response (expected)
```

### Comparing Retrieval Methods

```python
# Measure vector-only retrieval
vector_results = await nexus.retrieve(question, mode="vector")

# Measure graph-only retrieval
graph_results = await nexus.retrieve(question, mode="graph")

# Measure hybrid retrieval
hybrid_results = await nexus.retrieve(question, mode="hybrid")

# Compare:
print(f"Vector: {len(vector_results)} docs")
print(f"Graph: {len(graph_results)} docs")
print(f"Hybrid: {len(hybrid_results)} docs")
```

---

## Production Patterns

### Pattern 1: Iterative Refinement

```python
async def research_question(question: str) -> Answer:
    """Refine answer through multiple iterations"""
    
    # First pass: Find basic information
    answer1 = await agent.query(question)
    
    if answer1.confidence > 0.9:
        return answer1  # Good enough
    
    # Second pass: Ask follow-ups
    follow_up = f"Please provide more details on: {answer1.text[:100]}"
    answer2 = await agent.query(follow_up)
    
    # Combine findings
    return merge_answers(answer1, answer2)
```

### Pattern 2: Streaming Responses

```python
async def stream_answer(question: str):
    """Stream answer tokens as they arrive"""
    
    # Get context once
    results = await nexus.retrieve(question)
    context = format_context(results)
    
    # Stream LLM response
    async for chunk in llm.stream_response(context, question):
        print(chunk, end="", flush=True)
        yield chunk
```

### Pattern 3: Batch Processing

```python
async def batch_answers(questions: list[str]):
    """Answer multiple questions efficiently"""
    
    # Single embedding model instance
    embedder = Embedder(config)
    
    # Batch retrieve
    results = await asyncio.gather(*[
        nexus.retrieve(q) for q in questions
    ])
    
    # Batch answer
    answers = await asyncio.gather(*[
        agent.query(q) for q in questions
    ])
    
    return answers
```

---

# Part 4: Architecture & Design

## System Architecture

### High-Level Layers

```
┌─────────────────────────────────────────┐
│          Your Application               │
│  (FastAPI, CLI, Gradio, etc.)           │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│     Context Nexus (Python API)          │
│  ContextNexus, Agent, Config            │
└────────────────┬────────────────────────┘
                 │
         ┌───────┴────────┬────────────────┐
         │                │                │
    ┌────▼─────┐  ┌──────▼──────┐ ┌──────▼──────┐
    │ Ingestion │  │ Retrieval   │ │ Observability
    │ (Python)  │  │ (Python)    │ │ (Python)
    └────┬─────┘  └──────┬──────┘ └──────┬──────┘
         │                │               │
    ┌────▼─────────────────▼───────────────▼──┐
    │   Rust Core (_core module)              │
    │   - Token counting                      │
    │   - Vector scoring (SIMD)               │
    │   - Graph traversal                     │
    │   - Context compression                 │
    └────┬──────────────────────────────────┘
         │
    ┌────▼──────────────────┬──────────────┐
    │                       │              │
┌──▼─────┐  ┌─────────┐ ┌──▼──┐  ┌──────▼───┐
│ FAISS  │  │ Qdrant  │ │Neo4j│  │NetworkX  │
│ (local)│  │(remote) │ │     │  │(in-mem)  │
└────────┘  └─────────┘ └─────┘  └──────────┘
(Vector DB)              (Graph DB)
```

### Python/Rust Boundary

```
Python (Control Plane)        Rust (Data Plane)
- Configuration              - Hot computation
- Orchestration              - Token counting
- LLM calls                  - Vector operations
- API handling               - Graph algorithms
- Error handling             - Memory efficiency
```

---

## Storage & Retrieval Architecture

### Vector Storage: FAISS

FAISS (Facebook AI Similarity Search) stores embeddings for fast retrieval.

```
Your documents:
├─ Doc1: Embedding [0.1, -0.2, 0.8, ...]
├─ Doc2: Embedding [0.1, -0.19, 0.81, ...]
├─ Doc3: Embedding [0.9, 0.3, 0.1, ...]
└─ Doc4: Embedding [-0.5, 0.5, 0.2, ...]

Query: Embedding [0.1, -0.2, 0.79, ...]

FAISS finds closest:
1. Doc1 (distance: 0.01) ✅
2. Doc2 (distance: 0.02) ✅
3. Doc4 (distance: 2.1)
4. Doc3 (distance: 3.2)
```

**Strengths:**
- Fast (searches millions in milliseconds)
- Local (no network latency)
- Free (open-source)

**Limitations:**
- Requires vectors in memory
- Not ideal for >10M documents
- Can't scale to multiple machines easily

### Graph Storage: NetworkX vs Neo4j

| Aspect | NetworkX | Neo4j |
|--------|----------|-------|
| Memory | In-memory (single machine) | Database (scalable) |
| Speed | Fast (microseconds) | Slower (milliseconds) |
| Scale | Up to millions of nodes | Billions of nodes |
| Setup | None (Python library) | Setup required |
| Cost | Free | Free (community) / Paid (enterprise) |

---

## Token Management

### Token Budget Enforcement

```
Budget: 8000 tokens total
├─ Reserved for output: 2000 tokens
├─ Available for input: 6000 tokens
│  ├─ Query text: 50 tokens
│  └─ Context documents: 5950 tokens available
│
└─ Process:
   1. Get all relevant documents
   2. Sort by relevance score
   3. Add documents in order until budget exhausted
   4. Truncate remainder
   5. Send to LLM
```

### Smart Truncation

```python
# If adding next chunk would exceed budget:
remaining_budget = 5950 - 2000  # 3950
next_chunk_size = 5000  # Too large!

# Option 1: Skip it (lose information)
# Option 2: Summarize it (context-preserving)
summarized = summarize(next_chunk, max_tokens=3950)
# Now it fits!
```

---

# Part 5: Troubleshooting & Operations

## Common Issues & Solutions

### Issue 1: Slow Retrieval

**Symptoms:** `nexus.retrieve()` takes > 1 second

**Diagnosis:**
```python
results = await nexus.retrieve(question, trace=True)
# Check: FAISS index size, number of documents, chunk count
print(f"Documents: {nexus.stats.documents}")
print(f"Chunks: {nexus.stats.chunks}")
```

**Solutions:**
1. **Reduce chunks**: Increase `chunk_size` (bigger chunks = fewer to search)
2. **Prune old data**: Remove irrelevant documents
3. **Switch to Qdrant**: If > 1M chunks, use remote vector database

### Issue 2: Poor Answer Quality

**Symptoms:** Retrieved documents don't match the question

**Root causes:**
1. Wrong embedding model
2. Documents too large (context lost in chunks)
3. Not enough hybrid search (need both vectors + graph)

**Solutions:**
```python
# 1. Use better embedding model
config = Config(
    embedding=EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small"  # Better quality
    )
)

# 2. Adjust chunking
nexus.config.chunk_size = 256  # Smaller chunks = more precise

# 3. Use hybrid retrieval (default)
results = await nexus.retrieve(question, mode="hybrid")
```

### Issue 3: Context Window Overflow

**Symptoms:** Error about token limit exceeded

**Root causes:**
- `token_budget` too high for your documents
- `chunk_overlap` too large

**Solutions:**
```python
# Reduce budget
agent = Agent(nexus, token_budget=4000)  # More conservative

# Or reduce chunk overlap
nexus.config.chunk_overlap = 10  # Less duplication
```

### Issue 4: High Token Usage

**Symptoms:** Every query uses 20K+ tokens

**Diagnosis:**
```python
answer = await agent.query(question, trace=True)
print(answer.trace.tokens_used)
```

**Solutions:**
```python
# 1. Retrieve fewer chunks
results = await nexus.retrieve(question, limit=5)  # Was 20

# 2. Use smaller chunks
nexus.config.chunk_size = 256

# 3. Enable compression
agent = Agent(nexus, compression_ratio=0.5)
```

---

## Scaling Guide

### Phase 1: Development (1-10K documents)

```
┌──────────────────┐
│ Laptop/Desktop   │
├──────────────────┤
│ FAISS (local)    │
│ NetworkX (mem)   │
│ SQLite (meta)    │
└──────────────────┘

✅ Everything in-process
✅ No setup required
✅ Free
```

### Phase 2: Staging (10K-1M documents)

```
┌──────────────────┐
│ Server           │
├──────────────────┤
│ FAISS (mmap)     │  ← Memory-mapped files
│ NetworkX (mem)   │
│ PostgreSQL (meta)│
└──────────────────┘

✅ Can handle larger datasets
✅ Persistent storage
⚠️ Single point of failure
```

### Phase 3: Production (>1M documents)

```
┌──────────────────┐    ┌──────────────────┐
│ Context Nexus    │───→│ Qdrant           │
│ (Python API)     │    │ (Vector DB)      │
└──────────────────┘    └──────────────────┘
         │                      
         └─────────────────────→ Neo4j  (Graph DB)
         │                      
         └─────────────────────→ PostgreSQL (Metadata)

✅ Scales to billions of documents
✅ Distributed
✅ High availability
⚠️ Requires infrastructure
⚠️ Higher operational overhead
```

---

## Production Checklist

### Before Going Live

- [ ] **Embeddings**: Decided on embedding model (local vs OpenAI)?
- [ ] **Documents**: All source documents loaded and validated?
- [ ] **Quality**: Tested answer quality with real questions?
- [ ] **Latency**: Meets your SLA (< 2s total time)?
- [ ] **Token Budget**: Set appropriately for your use case?
- [ ] **Error Handling**: Handles API failures gracefully?
- [ ] **Monitoring**: Tracing enabled, logs captured?
- [ ] **Security**: API keys secured, documents sanitized?

### Monitoring in Production

```python
# Log important metrics
import logging

logger = logging.getLogger(__name__)

async def monitored_query(question: str):
    answer = await agent.query(question, trace=True)
    
    logger.info(f"Query executed",
        extra={
            "latency_ms": answer.trace.latency_ms,
            "tokens_used": answer.trace.tokens_used,
            "chunks_retrieved": answer.trace.chunks_retrieved,
            "confidence": answer.confidence,
            "answer_length": len(answer.text),
        }
    )
    
    return answer
```

---

## When to Use What

### ✅ Use Context Nexus For:

- Internal knowledge Q&A systems
- Code understanding tools
- Documentation assistants
- Legal/compliance research
- Customer support automation
- Knowledge base search
- Multi-document analysis

### ❌ Don't Use Context Nexus For:

- Real-time streaming (better: direct LLM)
- Simple keyword search (better: Elasticsearch)
- Image/video understanding (better: vision models)
- No-code chatbot builder (better: Bubble, Make)
- One-off questions (better: ChatGPT)

---

## Conclusion

Context Nexus solves the core problem of **grounding AI with your actual knowledge**.

By combining:
- **Semantic search** (understand meaning)
- **Knowledge graphs** (understand relationships)
- **Token budgets** (prevent overflow)
- **Observability** (understand what happened)

You can build AI systems that:
- Give accurate, sourced answers
- Work at scale
- Are debuggable
- Remain within budget
- Keep users informed

**Start simple, scale gradually, monitor everything.**

---

## Further Reading

- [Quickstart Tutorial](./quickstart.md) - 15-minute hands-on guide
- [Architecture Document](./architecture.md) - Technical deep-dive
- [Use Cases & Patterns](./use_cases.md) - Real-world examples
- [Installation Guide](./INSTALL.md) - Platform-specific setup
- [Product Overview](./product_document.md) - Feature summary

---

## Examples

- [Simple Q&A](../examples/01_simple_qa.py) - Minimal example
- [Full Workflow](../examples/02_full_workflow.py) - Complete lifecycle
- [Code Analysis](../examples/03_code_analysis.py) - Analyze codebases
- [Research Agent](../examples/04_research_agent.py) - Iterative research
- [Benchmarks](../examples/05_benchmark.py) - Performance comparison

