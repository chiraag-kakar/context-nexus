# Building Agentic Workflows with Context Nexus

This guide shows you how to build real-world agentic AI systems using Context Nexus.

---

## What is an Agentic Workflow?

An **agentic workflow** is an AI system that:
1. **Decides** what information it needs
2. **Retrieves** that information from various sources
3. **Reasons** over the information step-by-step
4. **Acts** on the conclusions (or asks for human input)

Unlike simple chatbots that just respond to prompts, agentic systems **take action** and **follow multi-step plans**.

---

## Workflow 1: Document Q&A Agent

**Goal**: Answer questions about your documentation with sources.

### Step 1: Set Up

```python
from context_nexus import ContextNexus, Agent

# Create a nexus (uses sensible defaults)
nexus = ContextNexus()
```

### Step 2: Ingest Documents

```python
# Ingest your documentation
await nexus.ingest([
    "./docs/",           # Local markdown/text files
    "./pdfs/",           # PDF documents
])

# See what was ingested
print(nexus.stats)
# Output: Documents: 150, Chunks: 2,340, Graph Nodes: 890
```

### Step 3: Create an Agent

```python
agent = Agent(
    nexus,
    token_budget=8000,  # Never exceed this many tokens per query
)
```

### Step 4: Ask Questions

```python
answer = await agent.query("How do I set up authentication?")

print(answer.text)
# Output: "To set up authentication, you need to..."

print(answer.sources)
# Output: [
#   Source(title="Auth Guide", relevance=0.94),
#   Source(title="Security Best Practices", relevance=0.87),
# ]

print(answer.confidence)
# Output: 0.91
```

---

## Workflow 2: Multi-Step Research Agent

**Goal**: Research a topic by gathering information, validating it, and synthesizing an answer.

### The Workflow

```
[Query] → [Retrieve] → [Validate] → [Expand if needed] → [Synthesize] → [Answer]
```

### Implementation

```python
from context_nexus import ContextNexus, Workflow, Step

nexus = ContextNexus()

# Define the workflow
workflow = Workflow("research")

@workflow.step("retrieve")
async def retrieve(query: str, context: Context) -> Context:
    """Find relevant documents"""
    results = await nexus.retrieve(query, limit=20)
    return context.add_documents(results)

@workflow.step("validate")
async def validate(context: Context) -> Context:
    """Check if we have enough information"""
    if context.confidence < 0.7:
        # Need more information
        return context.needs_expansion()
    return context

@workflow.step("expand")
async def expand(context: Context) -> Context:
    """Get related documents via graph traversal"""
    related = await nexus.expand_graph(context.entities, hops=2)
    return context.add_documents(related)

@workflow.step("synthesize")
async def synthesize(context: Context) -> Answer:
    """Generate final answer"""
    return await nexus.generate(
        context.documents,
        output_type="structured_answer"
    )

# Connect steps
workflow.connect("retrieve", "validate")
workflow.connect("validate", "expand", condition="needs_expansion")
workflow.connect("validate", "synthesize", condition="has_enough")
workflow.connect("expand", "validate")  # Loop back

# Run the workflow
answer = await workflow.run("What are the architectural risks in our codebase?")
```

---

## Workflow 3: Incident Analysis Agent

**Goal**: Analyze an incident by connecting logs, tickets, and documentation.

### Set Up Multiple Sources

```python
from context_nexus import ContextNexus
from context_nexus.sources import LogSource, TicketSource, WikiSource

nexus = ContextNexus()

# Connect to multiple data sources
await nexus.add_source(LogSource(
    provider="datadog",
    api_key=os.environ["DATADOG_API_KEY"]
))

await nexus.add_source(TicketSource(
    provider="jira",
    api_key=os.environ["JIRA_API_KEY"]
))

await nexus.add_source(WikiSource(
    provider="confluence",
    api_key=os.environ["CONFLUENCE_API_KEY"]
))
```

### Create an Incident Analysis Workflow

```python
from context_nexus import Agent, AnalysisSchema

# Define what we want to extract
class IncidentReport(AnalysisSchema):
    timeline: list[Event]
    root_cause: str
    affected_services: list[str]
    similar_incidents: list[Incident]
    recommendations: list[str]

# Create agent with structured output
agent = Agent(nexus, output_schema=IncidentReport)

# Analyze an incident
report = await agent.analyze(
    "What caused the payment service outage on Jan 10, 2026?"
)

print(report.root_cause)
# Output: "Database connection pool exhaustion due to..."

print(report.similar_incidents)
# Output: [
#   Incident(id="INC-1234", date="2025-11-15", similarity=0.89),
#   Incident(id="INC-0987", date="2025-08-03", similarity=0.76),
# ]
```

---

## Workflow 4: Contract Analysis Agent

**Goal**: Analyze contracts for compliance issues.

```python
from context_nexus import ContextNexus, Agent

nexus = ContextNexus()

# Ingest contracts and policies
await nexus.ingest([
    "./contracts/",  # Your contracts
    "./policies/",   # Company policies
])

# Define analysis schema
class ComplianceFinding(AnalysisSchema):
    contract: str
    clause: str
    issue: str
    severity: Literal["low", "medium", "high", "critical"]
    excerpt: str
    recommendation: str

agent = Agent(nexus, output_schema=list[ComplianceFinding])

# Run analysis
findings = await agent.analyze(
    "Find clauses that conflict with our 2026 data protection policy"
)

for finding in findings:
    print(f"[{finding.severity.upper()}] {finding.contract}")
    print(f"  Issue: {finding.issue}")
    print(f"  Clause: {finding.excerpt[:100]}...")
    print()
```

---

## Common Patterns

### Pattern 1: Hybrid Retrieval

Combine semantic search with graph-based reasoning:

```python
from context_nexus import HybridRetriever

retriever = HybridRetriever(
    vector_weight=0.6,  # 60% semantic similarity
    graph_weight=0.4,   # 40% relationship-based
)

agent = Agent(nexus, retriever=retriever)
```

### Pattern 2: Token Budget Management

Never overflow the LLM's context window:

```python
agent = Agent(
    nexus,
    token_budget=8000,           # Hard limit
    compression="hierarchical",  # Compress if needed
)
```

### Pattern 3: Observability

See exactly what your agent is doing:

```python
answer = await agent.query("...", trace=True)

# See every step
for step in answer.trace.steps:
    print(f"{step.name}: {step.duration_ms}ms, {step.tokens} tokens")

# See what documents were used
for doc in answer.trace.documents_used:
    print(f"  - {doc.title} (relevance: {doc.relevance:.0%})")
```

### Pattern 4: Fallback Strategies

Handle failures gracefully:

```python
agent = Agent(
    nexus,
    fallback_strategy="graceful",  # Return partial results on failure
    timeout_seconds=30,
    retry_count=3,
)
```

---

## Deployment

### Local Development

```bash
# Install
pip install context-nexus

# Uses FAISS (local) and NetworkX (in-memory graph)
# No external dependencies needed
```

### Production

```bash
# Use production backends
pip install context-nexus[production]

# Requires:
# - Qdrant for vector storage
# - Neo4j for graph storage
# - Redis for caching (optional)
```

### Configuration

```python
# Development (default)
nexus = ContextNexus()

# Production
nexus = ContextNexus(
    vector_store="qdrant://localhost:6333",
    graph_store="neo4j://localhost:7687",
    cache="redis://localhost:6379",
)
```

---

## Next Steps

1. **Try the examples**: Clone the repo and run the example workflows
2. **Ingest your data**: Point Context Nexus at your documents
3. **Build your agent**: Start simple, add complexity as needed
4. **Monitor & iterate**: Use tracing to understand and improve

---

## Getting Help

- **GitHub Issues**: Report bugs, request features
- **Discussions**: Ask questions, share what you've built
- **Discord**: Real-time chat with the community
