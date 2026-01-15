# Context Nexus Examples

This directory contains working examples that demonstrate the core features of Context Nexus.

---

## Prerequisites

Before running these examples, you'll need:

### 1. Python 3.10+
```bash
python --version  # Should be 3.10 or higher
```

### 2. Install Context Nexus
```bash
pip install context-nexus

# Or install from source (if you're in the project directory)
pip install -e .
```

### 3. OpenAI API Key (Required for Examples 01-05)

All current examples use OpenAI for embeddings and LLM generation.

**Get a key**: https://platform.openai.com/api-keys

**Set it up**:
```bash
# Mac/Linux
export OPENAI_API_KEY='sk-...'

# Windows
set OPENAI_API_KEY=sk-...

# Or create a .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

**Cost**: OpenAI offers $5 in free credits for new accounts. Each example run costs ~$0.01-0.05.

---

## Examples Overview

| Example | What It Shows | Data Source | Complexity |
|---------|---------------|-------------|------------|
| [01_simple_qa.py](01_simple_qa.py) | Basic Q&A in ~50 lines | Inline text | ⭐ Simple |
| [02_full_workflow.py](02_full_workflow.py) | Complete lifecycle with tracing | Inline docs | ⭐⭐ Intermediate |
| [03_code_analysis.py](03_code_analysis.py) | Analyze codebases | Local Python files | ⭐⭐ Intermediate |
| [04_research_agent.py](04_research_agent.py) | Iterative research workflow | Generated corpus | ⭐⭐⭐ Advanced |
| [05_benchmark.py](05_benchmark.py) | **Performance vs baseline** | **Wikipedia + arXiv** | ⭐⭐⭐ Advanced |

**Set it up**:
```bash
# Mac/Linux
export OPENAI_API_KEY='sk-...'

# Windows
set OPENAI_API_KEY=sk-...

# Or create a .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

**Cost**: OpenAI offers $5 in free credits for new accounts. Each example run costs ~$0.01-0.05.

---

## Examples Overview

| Example | What It Shows | Complexity |
|---------|---------------|------------|
| [01_simple_qa.py](01_simple_qa.py) | Basic Q&A in ~50 lines | ⭐ Simple |
| [02_full_workflow.py](02_full_workflow.py) | Complete lifecycle with tracing | ⭐⭐ Intermediate |
| [03_code_analysis.py](03_code_analysis.py) | Analyze codebases | ⭐⭐ Intermediate |
| [04_research_agent.py](04_research_agent.py) | Iterative research workflow | ⭐⭐⭐ Advanced |
| [05_benchmark.py](05_benchmark.py) | Performance at scale (100+ docs) | ⭐⭐⭐ Advanced |

---

## Running the Examples

### Start Simple
```bash
# Make sure your API key is set
echo $OPENAI_API_KEY

# Run the simplest example
python examples/01_simple_qa.py
```

### Try the Full Workflow
```bash
python examples/02_full_workflow.py
```

This shows the complete pipeline:
- Document ingestion
- Vectorization
- Hybrid search (semantic + graph)
- Agent query with token budgets
- Full tracing and source attribution

### Analyze Code
```bash
python examples/03_code_analysis.py
```

This ingests the Context Nexus codebase itself and lets you ask questions about it.

### Research Agent
```bash
python examples/04_research_agent.py
```

Shows an iterative research workflow with confidence tracking and rate limiting.

### Performance Benchmark
```bash
python examples/05_benchmark.py
```

**What it does**:
- Generates 100 realistic documents (technical documentation)
- Processes ~100KB of text
- Runs search and query benchmarks
- Provides production readiness assessment

**Metrics reported**:
- Ingestion throughput (docs/sec, chunks/sec)
- Search latency (average, min, max)
- Agent query latency
- Token usage statistics

**Run time**: ~60 seconds (including embeddings)

**Use this to**:
- Validate performance on your hardware
- Estimate costs for your use case
- Decide between local (FAISS) vs production (Qdrant) backends

---

## Understanding the Dependencies

### What's Being Used

| Dependency | What For | Where It's Used |
|------------|----------|-----------------|
| **OpenAI API** | Embeddings + LLM generation | All examples |
| **FAISS** | Vector similarity search | Installed with `context-nexus` |
| **NetworkX** | Knowledge graph (in-memory) | Installed with `context-nexus` |

### Local Development (No External APIs)

Want to run without OpenAI? You can use local models:

```bash
# Install local dependencies
pip install context-nexus[local]

# Install Ollama for local LLM
# Mac: brew install ollama
# Linux/Windows: https://ollama.ai

# Pull a model
ollama pull llama3

# Then in your code:
nexus = ContextNexus(
    embedding_provider="local",  # Uses sentence-transformers
    llm_provider="local"          # Uses Ollama
)
```

**Note**: Local mode examples coming soon!

---

## Production Backends

For production, you can swap to scalable backends:

### Qdrant (Vector Database)
```bash
# Install
pip install context-nexus[production]

# Run Qdrant locally
docker run -p 6333:6333 qdrant/qdrant

# Use it
nexus = ContextNexus(vector_store="qdrant")
```

### Neo4j (Graph Database)
```bash
# Run Neo4j locally
docker run -p 7474:7474 -p 7687:7687 neo4j

# Use it
nexus = ContextNexus(graph_store="neo4j")
```

**Current examples use FAISS + NetworkX** (no external services needed).

---

## Troubleshooting

### "OpenAI API key not set"
```bash
export OPENAI_API_KEY='your-key-here'
```

### "ModuleNotFoundError: No module named 'context_nexus'"
```bash
# Make sure you're in the right directory and installed it
pip install -e .
```

### "Rate limit exceeded"
You're hitting OpenAI's rate limits. The examples include pauses to avoid this, but if you're running multiple examples rapidly:
- Wait a minute between runs
- Or upgrade your OpenAI account tier

### Examples are slow
First run is slower because:
- Documents are being embedded (calls to OpenAI)
- FAISS index is being built

Subsequent queries are fast (<1 second).

---

## Next Steps

1. **Modify the examples** - Change the documents, queries, or parameters
2. **Read the docs** - Check out [docs/quickstart.md](../docs/quickstart.md)
3. **Build your own** - Use these as templates for your use case

---

## Questions?

- **GitHub Issues**: https://github.com/chiraag-kakar/context-nexus/issues
- **Documentation**: [docs/](../docs/)
