<h1 align="center">Context Nexus</h1>

<p align="center">
  <strong>SDK for building agentic AI systems</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/context-nexus/"><img src="https://img.shields.io/pypi/v/context-nexus?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/context-nexus/"><img src="https://img.shields.io/pypi/pyversions/context-nexus" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

<p align="center">
  <img src="docs/images/feature_overview.png" alt="Context Nexus" width="400">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="docs/quickstart.md">Tutorial</a> •
  <a href="docs/INDEX.md">Docs Index</a> •
  <a href="docs/blog.md">Complete Guide</a> •
  <a href="#benchmark">Benchmark</a>
</p>

---

## What is Context Nexus?

Build AI applications that can search, reason, and answer questions over your documents.

**Unlike basic RAG implementations**, Context Nexus adds:
- 🔗 **Knowledge graphs** for relationship-aware retrieval
- 📊 **Token budgets** that never overflow
- 🔍 **Hybrid search** combining vectors + graphs
- 📈 **Full observability** for every query

<p align="center">
  <img src="docs/images/how_it_works.png" alt="How it works" width="500">
</p>

```python
nexus = ContextNexus()
await nexus.ingest(["./docs/", "./papers.pdf"])  # PDFs, HTML, URLs supported

agent = Agent(nexus, token_budget=8000)
answer = await agent.query("What services depend on payments?")
print(answer.text, answer.sources)  # Answer with citations
```

---

## Why Context Nexus?

| Problem | Baseline RAG | Context Nexus |
|---------|--------------|---------------|
| Vector search alone isn't enough | ❌ Keyword fallback | ✅ Hybrid: vectors + graph |
| Context windows overflow | ❌ Hope for the best | ✅ Enforced token budgets |
| "Why did AI say that?" | ❌ Black box | ✅ Full trace for every query |
| Python is slow for hot paths | ❌ Pure Python | ✅ Rust core for 10-100x speedup |
| Only handles plain text | ❌ Just .txt, .md | ✅ PDF, HTML, URLs, code |

---

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Text | `.txt`, `.md`, code files | Direct loading |
| PDF | `.pdf` | Extracts text with pypdf |
| HTML | `.html`, `.htm` | Extracts text, removes scripts |
| URLs | `https://...` | Fetches and parses content |
| Code | `.py`, `.js`, `.ts`, `.go`, `.rs`, etc. | 20+ languages |

---

## Installation

```bash
pip install context-nexus
```

**Requirements:**
- Python 3.10 or higher
- Pre-compiled Rust binaries included for major platforms (macOS, Linux, Windows)
- No Rust compiler needed - wheels contain native code for optimal performance

**Supported Platforms:**
- macOS (ARM64 / Apple Silicon, x86_64 / Intel)
- Linux (x86_64, ARM64)
- Windows (x86_64)

The package automatically uses Rust-accelerated implementations where available, with transparent fallback to Python for maximum compatibility.

---

## Quick Start

```python
import asyncio
from context_nexus import ContextNexus, Agent

async def main():
    nexus = ContextNexus()
    
    # Ingest from multiple sources
    await nexus.ingest([
        "./my-docs/",           # Directories (all supported files)
        "./research/paper.pdf", # PDF files
        "https://example.com",  # URLs
    ])
    
    agent = Agent(nexus, token_budget=8000)
    answer = await agent.query("What is our refund policy?")
    print(answer.text)

asyncio.run(main())
```

---

## Performance Benchmark

We benchmark Context Nexus against baseline vector-only search using real unstructured data from Wikipedia and arXiv.

### Quick Run

```bash
# Install local embedding model (one-time, 90MB download)
pip install sentence-transformers

# Run comprehensive benchmark
python examples/05_benchmark.py
```

### Results Summary

**Hybrid Retrieval Performance:**

| Metric | Baseline (Vector-Only) | Context Nexus (Hybrid) | Difference |
|--------|----------------------|----------------------|------------|
| Search latency (avg) | 0.07ms | 0.05ms | 29% faster |
| Search latency (p99) | 2.36ms | 0.27ms | 88% faster |
| Graph construction | N/A | <0.01s | Negligible overhead |
| Knowledge graph | N/A | ✅ 1,526 nodes, 1,511 edges | Relationship reasoning |

**Rust vs Python Performance:**

| Implementation | Time per 800KB | Throughput | Speedup |
|----------------|---------------|------------|---------|
| Rust (native) | ~2-5ms | ~400 MB/sec | 2-10x faster |
| Python (fallback) | ~10-20ms | ~80 MB/sec | Baseline |

*Rust acceleration applies to: text chunking, vector scoring, graph traversal, and RRF fusion.*

### What This Means

- **Graph construction overhead**: Negligible (<0.01s for 1,500+ chunks)
- **Search performance**: Hybrid retrieval is faster than vector-only despite added complexity
- **Rust acceleration**: Hot paths run 2-10x faster with zero code changes
- **Real-world data**: Tests use actual Wikipedia articles and arXiv papers
- **Free embeddings**: Uses sentence-transformers (local, no API costs)

---

## Examples

Ready-to-run examples in [`examples/`](examples/):

| Example | What It Shows | Data Source |
|---------|---------------|-------------|
| [01_simple_qa.py](examples/01_simple_qa.py) | Quick start guide | Inline text |
| [02_full_workflow.py](examples/02_full_workflow.py) | Complete lifecycle | Inline docs |
| [03_code_analysis.py](examples/03_code_analysis.py) | Codebase analysis | Local files |
| [04_research_agent.py](examples/04_research_agent.py) | Research workflows | Generated corpus |
| [05_benchmark.py](examples/05_benchmark.py) | Performance comparison | Wikipedia + arXiv |

See [examples/README.md](examples/README.md) for setup instructions and detailed descriptions.

---

## Features

- **Hybrid Retrieval** — Semantic search + graph reasoning for better results
- **PDF & HTML Support** — Process real documents, not just plain text  
- **Token Budget Management** — Automatic context window management, never overflow
- **Full Observability** — Trace every decision with detailed query analytics
- **Rust-Accelerated Performance** — Hot paths optimized for 2-10x speedup
- **Seamless Integration** — Pre-compiled binaries included, no setup required
- **Multi-Source Support** — Built-in fetchers for Wikipedia, arXiv, and more

---

## 📚 Documentation

### Getting Started

| Doc | Description | Time |
|-----|-------------|------|
| [Quickstart](docs/quickstart.md) | Build your first agent in 15 minutes | 15 min |
| [Complete Blog Guide](docs/blog.md) | Everything from basics to production (beginner to advanced) | 2-3 hours |
| [Documentation Index](docs/INDEX.md) | **Full navigation guide and feature matrix** | 5 min |

### Learn By Doing

| Example | Focus | Lines |
|---------|-------|-------|
| [01_simple_qa.py](examples/01_simple_qa.py) | Minimal setup (copy & run) | ~40 |
| [02_full_workflow.py](examples/02_full_workflow.py) | Complete production example | ~200 |
| [03_code_analysis.py](examples/03_code_analysis.py) | Analyzing codebases | ~150 |
| [04_research_agent.py](examples/04_research_agent.py) | Iterative research & refinement | ~180 |
| [05_benchmark.py](examples/05_benchmark.py) | Performance comparison (Rust vs Python) | ~220 |

### In-Depth Guides

| Guide | Topic | Audience |
|-------|-------|----------|
| [Observability Guide](docs/OBSERVABILITY.md) | Tracing, debugging, monitoring queries | Developers |
| [Use Cases & Patterns](docs/use_cases.md) | Real-world workflows and patterns | Developers |
| [Architecture Document](docs/architecture.md) | System design and Python/Rust boundary | Engineers |
| [Installation Guide](docs/INSTALL.md) | Platform-specific setup | Everyone |
| [Product Overview](docs/product_document.md) | Feature summary and comparisons | Decision Makers |

### Quick Navigation

**Find What You Need:** Start with [Documentation Index](docs/INDEX.md) for a complete feature matrix, topic coverage, and suggested learning paths.

---

## vs. Other Tools

| Feature | LangChain | LlamaIndex | Context Nexus |
|---------|-----------|------------|---------------|
| Vector search | ✅ | ✅ | ✅ |
| Knowledge graph | Plugin | ✅ | ✅ Built-in |
| Token budgets | Manual | Manual | ✅ Automatic |
| Rust performance | ❌ | ❌ | ✅ Native (2-10x faster) |
| PDF support | Plugin | ✅ | ✅ Built-in |
| Observability | LangSmith ($) | ✅ | ✅ Built-in |
| Install complexity | Medium | Medium | ✅ One command (`pip install`) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

```bash
git clone https://github.com/chiraag-kakar/context-nexus
cd context-nexus
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## License

MIT © [Chiraag Kakar](https://github.com/chiraag-kakar)
