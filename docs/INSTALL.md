# Installation & Setup Guide

## Quick Install (Recommended for Most Users)

### Python Package from PyPI

```bash
pip install context-nexus
```

**That's it!** Pre-compiled binaries with Rust acceleration are included for:
- macOS (ARM64/Apple Silicon, x86_64/Intel)
- Linux (x86_64, ARM64)
- Windows (x86_64)

**Verification:**
```bash
python -c "import context_nexus; print('✅ Installed successfully')"
```

---

## What You Get

When you `pip install context-nexus`, you automatically get:

### ✅ Python Control Plane
- All application logic and APIs
- LLM integrations (OpenAI, etc.)
- Document loaders and parsers
- Configuration management

### ✅ Rust Data Plane (Pre-compiled)
- High-performance text chunking (2-10x faster)
- Optimized vector scoring (5-10x faster)
- Graph traversal algorithms (3-5x faster)
- RRF fusion (2-3x faster)

### ✅ Automatic Fallback
- If Rust binaries aren't available for your platform
- System automatically uses Python fallback implementations
- **Your code doesn't change** - it just works

---

## System Requirements

### Minimum

- **Python**: 3.10 or higher
- **RAM**: 2GB minimum (4GB+ recommended for larger datasets)
- **Disk**: 500MB for package + models

### Recommended

- **Python**: 3.11+ (best performance)
- **RAM**: 8GB+ for production workloads
- **Disk**: 2GB+ if using local embedding models

---

## Installation Options

### Option 1: Basic Install (Most Users)

```bash
pip install context-nexus
```

Uses OpenAI for embeddings (requires API key).

### Option 2: Local Embeddings (No API Costs)

```bash
pip install context-nexus sentence-transformers
```

Runs embeddings locally - no API costs, no rate limits.

### Option 3: Production Features

```bash
pip install context-nexus[production]
```

Includes:
- Qdrant (scalable vector database)
- Neo4j (production graph database)
- Redis (caching layer)

### Option 4: Development Setup

```bash
pip install context-nexus[dev]
```

Includes testing and development tools.

---

## Verifying Rust Acceleration

Check if Rust native modules are working:

```python
try:
    from context_nexus._core import chunk_text, score_vectors
    print("✅ Rust acceleration active")
    print("   Functions available:", dir())
except ImportError:
    print("⚠️ Using Python fallback (still works, just slower)")
```

---

## Platform-Specific Notes

### macOS

**Apple Silicon (M1/M2/M3):**
```bash
pip install context-nexus  # ARM64 binaries included
```

**Intel Macs:**
```bash
pip install context-nexus  # x86_64 binaries included
```

### Linux

**Ubuntu/Debian:**
```bash
pip install context-nexus  # Works out of the box
```

**CentOS/RHEL:**
```bash
pip install context-nexus  # x86_64 binaries included
```

**ARM (Raspberry Pi, AWS Graviton):**
```bash
pip install context-nexus  # ARM64 binaries included
```

### Windows

```cmd
pip install context-nexus  # x86_64 binaries included
```

Note: Windows ARM is not yet supported (uses Python fallback).

---

## Building from Source (Optional - Advanced Users Only)

**You don't need this unless:**
- You're contributing to Context Nexus development
- Your platform isn't supported by pre-built wheels
- You want to modify the Rust code

### Prerequisites

1. Install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

2. Clone repository:
```bash
git clone https://github.com/chiraag-kakar/context-nexus
cd context-nexus
```

3. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install maturin (Rust/Python build tool):
```bash
pip install maturin
```

5. Build and install:
```bash
maturin develop --release
```

6. Verify:
```bash
pytest tests/  # Should pass 26/26 tests
```

See [DEVELOPMENT.md](../DEVELOPMENT.md) for detailed development setup.

---

## Environment Configuration

### Required: OpenAI API Key (if using OpenAI embeddings)

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or create `.env` file:
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Optional: Local Embeddings (No API Key Needed)

```python
from context_nexus import ContextNexus
from context_nexus.core.config import EmbeddingConfig

config = EmbeddingConfig()
config.provider = "local"  # Uses sentence-transformers

nexus = ContextNexus(embedding_config=config)
```

---

## Common Installation Issues

### Issue: "No module named 'context_nexus'"

**Solution:**
```bash
pip install context-nexus
# If still fails, try:
pip install --upgrade --force-reinstall context-nexus
```

### Issue: "ImportError: dynamic module does not define module export function"

**Cause:** Stale Rust binaries in source directory.

**Solution:**
```bash
# Find and remove stale binaries
find . -name "_core*.so" -delete
pip install --force-reinstall context-nexus
```

### Issue: Performance slower than expected

**Check Rust acceleration:**
```python
from context_nexus._core import chunk_text
print("✅ Rust active")
```

If import fails:
```bash
# Verify your platform has pre-built wheels
python -c "import platform; print(platform.machine())"
# Should be: x86_64, arm64, or aarch64

# Force reinstall
pip install --force-reinstall context-nexus
```

### Issue: "sentence-transformers" model download fails

**Solution:**
```bash
# Download with better connection settings
pip install sentence-transformers --timeout=300

# Or set HuggingFace cache directory
export TRANSFORMERS_CACHE=/path/to/cache
```

---

## Docker Installation (Alternative)

```dockerfile
FROM python:3.11-slim

# Install context-nexus with all pre-compiled binaries
RUN pip install context-nexus sentence-transformers

# Your application code
COPY . /app
WORKDIR /app

CMD ["python", "your_app.py"]
```

---

## Next Steps

After installation:

1. **Try the quickstart:** [docs/quickstart.md](quickstart.md) (15 minutes)
2. **Run examples:** [examples/](../examples/) directory
3. **Read architecture:** [docs/architecture.md](architecture.md)
4. **Run benchmarks:** `python examples/05_benchmark.py`

---

## Support

- **Issues:** https://github.com/chiraag-kakar/context-nexus/issues
- **Discussions:** https://github.com/chiraag-kakar/context-nexus/discussions
- **Email:** connect.with.chiraag@gmail.com
