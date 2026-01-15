# Development Setup

## Prerequisites

### Required

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | [python.org](https://python.org/downloads) |
| Rust | 1.70+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Git | 2.0+ | [git-scm.com](https://git-scm.com) |

### VS Code Extensions

| Extension | ID |
|-----------|-----|
| Python | `ms-python.python` |
| Pylance | `ms-python.vscode-pylance` |
| Ruff | `charliermarsh.ruff` |
| rust-analyzer | `rust-lang.rust-analyzer` |
| Even Better TOML | `tamasfe.even-better-toml` |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/context-nexus
cd context-nexus

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e ".[dev]"

# build rust extension
cd crates/nexus-core
maturin develop
cd ../..

pytest
```

---

## Project Structure

```
context-nexus/
├── context_nexus/       # Python package
├── crates/
│   └── nexus-core/      # Rust crate
├── tests/
├── examples/
├── docs/
├── pyproject.toml
└── Cargo.toml
```

---

## Commands

```bash
# tests
pytest
pytest --cov=context_nexus

# lint
ruff check .
ruff format .

# rust
cargo test
cargo clippy
cargo fmt
```

---

## Environment Variables

Create `.env` (don't commit):

```
OPENAI_API_KEY=sk-...
```

---

## External Services (optional)

| Service | Default | Production |
|---------|---------|------------|
| Vector DB | FAISS (local) | Qdrant |
| Graph DB | NetworkX (in-memory) | Neo4j |
| LLM | OpenAI | Any |
