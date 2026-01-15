# Contributing to Context Nexus

Thank you for your interest in contributing! This guide will help you get started.

## Quick Start

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/context-nexus
cd context-nexus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Install Rust (for core development)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Run tests
pytest
```

## Development Setup

### Prerequisites

- Python 3.10+
- Rust 1.70+ (for core development)
- Git

### Project Structure

```
context-nexus/
├── python/              # Python package
│   └── context_nexus/
├── rust/                # Rust crate (performance core)
│   └── nexus_core/
├── tests/               # Test suite
├── docs/                # Documentation
└── examples/            # Example scripts
```

## How to Contribute

### Reporting Bugs

1. Search existing issues first
2. Use the bug report template
3. Include: Python version, OS, minimal reproducer

### Suggesting Features

1. Open a discussion first
2. Explain the use case
3. Be open to alternative solutions

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Run linting: `ruff check .`
7. Submit a pull request

## Code Standards

### Python

- Use type hints everywhere
- Follow PEP 8 (enforced by ruff)
- Write docstrings for public APIs
- Aim for 80%+ test coverage

### Rust

- Run `cargo fmt` before committing
- Run `cargo clippy` and address warnings
- Add benchmarks for performance-critical code

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=context_nexus

# Run specific test
pytest tests/test_retrieval.py -k "test_hybrid"

# Run Rust tests
cargo test
```

## Pull Request Guidelines

- Keep PRs focused (one feature/fix per PR)
- Update documentation if needed
- Add a changelog entry for user-facing changes
- Ensure CI passes before requesting review

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open an Issue
- **Chat**: Join our Discord

## Recognition

Contributors are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md) and thanked in release notes.

---

Thank you for helping make Context Nexus better! 🙏
