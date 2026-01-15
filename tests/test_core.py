import pytest
from context_nexus.core.types import Document, Chunk, Answer, Source
from context_nexus.core.config import Config


def test_document():
    doc = Document(content="hello", source="test.txt")
    assert doc.content == "hello"


def test_document_metadata():
    doc = Document(content="x", metadata={"key": "val"})
    assert doc.metadata["key"] == "val"


def test_config_defaults():
    cfg = Config()
    assert cfg.vector_store.backend == "faiss"
    assert cfg.llm.provider == "openai"


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cfg = Config.from_env()
    assert cfg.llm.api_key == "test-key"


def test_answer_with_sources():
    ans = Answer(
        text="response",
        sources=[Source(title="doc.md", chunk_id="1", relevance=0.9)],
    )
    assert len(ans.sources) == 1
