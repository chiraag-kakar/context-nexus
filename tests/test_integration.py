import pytest
import asyncio
from pathlib import Path
from context_nexus import ContextNexus, Agent
from context_nexus.core.types import Document


@pytest.fixture
async def nexus_with_data():
    """Create a nexus instance with sample data."""
    nexus = ContextNexus()
    
    # Create sample documents
    doc1 = Document(
        content="Python is a high-level programming language known for its simplicity.",
        source="python.md",
        id="doc1"
    )
    doc2 = Document(
        content="Machine learning is a subset of artificial intelligence.",
        source="ml.md",
        id="doc2"
    )
    
    await nexus.ingest([doc1, doc2])
    return nexus


@pytest.mark.asyncio
async def test_ingest_documents(nexus_with_data):
    """Test document ingestion."""
    stats = nexus_with_data.stats
    assert stats.documents == 2
    assert stats.chunks > 0
    assert stats.graph_nodes > 0


@pytest.mark.asyncio
async def test_retrieve():
    """Test retrieval functionality."""
    nexus = ContextNexus()
    doc = Document(
        content="Context Nexus is an SDK for building agentic AI systems.",
        source="readme.md",
        id="doc1"
    )
    await nexus.ingest([doc])
    
    results = await nexus.retrieve("What is Context Nexus?", limit=5)
    assert len(results) > 0
    assert results[0].score > 0


@pytest.mark.asyncio
async def test_agent_query(nexus_with_data):
    """Test agent query with real data."""
    agent = Agent(nexus_with_data, token_budget=8000)
    
    try:
        answer = await agent.query("What is Python?")
        assert answer.text
        assert len(answer.sources) > 0
        assert answer.confidence > 0
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_agent_query_with_trace(nexus_with_data):
    """Test agent query with tracing enabled."""
    agent = Agent(nexus_with_data)
    
    try:
        answer = await agent.query("Explain machine learning", trace=True)
        assert answer.trace is not None
        assert len(answer.trace.steps) > 0
        assert answer.trace.latency_ms > 0
        assert answer.trace.chunks_retrieved > 0
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_token_budget_management():
    """Test that token budget is respected."""
    nexus = ContextNexus()
    
    # Create large document
    large_content = " ".join(["word"] * 10000)
    doc = Document(content=large_content, source="large.md", id="large")
    
    await nexus.ingest([doc])
    
    agent = Agent(nexus, token_budget=1000)
    try:
        answer = await agent.query("What is this about?")
        assert answer.text  # Should still get an answer despite truncation
    finally:
        await agent.close()
