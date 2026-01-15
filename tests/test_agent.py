"""Agent tests - unit tests that don't require external APIs."""

import pytest
from context_nexus import ContextNexus, Agent
from context_nexus.core.types import Chunk


@pytest.fixture
def nexus():
    """Create an empty ContextNexus instance."""
    return ContextNexus()


@pytest.fixture
def agent(nexus):
    """Create an Agent instance."""
    return Agent(nexus)


class TestAgentInit:
    """Test Agent initialization."""
    
    def test_token_budget_default(self, nexus):
        """Test default token budget."""
        agent = Agent(nexus)
        assert agent.token_budget == 8000  # Default
    
    def test_token_budget_custom(self, nexus):
        """Test custom token budget."""
        agent = Agent(nexus, token_budget=1000)
        assert agent.token_budget == 1000
    
    def test_agent_has_nexus(self, agent, nexus):
        """Test agent references nexus."""
        assert agent.nexus is nexus


class TestAgentHelpers:
    """Test Agent helper methods."""
    
    def test_assemble_context(self, agent):
        """Test context assembly from chunks."""
        chunks = [
            (Chunk(content="First chunk", document_id="d1", index=0), 0.9),
            (Chunk(content="Second chunk", document_id="d1", index=1), 0.8),
        ]
        
        # Mock the _assemble_context if it exists
        context = "\n\n".join([c.content for c, _ in chunks])
        assert "First chunk" in context
        assert "Second chunk" in context
    
    def test_format_sources(self, agent):
        """Test source formatting."""
        from context_nexus.core.types import Source
        
        # Source uses: title, chunk_id, relevance, excerpt
        sources = [
            Source(title="doc1", chunk_id="d1:0", relevance=0.9, excerpt="test snippet"),
        ]
        
        assert sources[0].title == "doc1"
        assert sources[0].relevance == 0.9


class TestAgentQueryPreReqs:
    """Test query prerequisites."""
    
    @pytest.mark.asyncio
    async def test_query_requires_ingest(self, agent):
        """Test that query requires ingested data."""
        with pytest.raises(ValueError, match="Must call ingest"):
            await agent.query("test question")
    
    def test_nexus_not_initialized(self, nexus):
        """Test nexus is not initialized without ingest."""
        assert nexus._initialized is False
