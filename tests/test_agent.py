import pytest
from context_nexus import ContextNexus, Agent


@pytest.fixture
def nexus():
    return ContextNexus()


@pytest.fixture
def agent(nexus):
    return Agent(nexus)


@pytest.mark.asyncio
async def test_query(agent):
    ans = await agent.query("test question")
    assert ans.text


@pytest.mark.asyncio
async def test_query_with_trace(agent):
    ans = await agent.query("test", trace=True)
    assert ans.trace is not None


def test_token_budget(nexus):
    agent = Agent(nexus, token_budget=1000)
    assert agent.token_budget == 1000
