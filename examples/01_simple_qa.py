"""
Simple example showing how to build a Q&A agent in just a few lines.

This is the absolute basics - perfect for getting started quickly.
"""

import asyncio
from context_nexus import ContextNexus, Agent
from context_nexus.core.types import Document


async def main():
    # Create a nexus instance
    nexus = ContextNexus()
    
    # Add some sample content
    sample_doc = Document(
        content="""
        Context Nexus is an SDK for building agentic AI systems.
        It combines vector search with knowledge graphs to help
        your AI agents find and reason about information.
        
        Key features:
        - Hybrid retrieval (semantic + graph)
        - Automatic token budget management
        - Full observability and tracing
        - Rust-optimized hot paths for performance
        """,
        source="readme.md"
    )
    
    await nexus.ingest([sample_doc])
    print(f"✅ Ingested {nexus.stats.documents} documents\n")
    
    # Create an agent
    agent = Agent(nexus, token_budget=8000)
    
    try:
        # Ask a question
        answer = await agent.query("What is Context Nexus?")
        
        print("Answer:")
        print(f"  {answer.text}\n")
        
        print(f"Confidence: {answer.confidence:.0%}")
        print(f"Sources: {len(answer.sources)}")
        
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
