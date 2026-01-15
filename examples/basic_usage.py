"""
Example: Basic usage of Context Nexus
"""

import asyncio
from context_nexus import ContextNexus, Agent
from context_nexus.core.types import Document


async def main():
    # Initialize Context Nexus
    nexus = ContextNexus()
    
    # Option 1: Ingest from files/directories
    print("📁 Ingesting documents from directory...")
    stats = await nexus.ingest(["./docs/"])
    print(f"✅ Ingested {stats.documents} documents into {stats.chunks} chunks")
    print(f"   Graph: {stats.graph_nodes} nodes, {stats.graph_edges} edges\n")
    
    # Option 2: Ingest from Document objects
    doc = Document(
        content="Context Nexus combines vector search, graph reasoning, and LLMs.",
        source="inline.md"
    )
    await nexus.ingest([doc])
    
    # Create an agent
    agent = Agent(nexus, token_budget=8000)
    
    try:
        # Query with basic answer
        print("💬 Question: What is Context Nexus?")
        answer = await agent.query("What is Context Nexus?")
        print(f"🤖 Answer: {answer.text}\n")
        
        print(f"📚 Sources ({len(answer.sources)}):")
        for src in answer.sources:
            print(f"   - {src.title} (relevance: {src.relevance:.2f})")
        print()
        
        # Query with trace
        print("💬 Question: How does hybrid retrieval work?")
        answer = await agent.query("How does hybrid retrieval work?", trace=True)
        print(f"🤖 Answer: {answer.text}\n")
        
        if answer.trace:
            print("🔍 Trace:")
            print(f"   Total time: {answer.trace.latency_ms}ms")
            print(f"   Tokens used: {answer.trace.tokens_used}")
            print(f"   Chunks retrieved: {answer.trace.chunks_retrieved}")
            print(f"   Steps:")
            for step in answer.trace.steps:
                print(f"      - {step['name']}: {step['duration_ms']}ms")
    
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
