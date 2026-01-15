"""
Code analysis example - ingest a codebase and ask questions about it.

Shows how to analyze code using semantic search and graph traversal
to understand structure, dependencies, and impact of changes.
"""

import asyncio
import os
from pathlib import Path
from context_nexus import ContextNexus, Agent


async def main():
    print("=" * 70)
    print("CODE ANALYSIS AGENT")
    print("=" * 70)
    print()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY environment variable")
        return
    
    # ---- SETUP ----
    print("⚙️  PHASE 1: SETUP")
    print("-" * 70)
    
    nexus = ContextNexus()
    
    # Use larger chunks for code to preserve more context
    nexus.config.chunk_size = 800
    nexus.config.chunk_overlap = 100
    
    print(f"✅ Configured for code analysis")
    print(f"   Chunk size: {nexus.config.chunk_size} (larger for code)")
    print(f"   Chunk overlap: {nexus.config.chunk_overlap}")
    print()
    
    # ---- INGEST CODEBASE ----
    print("📂 PHASE 2: INGEST CODEBASE")
    print("-" * 70)
    
    # Point to the codebase (using this project as an example)
    codebase_path = Path(__file__).parent.parent / "context_nexus"
    
    print(f"Ingesting codebase from: {codebase_path}")
    print()
    
    if codebase_path.exists():
        stats = await nexus.ingest([str(codebase_path)])
        
        print(f"✅ Codebase ingested!")
        print(f"   Files processed: {stats.documents}")
        print(f"   Code chunks: {stats.chunks}")
        print(f"   Graph nodes: {stats.graph_nodes}")
        print(f"   Dependencies tracked: {stats.graph_edges}")
        print()
    else:
        print(f"⚠️  Path not found: {codebase_path}")
        print("   This example needs to run from the project directory")
        return
    
    # ---- CODE UNDERSTANDING ----
    print("🔍 PHASE 3: CODE UNDERSTANDING")
    print("-" * 70)
    
    agent = Agent(nexus, token_budget=12000)  # Higher budget for code
    
    try:
        # Architectural overview
        print("Query 1: What are the main components?")
        print()
        
        answer = await agent.query(
            "What are the main components of this codebase and how do they interact?",
            trace=True
        )
        
        print("Answer:")
        print(f"  {answer.text}")
        print()
        
        if answer.trace:
            print(f"  ⏱  Retrieved in {answer.trace.latency_ms}ms")
            print(f"  📄 Used {answer.trace.chunks_retrieved} code chunks")
            print()
        
        # Find dependencies
        print("Query 2: What files depend on the indexer module?")
        print()
        
        answer2 = await agent.query(
            "Which modules or files import or depend on the indexer module?"
        )
        
        print("Answer:")
        print(f"  {answer2.text}")
        print()
        print(f"  Sources: {len(answer2.sources)}")
        for src in answer2.sources[:3]:
            print(f"    - {src.title}")
        print()
        
        # ---- SEMANTIC CODE SEARCH ----
        print("🔎 PHASE 4: SEMANTIC CODE SEARCH")
        print("-" * 70)
        
        search_query = "async function that embeds text chunks"
        print(f"Searching: '{search_query}'")
        print()
        
        results = await nexus.retrieve(search_query, limit=5)
        
        print(f"Found {len(results)} relevant code snippets:")
        print()
        
        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result.metadata.get('file_name', 'unknown')}")
            print(f"      Score: {result.score:.4f}")
            print(f"      Preview:")
            lines = result.content.split('\n')[:2]
            for line in lines:
                print(f"        {line}")
            print()
        
        # ---- IMPACT ANALYSIS ----
        print("🕸️  PHASE 5: IMPACT ANALYSIS")
        print("-" * 70)
        
        print("Exploring code relationships through graph traversal:")
        print()
        
        impact_query = await agent.query(
            "If I modify the Loader class, what other components might be affected?"
        )
        
        print("Impact Analysis:")
        print(f"  {impact_query.text}")
        print()
        
        # ---- SUMMARY ----
        print("=" * 70)
        print("✅ CODE ANALYSIS COMPLETE")
        print("=" * 70)
        print()
        print("What we demonstrated:")
        print("  ✅ Ingested entire codebase with dependency tracking")
        print("  ✅ Semantic code search (find by meaning, not keywords)")
        print("  ✅ Knowledge graph for impact analysis")
        print("  ✅ Configurable chunk sizes for code vs docs")
        print()
        print("Useful for:")
        print("  - Onboarding new developers")
        print("  - Code review assistance")
        print("  - Impact analysis for refactoring")
        print("  - Security audit automation")
        print("  - Documentation generation")
        print()
        
    finally:
        await agent.close()
        print("🧹 Cleaned up resources")


if __name__ == "__main__":
    asyncio.run(main())
