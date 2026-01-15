"""
Complete Workflow Example 2: Code Analysis Agent

This example demonstrates:
- Ingesting code files from a directory
- Multi-step reasoning workflow
- Graph traversal for finding related code
- Structured output with custom schemas
"""

import asyncio
import os
from pathlib import Path
from dataclasses import dataclass
from context_nexus import ContextNexus, Agent


@dataclass
class SecurityFinding:
    """Structured output for security analysis"""
    file: str
    line_range: str
    severity: str  # low, medium, high, critical
    issue: str
    recommendation: str


async def main():
    print("=" * 70)
    print("CODE ANALYSIS AGENT - COMPLETE WORKFLOW")
    print("=" * 70)
    print()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY environment variable")
        return
    
    # ===================================================================
    # PHASE 1: SETUP & CONFIGURATION
    # ===================================================================
    print("⚙️  PHASE 1: SETUP & CONFIGURATION")
    print("-" * 70)
    
    nexus = ContextNexus()
    
    # Configure for code analysis
    nexus.config.chunk_size = 800  # Larger chunks for code context
    nexus.config.chunk_overlap = 100
    
    print(f"✅ Configured for code analysis")
    print(f"   Chunk size: {nexus.config.chunk_size} (larger for code)")
    print(f"   Chunk overlap: {nexus.config.chunk_overlap}")
    print()
    
    # ===================================================================
    # PHASE 2: INGEST CODEBASE
    # ===================================================================
    print("📂 PHASE 2: INGEST CODEBASE")
    print("-" * 70)
    
    # Point to your codebase (we'll use the project itself as example)
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
    
    # ===================================================================
    # PHASE 3: CODE UNDERSTANDING QUERIES
    # ===================================================================
    print("🔍 PHASE 3: CODE UNDERSTANDING QUERIES")
    print("-" * 70)
    
    agent = Agent(nexus, token_budget=12000)  # Higher budget for code
    
    try:
        # Query 1: Architectural overview
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
        
        # Query 2: Find dependencies
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
        
        # ===================================================================
        # PHASE 4: SEMANTIC CODE SEARCH
        # ===================================================================
        print("🔎 PHASE 4: SEMANTIC CODE SEARCH")
        print("-" * 70)
        
        # Use retrieval directly for code search
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
            # Show first 2 lines of code
            lines = result.content.split('\n')[:2]
            for line in lines:
                print(f"        {line}")
            print()
        
        # ===================================================================
        # PHASE 5: MULTI-STEP ANALYSIS WORKFLOW
        # ===================================================================
        print("🔄 PHASE 5: MULTI-STEP ANALYSIS WORKFLOW")
        print("-" * 70)
        
        print("Running multi-step workflow:")
        print("  Step 1: Find all async functions")
        print("  Step 2: Check for proper error handling")
        print("  Step 3: Identify improvement opportunities")
        print()
        
        # Step 1: Find async functions
        step1 = await agent.query(
            "List all async functions in the codebase with their purposes"
        )
        
        print("Step 1 Results:")
        print(f"  {step1.text[:200]}...")
        print()
        
        # Step 2: Error handling check
        step2 = await agent.query(
            "Do these async functions have proper error handling? "
            "Are there try-except blocks and cleanup (finally)?"
        )
        
        print("Step 2 Results:")
        print(f"  {step2.text[:200]}...")
        print()
        
        # ===================================================================
        # PHASE 6: KNOWLEDGE GRAPH TRAVERSAL
        # ===================================================================
        print("🕸️  PHASE 6: KNOWLEDGE GRAPH TRAVERSAL")
        print("-" * 70)
        
        print("Exploring code relationships through graph traversal:")
        print()
        
        # Query that requires understanding relationships
        graph_query = await agent.query(
            "If I modify the Loader class, what other components might be affected?"
        )
        
        print("Impact Analysis:")
        print(f"  {graph_query.text}")
        print()
        
        # ===================================================================
        # PHASE 7: SUMMARY
        # ===================================================================
        print("=" * 70)
        print("✅ CODE ANALYSIS WORKFLOW COMPLETE")
        print("=" * 70)
        print()
        print("Demonstrated Capabilities:")
        print("  ✅ Ingested entire codebase with dependency tracking")
        print("  ✅ Semantic code search (find by meaning, not keywords)")
        print("  ✅ Multi-step reasoning workflows")
        print("  ✅ Knowledge graph for impact analysis")
        print("  ✅ Configurable chunk sizes for code vs docs")
        print()
        print("Use Cases:")
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
