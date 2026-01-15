"""
Full-featured example showing document ingestion, retrieval, and agent queries.

This covers the complete workflow: loading docs, vectorizing them, searching,
and getting answers with full tracing and source attribution.
"""

import asyncio
import os
from pathlib import Path
from context_nexus import ContextNexus, Agent
from context_nexus.core.types import Document


async def main():
    print("=" * 70)
    print("DOCUMENT Q&A AGENT - COMPLETE WORKFLOW")
    print("=" * 70)
    print()
    
    # Make sure we have an API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Set it to use embeddings.")
        print("   export OPENAI_API_KEY='your-key-here'")
        return
    
    # ---- INITIALIZATION ----
    print("📋 PHASE 1: INITIALIZATION")
    print("-" * 70)
    
    nexus = ContextNexus(
        vector_store="faiss",      # Local FAISS for development
        graph_store="networkx",    # In-memory graph
        llm_provider="openai"      # OpenAI for generation
    )
    
    print(f"✅ Initialized ContextNexus")
    print(f"   Vector Store: FAISS (local)")
    print(f"   Graph Store: NetworkX (in-memory)")
    print(f"   LLM Provider: OpenAI ({nexus.config.llm.model})")
    print(f"   Embedding Model: {nexus.config.embedding.model}")
    print(f"   Chunk Size: {nexus.config.chunk_size} tokens")
    print(f"   Chunk Overlap: {nexus.config.chunk_overlap} tokens")
    print()
    
    # ---- DATA INGESTION ----
    print("📁 PHASE 2: DATA INGESTION")
    print("-" * 70)
    
    # Create sample documents to work with
    sample_docs = [
        Document(
            content="""
            # Context Nexus Architecture
            
            Context Nexus is built on three core pillars:
            
            1. **Hybrid Retrieval**: Combines semantic vector search with graph-based reasoning.
               The system uses FAISS for efficient similarity search and NetworkX for 
               relationship traversal.
            
            2. **Token Budget Management**: Automatically manages context window limits by
               intelligently selecting and compressing the most relevant chunks.
            
            3. **Rust Performance**: Critical hot paths are implemented in Rust for 10-100x
               speedup on token counting, vector operations, and graph algorithms.
            
            The ingestion pipeline follows a 5-step process:
            - Document loading from various sources
            - Smart chunking with configurable overlap
            - Embedding generation (OpenAI or local models)
            - Vector indexing with normalization
            - Knowledge graph construction
            """,
            source="architecture.md",
            id="doc_arch"
        ),
        Document(
            content="""
            # Getting Started with Context Nexus
            
            ## Installation
            
            ```bash
            pip install context-nexus
            ```
            
            For production deployments, install with production backends:
            
            ```bash
            pip install context-nexus[production]
            ```
            
            ## Quick Start
            
            1. Import and initialize:
               ```python
               from context_nexus import ContextNexus, Agent
               nexus = ContextNexus()
               ```
            
            2. Ingest your documents:
               ```python
               await nexus.ingest(["./docs/"])
               ```
            
            3. Create an agent and query:
               ```python
               agent = Agent(nexus, token_budget=8000)
               answer = await agent.query("How do I deploy?")
               ```
            
            ## Configuration
            
            You can configure Context Nexus using environment variables:
            - `OPENAI_API_KEY`: Your OpenAI API key
            - `VECTOR_BACKEND`: faiss or qdrant
            - `GRAPH_BACKEND`: networkx or neo4j
            """,
            source="quickstart.md",
            id="doc_start"
        ),
        Document(
            content="""
            # Context Nexus API Reference
            
            ## ContextNexus Class
            
            Main entry point for the SDK.
            
            ### Methods
            
            **ingest(sources, incremental=True)**
            - Ingests documents from paths, URLs, or Document objects
            - Returns: Stats object with counts
            - Async: Yes
            
            **retrieve(query, limit=20, mode='hybrid')**
            - Retrieves relevant chunks for a query
            - mode: 'hybrid', 'vector', or 'graph'
            - Returns: List of RetrievalResult objects
            - Async: Yes
            
            ## Agent Class
            
            AI agent that answers questions about ingested content.
            
            ### Methods
            
            **query(question, trace=False)**
            - Asks a question and gets an answer with sources
            - trace: Include detailed execution trace
            - Returns: Answer object
            - Async: Yes
            
            **analyze(question, output_schema=None)**
            - Analyzes content with optional structured output
            - output_schema: Pydantic model for structured response
            - Returns: Answer or structured object
            - Async: Yes
            """,
            source="api.md",
            id="doc_api"
        )
    ]
    
    print(f"📄 Ingesting {len(sample_docs)} documents...")
    print()
    
    # The full 5-step pipeline happens here: load → chunk → embed → index → graph
    stats = await nexus.ingest(sample_docs)
    
    print(f"✅ Ingestion complete!")
    print(f"   Documents processed: {stats.documents}")
    print(f"   Chunks created: {stats.chunks}")
    print(f"   Graph nodes: {stats.graph_nodes}")
    print(f"   Graph edges: {stats.graph_edges}")
    print()
    
    # ---- SEMANTIC SEARCH & RETRIEVAL ----
    print("🔍 PHASE 3: SEMANTIC SEARCH & RETRIEVAL")
    print("-" * 70)
    
    test_query = "How does hybrid retrieval work?"
    print(f"Query: '{test_query}'")
    print()
    
    # This does hybrid search: vector + graph
    results = await nexus.retrieve(test_query, limit=5)
    
    print(f"✅ Retrieved {len(results)} chunks")
    print()
    print("Top results:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n  [{i}] Score: {result.score:.4f} | Source: {result.source}")
        print(f"      {result.content[:120]}...")
    print()
    
    # ---- AGENT QUERY WITH TOKEN MANAGEMENT ----
    print("🤖 PHASE 4: AGENT QUERY WITH TOKEN MANAGEMENT")
    print("-" * 70)
    
    # Create agent with strict token budget
    agent = Agent(nexus, token_budget=8000)  # Hard limit: never exceed this
    
    print(f"✅ Agent created with token budget: {agent.token_budget}")
    print()
    
    try:
        # Basic question
        print("Query 1: What is Context Nexus?")
        print()
        
        answer1 = await agent.query("What is Context Nexus?")
        
        print("Answer:")
        print(f"  {answer1.text}")
        print()
        print(f"  Confidence: {answer1.confidence:.2%}")
        print(f"  Sources: {len(answer1.sources)}")
        print()
        
        # With tracing for full visibility
        print("Query 2: How do I get started? (with trace)")
        print()
        
        answer2 = await agent.query(
            "How do I get started with Context Nexus?",
            trace=True
        )
        
        print("Answer:")
        print(f"  {answer2.text}")
        print()
        
        if answer2.trace:
            print("📊 Execution Trace:")
            print(f"   Total latency: {answer2.trace.latency_ms}ms")
            print(f"   Tokens used: {answer2.trace.tokens_used}")
            print(f"   Chunks retrieved: {answer2.trace.chunks_retrieved}")
            print()
            print("   Steps:")
            for step in answer2.trace.steps:
                print(f"      - {step['name']}: {step['duration_ms']}ms")
                if 'chunks_found' in step:
                    print(f"        (found {step['chunks_found']} chunks)")
            print()
        
        # ---- SOURCE ATTRIBUTION ----
        print("📚 PHASE 5: SOURCE ATTRIBUTION")
        print("-" * 70)
        
        print(f"Answer was based on {len(answer2.sources)} sources:")
        print()
        
        for i, source in enumerate(answer2.sources, 1):
            print(f"  [{i}] {source.title}")
            print(f"      Relevance: {source.relevance:.2%}")
            print(f"      Chunk ID: {source.chunk_id}")
            if source.excerpt:
                print(f"      Excerpt: {source.excerpt[:80]}...")
            print()
        
        # ---- SUMMARY ----
        print("=" * 70)
        print("✅ WORKFLOW COMPLETE")
        print("=" * 70)
        print()
        print("Key Takeaways:")
        print("  1. ✅ Documents automatically chunked and embedded")
        print("  2. ✅ Hybrid search combines semantic + graph reasoning")
        print("  3. ✅ Token budgets enforced automatically")
        print("  4. ✅ Full traceability for every answer")
        print("  5. ✅ Source attribution with relevance scores")
        print()
        print("Production Tips:")
        print("  - Use `incremental=True` for continuous ingestion")
        print("  - Enable tracing in development, disable in production")
        print("  - Monitor token usage to optimize costs")
        print("  - Set appropriate token budgets for your use case")
        print()
        
    finally:
        await agent.close()
        print("🧹 Cleaned up resources")


if __name__ == "__main__":
    asyncio.run(main())
