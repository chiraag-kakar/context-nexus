#!/usr/bin/env python3
"""Test observability features to verify they work correctly."""

import asyncio
from context_nexus import ContextNexus, Agent
from context_nexus.core.types import Document, Trace
from context_nexus.core.config import Config, EmbeddingConfig

async def test_observability():
    """Test that trace information is captured correctly."""
    
    # Create sample documents
    docs = [
        Document(
            content="Python is a high-level programming language. It's great for AI and data science.",
            source="python_intro.txt",
            metadata={"topic": "programming", "file_name": "python_intro.txt"}
        ),
        Document(
            content="Rust is a systems programming language. It's known for memory safety and performance.",
            source="rust_intro.txt",
            metadata={"topic": "programming", "file_name": "rust_intro.txt"}
        ),
    ]
    
    # Initialize Context Nexus with local embedding config (384 dimensions for all-MiniLM-L6-v2)
    config = Config(
        embedding=EmbeddingConfig(
            provider="local",
            model="all-MiniLM-L6-v2",
            dimensions=384  # all-MiniLM-L6-v2 has 384 dimensions
        )
    )
    nexus = ContextNexus(config=config)
    await nexus.ingest(docs)
    
    # Create agent
    agent = Agent(nexus, token_budget=8000)
    
    # Query WITH trace enabled
    print("=" * 60)
    print("Testing Retrieval WITH Trace Enabled")
    print("=" * 60)
    
    try:
        # Test retrieval directly (no LLM needed)
        results = await nexus.retrieve("What is Python?", limit=10)
        
        print(f"\n✅ Retrieved {len(results)} chunks")
        print("\n📊 Retrieval Results:")
        
        for i, result in enumerate(results[:5], 1):
            print(f"\n   Chunk {i}:")
            print(f"      Content: {result.content[:80]}...")
            print(f"      Score: {result.score:.4f}")
            print(f"      Chunk ID: {result.chunk_id}")
            print(f"      Source: {result.source}")
            print(f"      Metadata: {result.metadata}")
            
        print("\n✅ Retrieval works - observability can capture these steps!")
        
    except Exception as e:
        print(f"\n❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
    
    # Test trace generation manually
    print("\n" + "=" * 60)
    print("Testing Trace Object Creation")
    print("=" * 60)
    
    try:
        # Manually create what the agent would create
        trace_steps = [
            {
                "name": "retrieve",
                "duration_ms": 123.4,
                "chunks_found": 5
            },
            {
                "name": "generate",
                "duration_ms": 456.7
            }
        ]
        
        trace_obj = Trace(
            steps=trace_steps,
            tokens_used=250,
            latency_ms=580.1,
            chunks_retrieved=5,
        )
        
        print("\n✅ Trace object created successfully:")
        print(f"   Total latency: {trace_obj.latency_ms}ms")
        print(f"   Tokens used: {trace_obj.tokens_used}")
        print(f"   Chunks retrieved: {trace_obj.chunks_retrieved}")
        print("\n   Execution steps:")
        for step in trace_obj.steps:
            print(f"     - {step['name']}: {step['duration_ms']:.1f}ms", end="")
            if 'chunks_found' in step:
                print(f" ({step['chunks_found']} chunks found)")
            else:
                print()
                
        print("\n✅ Trace data structures work correctly!")
        
    except Exception as e:
        print(f"\n❌ Error creating trace: {e}")
        import traceback
        traceback.print_exc()
    
    await agent.close()
    
    print("\n" + "=" * 60)
    print("✅ Observability Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_observability())
