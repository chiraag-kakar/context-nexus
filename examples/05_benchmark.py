"""
Benchmark example - demonstrates Context Nexus performance at scale.

This processes a large dataset (100+ documents, 1000+ chunks) to show:
- Ingestion throughput
- Search performance
- Query latency
- Memory efficiency

Run this to benchmark your system and validate production readiness.
"""

import asyncio
import os
import time
from pathlib import Path
from context_nexus import ContextNexus, Agent
from context_nexus.core.types import Document


def generate_large_dataset(num_docs=100):
    """
    Generate a realistic dataset for benchmarking.
    
    Simulates a technical documentation corpus with varied content.
    """
    docs = []
    
    topics = [
        ("authentication", "API authentication and security best practices"),
        ("database", "Database schema design and optimization"),
        ("deployment", "Deployment pipelines and CI/CD workflows"),
        ("monitoring", "Application monitoring and observability"),
        ("scaling", "Horizontal scaling and load balancing"),
        ("caching", "Caching strategies and Redis integration"),
        ("testing", "Unit testing and integration testing"),
        ("api_design", "RESTful API design principles"),
        ("security", "Security best practices and OWASP guidelines"),
        ("performance", "Performance optimization and profiling"),
    ]
    
    for i in range(num_docs):
        topic_name, topic_desc = topics[i % len(topics)]
        
        content = f"""
        # {topic_name.title()} Guide - Part {i+1}
        
        ## Overview
        
        {topic_desc}. This document covers implementation details,
        best practices, and common pitfalls to avoid.
        
        ## Key Concepts
        
        1. **Concept A**: Understanding the fundamentals of {topic_name}
           - Definition and scope
           - When to use this approach
           - Common use cases
        
        2. **Concept B**: Advanced {topic_name} patterns
           - Implementation strategies
           - Performance considerations
           - Scalability concerns
        
        3. **Concept C**: {topic_name} in production
           - Monitoring and alerting
           - Debugging techniques
           - Optimization strategies
        
        ## Implementation Example
        
        ```python
        # Example code for {topic_name}
        class {topic_name.title()}Handler:
            def __init__(self, config):
                self.config = config
            
            async def process(self, data):
                # Process the data
                result = await self._internal_process(data)
                return result
            
            async def _internal_process(self, data):
                # Implementation details
                pass
        ```
        
        ## Best Practices
        
        - Always validate input data
        - Implement proper error handling
        - Use async/await for I/O operations
        - Monitor performance metrics
        - Write comprehensive tests
        
        ## Common Pitfalls
        
        1. Not handling edge cases properly
        2. Ignoring performance implications
        3. Poor error handling
        4. Insufficient logging
        5. Missing tests
        
        ## Performance Considerations
        
        When implementing {topic_name}, consider:
        - Memory usage and object pooling
        - Database connection pooling
        - Caching strategies
        - Async operations
        - Load testing
        
        ## Related Topics
        
        - Topic {(i-1) % len(topics)}: Related concept
        - Topic {(i+1) % len(topics)}: Next steps
        - Topic {(i+2) % len(topics)}: Advanced patterns
        """
        
        doc = Document(
            content=content,
            source=f"docs/{topic_name}_{i+1:03d}.md",
            id=f"doc_{i:04d}"
        )
        docs.append(doc)
    
    return docs


async def main():
    print("=" * 70)
    print("CONTEXT NEXUS PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY environment variable")
        return
    
    # Configuration
    num_documents = 100
    num_queries = 20
    
    print(f"Configuration:")
    print(f"  Documents: {num_documents}")
    print(f"  Queries: {num_queries}")
    print()
    
    # ---- BENCHMARK 1: INGESTION THROUGHPUT ----
    print("📥 BENCHMARK 1: INGESTION THROUGHPUT")
    print("-" * 70)
    
    nexus = ContextNexus()
    
    print(f"Generating {num_documents} documents...")
    docs = generate_large_dataset(num_documents)
    total_chars = sum(len(doc.content) for doc in docs)
    print(f"  Total size: {total_chars:,} characters (~{total_chars//1000}KB)")
    print()
    
    print("Starting ingestion...")
    start_time = time.time()
    
    stats = await nexus.ingest(docs)
    
    ingestion_time = time.time() - start_time
    
    print(f"✅ Ingestion complete!")
    print(f"   Time: {ingestion_time:.2f} seconds")
    print(f"   Documents: {stats.documents}")
    print(f"   Chunks: {stats.chunks}")
    print(f"   Graph nodes: {stats.graph_nodes}")
    print(f"   Graph edges: {stats.graph_edges}")
    print()
    print(f"📊 Throughput:")
    print(f"   {stats.documents/ingestion_time:.1f} docs/sec")
    print(f"   {stats.chunks/ingestion_time:.1f} chunks/sec")
    print(f"   {total_chars/ingestion_time:,.0f} chars/sec")
    print()
    
    # ---- BENCHMARK 2: SEARCH PERFORMANCE ----
    print("🔍 BENCHMARK 2: SEARCH PERFORMANCE")
    print("-" * 70)
    
    test_queries = [
        "How do I implement authentication?",
        "What are database optimization techniques?",
        "Explain deployment best practices",
        "How to monitor application performance?",
        "What are caching strategies?",
    ]
    
    search_times = []
    
    for query in test_queries:
        start = time.time()
        results = await nexus.retrieve(query, limit=10)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        search_times.append(elapsed)
        
        print(f"  Query: '{query[:40]}...'")
        print(f"    Time: {elapsed:.0f}ms | Results: {len(results)}")
    
    print()
    print(f"📊 Search Performance:")
    print(f"   Average: {sum(search_times)/len(search_times):.0f}ms")
    print(f"   Min: {min(search_times):.0f}ms")
    print(f"   Max: {max(search_times):.0f}ms")
    print()
    
    # ---- BENCHMARK 3: AGENT QUERY LATENCY ----
    print("🤖 BENCHMARK 3: AGENT QUERY LATENCY")
    print("-" * 70)
    
    agent = Agent(nexus, token_budget=8000)
    
    query_times = []
    token_usage = []
    
    try:
        for i in range(5):  # Sample 5 queries
            query = test_queries[i % len(test_queries)]
            
            start = time.time()
            answer = await agent.query(query, trace=True)
            elapsed = (time.time() - start) * 1000
            
            query_times.append(elapsed)
            if answer.trace:
                token_usage.append(answer.trace.tokens_used)
            
            print(f"  [{i+1}] {query[:50]}...")
            print(f"      Total: {elapsed:.0f}ms")
            if answer.trace:
                print(f"      Retrieve: {answer.trace.steps[0]['duration_ms']}ms")
                print(f"      Generate: {answer.trace.steps[1]['duration_ms']}ms")
                print(f"      Tokens: {answer.trace.tokens_used}")
        
        print()
        print(f"📊 Agent Performance:")
        print(f"   Average latency: {sum(query_times)/len(query_times):.0f}ms")
        print(f"   Average tokens: {sum(token_usage)/len(token_usage):.0f}")
        print()
        
    finally:
        await agent.close()
    
    # ---- BENCHMARK 4: MEMORY & RESOURCE USAGE ----
    print("💾 BENCHMARK 4: RESOURCE USAGE")
    print("-" * 70)
    
    print(f"  Vector index size: {nexus._vector_indexer.total_chunks} chunks")
    print(f"  Graph size: {nexus._graph_indexer.total_nodes} nodes, {nexus._graph_indexer.total_edges} edges")
    print()
    
    # ---- SUMMARY & RECOMMENDATIONS ----
    print("=" * 70)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 70)
    print()
    print("Performance Summary:")
    print(f"  Ingestion: {stats.documents/ingestion_time:.1f} docs/sec, {stats.chunks/ingestion_time:.1f} chunks/sec")
    print(f"  Search: {sum(search_times)/len(search_times):.0f}ms average")
    print(f"  Agent queries: {sum(query_times)/len(query_times):.0f}ms average")
    print()
    
    # Production readiness assessment
    avg_search = sum(search_times)/len(search_times)
    avg_query = sum(query_times)/len(query_times)
    
    print("Production Readiness Assessment:")
    
    if avg_search < 100:
        print("  ✅ Search performance: Excellent (<100ms)")
    elif avg_search < 500:
        print("  ⚠️  Search performance: Good (100-500ms)")
    else:
        print("  ❌ Search performance: Needs optimization (>500ms)")
    
    if avg_query < 2000:
        print("  ✅ Query latency: Excellent (<2s)")
    elif avg_query < 5000:
        print("  ⚠️  Query latency: Acceptable (2-5s)")
    else:
        print("  ❌ Query latency: Needs optimization (>5s)")
    
    print()
    print("Scaling Recommendations:")
    print(f"  Current size: {stats.documents} docs, {stats.chunks} chunks")
    
    if stats.documents < 1000:
        print("  📝 For <1K docs: Current setup (FAISS + NetworkX) is optimal")
    elif stats.documents < 10000:
        print("  📈 For 1-10K docs: Consider Qdrant for persistence")
    else:
        print("  🚀 For >10K docs: Use Qdrant + Neo4j for production")
    
    print()
    print("Next Steps:")
    print("  1. Run this benchmark on your actual data")
    print("  2. Monitor token costs (check OpenAI dashboard)")
    print("  3. Optimize chunk_size/overlap for your use case")
    print("  4. Consider local embeddings for cost reduction")
    print()


if __name__ == "__main__":
    asyncio.run(main())
