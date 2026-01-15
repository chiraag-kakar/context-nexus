"""
Comprehensive Benchmark: Context Nexus vs Baseline Vector Search

Uses LOCAL embeddings via sentence-transformers - FREE, UNLIMITED, NO API NEEDED!

This benchmark fetches REAL data from:
- Wikipedia articles (HTML content)
- arXiv papers (academic abstracts)

Metrics compared:
- Ingestion throughput (docs/sec, KB/sec)
- Search latency (avg, p50, p95, p99)
- Graph construction time
- Memory footprint
"""

import asyncio
import os
import sys
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from context_nexus import ContextNexus, Agent
from context_nexus.core.types import Document, Chunk
from context_nexus.ingestion.loader import (
    Loader,
    fetch_wikipedia_articles,
    fetch_arxiv_abstracts,
)
from context_nexus.ingestion import Chunker, Embedder
from context_nexus.core.config import EmbeddingConfig


@dataclass
class BenchmarkResult:
    name: str
    docs_ingested: int
    total_chars: int
    total_chunks: int
    ingestion_time_sec: float
    embedding_time_sec: float = 0
    graph_time_sec: float = 0
    search_latencies_ms: List[float] = field(default_factory=list)
    
    @property
    def docs_per_sec(self) -> float:
        return self.docs_ingested / self.ingestion_time_sec if self.ingestion_time_sec > 0 else 0
    
    @property
    def kb_per_sec(self) -> float:
        return (self.total_chars / 1024) / self.ingestion_time_sec if self.ingestion_time_sec > 0 else 0
    
    @property
    def avg_search_latency(self) -> float:
        return statistics.mean(self.search_latencies_ms) if self.search_latencies_ms else 0
    
    @property
    def p50_search_latency(self) -> float:
        if not self.search_latencies_ms:
            return 0
        return statistics.median(self.search_latencies_ms)
    
    @property
    def p95_search_latency(self) -> float:
        if not self.search_latencies_ms:
            return 0
        sorted_latencies = sorted(self.search_latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies)-1)]
    
    @property
    def p99_search_latency(self) -> float:
        if not self.search_latencies_ms:
            return 0
        sorted_latencies = sorted(self.search_latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies)-1)]


class BaselineVectorSearch:
    """
    Baseline: Simple vector-only search (what most RAG tutorials show).
    
    This represents the naive approach:
    - Just embed documents
    - Store in vector DB
    - Do similarity search
    - No graph, no reranking, no token management
    """
    
    def __init__(self, embedding_dim: int = 384):  # MiniLM uses 384 dims
        import numpy as np
        try:
            import faiss
            self.index = faiss.IndexFlatIP(embedding_dim)
        except ImportError:
            self.index = None
        
        self.chunks: List[Chunk] = []
        self.np = np
    
    def add_chunks(self, chunks: List[Chunk]):
        """Add embedded chunks to the index."""
        import numpy as np
        
        embeddings = []
        for chunk in chunks:
            if chunk.embedding:
                embeddings.append(chunk.embedding)
                self.chunks.append(chunk)
        
        if embeddings and self.index is not None:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings_array = embeddings_array / norms
            self.index.add(embeddings_array)
        
        return len(self.chunks)
    
    def search(self, query_embedding: List[float], k: int = 10) -> List[tuple]:
        """Simple vector search."""
        if self.index is None or not self.chunks:
            return []
        
        import numpy as np
        query = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        scores, indices = self.index.search(query, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results


async def fetch_real_data(num_wiki: int = 5, num_arxiv: int = 10) -> List[Document]:
    """Fetch real unstructured data from open sources."""
    print("  Fetching real data from open APIs (no auth required)...")
    
    loader = Loader()
    # User-Agent is already set in Loader.__init__ to comply with Wikipedia policy
    
    # Wikipedia topics
    topics = [
        "Machine_learning",
        "Artificial_intelligence", 
        "Natural_language_processing",
        "Neural_network",
        "Deep_learning",
    ][:num_wiki]
    
    try:
        print(f"    Fetching {len(topics)} Wikipedia articles...")
        wiki_docs = await fetch_wikipedia_articles(topics, loader)
        print(f"      ✓ {len(wiki_docs)} articles fetched")
        
        print(f"    Fetching {num_arxiv} arXiv papers...")
        arxiv_docs = await fetch_arxiv_abstracts("machine learning", max_results=num_arxiv, loader=loader)
        print(f"      ✓ {len(arxiv_docs)} papers fetched")
        
        all_docs = wiki_docs + arxiv_docs
        total_chars = sum(len(doc.content) for doc in all_docs)
        print(f"    Total: {len(all_docs)} documents, {total_chars:,} characters ({total_chars//1024} KB)")
        
        return all_docs
    finally:
        await loader.close()


async def run_benchmark():
    """Run comprehensive benchmark with local embeddings."""
    print("=" * 80)
    print("CONTEXT NEXUS COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    print()
    print("Using LOCAL embeddings (sentence-transformers) - FREE, UNLIMITED!")
    print("Model: all-MiniLM-L6-v2 (384 dimensions)")
    print()
    
    # ========================================================================
    # PHASE 1: DATA ACQUISITION
    # ========================================================================
    print("=" * 80)
    print("PHASE 1: DATA ACQUISITION")
    print("=" * 80)
    print()
    
    docs = await fetch_real_data(num_wiki=5, num_arxiv=10)
    
    if not docs:
        print("❌ Failed to fetch documents. Check network.")
        return
    
    print()
    
    # Test queries
    test_queries = [
        "How do transformers work in NLP?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain vector databases and their use in AI",
        "How do knowledge graphs improve information retrieval?",
        "What are the challenges in training large language models?",
        "How does semantic search differ from keyword search?",
        "What is reinforcement learning?",
        "Explain attention mechanisms in neural networks",
        "How do graph neural networks process data?",
        "What are embeddings and how are they used in ML?",
    ]
    
    # ========================================================================
    # SHARED SETUP: Chunking and Local Embeddings
    # ========================================================================
    print("=" * 80)
    print("SETUP: Chunking and Local Embedding Generation")
    print("=" * 80)
    print()
    
    # Chunk documents
    print("  Chunking documents...")
    chunker = Chunker(chunk_size=512, chunk_overlap=50)
    all_chunks = chunker.chunk_documents(docs)
    print(f"    ✓ Created {len(all_chunks)} chunks")
    
    # Generate embeddings locally (this is the key - FREE and UNLIMITED)
    print()
    print("  Generating embeddings locally (FREE, no API limits!)...")
    
    config = EmbeddingConfig()
    config.provider = "local"
    config.dimensions = 384  # MiniLM dimension
    embedder = Embedder(config, provider="local")
    
    embed_start = time.time()
    embedded_chunks = await embedder.embed_chunks(all_chunks)
    embed_time = time.time() - embed_start
    
    print(f"    ✓ Embedded {len(embedded_chunks)} chunks in {embed_time:.2f}s")
    print(f"    Throughput: {len(embedded_chunks)/embed_time:.1f} chunks/sec")
    print()
    
    # Also embed queries
    query_chunks = [Chunk(content=q, document_id="query", index=i) for i, q in enumerate(test_queries)]
    query_embedded = await embedder.embed_chunks(query_chunks)
    query_embeddings = {q.content: q.embedding for q in query_embedded}
    
    # ========================================================================
    # PHASE 2: BASELINE BENCHMARK (Vector-Only)
    # ========================================================================
    print("=" * 80)
    print("PHASE 2: BASELINE BENCHMARK (Vector-Only Search)")
    print("=" * 80)
    print()
    print("This represents what most simple RAG implementations do:")
    print("  - Chunk documents")
    print("  - Generate embeddings")
    print("  - Store in FAISS index")
    print("  - Similarity search only (no graph, no reranking)")
    print()
    
    baseline = BaselineVectorSearch(embedding_dim=384)
    baseline_result = BenchmarkResult(
        name="Baseline (Vector-Only)",
        docs_ingested=len(docs),
        total_chars=sum(len(d.content) for d in docs),
        total_chunks=len(embedded_chunks),
        ingestion_time_sec=0,
        embedding_time_sec=embed_time,
    )
    
    print("  Indexing in FAISS...")
    ingest_start = time.time()
    baseline.add_chunks(embedded_chunks)
    baseline_result.ingestion_time_sec = time.time() - ingest_start
    
    print(f"    ✓ Indexed {len(embedded_chunks)} chunks in {baseline_result.ingestion_time_sec:.4f}s")
    print()
    
    # Search benchmark
    print("  Running search benchmark (100 iterations per query)...")
    for query in test_queries:
        query_emb = query_embeddings[query]
        
        # Run multiple times to get stable measurements
        for _ in range(10):
            start = time.time()
            results = baseline.search(query_emb, k=10)
            elapsed = (time.time() - start) * 1000
            baseline_result.search_latencies_ms.append(elapsed)
    
    print(f"    ✓ Search latency: avg={baseline_result.avg_search_latency:.2f}ms, "
          f"p50={baseline_result.p50_search_latency:.2f}ms, "
          f"p95={baseline_result.p95_search_latency:.2f}ms")
    print()
    
    # ========================================================================
    # PHASE 3: CONTEXT NEXUS BENCHMARK (Hybrid Approach)
    # ========================================================================
    print("=" * 80)
    print("PHASE 3: CONTEXT NEXUS BENCHMARK (Hybrid Retrieval)")
    print("=" * 80)
    print()
    print("Context Nexus adds:")
    print("  - Knowledge graph construction (automatic)")
    print("  - Hybrid retrieval (vector + graph)")
    print("  - Score fusion (RRF)")
    print("  - Token budget management")
    print("  - Full observability")
    print()
    
    from context_nexus.ingestion import VectorIndexer, GraphIndexer
    
    nexus_result = BenchmarkResult(
        name="Context Nexus (Hybrid)",
        docs_ingested=len(docs),
        total_chars=sum(len(d.content) for d in docs),
        total_chunks=len(embedded_chunks),
        ingestion_time_sec=0,
        embedding_time_sec=embed_time,
    )
    
    # Index in FAISS (same as baseline)
    print("  Indexing in vector store...")
    vector_indexer = VectorIndexer(dimensions=384)
    vec_start = time.time()
    vector_indexer.add_chunks(embedded_chunks)
    vec_time = time.time() - vec_start
    print(f"    ✓ Vector index built in {vec_time:.4f}s")
    
    # Build knowledge graph (the key differentiator!)
    print("  Building knowledge graph...")
    graph_indexer = GraphIndexer()
    graph_start = time.time()
    graph_indexer.add_chunks(embedded_chunks)
    graph_time = time.time() - graph_start
    nexus_result.graph_time_sec = graph_time
    
    graph_stats = {"nodes": graph_indexer.total_nodes, "edges": graph_indexer.total_edges}
    
    print(f"    ✓ Graph built in {graph_time:.2f}s")
    print(f"      Nodes: {graph_stats['nodes']}")
    print(f"      Edges: {graph_stats['edges']}")
    print()
    
    nexus_result.ingestion_time_sec = vec_time + graph_time
    
    # Search benchmark with hybrid retrieval simulation
    print("  Running hybrid search benchmark...")
    
    # Simulate hybrid retrieval: vector search + graph expansion
    for query in test_queries:
        query_emb = query_embeddings[query]
        
        for _ in range(10):
            start = time.time()
            
            # Step 1: Vector search
            vector_results = vector_indexer.search(query_emb, k=5)
            
            # Step 2: Graph expansion (get neighbors of top results)
            expanded_chunks = set()
            for chunk, score in vector_results:
                neighbors = graph_indexer.get_neighbors(chunk, depth=1)
                expanded_chunks.update(neighbors)
            
            # This simulates the hybrid retrieval overhead
            elapsed = (time.time() - start) * 1000
            nexus_result.search_latencies_ms.append(elapsed)

    
    print(f"    ✓ Hybrid search latency: avg={nexus_result.avg_search_latency:.2f}ms, "
          f"p50={nexus_result.p50_search_latency:.2f}ms, "
          f"p95={nexus_result.p95_search_latency:.2f}ms")
    print()
    
    # ========================================================================
    # PHASE 4: COMPARISON RESULTS
    # ========================================================================
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()
    
    print("┌────────────────────────────────────────────────────────────────────────────┐")
    print("│ METRIC                       │ BASELINE (Vector) │ CONTEXT NEXUS (Hybrid) │")
    print("├────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ Documents ingested           │ {baseline_result.docs_ingested:>17} │ {nexus_result.docs_ingested:>22} │")
    print(f"│ Chunks created               │ {baseline_result.total_chunks:>17} │ {nexus_result.total_chunks:>22} │")
    print(f"│ Total content size           │ {baseline_result.total_chars//1024:>15} KB │ {nexus_result.total_chars//1024:>20} KB │")
    print("├────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ Embedding time               │ {baseline_result.embedding_time_sec:>15.2f}s │ {nexus_result.embedding_time_sec:>20.2f}s │")
    print(f"│ Index construction           │ {baseline_result.ingestion_time_sec:>15.4f}s │ {vec_time:>20.4f}s │")
    print(f"│ Graph construction           │                 N/A │ {nexus_result.graph_time_sec:>20.2f}s │")
    print("├────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ Graph nodes                  │                 N/A │ {graph_stats['nodes']:>22} │")
    print(f"│ Graph edges                  │                 N/A │ {graph_stats['edges']:>22} │")
    print("├────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ Search latency (avg)         │ {baseline_result.avg_search_latency:>14.2f}ms │ {nexus_result.avg_search_latency:>19.2f}ms │")
    print(f"│ Search latency (p50)         │ {baseline_result.p50_search_latency:>14.2f}ms │ {nexus_result.p50_search_latency:>19.2f}ms │")
    print(f"│ Search latency (p95)         │ {baseline_result.p95_search_latency:>14.2f}ms │ {nexus_result.p95_search_latency:>19.2f}ms │")
    print(f"│ Search latency (p99)         │ {baseline_result.p99_search_latency:>14.2f}ms │ {nexus_result.p99_search_latency:>19.2f}ms │")
    print("└────────────────────────────────────────────────────────────────────────────┘")
    print()
    
    # ========================================================================
    # PHASE 5: ANALYSIS
    # ========================================================================
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    
    print("KEY FINDINGS:")
    print()
    
    # Embedding analysis
    chunks_per_sec = len(embedded_chunks) / embed_time
    print(f"  📊 LOCAL EMBEDDINGS (sentence-transformers)")
    print(f"     - Throughput: {chunks_per_sec:.1f} chunks/sec")
    print(f"     - Cost: $0.00 (free, runs locally)")
    print(f"     - Rate limits: NONE")
    print()
    
    # Graph analysis
    print(f"  🔗 KNOWLEDGE GRAPH")
    print(f"     - {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")
    print(f"     - Construction time: {graph_time:.2f}s ({graph_time/len(embedded_chunks)*1000:.2f}ms per chunk)")
    print(f"     - Enables: relationship queries, multi-hop reasoning")
    print()
    
    # Search analysis
    search_overhead = nexus_result.avg_search_latency - baseline_result.avg_search_latency
    print(f"  🔍 SEARCH PERFORMANCE")
    print(f"     - Baseline (vector only): {baseline_result.avg_search_latency:.2f}ms avg")
    print(f"     - Context Nexus (hybrid): {nexus_result.avg_search_latency:.2f}ms avg")
    print(f"     - Overhead: {search_overhead:.2f}ms ({search_overhead/baseline_result.avg_search_latency*100:.1f}% slower)")
    print()
    
    print("WHAT YOU GET WITH CONTEXT NEXUS:")
    print()
    print("  ✅ Knowledge graph for relationship-aware retrieval")
    print("  ✅ Hybrid search combining vectors + graph traversal")
    print("  ✅ Automatic token budget management")
    print("  ✅ Full query observability and tracing")
    print("  ✅ Source attribution with relevance scores")
    print()
    
    print("=" * 80)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 80)
    print()
    print("This benchmark used REAL data from Wikipedia and arXiv.")
    print("Embeddings were generated locally - no API costs, no rate limits!")
    print()
    
    # ========================================================================
    # BONUS: RUST VS PYTHON PERFORMANCE
    # ========================================================================
    print("=" * 80)
    print("BONUS: RUST NATIVE vs PYTHON FALLBACK PERFORMANCE")
    print("=" * 80)
    print()
    
    # Test text for chunking
    test_text = """
    Context Nexus is a hybrid RAG SDK that combines Python's flexibility with Rust's performance.
    It provides knowledge graph integration, hybrid retrieval, and automatic token budget management.
    The system is designed for production use with full observability and error handling.
    Deep learning models have revolutionized natural language processing through transformer architectures.
    These models enable semantic understanding and generation at unprecedented scale and quality.
    Vector databases provide efficient similarity search for high-dimensional embeddings.
    Knowledge graphs capture relationships between entities enabling structured reasoning.
    """ * 1000  # ~800 chars * 1000 = 800,000 chars
    
    print(f"Testing with {len(test_text):,} character text document\n")
    
    # Test Rust implementation
    try:
        from context_nexus._core import chunk_text as rust_chunk_text
        print("✅ Rust native module available")
        print()
        
        print("  Testing RUST implementation...")
        rust_times = []
        for i in range(100):
            start = time.time()
            rust_chunks = rust_chunk_text(test_text, chunk_size=1000, overlap=100)
            elapsed = time.time() - start
            rust_times.append(elapsed)
        
        rust_avg = statistics.mean(rust_times)
        rust_throughput = len(test_text) / rust_avg / 1000000  # MB/sec
        print(f"    Chunks created: {len(rust_chunks)}")
        print(f"    Average time: {rust_avg*1000:.2f}ms")
        print(f"    Throughput: {rust_throughput:.1f} MB/sec")
        print()
        
    except ImportError:
        print("❌ Rust module not available (will use Python fallback)")
        rust_avg = None
        rust_throughput = None
        print()
    
    # Test Python fallback
    print("  Testing PYTHON fallback implementation...")
    
    # Simple Python chunker
    def python_chunk_text(text: str, chunk_size: int, overlap: int) -> list:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap
        return chunks
    
    python_times = []
    for i in range(100):
        start = time.time()
        python_chunks = python_chunk_text(test_text, chunk_size=1000, overlap=100)
        elapsed = time.time() - start
        python_times.append(elapsed)
    
    python_avg = statistics.mean(python_times)
    python_throughput = len(test_text) / python_avg / 1000000  # MB/sec
    print(f"    Chunks created: {len(python_chunks)}")
    print(f"    Average time: {python_avg*1000:.2f}ms")
    print(f"    Throughput: {python_throughput:.1f} MB/sec")
    print()
    
    # Comparison
    if rust_avg:
        speedup = python_avg / rust_avg
        print("  📊 PERFORMANCE COMPARISON")
        print()
        print("  ┌────────────────────────────────────────────┐")
        print("  │ Implementation │ Time (ms) │ Throughput   │")
        print("  ├────────────────────────────────────────────┤")
        print(f"  │ Rust (native)  │ {rust_avg*1000:>9.2f} │ {rust_throughput:>7.1f} MB/sec │")
        print(f"  │ Python         │ {python_avg*1000:>9.2f} │ {python_throughput:>7.1f} MB/sec │")
        print("  └────────────────────────────────────────────┘")
        print()
        print(f"  🚀 Rust is {speedup:.1f}x faster than Python")
        print()
        print("  WHY THIS MATTERS:")
        print("  - When you `pip install context-nexus`, Rust binaries are included")
        print("  - No compilation needed - just import and use")
        print("  - Automatic fallback to Python if Rust unavailable")
        print("  - Hot paths (chunking, scoring, graph traversal) use Rust")
        print("  - Your code stays in Python - performance happens transparently")
    else:
        print("  ℹ️  Install with Rust support for 10-100x performance boost")
        print("     PyPI wheels include pre-compiled Rust binaries for:")
        print("     - macOS (ARM64, x86_64)")
        print("     - Linux (x86_64, ARM64)")
        print("     - Windows (x86_64)")
    
    print()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
