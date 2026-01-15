"""Integration tests - works without external APIs in CI."""


from context_nexus.core.types import Document, Chunk
from context_nexus.ingestion import Chunker, VectorIndexer, GraphIndexer


# ============================================================================
# UNIT TESTS (No external dependencies - always run in CI)
# ============================================================================

class TestChunker:
    """Test document chunking."""
    
    def test_chunks_document(self):
        """Test that chunker properly splits documents."""
        doc = Document(
            content="This is a test. " * 100,  # Long enough to chunk
            source="test.md",
            id="doc1"
        )
        
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_documents([doc])
        
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.content for c in chunks)
        assert all(c.document_id == "doc1" for c in chunks)
    
    def test_handles_short_content(self):
        """Test short content (no chunking needed)."""
        doc = Document(content="Short text.", source="short.md", id="s1")
        chunker = Chunker(chunk_size=1000)
        chunks = chunker.chunk_documents([doc])
        
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."
    
    def test_preserves_metadata(self):
        """Test metadata preservation."""
        doc = Document(
            content="Test content. " * 50,
            source="meta.md",
            id="m1",
            metadata={"author": "test"}
        )
        
        chunker = Chunker(chunk_size=100)
        chunks = chunker.chunk_documents([doc])
        
        assert all(c.metadata.get("author") == "test" for c in chunks)


class TestVectorIndexer:
    """Test FAISS vector indexing."""
    
    def test_adds_chunks(self):
        """Test adding chunks to index."""
        indexer = VectorIndexer(dimensions=384)
        
        chunks = [
            Chunk(content="test1", document_id="d1", index=0, embedding=[0.1] * 384),
            Chunk(content="test2", document_id="d1", index=1, embedding=[0.2] * 384),
        ]
        
        count = indexer.add_chunks(chunks)
        assert count == 2
        assert indexer.total_chunks == 2
    
    def test_search_returns_results(self):
        """Test similarity search."""
        indexer = VectorIndexer(dimensions=384)
        
        # Create chunks with distinct embeddings
        chunks = [
            Chunk(content="python programming", document_id="d1", index=0, 
                  embedding=[1.0] + [0.0] * 383),
            Chunk(content="machine learning", document_id="d1", index=1, 
                  embedding=[0.0] + [1.0] + [0.0] * 382),
        ]
        indexer.add_chunks(chunks)
        
        # Search for something similar to first chunk
        query = [1.0] + [0.0] * 383
        results = indexer.search(query, k=2)
        
        assert len(results) == 2
        assert results[0][0].content == "python programming"
        assert results[0][1] > results[1][1]  # Higher score for match
    
    def test_handles_empty_index(self):
        """Test search on empty index."""
        indexer = VectorIndexer(dimensions=384)
        query = [0.1] * 384
        results = indexer.search(query, k=5)
        
        assert results == []


class TestGraphIndexer:
    """Test NetworkX graph indexing."""
    
    def test_builds_graph(self):
        """Test graph construction."""
        indexer = GraphIndexer()
        
        chunks = [
            Chunk(content="chunk1", document_id="doc1", index=0),
            Chunk(content="chunk2", document_id="doc1", index=1),
            Chunk(content="chunk3", document_id="doc1", index=2),
        ]
        
        count = indexer.add_chunks(chunks)
        assert count == 3
        assert indexer.total_nodes == 3
        assert indexer.total_edges == 2  # Sequential links
    
    def test_links_sequential_chunks(self):
        """Test that sequential chunks are linked."""
        indexer = GraphIndexer()
        
        chunks = [
            Chunk(content="first", document_id="doc1", index=0),
            Chunk(content="second", document_id="doc1", index=1),
        ]
        indexer.add_chunks(chunks)
        
        # Check edge exists
        neighbors = indexer.get_neighbors(chunks[0], depth=1)
        assert "doc1:1" in neighbors
    
    def test_separate_documents(self):
        """Test chunks from different docs aren't linked."""
        indexer = GraphIndexer()
        
        chunks = [
            Chunk(content="doc1 chunk", document_id="doc1", index=0),
            Chunk(content="doc2 chunk", document_id="doc2", index=0),
        ]
        indexer.add_chunks(chunks)
        
        assert indexer.total_nodes == 2
        assert indexer.total_edges == 0  # No cross-doc links


class TestCoreTypes:
    """Test core data types."""
    
    def test_document_creation(self):
        """Test Document dataclass."""
        doc = Document(
            content="Test content",
            source="test.md",
            id="doc1",
            metadata={"key": "value"}
        )
        
        assert doc.content == "Test content"
        assert doc.source == "test.md"
        assert doc.id == "doc1"
        assert doc.metadata["key"] == "value"
    
    def test_chunk_creation(self):
        """Test Chunk dataclass."""
        chunk = Chunk(
            content="Chunk content",
            document_id="doc1",
            index=0,
            embedding=[0.1, 0.2, 0.3]
        )
        
        assert chunk.content == "Chunk content"
        assert chunk.document_id == "doc1"
        assert chunk.index == 0
        assert len(chunk.embedding) == 3


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance benchmarks for CI."""
    
    def test_chunking_performance(self):
        """Test chunking is fast."""
        import time
        
        # Create large document
        large_doc = Document(
            content="This is a sentence. " * 10000,
            source="large.md",
            id="large"
        )
        
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        
        start = time.time()
        chunks = chunker.chunk_documents([large_doc])
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"Chunking too slow: {elapsed:.2f}s"
        assert len(chunks) > 100
    
    def test_indexing_performance(self):
        """Test vector indexing is fast."""
        import time
        
        indexer = VectorIndexer(dimensions=384)
        
        # Create 1000 chunks
        chunks = [
            Chunk(
                content=f"chunk {i}",
                document_id="d1",
                index=i,
                embedding=[float(i % 10) / 10] * 384
            )
            for i in range(1000)
        ]
        
        start = time.time()
        indexer.add_chunks(chunks)
        elapsed = time.time() - start
        
        assert elapsed < 0.5, f"Indexing too slow: {elapsed:.2f}s"
    
    def test_search_performance(self):
        """Test search latency is low."""
        import time
        
        indexer = VectorIndexer(dimensions=384)
        
        # Add chunks
        chunks = [
            Chunk(
                content=f"chunk {i}",
                document_id="d1",
                index=i,
                embedding=[float(i % 10) / 10] * 384
            )
            for i in range(1000)
        ]
        indexer.add_chunks(chunks)
        
        # Run searches
        query = [0.5] * 384
        latencies = []
        
        for _ in range(100):
            start = time.time()
            indexer.search(query, k=10)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 5, f"Search too slow: {avg_latency:.2f}ms"
