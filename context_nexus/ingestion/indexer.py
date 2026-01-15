"""Indexing chunks into vector and graph stores."""

from typing import Sequence
import numpy as np
from context_nexus.core.types import Chunk


class VectorIndexer:
    """Indexes chunks into a vector store (FAISS)."""

    def __init__(self, dimensions: int = 1536):
        try:
            import faiss
            self.faiss = faiss
            self.index = faiss.IndexFlatIP(dimensions)  # Inner product (cosine similarity)
            self.chunk_map: dict[int, Chunk] = {}
            self.dimensions = dimensions
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Add chunks to the vector index.
        
        Args:
            chunks: List of chunks with embeddings
            
        Returns:
            Number of chunks added
        """
        embeddings = []
        
        for chunk in chunks:
            if chunk.embedding:
                embeddings.append(chunk.embedding)
            else:
                raise ValueError(f"Chunk {chunk.index} has no embedding")
        
        if not embeddings:
            return 0
        
        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.faiss.normalize_L2(embeddings_array)
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings_array)
        
        # Map indices to chunks
        for i, chunk in enumerate(chunks):
            self.chunk_map[start_idx + i] = chunk
        
        return len(chunks)

    def search(self, query_embedding: list[float], k: int = 20) -> list[tuple[Chunk, float]]:
        """Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        query_array = np.array([query_embedding], dtype=np.float32)
        self.faiss.normalize_L2(query_array)
        
        scores, indices = self.index.search(query_array, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.chunk_map:
                results.append((self.chunk_map[idx], float(score)))
        
        return results

    @property
    def total_chunks(self) -> int:
        """Get total number of indexed chunks."""
        return self.index.ntotal


class GraphIndexer:
    """Indexes chunks into a knowledge graph (NetworkX)."""

    def __init__(self):
        try:
            import networkx as nx
            self.nx = nx
            self.graph = nx.DiGraph()
        except ImportError:
            raise ImportError("networkx is required. Install with: pip install networkx")

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Add chunks as nodes to the graph.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Number of chunks added
        """
        for chunk in chunks:
            node_id = f"{chunk.document_id}:{chunk.index}"
            self.graph.add_node(
                node_id,
                content=chunk.content[:200],  # Store snippet
                document_id=chunk.document_id,
                chunk_index=chunk.index,
                metadata=chunk.metadata
            )
            
            # Link sequential chunks
            if chunk.index > 0:
                prev_id = f"{chunk.document_id}:{chunk.index - 1}"
                if self.graph.has_node(prev_id):
                    self.graph.add_edge(prev_id, node_id, rel="next")
        
        return len(chunks)

    def get_neighbors(self, chunk: Chunk, depth: int = 1) -> list[str]:
        """Get neighboring chunks in the graph.
        
        Args:
            chunk: Source chunk
            depth: Traversal depth
            
        Returns:
            List of node IDs
        """
        node_id = f"{chunk.document_id}:{chunk.index}"
        if not self.graph.has_node(node_id):
            return []
        
        neighbors = set()
        for node in self.nx.single_source_shortest_path_length(
            self.graph, node_id, cutoff=depth
        ).keys():
            if node != node_id:
                neighbors.add(node)
        
        return list(neighbors)

    @property
    def total_nodes(self) -> int:
        """Get total number of nodes."""
        return self.graph.number_of_nodes()

    @property
    def total_edges(self) -> int:
        """Get total number of edges."""
        return self.graph.number_of_edges()
