from dataclasses import dataclass
from typing import Literal

try:
    from context_nexus._core import fuse_scores  # type: ignore
except ImportError:
    # Fallback RRF fusion
    def fuse_scores(scores_a: list[float], scores_b: list[float], 
                   weight_a: float = 0.5, weight_b: float = 0.5, k: int = 60) -> list[float]:
        """Python fallback for RRF fusion."""
        fused = []
        for a, b in zip(scores_a, scores_b):
            score = weight_a * a + weight_b * b
            fused.append(score)
        return fused


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    score: float
    source: Literal["vector", "graph", "keyword", "hybrid"]
    metadata: dict


class HybridRetriever:
    """Combines vector search, graph traversal, and keyword matching."""

    def __init__(
        self,
        vector_weight: float = 0.6,
        graph_weight: float = 0.3,
        keyword_weight: float = 0.1,
        rerank: bool = True,
    ):
        total = vector_weight + graph_weight + keyword_weight
        self.vector_weight = vector_weight / total if total > 0 else 0
        self.graph_weight = graph_weight / total if total > 0 else 0
        self.keyword_weight = keyword_weight / total if total > 0 else 0
        self.rerank = rerank

    async def retrieve(self, query: str, nexus, limit: int = 20) -> list[RetrievalResult]:
        """Retrieve using hybrid search.
        
        Args:
            query: Search query
            nexus: ContextNexus instance with indexers
            limit: Maximum results
            
        Returns:
            List of retrieval results ranked by relevance
        """
        # Get query embedding
        from context_nexus.ingestion import Embedder
        embedder = Embedder(nexus.config.embedding, nexus.config.llm.api_key)
        
        try:
            # Create a dummy chunk to get embedding
            from context_nexus.core.types import Chunk
            dummy = Chunk(content=query, document_id="query", index=0)
            embedded = await embedder.embed_chunks([dummy])
            query_embedding = embedded[0].embedding
        finally:
            await embedder.close()
        
        # Vector search
        vector_results = nexus._vector_indexer.search(query_embedding, k=limit * 2)
        
        # Build results
        results_map: dict[str, RetrievalResult] = {}
        
        for chunk, score in vector_results:
            chunk_id = f"{chunk.document_id}:{chunk.index}"
            results_map[chunk_id] = RetrievalResult(
                chunk_id=chunk_id,
                content=chunk.content,
                score=score * self.vector_weight,
                source="vector",
                metadata=chunk.metadata
            )
        
        # Graph expansion (get neighbors of top vector results)
        if self.graph_weight > 0:
            top_chunks = [chunk for chunk, _ in vector_results[:5]]
            for chunk in top_chunks:
                neighbors = nexus._graph_indexer.get_neighbors(chunk, depth=1)
                for neighbor_id in neighbors[:3]:  # Limit neighbors
                    if neighbor_id not in results_map:
                        # Add neighbor with graph score
                        results_map[neighbor_id] = RetrievalResult(
                            chunk_id=neighbor_id,
                            content="",  # Would need to fetch full content
                            score=self.graph_weight * 0.5,
                            source="graph",
                            metadata={}
                        )
        
        # Sort by final score and return top k
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_results[:limit]
