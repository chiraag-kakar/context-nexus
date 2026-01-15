from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from context_nexus.core.config import Config
from context_nexus.core.types import Document


@dataclass
class Stats:
    documents: int = 0
    chunks: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0


class ContextNexus:
    """Main entry point for Context Nexus."""

    def __init__(
        self,
        config: Config | None = None,
        vector_store: str = "faiss",
        graph_store: str = "networkx",
        llm_provider: str = "openai",
    ):
        if config:
            self.config = config
        else:
            self.config = Config.from_env()
            self.config.vector_store.backend = vector_store  # type: ignore
            self.config.graph_store.backend = graph_store  # type: ignore
            self.config.llm.provider = llm_provider  # type: ignore

        self._stats = Stats()
        self._initialized = False

    @property
    def stats(self) -> Stats:
        return self._stats

    async def ingest(
        self,
        sources: Sequence[str | Path | Document],
        incremental: bool = True,
    ) -> Stats:
        """Ingest documents from paths, URLs, or Document objects."""
        from context_nexus.ingestion import Loader, Chunker, Embedder, VectorIndexer, GraphIndexer
        
        # Initialize components if not already done
        if not self._initialized:
            self._vector_indexer = VectorIndexer(dimensions=self.config.embedding.dimensions)
            self._graph_indexer = GraphIndexer()
            self._initialized = True
        
        # Step 1: Load documents
        loader = Loader()
        documents = []
        
        # Convert sources to async iterable
        for source in sources:
            if isinstance(source, Document):
                documents.append(source)
            else:
                async for doc in loader.load([source]):
                    documents.append(doc)
        
        if not documents:
            return self._stats
        
        self._stats.documents += len(documents)
        
        # Step 2: Chunk documents
        chunker = Chunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        chunks = chunker.chunk_documents(documents)
        self._stats.chunks += len(chunks)
        
        # Step 3: Generate embeddings
        embedder = Embedder(self.config.embedding, self.config.llm.api_key)
        try:
            chunks = await embedder.embed_chunks(chunks)
        finally:
            await embedder.close()
        
        # Step 4: Index into vector store
        added = self._vector_indexer.add_chunks(chunks)
        
        # Step 5: Index into graph store
        graph_added = self._graph_indexer.add_chunks(chunks)
        self._stats.graph_nodes = self._graph_indexer.total_nodes
        self._stats.graph_edges = self._graph_indexer.total_edges
        
        return self._stats

    async def retrieve(self, query: str, limit: int = 20, mode: str = "hybrid"):
        """Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            mode: Retrieval mode ("hybrid", "vector", "graph")
            
        Returns:
            List of retrieval results
        """
        if not self._initialized:
            raise ValueError("Must call ingest() before retrieve()")
        
        from context_nexus.retrieval import HybridRetriever
        
        retriever = HybridRetriever()
        results = await retriever.retrieve(query, self, limit=limit)
        
        return results
