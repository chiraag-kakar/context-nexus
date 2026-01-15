"""Text chunking using Rust-optimized functions."""

from typing import Sequence
from context_nexus.core.types import Document, Chunk

try:
    from context_nexus._core import chunk_text  # type: ignore
except ImportError:
    # Fallback if Rust module not built
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
        """Python fallback for chunking."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks


class Chunker:
    """Chunks documents into smaller pieces for embedding."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: Sequence[Document]) -> list[Chunk]:
        """Chunk a list of documents into smaller pieces.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks

    def _chunk_document(self, document: Document) -> list[Chunk]:
        """Chunk a single document using Rust-optimized chunking."""
        text_chunks = chunk_text(
            document.content,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )
        
        chunks = []
        for idx, text in enumerate(text_chunks):
            chunk = Chunk(
                content=text,
                document_id=document.id or document.source,
                index=idx,
                metadata={
                    **document.metadata,
                    "chunk_index": idx,
                    "total_chunks": len(text_chunks),
                }
            )
            chunks.append(chunk)
        
        return chunks
