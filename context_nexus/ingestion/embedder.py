"""Embedding generation for chunks."""

import asyncio
from typing import Sequence
import httpx
from context_nexus.core.types import Chunk
from context_nexus.core.config import EmbeddingConfig


class Embedder:
    """Generates embeddings for text chunks."""

    def __init__(self, config: EmbeddingConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def embed_chunks(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Same chunks with embeddings added
        """
        # Process in batches to avoid rate limits
        batch_size = 100
        embedded_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embedded = await self._embed_batch(batch)
            embedded_chunks.extend(batch_embedded)
        
        return embedded_chunks

    async def _embed_batch(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Embed a batch of chunks."""
        texts = [chunk.content for chunk in chunks]
        
        if self.config.provider == "openai":
            embeddings = await self._openai_embed(texts)
        else:
            # Add support for local embeddings later
            raise NotImplementedError(f"Provider {self.config.provider} not yet supported")
        
        # Attach embeddings to chunks
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            result.append(chunk)
        
        return result

    async def _openai_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        response = await self.client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": texts,
                "model": self.config.model,
                "dimensions": self.config.dimensions,
            }
        )
        response.raise_for_status()
        
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
