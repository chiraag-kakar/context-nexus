"""Embedding generation for chunks - supports both OpenAI and local models."""

import asyncio
from typing import Sequence, Literal
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from context_nexus.core.types import Chunk
from context_nexus.core.config import EmbeddingConfig


class Embedder:
    """Generates embeddings for text chunks.
    
    Supports two modes:
    - "openai": Uses OpenAI API (requires API key, has rate limits)
    - "local": Uses sentence-transformers (free, unlimited, runs locally)
    """

    def __init__(
        self, 
        config: EmbeddingConfig, 
        api_key: str | None = None,
        provider: Literal["openai", "local"] | None = None
    ):
        self.config = config
        self.api_key = api_key
        
        # Auto-detect provider based on what's available
        if provider:
            self.provider = provider
        elif config.provider == "local" or not api_key:
            self.provider = "local"
        else:
            self.provider = config.provider
        
        # Initialize HTTP client for OpenAI
        if self.provider == "openai":
            self.client = httpx.AsyncClient(timeout=30.0)
        else:
            self.client = None
            
        # Lazy-load the local model
        self._local_model = None

    def _get_local_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use a lightweight but effective model
                model_name = "all-MiniLM-L6-v2"  # 384 dimensions, fast
                print(f"  Loading local embedding model: {model_name}...")
                self._local_model = SentenceTransformer(model_name)
                print("  ✓ Model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._local_model

    async def embed_chunks(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Same chunks with embeddings added
        """
        if self.provider == "local":
            return await self._embed_local(chunks)
        else:
            return await self._embed_openai_batched(chunks)

    async def _embed_local(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Generate embeddings using local sentence-transformers model."""
        model = self._get_local_model()
        texts = [chunk.content for chunk in chunks]
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            lambda: model.encode(texts, show_progress_bar=len(texts) > 50)
        )
        
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            # Convert numpy array to list
            chunk.embedding = embedding.tolist()
            result.append(chunk)
        
        return result

    async def _embed_openai_batched(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Generate embeddings using OpenAI API with batching."""
        batch_size = 100
        embedded_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            # Aggressive pause for rate-limited accounts
            if i > 0:
                print("  (Rate limit pause: 20s...)")
                await asyncio.sleep(21)
                
            batch = chunks[i:i + batch_size]
            batch_embedded = await self._embed_batch(batch)
            embedded_chunks.extend(batch_embedded)
        
        return embedded_chunks

    async def _embed_batch(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Embed a batch of chunks with OpenAI."""
        texts = [chunk.content for chunk in chunks]
        embeddings = await self._openai_embed(texts)
        
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            result.append(chunk)
        
        return result

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        reraise=True
    )
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
        
        if response.status_code == 429:
            response.raise_for_status()
            
        response.raise_for_status()
        
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
