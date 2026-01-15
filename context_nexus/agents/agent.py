from typing import Any, TypeVar
import httpx

from context_nexus.core.nexus import ContextNexus
from context_nexus.core.types import Answer, Source, Trace

T = TypeVar("T")


class Agent:
    """Agent that answers questions about ingested content."""

    def __init__(
        self,
        nexus: ContextNexus,
        token_budget: int = 8000,
        retriever: Any = None,
    ):
        self.nexus = nexus
        self.token_budget = token_budget
        self.retriever = retriever
        self.client = httpx.AsyncClient(timeout=60.0)

    async def query(self, question: str, trace: bool = False) -> Answer:
        """Ask a question and get an answer with sources.
        
        Args:
            question: User's question
            trace: Whether to include trace information
            
        Returns:
            Answer with text, sources, and optional trace
        """
        import time
        start_time = time.time()
        trace_steps = []
        
        # Step 1: Retrieve relevant chunks
        retrieve_start = time.time()
        results = await self.nexus.retrieve(question, limit=10)
        retrieve_time = int((time.time() - retrieve_start) * 1000)
        
        if trace:
            trace_steps.append({
                "name": "retrieve",
                "duration_ms": retrieve_time,
                "chunks_found": len(results)
            })
        
        # Step 2: Build context from results
        context_pieces = []
        sources = []
        
        for idx, result in enumerate(results[:5]):  # Limit to top 5 for context
            context_pieces.append(f"[{idx+1}] {result.content}")
            sources.append(Source(
                title=result.metadata.get("file_name", "unknown"),
                chunk_id=result.chunk_id,
                relevance=result.score,
                excerpt=result.content[:150]
            ))
        
        context = "\n\n".join(context_pieces)
        
        # Step 3: Generate answer with LLM
        generate_start = time.time()
        answer_text = await self._generate_answer(question, context, trace_steps)
        generate_time = int((time.time() - generate_start) * 1000)
        
        if trace:
            trace_steps.append({
                "name": "generate",
                "duration_ms": generate_time
            })
        
        # Build trace object if requested
        trace_obj = None
        if trace:
            total_time = int((time.time() - start_time) * 1000)
            trace_obj = Trace(
                steps=trace_steps,
                tokens_used=len(context.split()) + len(answer_text.split()),
                latency_ms=total_time,
                chunks_retrieved=len(results),
            )

        return Answer(
            text=answer_text,
            sources=sources,
            confidence=0.85 if sources else 0.3,
            trace=trace_obj,
        )

    async def _generate_answer(self, question: str, context: str, trace_steps: list) -> str:
        """Generate answer using OpenAI."""
        try:
            from context_nexus._core import count_tokens  # type: ignore
        except ImportError:
            count_tokens = lambda text, model: len(text.split())
        
        # Truncate context if needed
        context_tokens = count_tokens(context, "gpt-4")
        if context_tokens > self.token_budget * 0.7:
            # Rough truncation
            max_chars = int(len(context) * (self.token_budget * 0.7) / context_tokens)
            context = context[:max_chars]
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. " +
                          "Always cite sources using [Source N] notation."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            }
        ]
        
        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.nexus.config.llm.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.nexus.config.llm.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500,
            }
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def analyze(self, question: str, output_schema: type[T] | None = None) -> Answer | T:
        """Analyze content with optional structured output."""
        return await self.query(question)
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
