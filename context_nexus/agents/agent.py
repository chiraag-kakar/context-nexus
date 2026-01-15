from typing import Any, TypeVar

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

    async def query(self, question: str, trace: bool = False) -> Answer:
        """Ask a question and get an answer with sources."""
        # TODO: implement actual query pipeline
        trace_obj = None
        if trace:
            trace_obj = Trace(
                steps=[{"name": "retrieve", "duration_ms": 45}, {"name": "generate", "duration_ms": 890}],
                tokens_used=1250,
                latency_ms=935,
                chunks_retrieved=5,
            )

        return Answer(
            text=f"[Placeholder] Answer to: {question}",
            sources=[Source(title="example.md", chunk_id="chunk_001", relevance=0.92)],
            confidence=0.85,
            trace=trace_obj,
        )

    async def analyze(self, question: str, output_schema: type[T] | None = None) -> Answer | T:
        """Analyze content with optional structured output."""
        return await self.query(question)
