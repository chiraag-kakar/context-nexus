"""Context Nexus - SDK for building agentic AI systems."""

__version__ = "0.0.0"

from context_nexus.core.config import Config
from context_nexus.core.types import Document, Chunk, Answer
from context_nexus.core.nexus import ContextNexus
from context_nexus.agents.agent import Agent
from context_nexus.retrieval.hybrid import HybridRetriever
from context_nexus.observability.tracer import Tracer

__all__ = [
    "__version__",
    "Config",
    "Document",
    "Chunk",
    "Answer",
    "ContextNexus",
    "Agent",
    "HybridRetriever",
    "Tracer",
]
