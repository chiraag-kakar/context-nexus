from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


@dataclass
class Document:
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class Chunk:
    content: str
    document_id: str
    index: int
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Source:
    title: str
    chunk_id: str
    relevance: float
    excerpt: str = ""


@dataclass
class Trace:
    steps: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    latency_ms: float = 0.0
    chunks_retrieved: int = 0


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    confidence: float = 0.0
    trace: Trace | None = None
    timestamp: datetime = field(default_factory=datetime.now)
