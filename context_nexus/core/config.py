from dataclasses import dataclass, field
from typing import Literal
import os


@dataclass
class VectorStoreConfig:
    backend: Literal["faiss", "qdrant"] = "faiss"
    url: str = ""
    collection_name: str = "context_nexus"


@dataclass
class GraphStoreConfig:
    backend: Literal["networkx", "neo4j"] = "networkx"
    uri: str = ""
    user: str = ""
    password: str = ""


@dataclass
class LLMConfig:
    provider: Literal["openai", "anthropic", "local"] = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""

    def __post_init__(self):
        if not self.api_key:
            env_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
            self.api_key = os.getenv(env_map.get(self.provider, ""), "")


@dataclass
class EmbeddingConfig:
    provider: Literal["openai", "local"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536


@dataclass
class Config:
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    graph_store: GraphStoreConfig = field(default_factory=GraphStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    default_token_budget: int = 8000
    chunk_size: int = 512
    chunk_overlap: int = 50

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "openai"),  # type: ignore
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            ),
            vector_store=VectorStoreConfig(
                backend=os.getenv("VECTOR_BACKEND", "faiss"),  # type: ignore
                url=os.getenv("QDRANT_URL", ""),
            ),
            graph_store=GraphStoreConfig(
                backend=os.getenv("GRAPH_BACKEND", "networkx"),  # type: ignore
                uri=os.getenv("NEO4J_URI", ""),
            ),
        )
