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
        # TODO: implement full ingestion pipeline
        for source in sources:
            if isinstance(source, (str, Path)):
                path = Path(source)
                if path.is_dir():
                    files = [f for f in path.rglob("*") if f.is_file() and not f.name.startswith(".")]
                    self._stats.documents += len(files)
                elif path.is_file():
                    self._stats.documents += 1
            elif isinstance(source, Document):
                self._stats.documents += 1

        # rough estimates for now
        self._stats.chunks = self._stats.documents * 10
        self._stats.graph_nodes = self._stats.documents * 5
        self._initialized = True
        return self._stats

    async def retrieve(self, query: str, limit: int = 20, mode: str = "hybrid"):
        """Retrieve relevant chunks for a query."""
        # TODO: implement retrieval
        pass
