"""Document loading from various sources."""

import asyncio
from pathlib import Path
from typing import AsyncIterator, Sequence
from context_nexus.core.types import Document


class Loader:
    """Loads documents from files, directories, or URLs."""

    SUPPORTED_EXTENSIONS = {
        ".md", ".txt", ".py", ".js", ".ts", ".java", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp", ".json", ".yaml", ".yml", ".toml"
    }

    async def load(self, sources: Sequence[str | Path]) -> AsyncIterator[Document]:
        """Load documents from a list of sources.
        
        Args:
            sources: List of file paths, directory paths, or URLs
            
        Yields:
            Document objects with content and metadata
        """
        for source in sources:
            if isinstance(source, str):
                source = Path(source)
            
            if source.is_dir():
                async for doc in self._load_directory(source):
                    yield doc
            elif source.is_file():
                doc = await self._load_file(source)
                if doc:
                    yield doc

    async def _load_directory(self, directory: Path) -> AsyncIterator[Document]:
        """Recursively load all supported files from a directory."""
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in self.SUPPORTED_EXTENSIONS:
                if not self._should_ignore(file_path):
                    doc = await self._load_file(file_path)
                    if doc:
                        yield doc

    async def _load_file(self, file_path: Path) -> Document | None:
        """Load a single file as a Document."""
        try:
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            return Document(
                content=content,
                source=str(file_path),
                metadata={
                    "file_type": file_path.suffix,
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                }
            )
        except (UnicodeDecodeError, IOError) as e:
            # Skip files that can't be read
            return None

    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored."""
        ignore_patterns = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".egg-info", "dist", "build", ".pytest_cache"
        }
        return any(pattern in file_path.parts for pattern in ignore_patterns)
