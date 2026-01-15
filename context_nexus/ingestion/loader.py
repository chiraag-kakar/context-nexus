"""Document loading from various sources including PDFs, HTML, and URLs."""

import asyncio
import io
from pathlib import Path
from typing import AsyncIterator, Sequence
import httpx
from context_nexus.core.types import Document


class Loader:
    """
    Loads documents from files, directories, URLs, and various formats.
    
    Supports:
    - Text files (.txt, .md, code files)
    - PDF files (.pdf) - requires pypdf
    - HTML files (.html) - extracts text content
    - URLs - fetches and parses content
    """

    TEXT_EXTENSIONS = {
        ".md", ".txt", ".py", ".js", ".ts", ".java", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp", ".json", ".yaml", ".yml", ".toml",
        ".xml", ".csv", ".log", ".sh", ".bash", ".zsh"
    }
    
    BINARY_EXTENSIONS = {".pdf"}
    HTML_EXTENSIONS = {".html", ".htm"}
    
    ALL_EXTENSIONS = TEXT_EXTENSIONS | BINARY_EXTENSIONS | HTML_EXTENSIONS

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0, 
            follow_redirects=True,
            headers={"User-Agent": "ContextNexus/0.1.0 (contact: your-email@example.com)"}
        )
        self._pdf_available = self._check_pdf_support()

    def _check_pdf_support(self) -> bool:
        """Check if PDF parsing is available."""
        try:
            import pypdf
            return True
        except ImportError:
            return False

    async def load(self, sources: Sequence[str | Path]) -> AsyncIterator[Document]:
        """
        Load documents from a list of sources.
        
        Sources can be:
        - File paths (str or Path)
        - Directory paths (loads all supported files recursively)
        - URLs (http:// or https://)
        """
        for source in sources:
            source_str = str(source)
            
            # Handle URLs
            if source_str.startswith(("http://", "https://")):
                doc = await self._load_url(source_str)
                if doc:
                    yield doc
                continue
            
            path = Path(source)
            
            if path.is_dir():
                async for doc in self._load_directory(path):
                    yield doc
            elif path.is_file():
                doc = await self._load_file(path)
                if doc:
                    yield doc

    async def _load_directory(self, directory: Path) -> AsyncIterator[Document]:
        """Recursively load all supported files from a directory."""
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.ALL_EXTENSIONS:
                if not self._should_ignore(file_path):
                    doc = await self._load_file(file_path)
                    if doc:
                        yield doc

    async def _load_file(self, file_path: Path) -> Document | None:
        """Load a single file based on its type."""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".pdf":
                return await self._load_pdf(file_path)
            elif suffix in self.HTML_EXTENSIONS:
                return await self._load_html_file(file_path)
            else:
                return await self._load_text_file(file_path)
        except Exception as e:
            # Log but continue on errors
            return None

    async def _load_text_file(self, file_path: Path) -> Document | None:
        """Load a plain text file."""
        try:
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            return Document(
                content=content,
                source=str(file_path),
                metadata={
                    "file_type": file_path.suffix,
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "loader": "text"
                }
            )
        except (UnicodeDecodeError, IOError):
            return None

    async def _load_pdf(self, file_path: Path) -> Document | None:
        """Load a PDF file and extract text."""
        if not self._pdf_available:
            return None
        
        try:
            import pypdf
            
            def extract_pdf_text():
                reader = pypdf.PdfReader(str(file_path))
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n\n".join(text_parts)
            
            content = await asyncio.to_thread(extract_pdf_text)
            
            if not content.strip():
                return None
            
            return Document(
                content=content,
                source=str(file_path),
                metadata={
                    "file_type": ".pdf",
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "loader": "pdf"
                }
            )
        except Exception:
            return None

    async def _load_html_file(self, file_path: Path) -> Document | None:
        """Load an HTML file and extract text content."""
        try:
            raw_content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            content = self._extract_text_from_html(raw_content)
            
            if not content.strip():
                return None
            
            return Document(
                content=content,
                source=str(file_path),
                metadata={
                    "file_type": ".html",
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "loader": "html"
                }
            )
        except Exception:
            return None

    async def _load_url(self, url: str) -> Document | None:
        """Fetch and parse content from a URL."""
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "")
            
            if "pdf" in content_type:
                return await self._parse_pdf_bytes(response.content, url)
            elif "html" in content_type or "text/html" in content_type:
                content = self._extract_text_from_html(response.text)
            else:
                content = response.text
            
            if not content.strip():
                return None
            
            return Document(
                content=content,
                source=url,
                metadata={
                    "content_type": content_type,
                    "url": url,
                    "loader": "url"
                }
            )
        except Exception:
            return None

    async def _parse_pdf_bytes(self, pdf_bytes: bytes, source: str) -> Document | None:
        """Parse PDF from bytes."""
        if not self._pdf_available:
            return None
        
        try:
            import pypdf
            
            def extract():
                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n\n".join(text_parts)
            
            content = await asyncio.to_thread(extract)
            
            if not content.strip():
                return None
            
            return Document(
                content=content,
                source=source,
                metadata={"loader": "pdf_url"}
            )
        except Exception:
            return None

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML (simple approach without BeautifulSoup)."""
        import re
        
        # Remove script and style elements
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Replace block elements with newlines
        text = re.sub(r'<(p|div|br|h[1-6]|li|tr)[^>]*>', '\n', text, flags=re.IGNORECASE)
        
        # Remove all remaining tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()

    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored."""
        ignore_patterns = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".egg-info", "dist", "build", ".pytest_cache", ".tox"
        }
        return any(pattern in file_path.parts for pattern in ignore_patterns)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Convenience functions for fetching open data sources
async def fetch_wikipedia_articles(topics: list[str], loader: Loader) -> list[Document]:
    """
    Fetch Wikipedia articles for benchmarking.
    Uses Wikipedia's public API (no auth required).
    """
    docs = []
    
    for topic in topics:
        url = f"https://en.wikipedia.org/api/rest_v1/page/html/{topic.replace(' ', '_')}"
        doc = await loader._load_url(url)
        if doc:
            doc.metadata["source_type"] = "wikipedia"
            doc.metadata["topic"] = topic
            docs.append(doc)
    
    return docs


async def fetch_arxiv_abstracts(query: str, max_results: int = 50, loader: Loader = None) -> list[Document]:
    """
    Fetch arXiv paper abstracts for benchmarking.
    Uses arXiv's public API (no auth required).
    """
    if loader is None:
        loader = Loader()
    
    # arXiv API returns Atom XML
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
    
    try:
        response = await loader.client.get(url)
        response.raise_for_status()
        
        # Simple XML parsing for abstracts
        import re
        
        docs = []
        entries = re.findall(r'<entry>(.*?)</entry>', response.text, re.DOTALL)
        
        for i, entry in enumerate(entries):
            title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            abstract_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            
            if title_match and abstract_match:
                title = title_match.group(1).strip()
                abstract = abstract_match.group(1).strip()
                
                content = f"# {title}\n\n{abstract}"
                
                doc = Document(
                    content=content,
                    source=f"arxiv:{query}:{i}",
                    metadata={
                        "source_type": "arxiv",
                        "title": title,
                        "query": query
                    }
                )
                docs.append(doc)
        
        return docs
    except Exception:
        return []


async def fetch_gutenberg_books(book_ids: list[int], loader: Loader) -> list[Document]:
    """
    Fetch public domain books from Project Gutenberg.
    Uses Gutenberg's public mirrors (no auth required).
    """
    docs = []
    
    for book_id in book_ids:
        # Gutenberg plain text mirror
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        doc = await loader._load_url(url)
        if doc:
            doc.metadata["source_type"] = "gutenberg"
            doc.metadata["book_id"] = book_id
            docs.append(doc)
    
    return docs

