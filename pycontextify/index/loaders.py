"""Content loaders for PyContextify.

This module implements content loaders for different source types with
relationship extraction capabilities.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .metadata import SourceType

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract base class for content loaders."""

    @abstractmethod
    def load(self, source: str) -> List[Tuple[str, str]]:
        """Load content from source.

        Args:
            source: Source path or URL

        Returns:
            List of (file_path, content) tuples
        """
        pass


class CodeLoader(BaseLoader):
    """Loader for codebase directories with dependency extraction."""

    SUPPORTED_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".h",
        ".c",
        ".cs",
        ".rb",
        ".php",
        ".go",
        ".rs",
        ".swift",
        ".kt",
        ".scala",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
    }

    EXCLUDED_DIRS = {
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".env",
        "build",
        "dist",
        "target",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".tox",
        "htmlcov",
    }

    def __init__(self, max_file_size_mb: int = 10):
        """Initialize code loader.

        Args:
            max_file_size_mb: Maximum file size to process in MB
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def load(self, directory_path: str) -> List[Tuple[str, str]]:
        """Load code files from directory recursively.

        Args:
            directory_path: Path to directory to scan

        Returns:
            List of (file_path, content) tuples
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        files = []
        processed_count = 0

        for file_path in self._walk_directory(directory):
            try:
                content = self._read_file(file_path)
                if content:
                    files.append((str(file_path), content))
                    processed_count += 1

                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} code files")

            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        logger.info(f"Loaded {len(files)} code files from {directory_path}")
        return files

    def _walk_directory(self, directory: Path) -> List[Path]:
        """Walk directory and find supported code files."""
        files = []

        for path in directory.rglob("*"):
            # Skip directories
            if path.is_dir():
                continue

            # Skip excluded directories
            if any(excluded in path.parts for excluded in self.EXCLUDED_DIRS):
                continue

            # Check file extension
            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                # Check file size
                try:
                    if path.stat().st_size <= self.max_file_size_bytes:
                        files.append(path)
                    else:
                        logger.warning(f"Skipping large file: {path}")
                except OSError:
                    continue

        return files

    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content with encoding detection."""
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()

                    # Skip binary files (basic heuristic)
                    if "\x00" in content:
                        return None

                    return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                return None

        logger.warning(f"Could not decode file: {file_path}")
        return None


class DocumentLoader(BaseLoader):
    """Loader for individual documents with structure extraction."""

    def __init__(self, pdf_engine: str = "pymupdf"):
        """Initialize document loader.

        Args:
            pdf_engine: PDF processing engine to use
        """
        self.pdf_engine = pdf_engine
        self.pdf_loader = None

        # Initialize PDF loader if PDF support is available
        self._initialize_pdf_loader()

    def load(self, file_path: str) -> List[Tuple[str, str]]:
        """Load single document.

        Args:
            file_path: Path to document file

        Returns:
            List with single (file_path, content) tuple
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._load_document(path)
        if content:
            return [(str(path), content)]
        return []

    def _load_document(self, path: Path) -> Optional[str]:
        """Load document based on file extension."""
        extension = path.suffix.lower()

        if extension == ".pdf":
            return self._load_pdf(path)
        elif extension in [".md", ".txt"]:
            return self._load_text_file(path)
        else:
            logger.warning(f"Unsupported document type: {extension}")
            return None

    def _initialize_pdf_loader(self):
        """Initialize PDF loader with configured engine."""
        try:
            from .pdf_loader import PDFLoader

            self.pdf_loader = PDFLoader(preferred_engine=self.pdf_engine)
            logger.info(f"PDF loader initialized with engine: {self.pdf_engine}")
        except ImportError as e:
            logger.warning(f"Could not initialize PDF loader: {e}")
            self.pdf_loader = None

    def _load_pdf(self, path: Path) -> Optional[str]:
        """Load PDF file content using enhanced PDF loader."""
        if not self.pdf_loader:
            logger.error("PDF loader not available")
            return None

        try:
            text = self.pdf_loader.load_pdf(str(path))
            logger.info(f"Successfully loaded PDF: {path.name}")
            return text
        except Exception as e:
            logger.error(f"Failed to load PDF {path}: {e}")
            return None

    def _load_text_file(self, path: Path) -> Optional[str]:
        """Load text file with encoding detection."""
        encodings = ["utf-8", "utf-16", "latin-1"]

        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")
                return None

        return None


class WebpageLoader(BaseLoader):
    """Loader for web content with link extraction.

    Follows Scrapy-inspired best practices:
    - Depth limit is inclusive (max_depth=2 means starting URL + 1 level)
    - Configurable max links per page
    - Transparent logging when limits are reached
    """

    def __init__(
        self,
        delay_seconds: int = 1,
        max_depth: int = 2,
        max_links_per_page: int = 50,
    ):
        """Initialize webpage loader.

        Args:
            delay_seconds: Delay between requests in seconds
            max_depth: Maximum crawl depth (0 = no limit, 1 = starting URL only,
                      2 = starting URL + direct children, etc.)
            max_links_per_page: Maximum number of links to follow per page
        """
        self.delay_seconds = delay_seconds
        self.max_depth = max_depth
        self.max_links_per_page = max_links_per_page
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "PyContextify/1.0 (Educational MCP Server)"}
        )

    def load(
        self, url: str, recursive: bool = False, max_depth: int = 1
    ) -> List[Tuple[str, str]]:
        """Load webpage content.

        Args:
            url: URL to load
            recursive: Whether to follow links
            max_depth: Maximum depth for recursive loading (inclusive).
                      1 = starting URL only, 2 = starting URL + direct children, etc.
                      0 = unlimited depth (use with caution)

        Returns:
            List of (url, content) tuples
        """
        self.visited_urls.clear()
        pages = []

        if recursive:
            # Start at depth 1 (the starting URL is at depth 1)
            pages = self._crawl_recursive(url, max_depth, 1)
        else:
            soup, content = self._fetch_and_parse(url)
            if content:
                pages = [(url, content)]

        logger.info(f"Loaded {len(pages)} web pages from {url}")
        return pages

    def _crawl_recursive(
        self, url: str, max_depth: int, current_depth: int
    ) -> List[Tuple[str, str]]:
        """Recursively crawl web pages.

        Args:
            url: URL to crawl
            max_depth: Maximum depth allowed (inclusive)
            current_depth: Current depth in the crawl tree (starts at 1 for root)

        Returns:
            List of (url, content) tuples
        """
        # Check depth limit (inclusive) - depth 1 is the starting URL
        if max_depth > 0 and current_depth > max_depth:
            logger.debug(
                f"Skipping {url}: depth {current_depth} exceeds max_depth {max_depth}"
            )
            return []

        # Check if already visited
        if url in self.visited_urls:
            logger.debug(f"Skipping {url}: already visited")
            return []

        self.visited_urls.add(url)
        pages = []

        # Load current page - get both soup (for links) and content (for indexing)
        logger.info(f"Crawling {url} at depth {current_depth}")
        soup, content = self._fetch_and_parse(url)
        if content:
            pages.append((url, content))

            # Extract and follow links if we haven't reached max depth
            # (depth check: continue only if we're below max_depth or max_depth is unlimited)
            if soup and (max_depth == 0 or current_depth < max_depth):
                links = self._extract_links_from_soup(soup, url)
                total_links = len(links)

                # Apply link limit
                links_to_crawl = links[: self.max_links_per_page]

                if total_links > self.max_links_per_page:
                    logger.warning(
                        f"Found {total_links} links on {url}, "
                        f"but limiting to first {self.max_links_per_page} "
                        f"(set max_links_per_page to increase)"
                    )
                else:
                    logger.info(f"Found {total_links} links on {url}")

                for link in links_to_crawl:
                    if self._should_follow_link(link, url):
                        time.sleep(self.delay_seconds)  # Rate limiting
                        sub_pages = self._crawl_recursive(
                            link, max_depth, current_depth + 1
                        )
                        pages.extend(sub_pages)

        return pages

    def _fetch_and_parse(
        self, url: str
    ) -> Tuple[Optional[BeautifulSoup], Optional[str]]:
        """Fetch and parse a web page, returning both soup and extracted text.

        Args:
            url: URL to fetch

        Returns:
            Tuple of (BeautifulSoup object, extracted text content)
            Returns (None, None) on error
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Parse HTML and extract text
            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()

            # IMPROVED CONTENT EXTRACTION STRATEGY
            content_candidates = []

            # Strategy 1: Try semantic HTML elements
            main_content = soup.find("main")
            if main_content:
                text = main_content.get_text(separator=" ", strip=True)
                text = re.sub(r"\s+", " ", text).strip()
                content_candidates.append(("main", text))

            article_content = soup.find("article")
            if article_content:
                text = article_content.get_text(separator=" ", strip=True)
                text = re.sub(r"\s+", " ", text).strip()
                content_candidates.append(("article", text))

            # Strategy 2: Try common content containers
            content_selectors = [
                ("div.container", "container"),
                ("div[class*='content']", "content_div"),
                (".main-content", "main_content_class"),
                ("#content", "content_id"),
                ("#main", "main_id"),
            ]

            for selector, name in content_selectors:
                try:
                    elements = soup.select(selector)
                    for element in elements:
                        text = element.get_text(separator=" ", strip=True)
                        text = re.sub(r"\s+", " ", text).strip()
                        if text:  # Only add non-empty content
                            content_candidates.append((name, text))
                except Exception:
                    continue

            # Strategy 3: Try body as fallback
            if soup.body:
                text = soup.body.get_text(separator=" ", strip=True)
                text = re.sub(r"\s+", " ", text).strip()
                content_candidates.append(("body", text))

            # Strategy 4: Full soup as last resort
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            content_candidates.append(("full", text))

            # CHOOSE THE BEST CANDIDATE
            # Look for the candidate with the most substantial content
            # that contains key structural indicators

            best_candidate = None
            max_quality_score = 0

            for source, text in content_candidates:
                if len(text) < 100:  # Skip very short content
                    continue

                # Calculate quality score based on:
                # 1. Length (longer is generally better for main content)
                # 2. Presence of structural indicators (headings, sections)
                # 3. Avoid pure navigation/header content

                length_score = min(len(text) / 10000, 1.0)  # Normalize to 0-1

                # Look for structural indicators
                structure_indicators = [
                    "What",
                    "How",
                    "Why",  # Question headings
                    "Introduction",
                    "Overview",
                    "Summary",
                    "Product",
                    "Service",
                    "Solution",
                    "Architecture",
                    "Design",
                    "Implementation",
                ]

                structure_score = sum(
                    1
                    for indicator in structure_indicators
                    if indicator.lower() in text.lower()
                ) / len(structure_indicators)

                # Penalize if it looks like pure navigation
                nav_penalties = ["Home", "About", "Contact", "Privacy", "Terms"]
                nav_ratio = sum(1 for nav in nav_penalties if nav in text) / len(
                    nav_penalties
                )
                nav_penalty = min(nav_ratio, 0.5)

                quality_score = length_score + structure_score - nav_penalty

                logger.debug(
                    f"Content candidate '{source}': {len(text)} chars, quality={quality_score:.3f}"
                )

                if quality_score > max_quality_score:
                    max_quality_score = quality_score
                    best_candidate = (source, text)

            if best_candidate:
                source, final_text = best_candidate
                logger.info(
                    f"Selected content from '{source}': {len(final_text)} characters"
                )
                return soup, final_text
            else:
                logger.warning("No suitable content found")
                return soup, None

        except requests.RequestException as e:
            logger.warning(f"Failed to load {url}: {e}")
            return None, None
        except Exception as e:
            logger.warning(f"Error processing {url}: {e}")
            return None, None

    def _extract_links_from_soup(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract links from BeautifulSoup object.

        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs
        """
        try:
            links = []

            for link in soup.find_all("a", href=True):
                href = link["href"]
                absolute_url = urljoin(base_url, href)

                # Only include HTTP/HTTPS links
                if absolute_url.startswith(("http://", "https://")):
                    links.append(absolute_url)

            return links
        except Exception as e:
            logger.warning(f"Error extracting links: {e}")
            return []

    def _should_follow_link(self, link: str, base_url: str) -> bool:
        """Determine if a link should be followed."""
        try:
            base_domain = urlparse(base_url).netloc
            link_domain = urlparse(link).netloc

            # Only follow links within the same domain
            return link_domain == base_domain and link not in self.visited_urls

        except Exception:
            return False


class LoaderFactory:
    """Factory for selecting appropriate loader."""

    @staticmethod
    def get_loader(source_type: SourceType, **kwargs) -> BaseLoader:
        """Get appropriate loader for source type.

        Args:
            source_type: Type of content to load
            **kwargs: Additional parameters for loader

        Returns:
            Appropriate loader instance
        """
        if source_type == SourceType.CODE:
            return CodeLoader(**kwargs)
        elif source_type == SourceType.DOCUMENT:
            return DocumentLoader(**kwargs)
        elif source_type == SourceType.WEBPAGE:
            return WebpageLoader(**kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
