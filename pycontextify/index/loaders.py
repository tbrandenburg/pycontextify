"""Content loaders for PyContextify.

This module implements content loaders for different source types with
relationship extraction capabilities.
"""

import asyncio
import inspect
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from crawl4ai import AsyncWebCrawler as AsyncWebCrawlerType
    from crawl4ai import BrowserConfig as BrowserConfigType
    from crawl4ai import CacheMode as CacheModeType
    from crawl4ai import CrawlerRunConfig as CrawlerRunConfigType
    from crawl4ai import CrawlResult as CrawlResultType
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy as BFSDeepCrawlStrategyType
    from crawl4ai.models import CrawlResultContainer as CrawlResultContainerType
else:  # pragma: no cover - runtime fallbacks populated lazily
    AsyncWebCrawlerType = BrowserConfigType = CacheModeType = CrawlResultType = (
        CrawlerRunConfigType
    ) = BFSDeepCrawlStrategyType = CrawlResultContainerType = Any

from .metadata import SourceType

logger = logging.getLogger(__name__)


_async_crawler_cls: Optional[type] = None
_browser_config_cls: Optional[type] = None
_cache_mode_cls: Optional[type] = None
_crawl_result_cls: Optional[type] = None
_crawler_run_config_cls: Optional[type] = None
_deep_crawl_strategy_cls: Optional[type] = None
_crawl_result_container_cls: Optional[type] = None
_html2text_fn: Optional[Callable[[str], str]] = None


def _require_crawl4ai() -> None:
    """Load Crawl4AI lazily and provide a clear error if unavailable."""

    global _async_crawler_cls
    global _browser_config_cls
    global _cache_mode_cls
    global _crawl_result_cls
    global _crawler_run_config_cls
    global _deep_crawl_strategy_cls
    global _crawl_result_container_cls
    global _html2text_fn

    if _async_crawler_cls is not None:
        return

    try:
        from crawl4ai import AsyncWebCrawler as _AsyncWebCrawler
        from crawl4ai import BrowserConfig as _BrowserConfig
        from crawl4ai import CacheMode as _CacheMode
        from crawl4ai import CrawlerRunConfig as _CrawlerRunConfig
        from crawl4ai import CrawlResult as _CrawlResult
        from crawl4ai import html2text as _html2text
        from crawl4ai.deep_crawling import BFSDeepCrawlStrategy as _BFSDeepCrawlStrategy
        from crawl4ai.models import CrawlResultContainer as _CrawlResultContainer
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - exercised in import guard tests
        raise ModuleNotFoundError(
            "crawl4ai is required for web crawling support. "
            "Install it with 'pip install crawl4ai' or reinstall PyContextify with its web extras."
        ) from exc

    _async_crawler_cls = _AsyncWebCrawler
    _browser_config_cls = _BrowserConfig
    _cache_mode_cls = _CacheMode
    _crawl_result_cls = _CrawlResult
    _crawler_run_config_cls = _CrawlerRunConfig
    _deep_crawl_strategy_cls = _BFSDeepCrawlStrategy
    _crawl_result_container_cls = _CrawlResultContainer
    _html2text_fn = _html2text


def _playwright_browsers_installed(browser: str = "chromium") -> bool:
    """Return True if the requested Playwright browser is present locally."""

    browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if browsers_path:
        base_dir = Path(browsers_path).expanduser()
    else:
        base_dir = Path.home() / ".cache" / "ms-playwright"

    pattern = f"{browser}-*"
    try:
        return any(base_dir.glob(pattern))
    except OSError:
        # If the cache folder cannot be inspected we assume browsers are missing
        return False


def _install_crawl4ai_browsers() -> bool:
    """Install Crawl4AI's Playwright runtime if available."""

    try:
        from crawl4ai import install as crawl4ai_install  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Crawl4AI installation helpers unavailable: %s", exc)
        return False

    try:
        crawl4ai_install.install_playwright()
    except Exception as exc:
        logger.warning("Automatic Crawl4AI Playwright install failed: %s", exc)
        return False

    if _playwright_browsers_installed():
        logger.info("Installed Playwright Chromium runtime for Crawl4AI")
        return True

    logger.warning("Playwright install command completed but Chromium was not detected")
    return False


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
    - Depth limit follows Crawl4AI's breadth-first strategy semantics
    - Transparent logging when limits are reached
    """

    _playwright_ready: bool = False
    _playwright_install_attempted: bool = False

    def __init__(
        self,
        delay_seconds: int = 1,
        max_depth: int = 2,
        *,
        headless: bool = True,
        browser_mode: str = "dedicated",
        cache_mode: Optional["CacheModeType"] = None,
        excluded_tags: Optional[List[str]] = None,
    ):
        """Initialize webpage loader.

        Args:
            delay_seconds: Delay between requests in seconds
            max_depth: Maximum crawl depth (0 = no limit, 1 = starting URL +
                      direct children, 2 = children + grandchildren, etc.)
            headless: Whether to run the crawl browser in headless mode
            browser_mode: Browser mode used by Crawl4AI (e.g. "dedicated", "builtin")
            cache_mode: Cache mode for Crawl4AI requests (defaults to bypass)
            excluded_tags: HTML tags excluded when generating page content
        """
        _require_crawl4ai()

        if _browser_config_cls is None or _crawler_run_config_cls is None:
            raise RuntimeError("Crawl4AI dependencies failed to initialize")

        self.delay_seconds = delay_seconds
        self.max_depth = max_depth
        self.visited_urls: Set[str] = set()
        self._browser_config = _browser_config_cls(
            browser_mode=browser_mode,
            headless=headless,
            verbose=False,
        )
        default_cache_mode = None
        if _cache_mode_cls is not None:
            default_cache_mode = getattr(_cache_mode_cls, "BYPASS", None)

        effective_cache_mode = cache_mode
        if effective_cache_mode is None:
            effective_cache_mode = default_cache_mode

        self._crawler_config = _crawler_run_config_cls(
            cache_mode=effective_cache_mode,
            excluded_tags=excluded_tags or ["nav", "footer", "aside"],
            remove_overlay_elements=True,
            verbose=False,
        )
        self._crawler_config.mean_delay = max(0.0, float(self.delay_seconds))
        self._crawler_config.max_range = 0.0

    def load(
        self,
        url: str,
        recursive: bool = False,
        max_depth: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        """Load webpage content."""

        self.visited_urls.clear()
        effective_depth = self.max_depth if max_depth is None else max_depth

        try:
            if recursive:
                run_config = self._build_run_config(
                    deep_crawl_strategy=self._create_deep_crawl_strategy(
                        effective_depth
                    ),
                    stream=False,
                )
                results = self._execute_crawl(url, run_config)
                pages = self._results_to_pages(results, url)
            else:
                run_config = self._build_run_config()
                results = self._execute_crawl(url, run_config)
                pages = self._results_to_pages(results, url, stop_after_first=True)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = str(exc)
            hint = ""
            if "playwright" in message.lower() or "browser" in message.lower():
                hint = " Ensure Playwright browsers are installed via 'playwright install'."
            logger.warning(f"Error processing {url}: {exc}.{hint}")
            pages = []

        logger.info(f"Loaded {len(pages)} web pages from {url}")
        return pages

    def _extract_text(self, result: "CrawlResultType") -> Optional[str]:
        """Extract text content from a Crawl4AI result."""

        markdown_result = getattr(result, "markdown", None)
        if markdown_result:
            markdown_text = getattr(markdown_result, "fit_markdown", None) or getattr(
                markdown_result, "raw_markdown", None
            )
            if not markdown_text and isinstance(markdown_result, str):
                markdown_text = str(markdown_result)

            if markdown_text:
                stripped = markdown_text.strip()
                if stripped:
                    return stripped

        extracted = getattr(result, "extracted_content", None)
        if extracted:
            stripped = extracted.strip()
            if stripped:
                return stripped

        html_content = getattr(result, "cleaned_html", None) or getattr(
            result, "html", None
        )

        if not html_content:
            return None

        converter = _html2text_fn
        if converter is None:

            def converter(value):
                return re.sub(r"<[^>]+>", " ", value)

        try:
            converted = converter(html_content)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("html2text fallback failed for %s: %s", result.url, exc)
            converted = re.sub(r"<[^>]+>", " ", html_content)

        stripped = converted.strip()
        return stripped or None

    def _execute_crawl(
        self, url: str, run_config: "CrawlerRunConfigType"
    ) -> List["CrawlResultType"]:
        """Execute a Crawl4AI crawl synchronously."""

        self._ensure_runtime()

        try:
            return self._run_crawl(url, run_config)
        except Exception as exc:
            if self._retry_after_runtime_error(exc):
                return self._run_crawl(url, run_config)
            raise

    def _run_crawl(
        self, url: str, run_config: "CrawlerRunConfigType"
    ) -> List["CrawlResultType"]:
        if _async_crawler_cls is None or _crawl_result_cls is None:
            raise RuntimeError("Crawl4AI runtime not initialized")

        async def _crawl(target_url: str, config: "CrawlerRunConfigType"):
            async with _async_crawler_cls(config=self._browser_config) as crawler:
                response = crawler.arun(url=target_url, config=config)
                if inspect.isasyncgen(response):
                    return [item async for item in response]

                # Add timeout to prevent indefinite hangs (5 minutes for recursive crawls)
                try:
                    result = await asyncio.wait_for(response, timeout=300.0)
                except asyncio.TimeoutError:
                    logger.error(f"Crawl timed out after 300 seconds for {target_url}")
                    raise TimeoutError("Crawl operation timed out after 300 seconds")

                return self._coerce_results(result)

        try:
            return asyncio.run(_crawl(url, run_config))
        except RuntimeError as exc:
            if "event loop is already running" in str(exc).lower():
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(_crawl(url, run_config))
                finally:
                    loop.close()
            raise

    def _coerce_results(self, result: Any) -> List["CrawlResultType"]:
        """Convert Crawl4AI return values into a list of results."""

        if result is None:
            return []

        if isinstance(result, list):
            return result

        if _crawl_result_cls is not None and isinstance(result, _crawl_result_cls):
            return [result]

        if _crawl_result_container_cls is not None and isinstance(
            result, _crawl_result_container_cls
        ):
            return list(result)

        if isinstance(result, Iterable):
            return list(result)

        return []

    def _build_run_config(self, **overrides: Any) -> "CrawlerRunConfigType":
        return self._crawler_config.clone(**overrides)

    def _create_deep_crawl_strategy(self, max_depth: int) -> "BFSDeepCrawlStrategyType":
        if _deep_crawl_strategy_cls is None:
            raise RuntimeError("Crawl4AI runtime not initialized")

        strategy_depth = math.inf if max_depth <= 0 else max_depth
        # Limit max_pages to prevent crawling entire websites
        # With depth 1, this could be ~50 pages max (start + up to 49 children)
        # With depth 2, this limits exponential growth
        max_pages_limit = 50
        return _deep_crawl_strategy_cls(
            max_depth=strategy_depth,
            include_external=False,
            max_pages=max_pages_limit,
            logger=logger,
        )

    def _results_to_pages(
        self,
        results: List["CrawlResultType"],
        fallback_url: str,
        *,
        stop_after_first: bool = False,
    ) -> List[Tuple[str, str]]:
        pages: List[Tuple[str, str]] = []
        seen: Set[str] = set()

        for result in results:
            if not getattr(result, "success", False):
                continue

            canonical = self._canonical_url(result, fallback_url)
            if canonical in seen:
                continue

            text_content = self._extract_text(result)
            if text_content:
                pages.append((canonical, text_content))
                seen.add(canonical)

            if stop_after_first:
                break

        return pages

    @staticmethod
    def _canonical_url(result: "CrawlResultType", fallback: str) -> str:
        return (
            getattr(result, "redirected_url", None)
            or getattr(result, "url", None)
            or fallback
        )

    def _ensure_runtime(self) -> None:
        """Ensure a local Playwright runtime is available when required."""

        if not self._requires_local_browser():
            WebpageLoader._playwright_ready = True
            return

        if WebpageLoader._playwright_ready:
            return

        if _playwright_browsers_installed():
            WebpageLoader._playwright_ready = True
            return

        if not WebpageLoader._playwright_install_attempted:
            WebpageLoader._playwright_install_attempted = True
            if _install_crawl4ai_browsers():
                WebpageLoader._playwright_ready = True

    def _retry_after_runtime_error(self, exc: Exception) -> bool:
        """Attempt to recover from missing browser runtimes."""

        if not self._requires_local_browser():
            return False

        message = str(exc).lower()
        hints = [
            "playwright",  # generic - ensures we catch CLI hints
            "browser executable",  # playwright missing binary message
            "chromium",  # chromium not downloaded yet
        ]

        if not any(hint in message for hint in hints):
            return False

        if _install_crawl4ai_browsers():
            WebpageLoader._playwright_ready = True
            return True

        return False

    def _requires_local_browser(self) -> bool:
        """Determine if the loader needs a local Playwright runtime."""

        mode = getattr(self._browser_config, "browser_mode", "dedicated") or "dedicated"
        if mode.lower() == "api":
            return False

        return True


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
