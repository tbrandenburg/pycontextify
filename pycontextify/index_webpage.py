"""Webpage indexing orchestration for PyContextify."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple

from .types import SourceType

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - handled lazily
    from crawl4ai import AsyncWebCrawler as AsyncWebCrawlerType  # noqa: F401
    from crawl4ai import BrowserConfig as BrowserConfigType  # noqa: F401
    from crawl4ai import CacheMode as CacheModeType  # noqa: F401
    from crawl4ai import CrawlerRunConfig as CrawlerRunConfigType  # noqa: F401
    from crawl4ai import CrawlResult as CrawlResultType  # noqa: F401
    from crawl4ai.deep_crawling import (  # noqa: F401
        BFSDeepCrawlStrategy as BFSDeepCrawlStrategyType,
    )
    from crawl4ai.models import (  # noqa: F401
        CrawlResultContainer as CrawlResultContainerType,
    )
else:  # pragma: no cover - keep mypy happy
    AsyncWebCrawlerType = BrowserConfigType = CacheModeType = CrawlResultType = (
        CrawlerRunConfigType
    ) = BFSDeepCrawlStrategyType = CrawlResultContainerType = Any

_async_crawler_cls: Optional[type] = None
_browser_config_cls: Optional[type] = None
_cache_mode_cls: Optional[type] = None
_crawl_result_cls: Optional[type] = None
_crawler_run_config_cls: Optional[type] = None
_deep_crawl_strategy_cls: Optional[type] = None
_crawl_result_container_cls: Optional[type] = None
_html2text_fn: Optional[Any] = None


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
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests
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
    browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    base_dir = (
        Path(browsers_path).expanduser()
        if browsers_path
        else Path.home() / ".cache" / "ms-playwright"
    )
    pattern = f"{browser}-*"
    try:
        return any(base_dir.glob(pattern))
    except OSError:
        return False


def _install_crawl4ai_browsers() -> bool:
    try:
        from crawl4ai import install as crawl4ai_install  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Crawl4AI installation helpers unavailable: %s", exc)
        return False

    try:
        # Set encoding environment for Windows
        env = os.environ.copy()
        if sys.platform == "win32":
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

        crawl4ai_install.install_playwright()
    except UnicodeEncodeError as enc_exc:  # pragma: no cover - Windows specific
        logger.warning(
            "Automatic Crawl4AI Playwright install failed due to encoding: %s", enc_exc
        )
        return False
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Automatic Crawl4AI Playwright install failed: %s", exc)
        return False

    if _playwright_browsers_installed():
        logger.info("Installed Playwright Chromium runtime for Crawl4AI")
        return True

    logger.warning("Playwright install command completed but Chromium was not detected")
    return False


class WebpageLoader:
    """Loader for web content with link extraction."""

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
        default_cache_mode = (
            getattr(_cache_mode_cls, "BYPASS", None) if _cache_mode_cls else None
        )
        effective_cache_mode = (
            cache_mode if cache_mode is not None else default_cache_mode
        )

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
        """Load webpage content synchronously.

        This is a synchronous wrapper around the async implementation
        to maintain backwards compatibility with existing tests and usage.
        """
        try:
            # Try to use existing event loop if available
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop context, create a new one
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(
                        self._load_async(url, recursive, max_depth)
                    )
                finally:
                    new_loop.close()
            else:
                return loop.run_until_complete(
                    self._load_async(url, recursive, max_depth)
                )
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(self._load_async(url, recursive, max_depth))

    async def _load_async(
        self,
        url: str,
        recursive: bool = False,
        max_depth: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        """Load webpage content asynchronously."""
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
                results = await self._execute_crawl(url, run_config)
                pages = self._results_to_pages(results, url)
            else:
                run_config = self._build_run_config()
                results = await self._execute_crawl(url, run_config)
                pages = self._results_to_pages(results, url, stop_after_first=True)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = str(exc)
            hint = ""
            if "playwright" in message.lower() or "browser" in message.lower():
                hint = " Ensure Playwright browsers are installed via 'playwright install'."
            logger.warning("Error processing %s: %s.%s", url, exc, hint)
            pages = []

        logger.info("Loaded %d web pages from %s", len(pages), url)
        return pages

    def _build_run_config(
        self,
        deep_crawl_strategy: Optional["BFSDeepCrawlStrategyType"] = None,
        stream: bool = True,
    ) -> "CrawlerRunConfigType":
        if hasattr(self._crawler_config, "clone"):
            run_config = self._crawler_config.clone(stream=stream)
        else:  # pragma: no cover - defensive fallback
            run_config = self._crawler_config.copy()
            run_config.stream = stream
        if deep_crawl_strategy is not None:
            run_config.deep_crawl_strategy = deep_crawl_strategy
        return run_config

    async def _execute_crawl(
        self, url: str, run_config: "CrawlerRunConfigType"
    ) -> List["CrawlResultType"]:
        self._ensure_runtime()

        try:
            return await self._run_crawl(url, run_config)
        except Exception as exc:
            if self._retry_after_runtime_error(exc):
                return await self._run_crawl(url, run_config)
            raise

    async def _run_crawl(
        self, url: str, run_config: "CrawlerRunConfigType"
    ) -> List["CrawlResultType"]:
        if _async_crawler_cls is None:
            raise RuntimeError("Crawl4AI runtime not initialized")

        # Use proper async context manager directly without nested asyncio.run()
        async with _async_crawler_cls(config=self._browser_config) as crawler:
            try:
                response = crawler.arun(url=url, config=run_config)

                if inspect.isasyncgen(response):
                    results = []
                    async for item in response:
                        results.append(item)
                    return results

                try:
                    result = await asyncio.wait_for(response, timeout=300.0)
                except asyncio.TimeoutError:
                    logger.error("Crawl timed out after 300 seconds for %s", url)
                    raise TimeoutError("Crawl operation timed out after 300 seconds")

                return self._coerce_results(result)
            except Exception as exc:
                logger.error("Error during crawl execution for %s: %s", url, exc)
                raise

    def _coerce_results(self, result: Any) -> List["CrawlResultType"]:
        """Normalize Crawl4AI responses into a list."""

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

    def _create_deep_crawl_strategy(self, max_depth: int) -> "BFSDeepCrawlStrategyType":
        if _deep_crawl_strategy_cls is None:
            raise RuntimeError("Crawl4AI runtime not initialized")

        strategy_depth = math.inf if max_depth <= 0 else max_depth
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

    def _extract_text(self, result: "CrawlResultType") -> Optional[str]:
        markdown_result = getattr(result, "markdown", None)
        if markdown_result:
            markdown_text = getattr(markdown_result, "fit_markdown", None) or getattr(
                markdown_result, "raw_markdown", None
            )
            if not markdown_text and isinstance(markdown_result, str):
                markdown_text = str(markdown_result)
            if markdown_text and markdown_text.strip():
                return markdown_text.strip()

        extracted = getattr(result, "extracted_content", None)
        if extracted and extracted.strip():
            return extracted.strip()

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

    def _ensure_runtime(self) -> None:
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
        if not self._requires_local_browser():
            return False

        message = str(exc).lower()
        hints = ["playwright", "browser executable", "chromium"]
        if not any(hint in message for hint in hints):
            return False

        if _install_crawl4ai_browsers():
            WebpageLoader._playwright_ready = True
            return True
        return False

    def _requires_local_browser(self) -> bool:
        mode = getattr(self._browser_config, "browser_mode", "dedicated") or "dedicated"
        return mode.lower() != "api"


class WebpageIndexer:
    """Coordinate webpage ingestion using an :class:`IndexManager`."""

    def __init__(self, manager: "IndexManager") -> None:
        self._manager = manager
        self._loader = WebpageLoader(delay_seconds=manager.config.crawl_delay_seconds)

    async def index(
        self,
        url: str,
        *,
        recursive: bool = False,
        max_depth: int = 1,
    ) -> Dict[str, Any]:
        logger.info(
            "Starting webpage indexing: %s (recursive=%s, max_depth=%s)",
            url,
            recursive,
            max_depth,
        )
        try:
            pages = await self._loader._load_async(
                url, recursive=recursive, max_depth=max_depth
            )
            if not pages:
                return {"error": "Could not load any web pages"}

            chunks_added = 0
            for page_url, content in pages:
                chunks_added += self._manager.process_content(
                    content, page_url, SourceType.WEBPAGE
                )

            self._manager.auto_save()
            self._manager.ensure_embedder_loaded()

            stats = {
                "pages_processed": len(pages),
                "chunks_added": chunks_added,
                "source_type": "webpage",
                "recursive": recursive,
                "max_depth": max_depth,
                "embedding_provider": self._manager.embedder.get_provider_name(),
                "embedding_model": self._manager.embedder.get_model_name(),
            }
            logger.info("Completed webpage indexing: %s", stats)
            return stats
        except Exception as exc:  # pragma: no cover - defensive
            error_msg = f"Failed to index webpage {url}: {exc}"
            logger.error(error_msg)
            return {"error": error_msg}


__all__ = ["WebpageIndexer", "WebpageLoader"]
