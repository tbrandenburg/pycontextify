import math
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("crawl4ai")

from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.models import CrawlResult

from pycontextify.index.loaders import WebpageLoader


def _build_result(url: str, text: str, *, success: bool = True) -> CrawlResult:
    return CrawlResult(
        url=url,
        html="<html></html>",
        success=success,
        extracted_content=text if success else None,
        links={"internal": []},
    )


class TestRecursiveCrawlingIntegration:
    """Tests to ensure the Crawl4AI deep crawl integration is configured correctly."""

    @patch.object(WebpageLoader, "_run_crawl")
    def test_recursive_load_configures_bfs_strategy(self, mock_run: Mock):
        captured_config = {}

        def _fake_run(url: str, run_config):
            captured_config["config"] = run_config
            return [_build_result(url, "Root page")]

        mock_run.side_effect = _fake_run

        loader = WebpageLoader()
        WebpageLoader._playwright_ready = True
        pages = loader.load("https://example.com", recursive=True, max_depth=3)

        assert len(pages) == 1
        config = captured_config["config"]
        assert isinstance(config.deep_crawl_strategy, BFSDeepCrawlStrategy)
        assert config.deep_crawl_strategy.max_depth == 3
        assert config.stream is False

    @patch.object(WebpageLoader, "_run_crawl")
    def test_recursive_load_collects_successful_results(self, mock_run: Mock):
        def _fake_run(url: str, run_config):
            return [
                _build_result(url, "Root"),
                _build_result(url, "Duplicate"),
                _build_result("https://example.com/child", "Child page"),
                _build_result("https://example.com/error", "", success=False),
            ]

        mock_run.side_effect = _fake_run

        loader = WebpageLoader()
        WebpageLoader._playwright_ready = True
        pages = loader.load("https://example.com", recursive=True, max_depth=2)

        urls = [url for url, _ in pages]
        assert urls == ["https://example.com", "https://example.com/child"]


class TestDepthConfiguration:
    """Unit tests ensuring depth semantics are delegated to Crawl4AI."""

    @patch.object(WebpageLoader, "_run_crawl")
    def test_unlimited_depth_maps_to_infinite(self, mock_run: Mock):
        captured_config = {}

        def _fake_run(url: str, run_config):
            captured_config["config"] = run_config
            return [_build_result(url, "Root page")]

        mock_run.side_effect = _fake_run

        loader = WebpageLoader()
        WebpageLoader._playwright_ready = True
        loader.load("https://example.com", recursive=True, max_depth=0)

        config = captured_config["config"]
        assert isinstance(config.deep_crawl_strategy, BFSDeepCrawlStrategy)
        assert math.isinf(config.deep_crawl_strategy.max_depth)
