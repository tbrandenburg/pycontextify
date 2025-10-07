"""Integration scenarios for loader implementations."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("crawl4ai")

from crawl4ai.models import CrawlResult, MarkdownGenerationResult

from pycontextify.index.loaders import CodeLoader, WebpageLoader


def _mk_crawl_result(url: str, text: str) -> CrawlResult:
    """Create a lightweight CrawlResult for WebpageLoader tests."""

    markdown = MarkdownGenerationResult(
        raw_markdown=text,
        markdown_with_citations=text,
        references_markdown="",
        fit_markdown=text,
        fit_html=None,
    )

    return CrawlResult(
        url=url,
        html=f"<html><body>{text}</body></html>",
        fit_html=f"<html><body>{text}</body></html>",
        cleaned_html=None,
        markdown=markdown,
        links={"internal": []},
        success=True,
        extracted_content=None,
        redirected_url=None,
    )


def test_code_loader_real_files(tmp_path):
    """CodeLoader walks real directories and filters unsupported files."""
    (tmp_path / "test.py").write_text("print('Python')")
    (tmp_path / "test.js").write_text("console.log('JavaScript')")
    (tmp_path / "README.md").write_text("# Markdown")
    (tmp_path / "config.json").write_text('{"key": "value"}')

    excluded = tmp_path / "__pycache__"
    excluded.mkdir()
    (excluded / "cached.pyc").write_text("cache")

    loader = CodeLoader()
    files = loader.load(tmp_path)

    file_names = {Path(f[0]).name for f in files}
    assert {"test.py", "test.js", "README.md", "config.json"}.issubset(file_names)
    assert "cached.pyc" not in file_names


@patch.object(WebpageLoader, "_execute_crawl")
def test_webpage_recursive_crawling_mock(_mock_execute_crawl):
    """WebpageLoader stitches crawl results when running recursively."""
    loader = WebpageLoader()

    root_result = _mk_crawl_result("https://example.com", "Root")
    child_result = _mk_crawl_result("https://example.com/child", "Child")
    _mock_execute_crawl.return_value = [root_result, child_result]

    result = loader.load("https://example.com", recursive=True, max_depth=2)
    urls = [url for url, _ in result]

    assert urls == ["https://example.com", "https://example.com/child"]
