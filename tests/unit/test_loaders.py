"""Simple tests for pycontextify.indexer.loaders module to improve coverage."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("crawl4ai")

from crawl4ai.models import CrawlResult, CrawlResultContainer, MarkdownGenerationResult


def _mk_markdown(text: str) -> MarkdownGenerationResult:
    return MarkdownGenerationResult(
        raw_markdown=text,
        markdown_with_citations=text,
        references_markdown="",
        fit_markdown=text,
        fit_html=None,
    )


def _mk_crawl_result(url: str, text: str, *, links=None) -> CrawlResult:
    return CrawlResult(
        url=url,
        html=f"<html><body>{text}</body></html>",
        fit_html=f"<html><body>{text}</body></html>",
        cleaned_html=None,
        markdown=_mk_markdown(text),
        links=links or {"internal": []},
        success=True,
        extracted_content=None,
        redirected_url=None,
    )


from pycontextify.indexer.loaders import (
    CodeLoader,
    DocumentLoader,
    LoaderFactory,
    WebpageLoader,
)
from pycontextify.storage.metadata import SourceType

# Disable the automatic sentence transformer mocking for this file
pytestmark = pytest.mark.no_mock_st


class TestCodeLoaderSimple:
    """Simple tests for CodeLoader."""

    def test_init(self):
        """Test CodeLoader initialization."""
        loader = CodeLoader()
        assert loader.max_file_size_bytes == 10 * 1024 * 1024

        loader_custom = CodeLoader(max_file_size_mb=5)
        assert loader_custom.max_file_size_bytes == 5 * 1024 * 1024

    def test_constants(self):
        """Test that constants are defined correctly."""
        assert ".py" in CodeLoader.SUPPORTED_EXTENSIONS
        assert ".js" in CodeLoader.SUPPORTED_EXTENSIONS
        assert "__pycache__" in CodeLoader.EXCLUDED_DIRS
        assert ".git" in CodeLoader.EXCLUDED_DIRS

    def test_load_nonexistent_directory(self):
        """Test loading non-existent directory."""
        loader = CodeLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/this/path/does/not/exist")

    def test_read_text_file(self):
        """Test reading a simple text file."""
        loader = CodeLoader()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello world')")
            temp_path = f.name

        try:
            content = loader._read_file(Path(temp_path))
            assert content == "print('hello world')"
        finally:
            Path(temp_path).unlink()

    def test_read_binary_file(self):
        """Test that binary files are rejected."""
        loader = CodeLoader()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"\x00\x01\x02\x03")  # Binary content
            temp_path = f.name

        try:
            content = loader._read_file(Path(temp_path))
            assert content is None
        finally:
            Path(temp_path).unlink()


class TestDocumentLoaderSimple:
    """Simple tests for DocumentLoader."""

    def test_init(self):
        """Test DocumentLoader initialization."""
        with patch.object(DocumentLoader, "_initialize_pdf_loader"):
            loader = DocumentLoader()
            assert loader.pdf_engine == "pymupdf"

            loader_custom = DocumentLoader(pdf_engine="pypdf2")
            assert loader_custom.pdf_engine == "pypdf2"

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with patch.object(DocumentLoader, "_initialize_pdf_loader"):
            loader = DocumentLoader()
            with pytest.raises(FileNotFoundError):
                loader.load("/this/file/does/not/exist.pdf")

    def test_load_text_file(self):
        """Test loading a text file."""
        with patch.object(DocumentLoader, "_initialize_pdf_loader"):
            loader = DocumentLoader()

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write("Hello, world!")
                temp_path = f.name

            try:
                content = loader._load_text_file(Path(temp_path))
                assert content == "Hello, world!"
            finally:
                Path(temp_path).unlink()

    def test_unsupported_format(self):
        """Test loading unsupported format."""
        with patch.object(DocumentLoader, "_initialize_pdf_loader"):
            loader = DocumentLoader()

            result = loader._load_document(Path("/test/file.unknown"))
            assert result is None


class TestWebpageLoaderSimple:
    """Simple tests for WebpageLoader."""

    def test_init(self):
        """Test WebpageLoader initialization."""
        loader = WebpageLoader()
        assert loader.delay_seconds == 1
        assert loader.max_depth == 2
        assert isinstance(loader.visited_urls, set)

        loader_custom = WebpageLoader(delay_seconds=2, max_depth=3)
        assert loader_custom.delay_seconds == 2
        assert loader_custom.max_depth == 3

    @patch.object(WebpageLoader, "_execute_crawl")
    def test_load_simple_page(self, mock_crawl):
        """Test loading a simple webpage."""

        content = "Main Content " * 50
        mock_result = _mk_crawl_result("https://example.com", content)
        mock_result.links = {"internal": [], "external": []}
        mock_crawl.return_value = [mock_result]

        loader = WebpageLoader()
        result = loader.load("https://example.com")

        assert len(result) == 1
        assert result[0][0] == "https://example.com"
        assert "Main Content" in result[0][1]

    @patch.object(WebpageLoader, "_execute_crawl", side_effect=Exception("fail"))
    def test_load_request_error(self, _mock_crawl):
        """Test handling crawl errors."""

        loader = WebpageLoader()
        result = loader.load("https://example.com")

        assert result == []

    def test_visited_urls_cleared(self):
        """Test that visited URLs are cleared between loads."""
        loader = WebpageLoader()
        loader.visited_urls.add("https://old.com")

        with patch.object(loader, "_execute_crawl", return_value=[]):
            loader.load("https://new.com")

        # Should not contain old URL after new load
        assert "https://old.com" not in loader.visited_urls

    def test_coerce_results_accepts_container(self):
        """Ensure CrawlResultContainer is converted to a list."""

        loader = WebpageLoader()
        result = _mk_crawl_result("https://example.com", "Body")
        container = CrawlResultContainer(result)

        coerced = loader._coerce_results(container)

        assert isinstance(coerced, list)
        assert coerced[0].url == "https://example.com"


class TestWebpageLoaderRuntimeBootstrap:
    """Tests covering automatic Crawl4AI runtime preparation."""

    def setup_method(self):
        WebpageLoader._playwright_ready = False
        WebpageLoader._playwright_install_attempted = False

    @patch(
        "pycontextify.indexer.loaders._playwright_browsers_installed", return_value=False
    )
    @patch("pycontextify.indexer.loaders._install_crawl4ai_browsers", return_value=True)
    def test_ensure_runtime_installs_when_missing(
        self, mock_install: Mock, _mock_detect: Mock
    ):
        loader = WebpageLoader()
        loader._ensure_runtime()

        mock_install.assert_called_once()
        assert WebpageLoader._playwright_ready is True

    @patch(
        "pycontextify.indexer.loaders._playwright_browsers_installed", return_value=False
    )
    @patch("pycontextify.indexer.loaders._install_crawl4ai_browsers")
    def test_api_mode_skips_install(self, mock_install: Mock, mock_detect: Mock):
        loader = WebpageLoader(browser_mode="api")
        loader._ensure_runtime()

        mock_detect.assert_not_called()
        mock_install.assert_not_called()
        assert WebpageLoader._playwright_ready is True

    @patch("pycontextify.indexer.loaders._install_crawl4ai_browsers", return_value=True)
    def test_retry_after_runtime_error_success(self, mock_install: Mock):
        loader = WebpageLoader()
        WebpageLoader._playwright_ready = False

        should_retry = loader._retry_after_runtime_error(
            Exception("Executable doesn't exist. Please run Playwright install")
        )

        mock_install.assert_called_once()
        assert should_retry is True
        assert WebpageLoader._playwright_ready is True

    @patch("pycontextify.indexer.loaders._install_crawl4ai_browsers", return_value=False)
    def test_retry_after_runtime_error_failure(self, mock_install: Mock):
        loader = WebpageLoader()
        WebpageLoader._playwright_ready = False

        should_retry = loader._retry_after_runtime_error(Exception("timeout"))

        mock_install.assert_not_called()
        assert should_retry is False
        assert WebpageLoader._playwright_ready is False


class TestLoaderFactorySimple:
    """Simple tests for LoaderFactory."""

    def test_get_code_loader(self):
        """Test creating CodeLoader."""
        loader = LoaderFactory.get_loader(SourceType.CODE)
        assert isinstance(loader, CodeLoader)

    def test_get_document_loader(self):
        """Test creating DocumentLoader."""
        with patch.object(DocumentLoader, "_initialize_pdf_loader"):
            loader = LoaderFactory.get_loader(SourceType.DOCUMENT)
            assert isinstance(loader, DocumentLoader)

    def test_get_webpage_loader(self):
        """Test creating WebpageLoader."""
        loader = LoaderFactory.get_loader(SourceType.WEBPAGE)
        assert isinstance(loader, WebpageLoader)

    def test_unsupported_type(self):
        """Test error for unsupported type."""
        with pytest.raises(ValueError, match="Unsupported source type"):
            LoaderFactory.get_loader("invalid_type")

    def test_loader_with_params(self):
        """Test creating loaders with parameters."""
        # CodeLoader with params
        loader = LoaderFactory.get_loader(SourceType.CODE, max_file_size_mb=5)
        assert isinstance(loader, CodeLoader)
        assert loader.max_file_size_bytes == 5 * 1024 * 1024

        # DocumentLoader with params
        with patch.object(DocumentLoader, "_initialize_pdf_loader"):
            loader = LoaderFactory.get_loader(SourceType.DOCUMENT, pdf_engine="pypdf2")
            assert isinstance(loader, DocumentLoader)
            assert loader.pdf_engine == "pypdf2"

        # WebpageLoader with params
        loader = LoaderFactory.get_loader(SourceType.WEBPAGE, delay_seconds=3)
        assert isinstance(loader, WebpageLoader)
        assert loader.delay_seconds == 3


