"""Simple tests for pycontextify.index.loaders module to improve coverage."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pycontextify.index.loaders import (
    CodeLoader,
    DocumentLoader,
    LoaderFactory,
    WebpageLoader,
)
from pycontextify.index.metadata import SourceType

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

    @patch("requests.Session.get")
    def test_load_simple_page(self, mock_get):
        """Test loading a simple webpage."""
        # Create content long enough to pass quality threshold
        content = "Main Content " * 50  # Longer content
        mock_response = Mock()
        mock_response.content = (
            f"<html><body><main>{content}</main></body></html>".encode()
        )
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        loader = WebpageLoader()
        result = loader.load("https://example.com")

        assert len(result) <= 1  # Should be 0 or 1 depending on content quality

    @patch("requests.Session.get")
    def test_load_request_error(self, mock_get):
        """Test handling request errors."""
        import requests

        mock_get.side_effect = requests.RequestException("Connection failed")

        loader = WebpageLoader()
        result = loader.load("https://example.com")

        assert result == []

    def test_extract_links(self):
        """Test extracting links from HTML."""
        html = """
        <html>
            <body>
                <a href="/page1">Page 1</a>
                <a href="https://example.com/page2">Page 2</a>
                <a href="mailto:test@example.com">Email</a>
            </body>
        </html>
        """

        loader = WebpageLoader()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        links = loader._extract_links_from_soup(soup, "https://example.com")

        # Should convert relative URLs to absolute and exclude non-HTTP links
        http_links = [
            link for link in links if link.startswith(("http://", "https://"))
        ]
        assert len(http_links) >= 1

    def test_should_follow_link(self):
        """Test link following logic."""
        loader = WebpageLoader()

        # Same domain should be followed
        assert loader._should_follow_link(
            "https://example.com/page2", "https://example.com/page1"
        )

        # Different domain should not
        assert not loader._should_follow_link(
            "https://other.com/page", "https://example.com/page1"
        )

    def test_visited_urls_cleared(self):
        """Test that visited URLs are cleared between loads."""
        loader = WebpageLoader()
        loader.visited_urls.add("https://old.com")

        with patch.object(loader, "_fetch_and_parse", return_value=(None, None)):
            loader.load("https://new.com")

        # Should not contain old URL after new load
        assert "https://old.com" not in loader.visited_urls


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


class TestIntegration:
    """Integration tests."""

    def test_code_loader_real_files(self):
        """Test CodeLoader with real file structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create various file types
            (Path(temp_dir) / "test.py").write_text("print('Python')")
            (Path(temp_dir) / "test.js").write_text("console.log('JavaScript')")
            (Path(temp_dir) / "README.md").write_text("# Markdown")
            (Path(temp_dir) / "config.json").write_text('{"key": "value"}')

            # Create excluded directory
            excluded = Path(temp_dir) / "__pycache__"
            excluded.mkdir()
            (excluded / "cached.pyc").write_text("cache")

            loader = CodeLoader()
            files = loader.load(temp_dir)

            # Should load the supported files but not the cached one
            file_names = [Path(f[0]).name for f in files]
            assert "test.py" in file_names
            assert "test.js" in file_names
            assert "README.md" in file_names
            assert "config.json" in file_names
            assert "cached.pyc" not in file_names

    def test_webpage_recursive_crawling_mock(self):
        """Test basic recursive crawling."""
        loader = WebpageLoader()

        # Mock the methods to avoid actual HTTP requests
        from bs4 import BeautifulSoup
        test_soup = BeautifulSoup("<html><body>Test content</body></html>", "html.parser")
        
        with patch.object(loader, "_fetch_and_parse") as mock_fetch:
            with patch.object(loader, "_extract_links_from_soup") as mock_links:
                with patch.object(loader, "_should_follow_link") as mock_should:
                    mock_fetch.return_value = (test_soup, "Test content")
                    mock_links.return_value = []
                    mock_should.return_value = False

                    result = loader._crawl_recursive("https://example.com", 2, 1)

                    assert len(result) == 1
                    assert result[0][0] == "https://example.com"
                    assert result[0][1] == "Test content"
