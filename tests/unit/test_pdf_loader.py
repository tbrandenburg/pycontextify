"""Consolidated tests for PDF loader functionality.

This module combines the essential test cases from test_pdf_loader_enhanced.py,
test_pdf_loader_consolidated.py, and test_pdf_loader_quickwin.py, focusing
on the most portable and comprehensive test coverage.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from pycontextify.indexer_pdf_loader import PDFLoader


class TestPDFLoaderInitialization:
    """Test PDF loader initialization and engine detection."""

    def test_default_initialization(self):
        """Test PDF loader initialization with default settings."""
        with patch.object(
            PDFLoader, "_check_available_engines", return_value=["pymupdf", "pypdf2"]
        ):
            loader = PDFLoader()
            assert loader.preferred_engine == "pymupdf"
            assert isinstance(loader.available_engines, list)

    def test_custom_engine_initialization(self):
        """Test initialization with custom preferred engine."""
        with patch.object(
            PDFLoader, "_check_available_engines", return_value=["pymupdf", "pypdf2"]
        ):
            loader = PDFLoader(preferred_engine="pypdf2")
            assert loader.preferred_engine == "pypdf2"

    def test_no_engines_available_error(self):
        """Test error when no PDF engines are available."""
        with patch.object(PDFLoader, "_check_available_engines", return_value=[]):
            with pytest.raises(
                ImportError, match="No PDF processing engines available"
            ):
                PDFLoader()

    def test_engine_availability_detection(self):
        """Test engine availability detection."""
        # Test all engines available
        with patch("builtins.__import__") as mock_import:
            mock_import.return_value = Mock()
            loader = PDFLoader.__new__(PDFLoader)
            engines = loader._check_available_engines()

            expected_engines = {"pymupdf", "pypdf2", "pdfplumber"}
            for engine in engines:
                assert engine in expected_engines

        # Test no engines available
        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            loader = PDFLoader.__new__(PDFLoader)
            engines = loader._check_available_engines()
            assert engines == []

    def test_partial_engine_availability(self):
        """Test when only some engines are available."""

        def import_side_effect(name, *args, **kwargs):
            if name == "fitz":  # PyMuPDF available
                return Mock()
            elif name == "PyPDF2":  # PyPDF2 not available
                raise ImportError("No module named PyPDF2")
            elif name == "pdfplumber":  # pdfplumber available
                return Mock()
            else:
                return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_side_effect):
            loader = PDFLoader.__new__(PDFLoader)
            engines = loader._check_available_engines()
            assert "pymupdf" in engines
            assert "pdfplumber" in engines
            assert "pypdf2" not in engines


class TestPDFFileValidation:
    """Test PDF file validation and error handling."""

    def setup_method(self):
        """Set up test environment."""
        with patch.object(
            PDFLoader, "_check_available_engines", return_value=["pypdf2", "pymupdf"]
        ):
            self.loader = PDFLoader()

    def test_nonexistent_file_error(self):
        """Test error when PDF file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            self.loader.load_pdf("/nonexistent/file.pdf")

    def test_non_pdf_file_error(self):
        """Test error when file is not a PDF."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is not a PDF file")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="File is not a PDF"):
                self.loader.load_pdf(temp_file)
        finally:
            os.unlink(temp_file)

    def test_empty_text_extraction_error(self):
        """Test error when text extraction returns empty content."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "pathlib.Path.suffix",
                new_callable=lambda: property(lambda self: ".pdf"),
            ),
            patch.object(
                self.loader,
                "_extract_with_engine",
                return_value="",  # Empty text
            ),
        ):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                temp_file = f.name

            try:
                with pytest.raises(ValueError, match="Could not extract text from PDF"):
                    self.loader.load_pdf(temp_file)
            finally:
                os.unlink(temp_file)


class TestPDFEngineExtraction:
    """Test PDF engine-specific extraction methods."""

    def setup_method(self):
        """Set up test environment."""
        self.loader = PDFLoader.__new__(PDFLoader)
        self.loader.preferred_engine = "pymupdf"
        self.loader.available_engines = ["pymupdf", "pypdf2", "pdfplumber"]

    def test_extract_with_pymupdf_success(self):
        """Test PyMuPDF extraction with mocked fitz."""
        # Create mock PDF document structure
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "First page content"
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "Second page content"

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=2)
        mock_doc.__getitem__ = Mock(side_effect=[mock_page1, mock_page2])
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)

        mock_fitz = Mock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = self.loader._extract_with_pymupdf("/test.pdf")

        expected = (
            "--- Page 1 ---\nFirst page content\n\n--- Page 2 ---\nSecond page content"
        )
        assert result == expected
        mock_fitz.open.assert_called_once_with("/test.pdf")

    def test_extract_with_pymupdf_empty_pages(self):
        """Test PyMuPDF extraction with empty pages."""
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "  \n  "  # Whitespace only
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "Valid content"

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=2)
        mock_doc.__getitem__ = Mock(side_effect=[mock_page1, mock_page2])
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)

        mock_fitz = Mock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = self.loader._extract_with_pymupdf("/test.pdf")

        # Should only include page with valid content
        expected = "--- Page 2 ---\nValid content"
        assert result == expected

    def test_extract_with_pypdf2_success(self):
        """Test PyPDF2 extraction with mocked PyPDF2."""
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 text"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 text"

        mock_reader = Mock()
        mock_reader.pages = [mock_page1, mock_page2]

        mock_pypdf2 = Mock()
        mock_pypdf2.PdfReader.return_value = mock_reader

        with patch.dict("sys.modules", {"PyPDF2": mock_pypdf2}):
            with patch("builtins.open", MagicMock()):
                result = self.loader._extract_with_pypdf2("/test.pdf")

        expected = "--- Page 1 ---\nPage 1 text\n\n--- Page 2 ---\nPage 2 text"
        assert result == expected
        mock_pypdf2.PdfReader.assert_called_once()

    def test_extract_with_pdfplumber_success(self):
        """Test pdfplumber extraction with mocked pdfplumber."""
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Content from page 1"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Content from page 2"

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)

        mock_pdfplumber = Mock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch.dict("sys.modules", {"pdfplumber": mock_pdfplumber}):
            result = self.loader._extract_with_pdfplumber("/test.pdf")

        expected = (
            "--- Page 1 ---\nContent from page 1\n\n--- Page 2 ---\nContent from page 2"
        )
        assert result == expected
        mock_pdfplumber.open.assert_called_once_with("/test.pdf")

    def test_extract_with_pdfplumber_none_text(self):
        """Test pdfplumber extraction with None text responses."""
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = None
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Valid text"

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)

        mock_pdfplumber = Mock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch.dict("sys.modules", {"pdfplumber": mock_pdfplumber}):
            result = self.loader._extract_with_pdfplumber("/test.pdf")

        expected = "--- Page 2 ---\nValid text"
        assert result == expected


class TestPDFEngineFallback:
    """Test PDF engine fallback behavior."""

    def setup_method(self):
        """Set up test environment."""
        with patch.object(
            PDFLoader, "_check_available_engines", return_value=["pymupdf", "pypdf2"]
        ):
            self.loader = PDFLoader(preferred_engine="pymupdf")

    def test_successful_extraction_with_preferred_engine(self):
        """Test successful extraction with preferred engine."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "pathlib.Path.suffix",
                new_callable=lambda: property(lambda self: ".pdf"),
            ),
            patch.object(
                self.loader,
                "_extract_with_engine",
                return_value="Extracted text content",
            ),
        ):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                temp_file = f.name

            try:
                result = self.loader.load_pdf(temp_file)
                assert result == "Extracted text content"
            finally:
                os.unlink(temp_file)

    def test_engine_fallback_on_failure(self):
        """Test fallback to secondary engine when primary fails."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "pathlib.Path.suffix",
                new_callable=lambda: property(lambda self: ".pdf"),
            ),
        ):
            # Mock preferred engine fails, fallback succeeds
            def mock_extract(file_path, engine):
                if engine == "pymupdf":
                    raise Exception("Preferred engine failed")
                return "Fallback engine content"

            with patch.object(
                self.loader, "_extract_with_engine", side_effect=mock_extract
            ):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    temp_file = f.name

                try:
                    result = self.loader.load_pdf(temp_file)
                    assert result == "Fallback engine content"
                finally:
                    os.unlink(temp_file)

    def test_empty_text_fallback(self):
        """Test fallback when preferred engine returns empty text."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "pathlib.Path.suffix",
                new_callable=lambda: property(lambda self: ".pdf"),
            ),
        ):
            # Mock preferred engine returns empty, fallback succeeds
            def mock_extract(file_path, engine):
                if engine == "pymupdf":
                    return ""  # Empty text
                return "Fallback engine text"

            with patch.object(
                self.loader, "_extract_with_engine", side_effect=mock_extract
            ):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    temp_file = f.name

                try:
                    result = self.loader.load_pdf(temp_file)
                    assert result == "Fallback engine text"
                finally:
                    os.unlink(temp_file)

    def test_all_engines_fail(self):
        """Test error when all engines fail."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "pathlib.Path.suffix",
                new_callable=lambda: property(lambda self: ".pdf"),
            ),
            patch.object(
                self.loader,
                "_extract_with_engine",
                side_effect=Exception("All engines failed"),
            ),
        ):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                temp_file = f.name

            try:
                with pytest.raises(ValueError, match="Could not extract text from PDF"):
                    self.loader.load_pdf(temp_file)
            finally:
                os.unlink(temp_file)


class TestPDFPageContextExtraction:
    """Test page context extraction functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.loader = PDFLoader.__new__(PDFLoader)
        self.loader.available_engines = ["pypdf2"]

    def test_extract_page_context_basic_patterns(self):
        """Test basic page number pattern extraction."""
        test_cases = [
            ("--- Page 10 ---", 10, True),
            ("Page 25", 25, True),
            ("[Page 99]", 99, True),
            ("p. 42", 42, True),
            ("p.15", 15, True),
            ("No page markers here", None, False),
        ]

        for text, expected_page, expected_marker in test_cases:
            result = self.loader.extract_page_context(text)
            assert result["page_number"] == expected_page
            assert result["has_page_marker"] == expected_marker

    def test_extract_page_context_with_problematic_input(self):
        """Test regex error handling with problematic input."""
        # Test with very long text that might cause regex issues
        problematic_text = "Page " + "X" * 1000
        result = self.loader.extract_page_context(problematic_text)

        assert result["page_number"] is None
        assert result["has_page_marker"] is False

    def test_extract_page_context_section_titles(self):
        """Test section title extraction."""
        test_cases = [
            ("1.2 Introduction\nSome content", "Introduction"),
            ("CHAPTER OVERVIEW\nContent here", "CHAPTER OVERVIEW"),
            ("Chapter 5 Advanced Topics", "Chapter 5 Advanced Topics"),
            ("   Title Case Heading   \nContent", "Title Case Heading"),
        ]

        for text, expected_title in test_cases:
            result = self.loader.extract_page_context(text)
            assert result["section_title"] == expected_title

    def test_extract_page_context_comprehensive(self):
        """Test comprehensive context extraction."""
        test_text = """--- Page 42 ---

1.5 Data Processing Methods

This section covers advanced data processing techniques
used in modern applications. The methods described here
build upon the foundations established in earlier chapters.
"""

        result = self.loader.extract_page_context(test_text)
        assert result["page_number"] == 42
        assert result["has_page_marker"] is True
        assert result["section_title"] == "Data Processing Methods"


class TestPDFMetadataExtraction:
    """Test PDF metadata extraction functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.loader = PDFLoader.__new__(PDFLoader)
        self.loader.available_engines = ["pymupdf"]

    def test_get_pdf_info_basic_structure(self):
        """Test basic PDF info structure."""
        mock_stat = Mock()
        mock_stat.st_size = 1024
        mock_stat.st_ctime = 1609459200
        mock_stat.st_mtime = 1640995200

        with (
            patch("pathlib.Path.stat", return_value=mock_stat),
            patch("pathlib.Path.absolute", return_value=Path("/test/doc.pdf")),
            patch("pathlib.Path.name", "doc.pdf"),
            patch.object(
                self.loader,
                "_get_info_pymupdf",
                return_value={
                    "page_count": 5,
                    "has_text": True,
                    "word_count_estimate": 1000,
                },
            ),
        ):
            result = self.loader.get_pdf_info("/test/doc.pdf")

            # Check basic structure
            assert "file_path" in result
            assert "file_name" in result
            assert "file_size_bytes" in result
            assert "file_size_mb" in result
            assert "page_count" in result
            assert "has_text" in result
            assert "word_count_estimate" in result
            assert "reading_time_minutes" in result
            assert "engine_used" in result

            # Check calculated values
            assert result["file_size_bytes"] == 1024
            assert result["file_size_mb"] == 0.0  # 1024 bytes = 0.0 MB (rounded)
            assert result["page_count"] == 5
            assert result["has_text"] is True
            assert result["word_count_estimate"] == 1000
            assert result["reading_time_minutes"] == 5  # 1000 words / 200 wpm
            assert result["engine_used"] == "pymupdf"

    def test_get_pdf_info_reading_time_calculation(self):
        """Test reading time calculation."""
        test_cases = [
            (0, 0),  # No words
            (100, 1),  # Minimum 1 minute
            (200, 1),  # 1 minute
            (400, 2),  # 2 minutes
            (1000, 5),  # 5 minutes
        ]

        for word_count, expected_minutes in test_cases:
            mock_stat = Mock()
            mock_stat.st_size = 1024
            mock_stat.st_ctime = mock_stat.st_mtime = 1609459200

            with (
                patch("pathlib.Path.stat", return_value=mock_stat),
                patch("pathlib.Path.absolute", return_value=Path("/test/doc.pdf")),
                patch("pathlib.Path.name", "doc.pdf"),
                patch.object(
                    self.loader,
                    "_get_info_pymupdf",
                    return_value={
                        "page_count": 1,
                        "word_count_estimate": word_count,
                    },
                ),
            ):
                result = self.loader.get_pdf_info("/test/doc.pdf")
                assert result["reading_time_minutes"] == expected_minutes


class TestPDFLoaderIntegration:
    """Integration tests for PDF loader functionality."""

    def test_engine_method_mapping(self):
        """Test that engine names map to correct extraction methods."""
        loader = PDFLoader.__new__(PDFLoader)
        loader.available_engines = ["pymupdf", "pypdf2", "pdfplumber"]

        # Test method mapping
        with patch.object(
            loader, "_extract_with_pymupdf", return_value="pymupdf_result"
        ) as mock_pymupdf:
            result = loader._extract_with_engine("/test.pdf", "pymupdf")
            assert result == "pymupdf_result"
            mock_pymupdf.assert_called_once_with("/test.pdf")

        with patch.object(
            loader, "_extract_with_pypdf2", return_value="pypdf2_result"
        ) as mock_pypdf2:
            result = loader._extract_with_engine("/test.pdf", "pypdf2")
            assert result == "pypdf2_result"
            mock_pypdf2.assert_called_once_with("/test.pdf")

        with patch.object(
            loader, "_extract_with_pdfplumber", return_value="pdfplumber_result"
        ) as mock_pdfplumber:
            result = loader._extract_with_engine("/test.pdf", "pdfplumber")
            assert result == "pdfplumber_result"
            mock_pdfplumber.assert_called_once_with("/test.pdf")

    def test_unsupported_engine_error(self):
        """Test error when using unsupported engine."""
        loader = PDFLoader.__new__(PDFLoader)
        loader.available_engines = ["pymupdf"]

        with pytest.raises(ValueError, match="Unsupported engine"):
            loader._extract_with_engine("/test.pdf", "unsupported_engine")
