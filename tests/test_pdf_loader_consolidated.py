"""Consolidated PDF loader tests - combining best practices from original and enhanced tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from pycontextify.index.pdf_loader import PDFLoader


class TestPDFLoaderConsolidated:
    """Consolidated tests for PDF loader functionality."""

    def test_initialization_and_engine_detection(self):
        """Test PDF loader initialization and engine detection."""
        # Test with default engine
        loader = PDFLoader()
        assert loader.preferred_engine == "pymupdf"
        assert isinstance(loader.available_engines, list)

        # Test with custom engine
        loader = PDFLoader(preferred_engine="pypdf2")
        assert loader.preferred_engine == "pypdf2"

        # Test engine checking - should return valid engine names only
        engines = loader._check_available_engines()
        valid_engines = {"pymupdf", "pypdf2", "pdfplumber"}
        for engine in engines:
            assert engine in valid_engines

    def test_file_validation_errors(self):
        """Test file existence and type validation."""
        loader = PDFLoader()

        # Test non-existent file
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            loader.load_pdf("/nonexistent/file.pdf")

        # Test non-PDF file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is not a PDF file")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="File is not a PDF"):
                loader.load_pdf(temp_file)
        finally:
            os.unlink(temp_file)

    def test_engine_availability_and_fallback(self):
        """Test engine availability detection and fallback behavior."""
        # Test when preferred engine is not available
        with patch(
            "pycontextify.index.pdf_loader.PDFLoader._check_available_engines"
        ) as mock_check:
            mock_check.return_value = ["pymupdf", "pypdf2"]  # pdfplumber not available

            loader = PDFLoader(preferred_engine="pdfplumber")
            assert "pdfplumber" not in loader.available_engines
            assert "pymupdf" in loader.available_engines

    def test_no_engines_available_error(self):
        """Test error when no PDF engines are available."""
        with patch(
            "pycontextify.index.pdf_loader.PDFLoader._check_available_engines"
        ) as mock_check:
            mock_check.return_value = []  # No engines available

            with pytest.raises(
                ImportError, match="No PDF processing engines available"
            ):
                PDFLoader()

    def test_engine_specific_extraction(self):
        """Test extraction with specific engines using method mocking."""
        # Test PyMuPDF extraction
        loader = PDFLoader(preferred_engine="pymupdf")
        with patch.object(
            loader,
            "_extract_with_pymupdf",
            return_value="PyMuPDF content\n--- Page 1 ---",
        ) as mock_method:
            result = loader._extract_with_pymupdf("/fake/path.pdf")
            assert "PyMuPDF content" in result
            assert "--- Page 1 ---" in result
            mock_method.assert_called_once_with("/fake/path.pdf")

        # Test PyPDF2 extraction
        loader = PDFLoader(preferred_engine="pypdf2")
        with patch.object(
            loader,
            "_extract_with_pypdf2",
            return_value="PyPDF2 content\n--- Page 1 ---",
        ) as mock_method:
            result = loader._extract_with_pypdf2("/fake/path.pdf")
            assert "PyPDF2 content" in result
            mock_method.assert_called_once_with("/fake/path.pdf")

        # Test pdfplumber extraction
        loader = PDFLoader(preferred_engine="pdfplumber")
        with patch.object(
            loader,
            "_extract_with_pdfplumber",
            return_value="pdfplumber content\n--- Page 1 ---",
        ) as mock_method:
            result = loader._extract_with_pdfplumber("/fake/path.pdf")
            assert "pdfplumber content" in result
            mock_method.assert_called_once_with("/fake/path.pdf")

    def test_successful_extraction_with_preferred_engine(self):
        """Test successful extraction with preferred engine."""
        with patch(
            "pycontextify.index.pdf_loader.PDFLoader._check_available_engines"
        ) as mock_check:
            mock_check.return_value = ["pymupdf"]

            loader = PDFLoader(preferred_engine="pymupdf")

            with patch.object(loader, "_extract_with_engine") as mock_extract:
                mock_extract.return_value = (
                    "--- Page 1 ---\nSuccessful content extraction"
                )

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    temp_file = f.name

                try:
                    result = loader.load_pdf(temp_file)
                    assert "Successful content extraction" in result
                    assert "--- Page 1 ---" in result
                finally:
                    os.unlink(temp_file)

    def test_engine_fallback_on_failure(self):
        """Test fallback to secondary engine when primary fails."""
        with patch(
            "pycontextify.index.pdf_loader.PDFLoader._check_available_engines"
        ) as mock_check:
            mock_check.return_value = ["pymupdf", "pypdf2"]

            loader = PDFLoader(preferred_engine="pymupdf")

            with patch.object(loader, "_extract_with_engine") as mock_extract:
                # First call fails, second succeeds
                mock_extract.side_effect = [
                    Exception("Primary engine failed"),
                    "Fallback engine content",
                ]

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    temp_file = f.name

                try:
                    result = loader.load_pdf(temp_file)
                    assert result == "Fallback engine content"
                    assert mock_extract.call_count == 2  # Called twice
                finally:
                    os.unlink(temp_file)

    def test_empty_text_and_error_handling(self):
        """Test handling of empty text and complete extraction failures."""
        with patch(
            "pycontextify.index.pdf_loader.PDFLoader._check_available_engines"
        ) as mock_check:
            mock_check.return_value = ["pymupdf"]

            loader = PDFLoader()

            # Test empty text extraction
            with patch.object(loader, "_extract_with_engine") as mock_extract:
                mock_extract.return_value = ""  # Empty text

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    temp_file = f.name

                try:
                    with pytest.raises(
                        ValueError, match="Could not extract text from PDF"
                    ):
                        loader.load_pdf(temp_file)
                finally:
                    os.unlink(temp_file)

            # Test all engines fail
            with patch.object(loader, "_extract_with_engine") as mock_extract:
                mock_extract.side_effect = Exception("All engines failed")

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    temp_file = f.name

                try:
                    with pytest.raises(
                        ValueError, match="Could not extract text from PDF"
                    ):
                        loader.load_pdf(temp_file)
                finally:
                    os.unlink(temp_file)

    def test_pdf_info_extraction(self):
        """Test PDF metadata information extraction."""
        with patch(
            "pycontextify.index.pdf_loader.PDFLoader._check_available_engines"
        ) as mock_check:
            mock_check.return_value = ["pymupdf"]

            loader = PDFLoader()
            expected_info = {
                "page_count": 3,
                "has_text": True,
                "metadata": {"title": "Test PDF", "author": "Test Author"},
            }

            with patch.object(loader, "_get_info_pymupdf") as mock_info:
                mock_info.return_value = expected_info

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    temp_file = f.name

                try:
                    info = loader.get_pdf_info(temp_file)
                    assert info["page_count"] == 3
                    assert info["engine_used"] == "pymupdf"
                    assert info["metadata"]["title"] == "Test PDF"
                finally:
                    os.unlink(temp_file)

    def test_pdf_info_engine_fallback(self):
        """Test PDF info fallback when primary engine fails."""
        with patch(
            "pycontextify.index.pdf_loader.PDFLoader._check_available_engines"
        ) as mock_check:
            mock_check.return_value = ["pymupdf", "pypdf2"]

            loader = PDFLoader()

            with patch.object(loader, "_get_info_pymupdf") as mock_info_fitz:
                with patch.object(loader, "_get_info_pypdf2") as mock_info_pypdf2:
                    # First engine fails, second succeeds
                    mock_info_fitz.side_effect = Exception("PyMuPDF info failed")
                    mock_info_pypdf2.return_value = {"page_count": 2, "has_text": True}

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                        temp_file = f.name

                    try:
                        info = loader.get_pdf_info(temp_file)
                        assert info["engine_used"] == "pypdf2"
                        assert info["page_count"] == 2
                    finally:
                        os.unlink(temp_file)

    def test_unsupported_engine_error(self):
        """Test error handling for unsupported engines."""
        loader = PDFLoader()
        with pytest.raises(ValueError, match="Unsupported engine"):
            loader._extract_with_engine("test.pdf", "unsupported_engine")

    def test_multi_page_extraction(self):
        """Test extraction from multi-page PDF documents."""
        with patch(
            "pycontextify.index.pdf_loader.PDFLoader._check_available_engines"
        ) as mock_check:
            mock_check.return_value = ["pymupdf"]

            loader = PDFLoader()

            with patch.object(loader, "_extract_with_engine") as mock_extract:
                mock_extract.return_value = (
                    "--- Page 1 ---\nPage 1 content\n\n--- Page 2 ---\nPage 2 content"
                )

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    temp_file = f.name

                try:
                    result = loader.load_pdf(temp_file)
                    assert "Page 1 content" in result
                    assert "Page 2 content" in result
                    assert "--- Page 1 ---" in result
                    assert "--- Page 2 ---" in result
                finally:
                    os.unlink(temp_file)

    def test_edge_case_empty_text_then_fallback_success(self):
        """Test edge case: preferred engine returns empty text, fallback succeeds."""
        with patch(
            "pycontextify.index.pdf_loader.PDFLoader._check_available_engines"
        ) as mock_check:
            mock_check.return_value = ["pymupdf", "pypdf2"]

            loader = PDFLoader(preferred_engine="pymupdf")

            with patch.object(loader, "_extract_with_engine") as mock_extract:
                # First call returns whitespace, second call succeeds
                mock_extract.side_effect = [
                    "   ",  # Whitespace only (considered empty)
                    "--- Page 1 ---\nFallback successful text",
                ]

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    temp_file = f.name

                try:
                    result = loader.load_pdf(temp_file)
                    assert "Fallback successful text" in result
                    assert mock_extract.call_count == 2
                finally:
                    os.unlink(temp_file)
