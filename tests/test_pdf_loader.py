"""Tests for enhanced PDF loader functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from pycontextify.index.pdf_loader import PDFLoader


class TestPDFLoader:
    """Test PDF loader functionality."""

    def test_pdf_loader_initialization(self):
        """Test PDF loader initialization."""
        # Test with default engine
        loader = PDFLoader()
        assert loader.preferred_engine == "pymupdf"
        assert isinstance(loader.available_engines, list)
        
        # Test with custom engine
        loader = PDFLoader(preferred_engine="pypdf2")
        assert loader.preferred_engine == "pypdf2"
    
    def test_check_available_engines(self):
        """Test checking available PDF engines."""
        loader = PDFLoader()
        engines = loader._check_available_engines()
        
        # Should be a list (might be empty if no engines are installed)
        assert isinstance(engines, list)
        
        # All engines should be valid names
        valid_engines = {"pymupdf", "pypdf2", "pdfplumber"}
        for engine in engines:
            assert engine in valid_engines
    
    def test_load_pdf_file_not_found(self):
        """Test error handling when PDF file doesn't exist."""
        loader = PDFLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_pdf("/nonexistent/file.pdf")
    
    def test_load_pdf_not_pdf_file(self):
        """Test error handling for non-PDF files."""
        loader = PDFLoader()
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is not a PDF file")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="File is not a PDF"):
                loader.load_pdf(temp_file)
        finally:
            os.unlink(temp_file)
    
    @patch('pycontextify.index.pdf_loader.fitz')
    def test_extract_with_pymupdf(self, mock_fitz):
        """Test PDF extraction with PyMuPDF."""
        # Mock PyMuPDF behavior
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Test PDF content from page 1"
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz.open.return_value = mock_doc
        
        loader = PDFLoader(preferred_engine="pymupdf")
        
        # Test the extraction method directly
        result = loader._extract_with_pymupdf("/fake/path.pdf")
        
        assert "Test PDF content from page 1" in result
        assert "--- Page 1 ---" in result
        mock_fitz.open.assert_called_once_with("/fake/path.pdf")
        mock_page.get_text.assert_called_once()
    
    @patch('pycontextify.index.pdf_loader.PyPDF2')
    def test_extract_with_pypdf2(self, mock_pypdf2):
        """Test PDF extraction with PyPDF2."""
        # Mock PyPDF2 behavior
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test PDF content from PyPDF2"
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        loader = PDFLoader(preferred_engine="pypdf2")
        
        # Mock file opening
        mock_file_content = b"fake pdf content"
        with patch("builtins.open", mock=Mock()) as mock_open:
            mock_open.return_value.__enter__.return_value = Mock()
            
            result = loader._extract_with_pypdf2("/fake/path.pdf")
            
            assert "Test PDF content from PyPDF2" in result
            assert "--- Page 1 ---" in result
    
    @patch('pycontextify.index.pdf_loader.pdfplumber')
    def test_extract_with_pdfplumber(self, mock_pdfplumber):
        """Test PDF extraction with pdfplumber."""
        # Mock pdfplumber behavior
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test PDF content from pdfplumber"
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        mock_pdf.__exit__.return_value = None
        mock_pdfplumber.open.return_value = mock_pdf
        
        loader = PDFLoader(preferred_engine="pdfplumber")
        
        result = loader._extract_with_pdfplumber("/fake/path.pdf")
        
        assert "Test PDF content from pdfplumber" in result
        assert "--- Page 1 ---" in result
        mock_pdfplumber.open.assert_called_once_with("/fake/path.pdf")
    
    def test_get_pdf_info_file_not_found(self):
        """Test PDF info for non-existent file."""
        loader = PDFLoader()
        
        # This should not raise an error, but return basic info
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as f:
            temp_path = f.name
        
        # File is deleted, so it doesn't exist
        info = loader.get_pdf_info(temp_path)
        
        assert "file_path" in info
        assert "file_name" in info
        assert info["page_count"] == 0
        assert info["has_text"] is False
    
    @patch('pycontextify.index.pdf_loader.fitz')
    def test_get_pdf_info_with_pymupdf(self, mock_fitz):
        """Test getting PDF info with PyMuPDF."""
        # Mock PyMuPDF behavior
        mock_doc = Mock()
        mock_doc.__len__.return_value = 3  # 3 pages
        mock_doc.__getitem__.side_effect = [
            Mock(get_text=Mock(return_value="Page 1 content")),
            Mock(get_text=Mock(return_value="Page 2 content")),
            Mock(get_text=Mock(return_value="Page 3 content"))
        ]
        mock_doc.metadata = {"title": "Test PDF", "author": "Test Author"}
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz.open.return_value = mock_doc
        
        # Create temporary file for path info
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_file = f.name
        
        try:
            loader = PDFLoader()
            # Force PyMuPDF to be available for this test
            loader.available_engines = ["pymupdf"]
            
            info = loader.get_pdf_info(temp_file)
            
            assert info["page_count"] == 3
            assert info["has_text"] is True
            assert info["engine_used"] == "pymupdf"
            assert "metadata" in info
            assert info["metadata"]["title"] == "Test PDF"
            
        finally:
            os.unlink(temp_file)
    
    def test_unsupported_engine(self):
        """Test error handling for unsupported engines."""
        loader = PDFLoader()
        
        with pytest.raises(ValueError, match="Unsupported engine"):
            loader._extract_with_engine("/fake/path.pdf", "unsupported_engine")
    
    def test_engine_fallback(self):
        """Test fallback behavior when preferred engine fails."""
        with patch.object(PDFLoader, '_check_available_engines') as mock_check:
            mock_check.return_value = ["pypdf2", "pdfplumber"]  # pymupdf not available
            
            loader = PDFLoader(preferred_engine="pymupdf")  # Preferred not available
            
            # Available engines should not include the preferred one
            assert "pymupdf" not in loader.available_engines
            assert "pypdf2" in loader.available_engines or "pdfplumber" in loader.available_engines
    
    def test_no_engines_available(self):
        """Test behavior when no PDF engines are available."""
        with patch.object(PDFLoader, '_check_available_engines') as mock_check:
            mock_check.return_value = []  # No engines available
            
            with pytest.raises(ImportError, match="No PDF processing engines available"):
                PDFLoader()
    
    def test_empty_pdf_handling(self):
        """Test handling of PDFs with no extractable text."""
        with patch.object(PDFLoader, '_extract_with_engine') as mock_extract:
            mock_extract.return_value = ""  # Empty text
            
            loader = PDFLoader()
            loader.available_engines = ["pymupdf"]  # Ensure at least one engine
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                temp_file = f.name
            
            try:
                with pytest.raises(ValueError, match="Could not extract text from PDF"):
                    loader.load_pdf(temp_file)
            finally:
                os.unlink(temp_file)
    
    def test_fallback_engine_success(self):
        """Test successful fallback to secondary engine."""
        with patch.object(PDFLoader, '_extract_with_engine') as mock_extract:
            # First call (preferred engine) fails, second call (fallback) succeeds
            mock_extract.side_effect = [
                Exception("Primary engine failed"),
                "Successfully extracted text from fallback engine"
            ]
            
            loader = PDFLoader(preferred_engine="pymupdf")
            loader.available_engines = ["pymupdf", "pypdf2"]  # Both available
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                temp_file = f.name
            
            try:
                result = loader.load_pdf(temp_file)
                assert result == "Successfully extracted text from fallback engine"
                # Should have tried both engines
                assert mock_extract.call_count == 2
            finally:
                os.unlink(temp_file)