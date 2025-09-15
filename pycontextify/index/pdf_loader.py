"""Enhanced PDF loader with support for multiple PDF processing engines.

This module provides a unified interface for PDF text extraction using
different backends (PyMuPDF, PyPDF2, pdfplumber) with fallback support.
"""

import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFLoader:
    """Enhanced PDF loader with multiple engine support and fallback."""
    
    def __init__(self, preferred_engine: str = "pymupdf"):
        """Initialize PDF loader with preferred engine.
        
        Args:
            preferred_engine: Preferred PDF engine ('pymupdf', 'pypdf2', 'pdfplumber')
        """
        self.preferred_engine = preferred_engine
        self.available_engines = self._check_available_engines()
        
        if not self.available_engines:
            raise ImportError("No PDF processing engines available. Install PyMuPDF, PyPDF2, or pdfplumber.")
        
        logger.info(f"PDF loader initialized with engines: {self.available_engines}")
        
    def _check_available_engines(self) -> List[str]:
        """Check which PDF engines are available."""
        engines = []
        
        try:
            import fitz  # PyMuPDF # noqa: F401
            engines.append("pymupdf")
        except ImportError:
            pass
            
        try:
            import PyPDF2  # noqa: F401
            engines.append("pypdf2")
        except ImportError:
            pass
            
        try:
            import pdfplumber  # noqa: F401
            engines.append("pdfplumber")
        except ImportError:
            pass
            
        return engines
    
    def load_pdf(self, file_path: str) -> str:
        """Load and extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If no text could be extracted
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        if not path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")
        
        # Try preferred engine first
        if self.preferred_engine in self.available_engines:
            try:
                text = self._extract_with_engine(file_path, self.preferred_engine)
                if text.strip():
                    logger.info(f"Successfully extracted text using {self.preferred_engine}")
                    return text
                else:
                    logger.warning(f"{self.preferred_engine} extracted empty text, trying fallback")
            except Exception as e:
                logger.warning(f"{self.preferred_engine} failed: {e}, trying fallback")
        
        # Try fallback engines
        for engine in self.available_engines:
            if engine == self.preferred_engine:
                continue
                
            try:
                text = self._extract_with_engine(file_path, engine)
                if text.strip():
                    logger.info(f"Successfully extracted text using fallback engine: {engine}")
                    return text
            except Exception as e:
                logger.warning(f"Engine {engine} failed: {e}")
        
        raise ValueError(f"Could not extract text from PDF: {file_path}")
    
    def _extract_with_engine(self, file_path: str, engine: str) -> str:
        """Extract text using specified engine.
        
        Args:
            file_path: Path to PDF file
            engine: Engine to use ('pymupdf', 'pypdf2', 'pdfplumber')
            
        Returns:
            Extracted text
        """
        if engine == "pymupdf":
            return self._extract_with_pymupdf(file_path)
        elif engine == "pypdf2":
            return self._extract_with_pypdf2(file_path)
        elif engine == "pdfplumber":
            return self._extract_with_pdfplumber(file_path)
        else:
            raise ValueError(f"Unsupported engine: {engine}")
    
    def _extract_with_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF (fitz)."""
        import fitz
        
        text_parts = []
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2."""
        import PyPDF2
        
        text_parts = []
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber."""
        import pdfplumber
        
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        
        return "\n\n".join(text_parts)
    
    def get_pdf_info(self, file_path: str) -> dict:
        """Get PDF metadata information.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        path = Path(file_path)
        info = {
            "file_path": str(path.absolute()),
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "engine_used": None,
            "page_count": 0,
            "has_text": False
        }
        
        # Try to get metadata with available engines
        for engine in self.available_engines:
            try:
                if engine == "pymupdf":
                    info.update(self._get_info_pymupdf(file_path))
                elif engine == "pypdf2":
                    info.update(self._get_info_pypdf2(file_path))
                elif engine == "pdfplumber":
                    info.update(self._get_info_pdfplumber(file_path))
                    
                info["engine_used"] = engine
                break
            except Exception as e:
                logger.warning(f"Could not get PDF info with {engine}: {e}")
                continue
        
        return info
    
    def _get_info_pymupdf(self, file_path: str) -> dict:
        """Get PDF info using PyMuPDF."""
        import fitz
        
        info = {}
        with fitz.open(file_path) as doc:
            info["page_count"] = len(doc)
            info["has_text"] = any(doc[i].get_text().strip() for i in range(min(3, len(doc))))
            if doc.metadata:
                info["metadata"] = doc.metadata
        
        return info
    
    def _get_info_pypdf2(self, file_path: str) -> dict:
        """Get PDF info using PyPDF2."""
        import PyPDF2
        
        info = {}
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            info["page_count"] = len(reader.pages)
            info["has_text"] = any(reader.pages[i].extract_text().strip() 
                                 for i in range(min(3, len(reader.pages))))
            if reader.metadata:
                info["metadata"] = dict(reader.metadata)
        
        return info
    
    def _get_info_pdfplumber(self, file_path: str) -> dict:
        """Get PDF info using pdfplumber."""
        import pdfplumber
        
        info = {}
        with pdfplumber.open(file_path) as pdf:
            info["page_count"] = len(pdf.pages)
            info["has_text"] = any(pdf.pages[i].extract_text() 
                                 for i in range(min(3, len(pdf.pages))))
            if pdf.metadata:
                info["metadata"] = pdf.metadata
        
        return info