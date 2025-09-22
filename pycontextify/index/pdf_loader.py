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
    
    def extract_page_context(self, text: str) -> dict:
        """Extract page number and section context from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Dictionary with page and section information
        """
        import re
        
        context = {
            "page_number": None,
            "section_title": None,
            "has_page_marker": False
        }
        
        # Look for page markers
        page_patterns = [
            r'--- Page (\d+) ---',
            r'Page (\d+)',
            r'\[Page (\d+)\]',
            r'p\. ?(\d+)',
            r'©.*?(\d+)$'  # Copyright line with page number
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                try:
                    context["page_number"] = int(match.group(1))
                    context["has_page_marker"] = True
                    break
                except (ValueError, IndexError):
                    continue
        
        # Look for section titles (common patterns)
        section_patterns = [
            r'^\d+\.\d*\s+(.+)$',  # Numbered sections like "1.2 Introduction"
            r'^[A-Z][A-Z\s]{5,50}$',  # ALL CAPS headings
            r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\s*$',  # Title Case headings
            r'^\s*(Chapter|Section|Part|Annex|Appendix)\s+([\w\s]+)$'  # Explicit chapter/section markers
        ]
        
        # Check first few lines for section titles
        lines = text.split('\n')[:10]  # Check first 10 lines
        for line in lines:
            line = line.strip()
            if len(line) < 3 or len(line) > 100:  # Skip very short or very long lines
                continue
                
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # Extract the title part
                    if match.lastindex and match.lastindex > 1:
                        title = match.group(2).strip()
                    else:
                        title = match.group(1).strip()
                        
                    if title and len(title) > 2:
                        context["section_title"] = title
                        break
                        
            if context["section_title"]:
                break
        
        return context
    
    def get_pdf_info(self, file_path: str) -> dict:
        """Get comprehensive PDF metadata information.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with comprehensive PDF metadata
        """
        path = Path(file_path)
        import os
        from datetime import datetime
        
        stat = path.stat()
        info = {
            "file_path": str(path.absolute()),
            "file_name": path.name,
            "file_size_bytes": stat.st_size,
            "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "engine_used": None,
            "page_count": 0,
            "has_text": False,
            "document_title": None,
            "author": None,
            "subject": None,
            "creator": None,
            "producer": None,
            "creation_date": None,
            "modification_date": None,
            "word_count_estimate": 0,
            "reading_time_minutes": 0
        }
        
        # Try to get metadata with available engines
        for engine in self.available_engines:
            try:
                if engine == "pymupdf":
                    engine_info = self._get_info_pymupdf(file_path)
                elif engine == "pypdf2":
                    engine_info = self._get_info_pypdf2(file_path)
                elif engine == "pdfplumber":
                    engine_info = self._get_info_pdfplumber(file_path)
                else:
                    continue
                    
                info.update(engine_info)
                info["engine_used"] = engine
                
                # Calculate reading time estimate (average 200 words per minute)
                if info["word_count_estimate"] > 0:
                    info["reading_time_minutes"] = max(1, round(info["word_count_estimate"] / 200))
                
                break
            except Exception as e:
                logger.warning(f"Could not get PDF info with {engine}: {e}")
                continue
        
        return info
    
    def _get_info_pymupdf(self, file_path: str) -> dict:
        """Get comprehensive PDF info using PyMuPDF."""
        import fitz
        
        info = {}
        with fitz.open(file_path) as doc:
            info["page_count"] = len(doc)
            
            # Check if document has text
            sample_pages = min(3, len(doc))
            total_text = ""
            for i in range(sample_pages):
                page_text = doc[i].get_text()
                total_text += page_text
                
            info["has_text"] = bool(total_text.strip())
            
            # Estimate word count from sample
            if total_text:
                words_in_sample = len(total_text.split())
                info["word_count_estimate"] = int((words_in_sample / sample_pages) * len(doc))
            
            # Extract metadata
            if doc.metadata:
                metadata = doc.metadata
                info["document_title"] = metadata.get("title", "").strip() or None
                info["author"] = metadata.get("author", "").strip() or None
                info["subject"] = metadata.get("subject", "").strip() or None
                info["creator"] = metadata.get("creator", "").strip() or None
                info["producer"] = metadata.get("producer", "").strip() or None
                info["creation_date"] = metadata.get("creationDate", "").strip() or None
                info["modification_date"] = metadata.get("modDate", "").strip() or None
        
        return info
    
    def _get_info_pypdf2(self, file_path: str) -> dict:
        """Get comprehensive PDF info using PyPDF2."""
        import PyPDF2
        
        info = {}
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            info["page_count"] = len(reader.pages)
            
            # Check text and estimate word count
            sample_pages = min(3, len(reader.pages))
            total_text = ""
            for i in range(sample_pages):
                page_text = reader.pages[i].extract_text()
                total_text += page_text
                
            info["has_text"] = bool(total_text.strip())
            
            if total_text:
                words_in_sample = len(total_text.split())
                info["word_count_estimate"] = int((words_in_sample / sample_pages) * len(reader.pages))
            
            # Extract metadata
            if reader.metadata:
                metadata = reader.metadata
                info["document_title"] = str(metadata.get("/Title", "")).strip() or None
                info["author"] = str(metadata.get("/Author", "")).strip() or None
                info["subject"] = str(metadata.get("/Subject", "")).strip() or None
                info["creator"] = str(metadata.get("/Creator", "")).strip() or None
                info["producer"] = str(metadata.get("/Producer", "")).strip() or None
                info["creation_date"] = str(metadata.get("/CreationDate", "")).strip() or None
                info["modification_date"] = str(metadata.get("/ModDate", "")).strip() or None
        
        return info
    
    def _get_info_pdfplumber(self, file_path: str) -> dict:
        """Get comprehensive PDF info using pdfplumber."""
        import pdfplumber
        
        info = {}
        with pdfplumber.open(file_path) as pdf:
            info["page_count"] = len(pdf.pages)
            
            # Check text and estimate word count
            sample_pages = min(3, len(pdf.pages))
            total_text = ""
            for i in range(sample_pages):
                page_text = pdf.pages[i].extract_text() or ""
                total_text += page_text
                
            info["has_text"] = bool(total_text.strip())
            
            if total_text:
                words_in_sample = len(total_text.split())
                info["word_count_estimate"] = int((words_in_sample / sample_pages) * len(pdf.pages))
            
            # Extract metadata
            if pdf.metadata:
                metadata = pdf.metadata
                info["document_title"] = str(metadata.get("Title", "")).strip() or None
                info["author"] = str(metadata.get("Author", "")).strip() or None
                info["subject"] = str(metadata.get("Subject", "")).strip() or None
                info["creator"] = str(metadata.get("Creator", "")).strip() or None
                info["producer"] = str(metadata.get("Producer", "")).strip() or None
                info["creation_date"] = str(metadata.get("CreationDate", "")).strip() or None
                info["modification_date"] = str(metadata.get("ModDate", "")).strip() or None
        
        return info
