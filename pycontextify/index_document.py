"""Document indexing orchestration for PyContextify.

This module groups all document-specific logic including PDF extraction,
text loading, and coordination helpers used by :class:`IndexManager`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import SourceType

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
            raise ImportError(
                "No PDF processing engines available. Install PyMuPDF, PyPDF2, or pdfplumber."
            )

        logger.info("PDF loader initialized with engines: %s", self.available_engines)

    def _check_available_engines(self) -> List[str]:
        """Check which PDF engines are available."""
        engines: List[str] = []

        try:
            import fitz  # PyMuPDF  # noqa: F401

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
        """Load and extract text from PDF file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")

        if self.preferred_engine in self.available_engines:
            try:
                text = self._extract_with_engine(file_path, self.preferred_engine)
                if text.strip():
                    logger.info(
                        "Successfully extracted text using preferred engine %s",
                        self.preferred_engine,
                    )
                    return text
                logger.warning(
                    "%s extracted empty text, trying fallback", self.preferred_engine
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "%s failed: %s, trying fallback", self.preferred_engine, exc
                )

        for engine in self.available_engines:
            if engine == self.preferred_engine:
                continue
            try:
                text = self._extract_with_engine(file_path, engine)
                if text.strip():
                    logger.info(
                        "Successfully extracted text using fallback engine %s", engine
                    )
                    return text
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Engine %s failed: %s", engine, exc)

        raise ValueError(f"Could not extract text from PDF: {file_path}")

    def _extract_with_engine(self, file_path: str, engine: str) -> str:
        if engine == "pymupdf":
            return self._extract_with_pymupdf(file_path)
        if engine == "pypdf2":
            return self._extract_with_pypdf2(file_path)
        if engine == "pdfplumber":
            return self._extract_with_pdfplumber(file_path)
        raise ValueError(f"Unsupported engine: {engine}")

    def _extract_with_pymupdf(self, file_path: str) -> str:
        import fitz

        text_parts: List[str] = []
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        return "\n\n".join(text_parts)

    def _extract_with_pypdf2(self, file_path: str) -> str:
        import PyPDF2

        text_parts: List[str] = []
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        return "\n\n".join(text_parts)

    def _extract_with_pdfplumber(self, file_path: str) -> str:
        import pdfplumber

        text_parts: List[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        return "\n\n".join(text_parts)

    def extract_page_context(self, text: str) -> Dict[str, Any]:
        """Extract page number and section context from text."""

        context: Dict[str, Any] = {
            "page_number": None,
            "section_title": None,
            "has_page_marker": False,
        }

        page_patterns = [
            r"--- Page (\d+) ---",
            r"Page (\d+)",
            r"\[Page (\d+)\]",
            r"p\. ?(\d+)",
            r"©.*?(\d+)$",
        ]

        for pattern in page_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                try:
                    context["page_number"] = int(match.group(1))
                    context["has_page_marker"] = True
                    break
                except (ValueError, TypeError):  # pragma: no cover - defensive
                    continue

        section_patterns = [
            r"^\d+\.\d*\s+(.+)$",
            r"^([A-Z][A-Z\s]{5,50})$",
            r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\s*$",
            r"^\s*((Chapter|Section|Part|Annex|Appendix)\s+[\w\s]+)$",
        ]

        lines = text.split("\n")[:10]
        for line in lines:
            candidate = line.strip()
            if len(candidate) < 3 or len(candidate) > 100:
                continue

            for pattern in section_patterns:
                match = re.match(pattern, candidate)
                if match:
                    title = match.group(1).strip()
                    if title and len(title) > 2:
                        context["section_title"] = title
                        break

            if context["section_title"]:
                break

        return context

    def get_pdf_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive PDF metadata information."""

        path = Path(file_path)
        from datetime import datetime

        stat = path.stat()
        info: Dict[str, Any] = {
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
            "reading_time_minutes": 0,
        }

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

                if info["word_count_estimate"] > 0:
                    info["reading_time_minutes"] = max(
                        1, round(info["word_count_estimate"] / 200)
                    )

                break
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not get PDF info with %s: %s", engine, exc)
                continue

        return info

    def _get_info_pymupdf(self, file_path: str) -> Dict[str, Any]:
        import fitz

        info: Dict[str, Any] = {}
        with fitz.open(file_path) as doc:
            info["page_count"] = len(doc)

            sample_pages = min(3, len(doc))
            total_text = ""
            for i in range(sample_pages):
                total_text += doc[i].get_text()

            info["has_text"] = bool(total_text.strip())

            if total_text:
                words_in_sample = len(total_text.split())
                info["word_count_estimate"] = int(
                    (words_in_sample / sample_pages) * len(doc)
                )

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

    def _get_info_pypdf2(self, file_path: str) -> Dict[str, Any]:
        import PyPDF2

        info: Dict[str, Any] = {}
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            info["page_count"] = len(reader.pages)

            sample_pages = min(3, len(reader.pages))
            total_text = ""
            for i in range(sample_pages):
                page_text = reader.pages[i].extract_text()
                total_text += page_text

            info["has_text"] = bool(total_text.strip())

            if total_text:
                words_in_sample = len(total_text.split())
                info["word_count_estimate"] = int(
                    (words_in_sample / sample_pages) * len(reader.pages)
                )

            if reader.metadata:
                metadata = reader.metadata
                info["document_title"] = str(metadata.get("/Title", "")).strip() or None
                info["author"] = str(metadata.get("/Author", "")).strip() or None
                info["subject"] = str(metadata.get("/Subject", "")).strip() or None
                info["creator"] = str(metadata.get("/Creator", "")).strip() or None
                info["producer"] = str(metadata.get("/Producer", "")).strip() or None
                info["creation_date"] = (
                    str(metadata.get("/CreationDate", "")).strip() or None
                )
                info["modification_date"] = (
                    str(metadata.get("/ModDate", "")).strip() or None
                )

        return info

    def _get_info_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        import pdfplumber

        info: Dict[str, Any] = {}
        with pdfplumber.open(file_path) as pdf:
            info["page_count"] = len(pdf.pages)

            sample_pages = min(3, len(pdf.pages))
            total_text = ""
            for i in range(sample_pages):
                page_text = pdf.pages[i].extract_text() or ""
                total_text += page_text

            info["has_text"] = bool(total_text.strip())

            if total_text:
                words_in_sample = len(total_text.split())
                info["word_count_estimate"] = int(
                    (words_in_sample / sample_pages) * len(pdf.pages)
                )

            if pdf.metadata:
                metadata = pdf.metadata
                info["document_title"] = str(metadata.get("Title", "")).strip() or None
                info["author"] = str(metadata.get("Author", "")).strip() or None
                info["subject"] = str(metadata.get("Subject", "")).strip() or None
                info["creator"] = str(metadata.get("Creator", "")).strip() or None
                info["producer"] = str(metadata.get("Producer", "")).strip() or None
                info["creation_date"] = (
                    str(metadata.get("CreationDate", "")).strip() or None
                )
                info["modification_date"] = (
                    str(metadata.get("ModDate", "")).strip() or None
                )

        return info


class DocumentLoader:
    """Loader for individual documents with structure extraction."""

    def __init__(self, pdf_engine: str = "pymupdf"):
        self.pdf_engine = pdf_engine
        self._pdf_loader: Optional[PDFLoader] = None

    def load(self, file_path: str) -> List[Tuple[str, str]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._load_document(path)
        if content:
            return [(str(path), content)]
        return []

    def _load_document(self, path: Path) -> Optional[str]:
        extension = path.suffix.lower()

        if extension == ".pdf":
            return self._load_pdf(path)
        if extension in {".md", ".txt"}:
            return self._load_text_file(path)

        logger.warning("Unsupported document type: %s", extension)
        return None

    def _ensure_pdf_loader(self) -> Optional[PDFLoader]:
        if self._pdf_loader is None:
            try:
                self._pdf_loader = PDFLoader(preferred_engine=self.pdf_engine)
            except ImportError as exc:
                logger.warning("Could not initialize PDF loader: %s", exc)
                self._pdf_loader = None
        return self._pdf_loader

    def _load_pdf(self, path: Path) -> Optional[str]:
        loader = self._ensure_pdf_loader()
        if loader is None:
            logger.error("PDF loader not available")
            return None
        try:
            text = loader.load_pdf(str(path))
            logger.info("Successfully loaded PDF: %s", path.name)
            return text
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load PDF %s: %s", path, exc)
            return None

    @staticmethod
    def _load_text_file(path: Path) -> Optional[str]:
        encodings = ["utf-8", "utf-16", "latin-1"]
        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as handle:
                    return handle.read()
            except UnicodeDecodeError:
                continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Error reading %s: %s", path, exc)
                return None
        return None


class DocumentIndexer:
    """Coordinate document ingestion using an :class:`IndexManager`."""

    def __init__(self, manager: "IndexManager") -> None:
        self._manager = manager
        self._loader = DocumentLoader(pdf_engine=manager.config.pdf_engine)

    def index(self, path: str) -> Dict[str, Any]:
        logger.info("Starting document indexing: %s", path)

        try:
            files = self._loader.load(path)
            if not files:
                return {"error": "Could not load document"}

            file_path, content = files[0]
            chunks_added = self._manager.process_content(
                content, file_path, SourceType.DOCUMENT
            )

            self._manager.auto_save()
            self._manager.ensure_embedder_loaded()

            stats = {
                "file_processed": file_path,
                "chunks_added": chunks_added,
                "source_type": "document",
                "embedding_provider": self._manager.embedder.get_provider_name(),
                "embedding_model": self._manager.embedder.get_model_name(),
            }
            logger.info("Completed document indexing: %s", stats)
            return stats
        except Exception as exc:  # pragma: no cover - defensive
            error_msg = f"Failed to index document {path}: {exc}"
            logger.error(error_msg)
            return {"error": error_msg}


__all__ = ["DocumentIndexer", "DocumentLoader", "PDFLoader"]
