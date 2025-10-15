"""File loader factory for PyContextify filebase indexing.

This module provides unified file loading with MIME detection and normalization.
"""

import logging
import mimetypes
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Supported text/code file extensions (non-exhaustive, common ones)
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".tex",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".swift",
    ".kt",
    ".scala",
    ".rb",
    ".php",
    ".pl",
    ".sh",
    ".bash",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".sql",
    ".r",
    ".m",
    ".mat",
}


class FileLoaderFactory:
    """Factory for loading files and normalizing them to a standard format.

    This loader:
    - Detects file types via MIME/extension
    - Loads PDFs via LangChain PyPDFLoader
    - Loads text/code files directly
    - Skips binary files
    - Normalizes all outputs to {"text": str, "metadata": dict}

    Examples:
        >>> loader = FileLoaderFactory()
        >>> docs = loader.load("/path/to/file.pdf", tags="documentation")
        >>> # Returns list of normalized docs (one per PDF page)

        >>> docs = loader.load("/path/to/code.py", tags="codebase")
        >>> # Returns single normalized doc with code content
    """

    def __init__(self, default_encoding: str = "utf-8"):
        """Initialize loader with encoding settings.

        Args:
            default_encoding: Default encoding for text files
        """
        self.default_encoding = default_encoding
        logger.debug(f"FileLoaderFactory initialized: encoding={default_encoding}")

    def _normalize_tags(self, tags: Any) -> List[str]:
        """Normalize raw tag input into a list of distinct tags."""
        if not tags:
            return []

        if isinstance(tags, str):
            normalized = [tag.strip() for tag in tags.split(",") if tag.strip()]
        elif isinstance(tags, (list, tuple, set)):
            normalized = [str(tag).strip() for tag in tags if str(tag).strip()]
        else:
            normalized = []

        # Preserve order while removing duplicates
        return list(dict.fromkeys(normalized))

    def load(
        self, path: str, tags: str, base_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load and normalize a file to standard format.

        Args:
            path: Absolute path to file
            tags: Comma-separated tags (required for all files)
            base_path: Base directory path for computing relative_path (optional)

        Returns:
            List of normalized documents: [{"text": str, "metadata": dict}, ...]
            Returns empty list if file cannot be loaded or is binary.

        Raises:
            ValueError: If tags are empty or None
            FileNotFoundError: If path does not exist
        """
        if not tags or not isinstance(tags, str) or not tags.strip():
            raise ValueError("tags are required and must be a non-empty string")

        file_path = Path(path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        if not file_path.is_file():
            logger.warning(f"Path is not a file, skipping: {path}")
            return []

        start_time = time.time()

        try:
            # Detect MIME type and extension
            mime_type, _ = mimetypes.guess_type(str(file_path))
            file_extension = file_path.suffix.lower()

            # Build base metadata common to all docs from this file
            base_metadata = self._build_base_metadata(
                file_path=file_path,
                base_path=base_path,
                tags=tags,
                mime_type=mime_type,
                file_extension=file_extension,
            )

            # Route to appropriate loader
            if self._is_pdf(file_path, mime_type):
                docs = self._load_pdf(file_path, base_metadata)
            elif self._is_text_file(file_path, mime_type):
                docs = self._load_text(file_path, base_metadata)
            else:
                logger.debug(f"Skipping binary/unsupported file: {path}")
                return []

            # Add loading time to all docs
            loading_time_ms = (time.time() - start_time) * 1000
            for doc in docs:
                doc["metadata"]["loading_time_ms"] = loading_time_ms

            logger.debug(f"Loaded {len(docs)} doc(s) from {file_path.name}")
            return docs

        except Exception as exc:
            logger.error(f"Failed to load {path}: {exc}")
            return []

    def _build_base_metadata(
        self,
        file_path: Path,
        base_path: Optional[str],
        tags: str,
        mime_type: Optional[str],
        file_extension: str,
    ) -> Dict[str, Any]:
        """Build base metadata dictionary for a file.

        Args:
            file_path: Resolved Path object
            base_path: Optional base directory for relative path calculation
            tags: Raw tags string
            mime_type: Detected MIME type
            file_extension: File extension with dot

        Returns:
            Dictionary with base metadata fields
        """
        # Compute relative path
        if base_path:
            try:
                base_resolved = Path(base_path).resolve()
                relative_path = file_path.relative_to(base_resolved).as_posix()
            except ValueError:
                # File not under base_path, use filename
                relative_path = file_path.name
        else:
            relative_path = file_path.name

        # Clean extension (remove dot)
        clean_extension = file_extension.lstrip(".")

        return {
            "mime_type": mime_type,
            "full_path": str(file_path),
            "relative_path": relative_path,
            "filename_stem": file_path.stem,
            "file_extension": clean_extension,
            "date_loaded": datetime.now(timezone.utc).isoformat(),
            "tags": self._normalize_tags(tags),
            "language": None,  # To be filled by chunker if needed
            "links": None,  # To be filled by chunker if detected
        }

    def _is_pdf(self, file_path: Path, mime_type: Optional[str]) -> bool:
        """Check if file is a PDF.

        Args:
            file_path: File path
            mime_type: Detected MIME type

        Returns:
            True if file is PDF
        """
        return mime_type == "application/pdf" or file_path.suffix.lower() == ".pdf"

    def _is_text_file(self, file_path: Path, mime_type: Optional[str]) -> bool:
        """Check if file is a text/code file.

        Args:
            file_path: File path
            mime_type: Detected MIME type

        Returns:
            True if file is text-based
        """
        # Check extension against known text extensions
        if file_path.suffix.lower() in TEXT_EXTENSIONS:
            return True

        # Check MIME type
        if mime_type and mime_type.startswith("text/"):
            return True

        return False

    def _load_pdf(
        self, file_path: Path, base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Load PDF content, preferring PyMuPDF4LLM for Markdown extraction.

        Falls back to the previous LangChain-based implementation when
        PyMuPDF4LLM is unavailable to preserve backwards compatibility.
        """

        try:
            return self._load_pdf_with_pymupdf4llm(file_path, base_metadata)
        except ImportError:
            logger.debug(
                "PyMuPDF4LLM not available, falling back to LangChain PyPDFLoader"
            )
        except Exception as exc:
            logger.warning(
                "PyMuPDF4LLM PDF load failed for %s: %s. Falling back to LangChain",
                file_path,
                exc,
            )

        return self._load_pdf_with_langchain(file_path, base_metadata)

    def _load_pdf_with_pymupdf4llm(
        self, file_path: Path, base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Load PDF as Markdown using PyMuPDF4LLM."""

        import pymupdf  # type: ignore[import]
        import pymupdf4llm  # type: ignore[import]

        markdown_text = pymupdf4llm.to_markdown(str(file_path))
        markdown_text = self._clean_markdown(markdown_text)

        total_pages = None
        try:
            with pymupdf.open(str(file_path)) as pdf_doc:  # type: ignore[attr-defined]
                total_pages = pdf_doc.page_count
        except Exception as exc:
            logger.debug(
                "Unable to determine PDF page count for %s: %s", file_path, exc
            )

        metadata = base_metadata.copy()
        metadata["pdf_loader"] = "pymupdf4llm"
        if total_pages is not None:
            metadata["total_pages"] = total_pages

        # Provide a synthetic page number to keep downstream expectations intact
        metadata.setdefault("page_number", 1)

        logger.info("Loaded PDF via PyMuPDF4LLM: %s", file_path.name)
        return [{"text": markdown_text, "metadata": metadata}]

    def _load_pdf_with_langchain(
        self, file_path: Path, base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback PDF loader using LangChain's PyPDFLoader."""

        try:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(str(file_path))
            langchain_docs = loader.load()

            normalized_docs = []
            for lc_doc in langchain_docs:
                doc = {"text": lc_doc.page_content, "metadata": base_metadata.copy()}

                if lc_doc.metadata:
                    if "links" in lc_doc.metadata:
                        doc["metadata"]["links"] = lc_doc.metadata["links"]

                    if "page" in lc_doc.metadata:
                        doc["metadata"]["page_number"] = lc_doc.metadata["page"]

                    for key in ["source", "total_pages"]:
                        if key in lc_doc.metadata:
                            doc["metadata"][key] = lc_doc.metadata[key]

                normalized_docs.append(doc)

            logger.info(
                "Loaded PDF with %s pages via LangChain: %s",
                len(normalized_docs),
                file_path.name,
            )
            return normalized_docs

        except ImportError as e:
            logger.error(
                "LangChain PyPDFLoader not available: %s. Install with: pip install langchain-community",
                e,
            )
            return []
        except Exception as exc:
            logger.error("Failed to load PDF %s via LangChain: %s", file_path, exc)
            return []

    def _clean_markdown(self, markdown_text: str) -> str:
        """Normalize Markdown extracted from PDFs."""

        markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text)
        lines = markdown_text.splitlines()
        clean_lines = [
            line for line in lines if not re.match(r"^Page \d+$", line.strip())
        ]
        return "\n".join(clean_lines).strip()

    def _load_text(
        self, file_path: Path, base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Load text/code file.

        Args:
            file_path: Path to text file
            base_metadata: Base metadata to include

        Returns:
            List with single normalized document
        """
        # Try reading with default encoding
        try:
            with open(file_path, "r", encoding=self.default_encoding) as f:
                text = f.read()

            return [{"text": text, "metadata": base_metadata.copy()}]

        except UnicodeDecodeError:
            # Try fallback encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        text = f.read()

                    logger.debug(
                        f"Loaded {file_path.name} with fallback encoding: {encoding}"
                    )
                    return [{"text": text, "metadata": base_metadata.copy()}]
                except UnicodeDecodeError:
                    continue

            logger.warning(f"Could not decode text file {file_path}, skipping")
            return []

        except Exception as exc:
            logger.error(f"Failed to load text file {file_path}: {exc}")
            return []


__all__ = ["FileLoaderFactory"]
