"""Text chunking system for PyContextify.

This module implements various chunking strategies for different content types,
with relationship extraction capabilities for building lightweight knowledge graphs.
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from ..orchestrator.config import Config
from ..storage.metadata import ChunkMetadata, SourceType

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for text chunkers.

    This class defines the interface for text chunking with relationship
    extraction capabilities for lightweight knowledge graph construction.
    """

    def __init__(self, config: Config) -> None:
        """Initialize chunker with configuration.

        Args:
            config: Configuration object with chunking parameters
        """
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.enable_relationships = config.enable_relationships
        self.max_relationships_per_chunk = config.max_relationships_per_chunk

    @abstractmethod
    def chunk_text(
        self, text: str, source_path: str, embedding_provider: str, embedding_model: str
    ) -> List[ChunkMetadata]:
        """Chunk text into semantic units with relationship extraction.

        Args:
            text: Text to chunk
            source_path: Path or URL of the source
            embedding_provider: Embedding provider name
            embedding_model: Embedding model name

        Returns:
            List of ChunkMetadata objects with extracted relationships
        """
        pass

    def _create_chunk_metadata(
        self,
        chunk_text: str,
        source_path: str,
        source_type: SourceType,
        start_char: int,
        end_char: int,
        embedding_provider: str,
        embedding_model: str,
        **kwargs,
    ) -> ChunkMetadata:
        """Create chunk metadata with embedding provider information.

        Args:
            chunk_text: The chunk text content
            source_path: Source file path or URL
            source_type: Type of source content
            start_char: Starting character position
            end_char: Ending character position
            embedding_provider: Embedding provider name
            embedding_model: Embedding model name
            **kwargs: Additional metadata fields

        Returns:
            ChunkMetadata object with relationship information
        """
        metadata = ChunkMetadata(
            source_path=source_path,
            source_type=source_type,
            chunk_text=chunk_text,
            start_char=start_char,
            end_char=end_char,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            file_extension=Path(source_path).suffix.lower() if source_path else None,
            **kwargs,
        )

        # Extract relationships if enabled
        if self.enable_relationships:
            self._extract_relationships(metadata)

        return metadata

    def _extract_relationships(self, chunk: ChunkMetadata) -> None:
        """Extract relationship hints from chunk text.

        This base implementation extracts basic patterns. Subclasses
        should override for content-specific extraction.

        Args:
            chunk: ChunkMetadata object to populate with relationships
        """
        # Extract basic references (words that might be important entities)
        # Look for capitalized words that might be proper nouns
        capitalized_words = re.findall(r"\b[A-Z][a-zA-Z]+\b", chunk.chunk_text)
        for word in capitalized_words[: self.max_relationships_per_chunk]:
            if len(word) > 2:  # Skip short words
                chunk.add_reference(word)

    def _split_by_tokens(self, text: str, chunk_size: int, overlap: int) -> List[tuple]:
        """Split text by approximate token count.

        Args:
            text: Text to split
            chunk_size: Target chunk size in tokens (approximated as words * 1.3)
            overlap: Overlap size in tokens

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        words = text.split()
        if not words:
            return []

        # Approximate tokens as words * 1.3 (rough estimate for English)
        words_per_chunk = int(chunk_size / 1.3)
        words_overlap = int(overlap / 1.3)

        chunks = []
        start_idx = 0

        while start_idx < len(words):
            # Get chunk words
            end_idx = min(start_idx + words_per_chunk, len(words))
            chunk_words = words[start_idx:end_idx]

            if not chunk_words:
                break

            # Find character positions
            if start_idx == 0:
                start_char = 0
            else:
                # Find start position by joining previous words
                start_char = len(" ".join(words[:start_idx])) + (
                    1 if start_idx > 0 else 0
                )

            end_char = len(" ".join(words[:end_idx]))

            chunk_text = " ".join(chunk_words)
            chunks.append((chunk_text, start_char, end_char))

            # Move start position for next chunk (with overlap)
            if end_idx >= len(words):
                break

            start_idx = max(end_idx - words_overlap, start_idx + 1)

        return chunks


class SimpleChunker(BaseChunker):
    """Simple token-based chunker for generic text."""

    def chunk_text(
        self, text: str, source_path: str, embedding_provider: str, embedding_model: str
    ) -> List[ChunkMetadata]:
        """Chunk text using simple token-based approach.

        Args:
            text: Text to chunk
            source_path: Source file path
            embedding_provider: Embedding provider name
            embedding_model: Embedding model name

        Returns:
            List of ChunkMetadata objects
        """
        if not text.strip():
            return []

        # Split into chunks
        chunk_tuples = self._split_by_tokens(text, self.chunk_size, self.chunk_overlap)

        # Create metadata objects
        chunks = []
        for chunk_text, start_char, end_char in chunk_tuples:
            chunk = self._create_chunk_metadata(
                chunk_text=chunk_text,
                source_path=source_path,
                source_type=SourceType.DOCUMENT,
                start_char=start_char,
                end_char=end_char,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )
            chunks.append(chunk)

        return chunks


class CodeChunker(SimpleChunker):
    """Code-aware chunker that respects code structure and extracts code relationships."""

    def chunk_text(
        self, text: str, source_path: str, embedding_provider: str, embedding_model: str
    ) -> List[ChunkMetadata]:
        """Chunk code text with awareness of code structure.

        Args:
            text: Code text to chunk
            source_path: Source file path
            embedding_provider: Embedding provider name
            embedding_model: Embedding model name

        Returns:
            List of ChunkMetadata objects with code relationships
        """
        if not text.strip():
            return []

        # Try to split by code boundaries first
        code_chunks = self._split_by_code_boundaries(text)

        # If code chunks are too large, fall back to token-based splitting
        final_chunks = []
        for code_text, start_char, end_char in code_chunks:
            # Check if this chunk is too large
            estimated_tokens = len(code_text.split()) * 1.3
            if estimated_tokens <= self.chunk_size:
                # Chunk is good size, use as-is
                final_chunks.append((code_text, start_char, end_char))
            else:
                # Chunk too large, split further
                sub_chunks = self._split_by_tokens(
                    code_text, self.chunk_size, self.chunk_overlap
                )
                # Adjust character positions
                for sub_text, sub_start, sub_end in sub_chunks:
                    adjusted_start = start_char + sub_start
                    adjusted_end = start_char + sub_end
                    final_chunks.append((sub_text, adjusted_start, adjusted_end))

        # Create metadata objects
        chunks = []
        for chunk_text, start_char, end_char in final_chunks:
            chunk = self._create_chunk_metadata(
                chunk_text=chunk_text,
                source_path=source_path,
                source_type=SourceType.CODE,
                start_char=start_char,
                end_char=end_char,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )
            chunks.append(chunk)

        return chunks

    def _split_by_code_boundaries(self, text: str) -> List[tuple]:
        """Split code by function/class boundaries when possible.

        Args:
            text: Code text to split

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        # Simple approach: split by double newlines and function/class definitions
        lines = text.split("\n")
        chunks = []
        current_chunk_lines = []
        current_start_char = 0
        char_position = 0

        for i, line in enumerate(lines):
            # Check if line starts a new function/class (simple heuristic)
            is_boundary = line.strip().startswith(
                ("def ", "class ", "function ", "const ", "var ", "let ")
            ) or (
                line.strip().startswith(("public ", "private ", "protected "))
                and any(keyword in line for keyword in ["function", "class", "def"])
            )

            # If we hit a boundary and have content, create a chunk
            if is_boundary and current_chunk_lines and len(current_chunk_lines) > 5:
                chunk_text = "\n".join(current_chunk_lines)
                end_char = char_position - 1  # Don't include the current line
                chunks.append((chunk_text, current_start_char, end_char))

                # Start new chunk
                current_chunk_lines = [line]
                current_start_char = char_position
            else:
                current_chunk_lines.append(line)

            # Update character position
            char_position += len(line) + 1  # +1 for newline

        # Add remaining lines as final chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append((chunk_text, current_start_char, len(text)))

        # If we didn't create any meaningful chunks, fall back to simple splitting
        if not chunks or len(chunks) == 1:
            return self._split_by_tokens(text, self.chunk_size, self.chunk_overlap)

        return chunks

    def _extract_relationships(self, chunk: ChunkMetadata) -> None:
        """Extract code-specific relationships.

        Args:
            chunk: ChunkMetadata object to populate with relationships
        """
        text = chunk.chunk_text

        # Extract function definitions
        func_pattern = r"(?:def|function)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        for match in re.finditer(func_pattern, text):
            func_name = match.group(1)
            chunk.add_code_symbol(func_name)
            chunk.add_reference(func_name)

        # Extract class definitions
        class_pattern = r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        for match in re.finditer(class_pattern, text):
            class_name = match.group(1)
            chunk.add_code_symbol(class_name)
            chunk.add_reference(class_name)

        # Extract imports
        import_patterns = [
            r"import\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
            r"from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import",
            r'#include\s*[<"]([^>"]+)[">]',
            r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
        ]

        for pattern in import_patterns:
            for match in re.finditer(pattern, text):
                module_name = match.group(1)
                chunk.add_reference(module_name)
                chunk.add_tag("import")

        # Extract variable assignments (simple heuristic)
        var_names = re.findall(
            r"^(?:var|let|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)", text, re.MULTILINE
        )
        for var_name in var_names[:5]:  # Limit to avoid noise
            chunk.add_code_symbol(var_name)


class DocumentChunker(SimpleChunker):
    """Document-aware chunker that preserves document structure and extracts document relationships."""

    def chunk_text(
        self, text: str, source_path: str, embedding_provider: str, embedding_model: str
    ) -> List[ChunkMetadata]:
        """Chunk document text with awareness of document structure.

        Args:
            text: Document text to chunk
            source_path: Source file path
            embedding_provider: Embedding provider name
            embedding_model: Embedding_model name

        Returns:
            List of ChunkMetadata objects with document relationships
        """
        if not text.strip():
            return []

        # Try to split by document structure first
        doc_chunks = self._split_by_document_structure(text)

        # Process chunks and split further if needed
        final_chunks = []
        for chunk_text, start_char, end_char, section in doc_chunks:
            # Check if chunk is too large
            estimated_tokens = len(chunk_text.split()) * 1.3
            if estimated_tokens <= self.chunk_size:
                final_chunks.append((chunk_text, start_char, end_char, section))
            else:
                # Split large chunks further
                sub_chunks = self._split_by_tokens(
                    chunk_text, self.chunk_size, self.chunk_overlap
                )
                for sub_text, sub_start, sub_end in sub_chunks:
                    adjusted_start = start_char + sub_start
                    adjusted_end = start_char + sub_end
                    final_chunks.append(
                        (sub_text, adjusted_start, adjusted_end, section)
                    )

        # Create metadata objects
        chunks = []
        for chunk_text, start_char, end_char, section in final_chunks:
            chunk = self._create_chunk_metadata(
                chunk_text=chunk_text,
                source_path=source_path,
                source_type=SourceType.DOCUMENT,
                start_char=start_char,
                end_char=end_char,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                parent_section=section,
            )
            chunks.append(chunk)

        return chunks

    def _split_by_document_structure(self, text: str) -> List[tuple]:
        """Split document by headers and sections.

        Args:
            text: Document text to split

        Returns:
            List of (chunk_text, start_char, end_char, section_name) tuples
        """
        lines = text.split("\n")
        chunks = []
        current_chunk_lines = []
        current_section = None
        current_start_char = 0
        char_position = 0

        for line in lines:
            # Check for markdown headers
            header_match = re.match(r"^(#{1,6})\s+(.+)", line.strip())
            if header_match:
                # Found a header - save current chunk if it has content
                if (
                    current_chunk_lines
                    and len(" ".join(current_chunk_lines).strip()) > 50
                ):
                    chunk_text = "\n".join(current_chunk_lines)
                    end_char = char_position - 1
                    chunks.append(
                        (chunk_text, current_start_char, end_char, current_section)
                    )

                # Start new chunk
                current_section = header_match.group(2).strip()
                current_chunk_lines = [line]
                current_start_char = char_position
            else:
                current_chunk_lines.append(line)

            char_position += len(line) + 1

        # Add final chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append((chunk_text, current_start_char, len(text), current_section))

        # If no structure found, fall back to simple splitting
        if len(chunks) <= 1:
            simple_chunks = self._split_by_tokens(
                text, self.chunk_size, self.chunk_overlap
            )
            return [
                (chunk_text, start_char, end_char, None)
                for chunk_text, start_char, end_char in simple_chunks
            ]

        return chunks

    def _extract_relationships(self, chunk: ChunkMetadata) -> None:
        """Extract document-specific relationships.

        Args:
            chunk: ChunkMetadata object to populate with relationships
        """
        text = chunk.chunk_text

        # Extract markdown links
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        for match in re.finditer(link_pattern, text):
            link_text = match.group(1)
            link_url = match.group(2)
            chunk.add_reference(link_text)
            if not link_url.startswith(("http://", "https://")):
                chunk.add_reference(link_url)  # Internal link

        # Extract citations (simple patterns)
        citation_patterns = [
            r"\[(\d+)\]",  # [1], [2], etc.
            r"\(([A-Za-z]+\s+\d{4})\)",  # (Smith 2020)
            r"@([a-zA-Z][a-zA-Z0-9_]*)",  # @reference
        ]

        for pattern in citation_patterns:
            for match in re.finditer(pattern, text):
                citation = match.group(1)
                chunk.add_reference(citation)
                chunk.add_tag("citation")

        # Extract emphasized terms (might be important concepts)
        emphasis_patterns = [
            r"\*\*([^*]+)\*\*",  # **bold**
            r"\*([^*]+)\*",  # *italic*
            r"`([^`]+)`",  # `code`
        ]

        for pattern in emphasis_patterns:
            for match in re.finditer(pattern, text):
                term = match.group(1).strip()
                if len(term) > 2 and len(term) < 50:  # Reasonable term length
                    chunk.add_reference(term)

        # Add section as tag if available
        if chunk.parent_section:
            chunk.add_tag(chunk.parent_section.lower())

        # Call parent for basic extraction
        super()._extract_relationships(chunk)


class WebPageChunker(DocumentChunker):
    """Web page chunker with HTML structure awareness and web-specific relationship extraction."""

    def chunk_text(
        self, text: str, source_path: str, embedding_provider: str, embedding_model: str
    ) -> List[ChunkMetadata]:
        """Chunk web page content with HTML structure awareness.

        Args:
            text: Web page text content (should be cleaned HTML)
            source_path: Source URL
            embedding_provider: Embedding provider name
            embedding_model: Embedding model name

        Returns:
            List of ChunkMetadata objects with web-specific relationships
        """
        if not text.strip():
            return []

        # Parse HTML structure and extract semantic chunks
        html_chunks = self._split_by_html_structure(text, source_path)

        # Process chunks and split further if needed
        final_chunks = []
        for chunk_text, start_char, end_char, section, metadata in html_chunks:
            # Check if chunk is too large
            estimated_tokens = len(chunk_text.split()) * 1.3
            if estimated_tokens <= self.chunk_size:
                final_chunks.append(
                    (chunk_text, start_char, end_char, section, metadata)
                )
            else:
                # Split large chunks further
                sub_chunks = self._split_by_tokens(
                    chunk_text, self.chunk_size, self.chunk_overlap
                )
                for sub_text, sub_start, sub_end in sub_chunks:
                    adjusted_start = start_char + sub_start
                    adjusted_end = start_char + sub_end
                    final_chunks.append(
                        (sub_text, adjusted_start, adjusted_end, section, metadata)
                    )

        # Create metadata objects
        chunks = []
        for chunk_text, start_char, end_char, section, web_metadata in final_chunks:
            chunk = self._create_chunk_metadata(
                chunk_text=chunk_text,
                source_path=source_path,
                source_type=SourceType.WEBPAGE,
                start_char=start_char,
                end_char=end_char,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                parent_section=section,
                metadata=web_metadata,
            )
            chunks.append(chunk)

        return chunks

    def _split_by_html_structure(self, text: str, source_url: str) -> List[tuple]:
        """Split web content by HTML semantic structure.

        This is a simplified version that works with cleaned text.
        For full HTML parsing, would need BeautifulSoup integration.

        Args:
            text: Cleaned web page text
            source_url: Source URL for extracting domain info

        Returns:
            List of (chunk_text, start_char, end_char, section, metadata) tuples
        """
        # For now, treat as document structure but add web-specific processing
        doc_chunks = super()._split_by_document_structure(text)

        # Add web-specific metadata
        web_chunks = []
        for chunk_text, start_char, end_char, section in doc_chunks:
            # Extract web-specific metadata
            metadata = self._extract_web_metadata(chunk_text, source_url)
            web_chunks.append((chunk_text, start_char, end_char, section, metadata))

        return web_chunks

    def _extract_web_metadata(self, text: str, source_url: str) -> Dict[str, Any]:
        """Extract web-specific metadata from chunk text.

        Args:
            text: Chunk text
            source_url: Source URL

        Returns:
            Dictionary with web-specific metadata
        """
        metadata = {}

        # Extract links from text (simple pattern)
        url_pattern = r'https?://[^\s<>"\'`|(){}[\]]+'
        urls = re.findall(url_pattern, text)
        if urls:
            metadata["links"] = urls[:10]  # Limit to avoid noise

        # Extract domain from source URL
        if source_url.startswith(("http://", "https://")):
            from urllib.parse import urlparse

            try:
                parsed = urlparse(source_url)
                metadata["domain"] = parsed.netloc
                metadata["url_path"] = parsed.path
            except Exception:
                pass

        return metadata

    def _extract_relationships(self, chunk: ChunkMetadata) -> None:
        """Extract web-specific relationships.

        Args:
            chunk: ChunkMetadata object to populate with relationships
        """
        # Call parent document extraction first
        super()._extract_relationships(chunk)

        # Add web-specific relationship extraction
        text = chunk.chunk_text

        # Extract domain information
        if "domain" in chunk.metadata:
            domain = chunk.metadata["domain"]
            chunk.add_tag(domain)
            chunk.add_reference(domain)

        # Extract URL path segments
        if "url_path" in chunk.metadata:
            path = chunk.metadata["url_path"]
            path_parts = [p for p in path.split("/") if p and len(p) > 1]
            for part in path_parts[:3]:  # Limit to avoid noise
                chunk.add_tag(part.lower())

        # Extract links
        if "links" in chunk.metadata:
            for link in chunk.metadata["links"][:5]:  # Limit to avoid noise
                chunk.add_reference(link)
                chunk.add_tag("external_link")

        # Extract email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, text)
        for email in emails[:3]:  # Limit to avoid noise
            chunk.add_reference(email)
            chunk.add_tag("contact")


class ChunkerFactory:
    """Factory for selecting appropriate chunker based on content type."""

    @staticmethod
    def get_chunker(source_type: SourceType, config: Config) -> BaseChunker:
        """Get appropriate chunker for content type.

        Args:
            source_type: Type of content to chunk
            config: Configuration object

        Returns:
            Appropriate chunker instance
        """
        if source_type == SourceType.CODE:
            return CodeChunker(config)
        elif source_type == SourceType.DOCUMENT:
            return DocumentChunker(config)
        elif source_type == SourceType.WEBPAGE:
            return WebPageChunker(config)
        else:
            # Default to simple chunker
            return SimpleChunker(config)
