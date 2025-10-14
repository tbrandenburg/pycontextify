"""Text chunking system for PyContextify.

This module implements various chunking strategies for different content types,
with relationship extraction capabilities for building lightweight knowledge graphs.
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from .config import Config
from .types import Chunk, SourceType

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
    ) -> List[Chunk]:
        """Chunk text into semantic units with relationship extraction.

        Args:
            text: Text to chunk
            source_path: Path or URL of the source
            embedding_provider: Embedding provider name
            embedding_model: Embedding model name

        Returns:
            List of Chunk objects with extracted relationships
        """
        pass

    def _create_chunk(
        self,
        chunk_text: str,
        source_path: str,
        source_type: SourceType,
        start_char: int,
        end_char: int,
        embedding_provider: str,
        embedding_model: str,
        **kwargs,
    ) -> Chunk:
        """Create chunk with embedding provider information.

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
            Chunk object with relationship information
        """
        chunk = Chunk(
            chunk_text=chunk_text,
            source_path=source_path,
            source_type=source_type,
            start_char=start_char,
            end_char=end_char,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            file_extension=Path(source_path).suffix.lower() if source_path else None,
            **kwargs,
        )

        # Extract relationships if enabled
        if self.enable_relationships:
            self._extract_relationships(chunk)

        return chunk

    def _extract_relationships(self, chunk: Chunk) -> None:
        """Extract relationship hints from chunk text.

        This base implementation extracts basic patterns. Subclasses
        should override for content-specific extraction.

        Args:
            chunk: Chunk object to populate with relationships
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
    ) -> List[Chunk]:
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
            chunk = self._create_chunk(
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
    ) -> List[Chunk]:
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
            chunk = self._create_chunk(
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

    def _extract_relationships(self, chunk: Chunk) -> None:
        """Extract code-specific relationships.

        Args:
            chunk: Chunk object to populate with relationships
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
    ) -> List[Chunk]:
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
            chunk = self._create_chunk(
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

    def _extract_relationships(self, chunk: Chunk) -> None:
        """Extract document-specific relationships.

        Args:
            chunk: Chunk object to populate with relationships
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
        else:
            # Default to simple chunker
            return SimpleChunker(config)

    @staticmethod
    def chunk_normalized_docs(
        normalized_docs: List[Dict[str, Any]],
        config: Config,
    ) -> List[Dict[str, Any]]:
        """Chunk normalized documents from loader into normalized chunks.

        This is the new adapter for the filebase pipeline that works with
        normalized dict format instead of Chunk dataclass.

        Args:
            normalized_docs: List of normalized docs from loader
                Format: [{"text": str, "metadata": dict}, ...]
            config: Configuration object

        Returns:
            List of normalized chunks with chunk-specific metadata added
                Format: [{"text": str, "metadata": dict}, ...]
        """
        all_chunks = []

        for doc_index, doc in enumerate(normalized_docs):
            text = doc["text"]
            base_metadata = doc["metadata"].copy()

            # Determine source type from file extension
            file_ext = base_metadata.get("file_extension", "")
            source_type = ChunkerFactory._infer_source_type(file_ext)

            # Get appropriate chunker
            chunker = ChunkerFactory.get_chunker(source_type, config)

            # Get embedding info from metadata (should be set by loader)
            embedding_provider = base_metadata.get(
                "embedding_provider", "sentence_transformers"
            )
            embedding_model = base_metadata.get("embedding_model", "all-mpnet-base-v2")
            source_path = base_metadata.get("full_path", "")

            # Chunk using existing chunker (returns Chunk objects)
            chunk_objects = chunker.chunk_text(
                text=text,
                source_path=source_path,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )

            # Convert Chunk objects to normalized dict format
            for chunk_idx, chunk_obj in enumerate(chunk_objects):
                chunk_dict = {
                    "text": chunk_obj.chunk_text,
                    "metadata": base_metadata.copy(),
                }

                # Add chunk-specific metadata
                chunk_dict["metadata"]["chunk_index"] = chunk_idx
                chunk_dict["metadata"]["chunk_name"] = chunk_obj.parent_section
                chunk_dict["metadata"]["language"] = base_metadata.get("language")
                chunk_dict["metadata"]["start_char"] = chunk_obj.start_char
                chunk_dict["metadata"]["end_char"] = chunk_obj.end_char

                # Preserve links from base metadata or chunk
                if base_metadata.get("links"):
                    chunk_dict["metadata"]["links"] = base_metadata["links"]

                # Add relationship metadata if relationships are enabled
                if config.enable_relationships:
                    if chunk_obj.tags:
                        chunk_dict["metadata"]["tags"] = chunk_obj.tags
                    if chunk_obj.references:
                        chunk_dict["metadata"]["references"] = chunk_obj.references
                    if chunk_obj.code_symbols:
                        chunk_dict["metadata"]["code_symbols"] = chunk_obj.code_symbols

                all_chunks.append(chunk_dict)

        return all_chunks

    @staticmethod
    def _infer_source_type(file_extension: str) -> SourceType:
        """Infer source type from file extension.

        Args:
            file_extension: File extension (without dot)

        Returns:
            SourceType enum value
        """
        code_extensions = {
            "py",
            "js",
            "ts",
            "jsx",
            "tsx",
            "java",
            "cpp",
            "c",
            "h",
            "hpp",
            "cs",
            "go",
            "rs",
            "swift",
            "kt",
            "scala",
            "rb",
            "php",
        }

        if file_extension in code_extensions:
            return SourceType.CODE
        else:
            return SourceType.DOCUMENT
