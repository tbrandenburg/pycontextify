"""Text chunking system for PyContextify.

This module implements various chunking strategies for different content types,
with relationship extraction capabilities for building lightweight knowledge graphs.
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
        **kwargs: Any,
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

    def _split_by_tokens(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[Tuple[str, int, int]]:
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
    """Code-aware chunker that respects code structure and extracts relationships."""

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

    def _split_by_code_boundaries(self, text: str) -> List[Tuple[str, int, int]]:
        """Split code by function/class boundaries when possible.

        Args:
            text: Code text to split

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        # Simple approach: split by double newlines and function/class definitions
        lines = text.split("\n")
        chunks = []
        current_chunk_lines: List[str] = []
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


class LanguageAwareChunker(SimpleChunker):
    """Universal code-aware chunker using LangChain's language-specific splitter.

    Supports all LangChain-supported languages with proper syntax awareness,
    keeping decorators/annotations with their methods and preserving code structure.
    """

    def __init__(self, config: Config, language: str):
        """Initialize language-aware chunker.

        Args:
            config: Configuration object
            language: Language identifier (e.g., 'python', 'java', 'js', 'ts')
        """
        super().__init__(config)
        self.language = language

    def chunk_text(
        self, text: str, source_path: str, embedding_provider: str, embedding_model: str
    ) -> List[Chunk]:
        """Chunk code text with awareness of language-specific syntax.

        Uses LangChain's language-specific RecursiveCharacterTextSplitter that
        understands the target language syntax and preserves code structure.

        Args:
            text: Code text to chunk
            source_path: Source file path
            embedding_provider: Embedding provider name
            embedding_model: Embedding model name

        Returns:
            List of Chunk objects with preserved code structure
        """
        if not text.strip():
            return []

        # Convert token-based chunk size to character estimate
        # Different languages have different verbosity
        char_multiplier = {
            "python": 5,  # Python has longer tokens due to indentation
            "java": 4,  # Java is verbose
            "js": 4,  # JavaScript moderate verbosity
            "ts": 4,  # TypeScript similar to JavaScript
            "cpp": 4,  # C++ moderate verbosity
            "go": 4,  # Go moderate verbosity
            "rust": 4,  # Rust moderate verbosity
            "scala": 4,  # Scala moderate verbosity
        }.get(
            self.language, 4
        )  # Default multiplier

        char_chunk_size = self.chunk_size * char_multiplier
        char_overlap = self.chunk_overlap * char_multiplier

        # Use language-specific LangChain splitter
        chunks_tuples = self._split_with_langchain_language(
            text, char_chunk_size, char_overlap
        )

        # Create metadata objects
        chunks = []
        for chunk_text, start_char, end_char in chunks_tuples:
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

    def _split_with_langchain_language(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[Tuple[str, int, int]]:
        """Split code using LangChain's language-specific splitter.

        Args:
            text: Code to split
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap size in characters

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        try:
            from langchain_text_splitters import (
                Language,
                RecursiveCharacterTextSplitter,
            )
        except ImportError as e:
            logger.warning(
                "LangChain text splitters not available (%s). Using fallback.",
                e,
            )
            return self._split_by_tokens(text, self.chunk_size, self.chunk_overlap)

        # Map file extensions to LangChain Language enum values
        language_map = {
            "python": Language.PYTHON,
            "py": Language.PYTHON,
            "java": Language.JAVA,
            "javascript": Language.JS,
            "js": Language.JS,
            "jsx": Language.JS,
            "typescript": Language.TS,
            "ts": Language.TS,
            "tsx": Language.TS,
            "cpp": Language.CPP,
            "c": Language.C,
            "go": Language.GO,
            "rust": Language.RUST,
            "scala": Language.SCALA,
            "ruby": Language.RUBY,
            "php": Language.PHP,
            "swift": Language.SWIFT,
            "kotlin": Language.KOTLIN,
            "csharp": Language.CSHARP,
            "cs": Language.CSHARP,
        }

        langchain_language = language_map.get(self.language)
        if not langchain_language:
            logger.warning(
                "Language '%s' not supported by LangChain. Using fallback.",
                self.language,
            )
            return self._split_by_tokens(text, self.chunk_size, self.chunk_overlap)

        try:
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=langchain_language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )

            langchain_chunks = text_splitter.split_text(text)

            chunks = []
            current_pos = 0

            for chunk_text in langchain_chunks:
                chunk_start = text.find(chunk_text, current_pos)
                if chunk_start == -1:
                    chunk_start = current_pos

                chunk_end = chunk_start + len(chunk_text)
                chunks.append((chunk_text, chunk_start, chunk_end))
                current_pos = chunk_start + len(chunk_text)

            return chunks

        except Exception as exc:
            logger.error(
                "LangChain %s splitter failed: %s. Using fallback.",
                self.language,
                exc,
            )
            return self._split_by_tokens(text, self.chunk_size, self.chunk_overlap)

    def _extract_relationships(self, chunk: Chunk) -> None:
        """Extract language-specific relationships based on language.

        Args:
            chunk: Chunk object to populate with relationships
        """
        text = chunk.chunk_text

        # Language-specific relationship extraction
        if self.language in ["python", "py"]:
            self._extract_python_relationships(chunk, text)
        elif self.language == "java":
            self._extract_java_relationships(chunk, text)
        elif self.language in ["js", "jsx", "ts", "tsx", "javascript", "typescript"]:
            self._extract_javascript_relationships(chunk, text)
        else:
            # Generic code relationship extraction
            self._extract_generic_code_relationships(chunk, text)

    def _extract_python_relationships(self, chunk: Chunk, text: str) -> None:
        """Extract Python-specific relationships."""
        # Function definitions (including methods with decorators)
        func_pattern = (
            r"(?:@\w+\s*\n\s*)*(?:def|async\s+def)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        )
        for match in re.finditer(func_pattern, text, re.MULTILINE):
            func_name = match.group(1)
            chunk.add_code_symbol(func_name)
            chunk.add_reference(func_name)

        # Class definitions
        class_pattern = r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        for match in re.finditer(class_pattern, text):
            class_name = match.group(1)
            chunk.add_code_symbol(class_name)
            chunk.add_reference(class_name)

        # Python imports
        import_patterns = [
            r"import\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
            r"from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import",
        ]
        for pattern in import_patterns:
            for match in re.finditer(pattern, text):
                module_name = match.group(1)
                chunk.add_reference(module_name)
                chunk.add_tag("import")

        # Decorators
        decorator_pattern = r"@([a-zA-Z_][a-zA-Z0-9_.]*)"
        for match in re.finditer(decorator_pattern, text):
            decorator_name = match.group(1)
            chunk.add_reference(decorator_name)
            chunk.add_tag("decorator")

    def _extract_java_relationships(self, chunk: Chunk, text: str) -> None:
        """Extract Java-specific relationships."""
        # Method definitions (including those with annotations)
        method_pattern = r"(?:@\w+\s*\n\s*)*(?:public|private|protected|static)?\s*(?:\w+\s+)*(?:\w+\s+)*([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        for match in re.finditer(method_pattern, text, re.MULTILINE):
            method_name = match.group(1)
            if method_name not in ["if", "for", "while", "switch", "catch"]:
                chunk.add_code_symbol(method_name)
                chunk.add_reference(method_name)

        # Class and interface definitions
        for pattern, tag in [
            (
                r"(?:public|private|protected)?\s*(?:abstract\s+)?(?:final\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                "class",
            ),
            (
                r"(?:public|private|protected)?\s*interface\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                "interface",
            ),
        ]:
            for match in re.finditer(pattern, text):
                name = match.group(1)
                chunk.add_code_symbol(name)
                chunk.add_reference(name)
                chunk.add_tag(tag)

        # Imports and annotations
        import_pattern = r"import\s+(?:static\s+)?([a-zA-Z_][a-zA-Z0-9_.]*)"
        for match in re.finditer(import_pattern, text):
            import_name = match.group(1)
            chunk.add_reference(import_name)
            chunk.add_tag("import")

        annotation_pattern = r"@([a-zA-Z_][a-zA-Z0-9_.]*)"
        for match in re.finditer(annotation_pattern, text):
            annotation_name = match.group(1)
            chunk.add_reference(annotation_name)
            chunk.add_tag("annotation")

    def _extract_javascript_relationships(self, chunk: Chunk, text: str) -> None:
        """Extract JavaScript/TypeScript-specific relationships."""
        # Function definitions (various patterns)
        function_patterns = [
            r"(?:async\s+)?function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(",
            r"(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s+)?(?:function|\()",
            r"([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:\s*(?:async\s+)?(?:function|\()",
            r"([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(.*?\)\s*=>",
        ]
        for pattern in function_patterns:
            for match in re.finditer(pattern, text):
                func_name = match.group(1)
                chunk.add_code_symbol(func_name)
                chunk.add_reference(func_name)

        # Class definitions
        class_pattern = r"class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)"
        for match in re.finditer(class_pattern, text):
            class_name = match.group(1)
            chunk.add_code_symbol(class_name)
            chunk.add_reference(class_name)

        # Imports/exports
        import_patterns = [
            r"import\s+.*?from\s+['\"]([^'\"]+)['\"]",
            r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
            r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
        ]
        for pattern in import_patterns:
            for match in re.finditer(pattern, text):
                module_name = match.group(1)
                chunk.add_reference(module_name)
                chunk.add_tag("import")

        # TypeScript-specific patterns
        if self.language in ["ts", "tsx", "typescript"]:
            # Interfaces and types
            for pattern, tag in [
                (r"interface\s+([a-zA-Z_$][a-zA-Z0-9_$]*)", "interface"),
                (r"type\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=", "type"),
            ]:
                for match in re.finditer(pattern, text):
                    name = match.group(1)
                    chunk.add_code_symbol(name)
                    chunk.add_reference(name)
                    chunk.add_tag(tag)

    def _extract_generic_code_relationships(self, chunk: Chunk, text: str) -> None:
        """Extract generic code relationships for unsupported languages."""
        # Basic function/method patterns
        func_patterns = [
            r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # Python, Ruby style
            r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # JavaScript style
            r"fn\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # Rust style
        ]
        for pattern in func_patterns:
            for match in re.finditer(pattern, text):
                func_name = match.group(1)
                chunk.add_code_symbol(func_name)
                chunk.add_reference(func_name)

        # Basic class patterns
        class_patterns = [
            r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # Most languages
            r"struct\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # C/C++/Rust/Go style
        ]
        for pattern in class_patterns:
            for match in re.finditer(pattern, text):
                class_name = match.group(1)
                chunk.add_code_symbol(class_name)
                chunk.add_reference(class_name)


class DocumentChunker(SimpleChunker):
    """Document-aware chunker using LangChain's RecursiveCharacterTextSplitter."""

    def chunk_text(
        self, text: str, source_path: str, embedding_provider: str, embedding_model: str
    ) -> List[Chunk]:
        """Chunk document text with awareness of document structure.

        Uses LangChain's RecursiveCharacterTextSplitter for intelligent boundaries,
        preventing mid-sentence breaks and preserving semantic structure.

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

        # First try to split by document structure (headers)
        doc_chunks = self._split_by_document_structure(text)

        # Process chunks with LangChain RecursiveCharacterTextSplitter
        final_chunks = []
        for chunk_text, start_char, end_char, section in doc_chunks:
            # Convert token-based chunk size to character estimate
            # Average token ≈ 4 characters for English text
            char_chunk_size = self.chunk_size * 4
            char_overlap = self.chunk_overlap * 4

            if len(chunk_text) <= char_chunk_size:
                # Chunk is small enough, use as-is
                final_chunks.append((chunk_text, start_char, end_char, section))
            else:
                # Use LangChain RecursiveCharacterTextSplitter for large chunks
                sub_chunks = self._split_with_langchain(
                    chunk_text, char_chunk_size, char_overlap, start_char
                )
                for sub_text, sub_start, sub_end in sub_chunks:
                    final_chunks.append((sub_text, sub_start, sub_end, section))

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

    def _split_with_langchain(
        self, text: str, chunk_size: int, chunk_overlap: int, base_start_char: int
    ) -> List[Tuple[str, int, int]]:
        """Split text using LangChain's RecursiveCharacterTextSplitter.

        Args:
            text: Text to split
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap size in characters
            base_start_char: Base character offset for position calculation

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as e:
            logger.warning(
                "LangChain text splitters not available (%s). Using fallback.",
                e,
            )
            return self._split_by_tokens(
                text, self.chunk_size, self.chunk_overlap, base_start_char
            )

        # Initialize RecursiveCharacterTextSplitter with document-friendly separators
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraphs (highest priority)
                "\n",  # Lines
                ". ",  # Sentences
                "? ",  # Questions
                "! ",  # Exclamations
                "; ",  # Semicolons
                ", ",  # Commas
                " ",  # Spaces
                "",  # Characters (fallback)
            ],
            is_separator_regex=False,
            add_start_index=True,
        )

        try:
            # Split text into chunks
            langchain_chunks = text_splitter.split_text(text)

            # Convert to our format with proper character positions
            chunks = []
            current_pos = 0

            for chunk_text in langchain_chunks:
                # Find the actual position of this chunk in the original text
                chunk_start = text.find(chunk_text, current_pos)
                if chunk_start == -1:
                    # Fallback: use sequential position
                    chunk_start = current_pos

                chunk_end = chunk_start + len(chunk_text)

                # Adjust positions relative to the document
                absolute_start = base_start_char + chunk_start
                absolute_end = base_start_char + chunk_end

                chunks.append((chunk_text, absolute_start, absolute_end))
                current_pos = chunk_start + len(chunk_text)

            return chunks

        except Exception as exc:
            logger.error(
                "LangChain RecursiveCharacterTextSplitter failed: %s. Using fallback.",
                exc,
            )
            return self._split_by_tokens(
                text, self.chunk_size, self.chunk_overlap, base_start_char
            )

    def _split_by_tokens(
        self, text: str, chunk_size: int, overlap: int, base_start_char: int = 0
    ) -> List[Tuple[str, int, int]]:
        """Fallback method: Split text by approximate token count.

        Args:
            text: Text to split
            chunk_size: Target chunk size in tokens
            overlap: Overlap size in tokens
            base_start_char: Base character offset for position calculation

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
                start_char = base_start_char
            else:
                # Find start position by joining previous words
                start_char = (
                    base_start_char
                    + len(" ".join(words[:start_idx]))
                    + (1 if start_idx > 0 else 0)
                )

            end_char = base_start_char + len(" ".join(words[:end_idx]))

            chunk_text = " ".join(chunk_words)
            chunks.append((chunk_text, start_char, end_char))

            # Move start position for next chunk (with overlap)
            if end_idx >= len(words):
                break

            start_idx = max(end_idx - words_overlap, start_idx + 1)

        return chunks

    def _split_by_document_structure(
        self, text: str
    ) -> List[Tuple[str, int, int, Optional[str]]]:
        """Split document by headers and sections.

        Args:
            text: Document text to split

        Returns:
            List of (chunk_text, start_char, end_char, section_name) tuples
        """
        lines = text.split("\n")
        chunks: List[Tuple[str, int, int, Optional[str]]] = []
        current_chunk_lines: List[str] = []
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

        # If no structure found, treat entire text as one chunk
        if len(chunks) <= 1:
            return [(text, 0, len(text), None)]

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
    def get_chunker(
        source_type: SourceType, config: Config, file_extension: str = ""
    ) -> BaseChunker:
        """Get appropriate chunker for content type.

        Args:
            source_type: Type of content to chunk
            config: Configuration object
            file_extension: File extension to select language-specific chunker

        Returns:
            Appropriate chunker instance
        """
        if source_type == SourceType.CODE:
            # Use language-aware chunker for supported languages
            supported_languages = {
                "py": "python",
                "java": "java",
                "js": "js",
                "jsx": "js",
                "ts": "ts",
                "tsx": "ts",
                "cpp": "cpp",
                "c": "c",
                "go": "go",
                "rs": "rust",
                "rb": "ruby",
                "php": "php",
                "swift": "swift",
                "kt": "kotlin",
                "scala": "scala",
                "cs": "csharp",
            }

            if file_extension in supported_languages:
                language = supported_languages[file_extension]
                return LanguageAwareChunker(config, language)
            else:
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

            # Get appropriate chunker (pass file extension for language-specific selection)
            chunker = ChunkerFactory.get_chunker(source_type, config, file_ext)

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
