"""Common type definitions for PyContextify.

This module contains shared enums and type definitions used across
the indexing and storage layers, ensuring proper separation of concerns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SourceType(Enum):
    """Types of content sources for chunks."""

    CODE = "code"
    DOCUMENT = "document"


@dataclass
class Chunk:
    """Lightweight chunk data structure for indexing pipeline.

    This is a simple DTO (Data Transfer Object) used by chunkers to represent
    text chunks without coupling to storage implementation details.
    """

    chunk_text: str
    source_path: str
    source_type: SourceType
    start_char: int = 0
    end_char: int = 0
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-mpnet-base-v2"

    # Optional metadata fields
    file_extension: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    parent_section: Optional[str] = None
    code_symbols: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)

    def add_reference(self, reference: str) -> None:
        """Add a reference if not already present."""
        if reference not in self.references:
            self.references.append(reference)

    def add_code_symbol(self, symbol: str) -> None:
        """Add a code symbol if not already present."""
        if symbol not in self.code_symbols:
            self.code_symbols.append(symbol)
