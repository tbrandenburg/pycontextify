"""Common type definitions for PyContextify.

This module contains shared enums and type definitions used across
the indexing and storage layers, ensuring proper separation of concerns.
"""

from enum import Enum


class SourceType(Enum):
    """Types of content sources for chunks."""

    CODE = "code"
    DOCUMENT = "document"
    WEBPAGE = "webpage"
