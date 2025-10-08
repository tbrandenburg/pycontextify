"""Compatibility shims for loader classes.

The indexing system now exposes dedicated modules (:mod:`index_codebase`,
:mod:`index_document`, :mod:`index_webpage`). This module remains to keep
backwards compatibility for external imports while delegating to the new
implementations.
"""

from __future__ import annotations

import warnings
from typing import Any

from .index_codebase import CodeLoader
from .index_document import DocumentLoader
from .index_webpage import WebpageLoader
from .storage_metadata import SourceType

__all__ = [
    "CodeLoader",
    "DocumentLoader",
    "WebpageLoader",
    "LoaderFactory",
]


class LoaderFactory:
    """Deprecated facade that forwards to the dedicated loader classes."""

    @staticmethod
    def get_loader(source_type: SourceType, **kwargs: Any):
        warnings.warn(
            "LoaderFactory is deprecated; import the dedicated loader from "
            "pycontextify.index_codebase/index_document/index_webpage instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if source_type == SourceType.CODE:
            return CodeLoader(**kwargs)
        if source_type == SourceType.DOCUMENT:
            return DocumentLoader(**kwargs)
        if source_type == SourceType.WEBPAGE:
            return WebpageLoader(**kwargs)
        raise ValueError(f"Unsupported source type: {source_type}")
