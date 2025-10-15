"""PyContextify - a relationship-aware semantic search MCP server."""

import warnings
import os
from importlib import metadata as _metadata

# Suppress SWIG-related deprecation warnings from FAISS
# These warnings come from SWIG-generated bindings and are harmless
# but can clutter output. Set PYCONTEXTIFY_SHOW_SWIG_WARNINGS=1 to see them.
if not os.getenv("PYCONTEXTIFY_SHOW_SWIG_WARNINGS"):
    warnings.filterwarnings(
        "ignore",
        message="builtin type .* has no __module__ attribute",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*SwigPy.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*swigvarlink.*",
        category=DeprecationWarning,
    )

from . import mcp as mcp
from .config import Config
from .indexer import IndexManager
from .types import SourceType

try:  # pragma: no cover - exercised when installed as a package
    __version__ = _metadata.version("pycontextify")
except _metadata.PackageNotFoundError:  # pragma: no cover - local editable installs
    __version__ = "0.1.0"

__author__ = "Tom Brandenburg"
__email__ = "kabelkaspertom@googlemail.com"
__license__ = "MIT"

__all__ = [
    "IndexManager",
    "Config",
    "SourceType",
    "mcp",
    "__version__",
    "__author__",
    "__email__",
]
