"""PyContextify - a relationship-aware semantic search MCP server."""

from importlib import metadata as _metadata

from .index.config import Config
from .index.manager import IndexManager


try:  # pragma: no cover - exercised when installed as a package
    __version__ = _metadata.version("pycontextify")
except _metadata.PackageNotFoundError:  # pragma: no cover - local editable installs
    __version__ = "0.1.0"

__author__ = "Tom Brandenburg"
__email__ = "kabelkaspertom@googlemail.com"
__license__ = "MIT"

__all__ = ["IndexManager", "Config", "__version__", "__author__", "__email__"]
