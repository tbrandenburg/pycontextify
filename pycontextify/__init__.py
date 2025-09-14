"""PyContextify - A Python-based MCP server for semantic search.

This package provides semantic search capabilities over codebases, documents,
and webpages using FAISS vector similarity search, various embedding providers,
and lightweight knowledge graph capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Main exports
from .index.manager import IndexManager
from .index.config import Config

__all__ = ["IndexManager", "Config", "__version__"]