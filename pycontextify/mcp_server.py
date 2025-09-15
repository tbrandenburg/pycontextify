"""FastMCP server for PyContextify.

This module implements the MCP server with simplified interface providing
6 essential functions for semantic search over codebases, documents, and webpages.
"""

import logging
import os
import signal
from pathlib import Path
from typing import Any, Dict, List

from fastmcp import FastMCP

from .index.config import Config
from .index.manager import IndexManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global manager instance
manager = None


def initialize_manager() -> IndexManager:
    """Initialize the global IndexManager."""
    global manager
    if manager is None:
        try:
            config = Config()
            manager = IndexManager(config)
            logger.info("IndexManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize IndexManager: {e}")
            raise
    return manager


# Create FastMCP application
mcp = FastMCP("PyContextify")


@mcp.tool
def index_code(path: str) -> Dict[str, Any]:
    """Index a codebase directory for semantic search.

    This function recursively scans a directory for code files and indexes them
    for semantic search. It supports various programming languages and automatically
    extracts relationships between code elements.

    Args:
        path: Path to the codebase directory to index

    Returns:
        Dictionary with indexing statistics including files processed and chunks added
    """
    try:
        # Validate path
        if not path or not isinstance(path, str):
            return {"error": "Path must be a non-empty string"}

        path_obj = Path(path).resolve()
        if not path_obj.exists():
            return {"error": f"Path does not exist: {path}"}

        if not path_obj.is_dir():
            return {"error": f"Path is not a directory: {path}"}

        # Initialize manager and index
        mgr = initialize_manager()
        result = mgr.index_codebase(str(path_obj))

        logger.info(f"Code indexing completed for {path}: {result}")
        return result

    except Exception as e:
        error_msg = f"Failed to index code at {path}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool
def index_document(path: str) -> Dict[str, Any]:
    """Index a single document (PDF, Markdown, or text file) for semantic search.

    This function indexes individual documents and extracts their structure
    and relationships for enhanced search capabilities.

    Args:
        path: Path to the document file to index

    Returns:
        Dictionary with indexing statistics including chunks added
    """
    try:
        # Validate path
        if not path or not isinstance(path, str):
            return {"error": "Path must be a non-empty string"}

        path_obj = Path(path).resolve()
        if not path_obj.exists():
            return {"error": f"File does not exist: {path}"}

        if not path_obj.is_file():
            return {"error": f"Path is not a file: {path}"}

        # Check supported extensions
        supported_extensions = {".pdf", ".md", ".txt"}
        if path_obj.suffix.lower() not in supported_extensions:
            return {
                "error": f"Unsupported file type. Supported: {', '.join(supported_extensions)}"
            }

        # Initialize manager and index
        mgr = initialize_manager()
        result = mgr.index_document(str(path_obj))

        logger.info(f"Document indexing completed for {path}: {result}")
        return result

    except Exception as e:
        error_msg = f"Failed to index document at {path}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool
def index_webpage(
    url: str, recursive: bool = False, max_depth: int = 1
) -> Dict[str, Any]:
    """Index web content for semantic search with optional recursive crawling.

    This function indexes web pages and can optionally follow links to index
    related pages. It extracts web-specific relationships and structure.

    Args:
        url: URL of the webpage to index
        recursive: Whether to follow links and index linked pages
        max_depth: Maximum depth for recursive crawling (ignored if recursive=False)

    Returns:
        Dictionary with indexing statistics including pages processed and chunks added
    """
    try:
        # Validate URL
        if not url or not isinstance(url, str):
            return {"error": "URL must be a non-empty string"}

        if not url.startswith(("http://", "https://")):
            return {"error": "URL must start with http:// or https://"}

        # Validate parameters
        if not isinstance(recursive, bool):
            return {"error": "recursive must be a boolean"}

        if not isinstance(max_depth, int) or max_depth < 1:
            return {"error": "max_depth must be a positive integer"}

        # Limit max_depth to prevent excessive crawling
        if max_depth > 3:
            max_depth = 3
            logger.warning("Limited max_depth to 3 for safety")

        # Initialize manager and index
        mgr = initialize_manager()
        result = mgr.index_webpage(url, recursive=recursive, max_depth=max_depth)

        logger.info(f"Webpage indexing completed for {url}: {result}")
        return result

    except Exception as e:
        error_msg = f"Failed to index webpage {url}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool
def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search across all indexed content.

    This function searches for content similar to the provided query across
    all indexed codebases, documents, and webpages using vector similarity.

    Args:
        query: Search query text
        top_k: Maximum number of results to return (default: 5)

    Returns:
        List of search results with similarity scores and source information
    """
    try:
        # Validate query
        if not query or not isinstance(query, str):
            return []

        query = query.strip()
        if not query:
            return []

        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1:
            top_k = 5

        # Limit top_k to reasonable range
        top_k = min(max(top_k, 1), 50)

        # Initialize manager and search
        mgr = initialize_manager()
        results = mgr.search(query, top_k)

        logger.info(f"Search completed for '{query}': {len(results)} results")
        return results

    except Exception as e:
        error_msg = f"Search failed for query '{query}': {str(e)}"
        logger.error(error_msg)
        return []


@mcp.tool
def search_with_context(
    query: str, top_k: int = 5, include_related: bool = False
) -> List[Dict[str, Any]]:
    """Perform enhanced semantic search with optional relationship context.

    This function provides enhanced search capabilities that can include related
    content based on the lightweight knowledge graph built from indexed content.
    This allows for more comprehensive search results that consider relationships
    between entities.

    Args:
        query: Search query text
        top_k: Maximum number of results to return (default: 5)
        include_related: Whether to include related chunks based on relationships

    Returns:
        List of enhanced search results with relationship context when enabled
    """
    try:
        # Validate query
        if not query or not isinstance(query, str):
            return []

        query = query.strip()
        if not query:
            return []

        # Validate parameters
        if not isinstance(top_k, int) or top_k < 1:
            top_k = 5

        if not isinstance(include_related, bool):
            include_related = False

        # Limit top_k to reasonable range
        top_k = min(max(top_k, 1), 50)

        # Initialize manager and search
        mgr = initialize_manager()
        results = mgr.search_with_context(query, top_k, include_related)

        logger.info(
            f"Context search completed for '{query}': {len(results)} results "
            f"(include_related={include_related})"
        )
        return results

    except Exception as e:
        error_msg = f"Context search failed for query '{query}': {str(e)}"
        logger.error(error_msg)
        return []


@mcp.tool
def status() -> Dict[str, Any]:
    """Get system status and comprehensive statistics.

    This function returns detailed information about the current state of the
    indexing system, including memory usage, indexed content statistics,
    embedding provider information, and persistence status.

    Returns:
        Dictionary with comprehensive system status and statistics
    """
    try:
        # Try to get status from manager
        if manager is not None:
            result = manager.get_status()
        else:
            # Try to initialize manager to get status
            try:
                mgr = initialize_manager()
                result = mgr.get_status()
            except Exception as init_error:
                return {
                    "status": "error",
                    "error": f"Failed to initialize system: {str(init_error)}",
                    "manager_initialized": False,
                }

        # Add system-level information
        result["manager_initialized"] = manager is not None
        result["mcp_server"] = {
            "name": "PyContextify",
            "version": "0.1.0",
            "mcp_functions": [
                "index_code",
                "index_document",
                "index_webpage",
                "search",
                "search_with_context",
                "status",
            ],
            "interface": "simplified",  # 6-function interface
        }

        logger.info("Status request completed successfully")
        return result

    except Exception as e:
        error_msg = f"Failed to get system status: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "manager_initialized": manager is not None,
        }


def cleanup_handler(signum, frame):
    """Handle graceful shutdown signals."""
    logger.info("Received shutdown signal, cleaning up...")
    if manager:
        try:
            # Save index before shutdown if auto-persist is enabled
            manager.save_index()
        except Exception as e:
            logger.warning(f"Failed to save index during shutdown: {e}")

        # Cleanup embedder
        try:
            manager.__exit__(None, None, None)
        except Exception as e:
            logger.warning(f"Failed to cleanup manager: {e}")

    logger.info("Cleanup completed")
    os._exit(0)


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)


def main():
    """Main entry point for the MCP server."""
    try:
        logger.info("Starting PyContextify MCP Server...")
        logger.info("Server provides 6 simplified MCP functions:")
        logger.info("  - index_code(path): Index codebase directory")
        logger.info("  - index_document(path): Index document")
        logger.info("  - index_webpage(url, recursive, max_depth): Index web content")
        logger.info("  - search(query, top_k): Basic semantic search")
        logger.info(
            "  - search_with_context(query, top_k, include_related): Enhanced search with relationships"
        )
        logger.info("  - status(): Get system status and statistics")

        # Initialize manager early to catch configuration issues
        initialize_manager()

        # Run the MCP server
        mcp.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        cleanup_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise


if __name__ == "__main__":
    main()
