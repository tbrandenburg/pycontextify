"""FastMCP server for PyContextify.

This module implements the MCP server with simplified interface providing
6 essential functions for semantic search over codebases, documents, and webpages.
"""

import argparse
import logging
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from .index.config import Config
from .index.manager import IndexManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global manager instance
manager = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the MCP server."""
    parser = argparse.ArgumentParser(
        prog="pycontextify",
        description="PyContextify MCP Server - Semantic search over codebases, documents, and webpages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Start server with custom index path
  pycontextify --index-path ./my_index
  
  # Start with initial documents
  pycontextify --initial-documents ./docs/readme.md ./docs/api.md
  
  # Start with initial codebase and custom index
  pycontextify --index-path ./project_index --initial-codebase ./src
  
  # Start with initial webpages
  pycontextify --initial-webpages https://docs.python.org https://github.com/user/repo
  
  # Recursive webpage crawling with depth limit
  pycontextify --initial-webpages https://docs.example.com \
               --recursive-crawling --max-crawl-depth 2 --crawl-delay 2
  
  # Full example with all content types
  pycontextify --index-path ./my_index --index-name project_search \
               --initial-documents ./README.md --initial-codebase ./src ./tests \
               --initial-webpages https://api-docs.com --recursive-crawling

Configuration priority: CLI arguments > Environment variables > Defaults

Environment variables can still be used for all settings. Use --help for details.
        """,
    )

    # Index configuration
    parser.add_argument(
        "--index-path",
        type=str,
        help="Directory path for vector storage and index files (overrides PYCONTEXTIFY_INDEX_DIR)",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        help="Custom index name (overrides PYCONTEXTIFY_INDEX_NAME, default: semantic_index)",
    )

    # Initial indexing
    parser.add_argument(
        "--initial-documents",
        nargs="*",
        type=str,
        help="File paths to documents to index at startup (PDF, Markdown, Text files)",
    )
    parser.add_argument(
        "--initial-codebase",
        nargs="*",
        type=str,
        help="Directory paths to codebases to index at startup",
    )
    parser.add_argument(
        "--initial-webpages",
        nargs="*",
        type=str,
        help="URLs to webpages to index at startup (http/https only)",
    )

    # Webpage crawling options
    parser.add_argument(
        "--recursive-crawling",
        action="store_true",
        help="Enable recursive crawling for initial webpages",
    )
    parser.add_argument(
        "--max-crawl-depth",
        type=int,
        default=1,
        help="Maximum crawl depth for recursive crawling (1-3, default: 1)",
    )
    parser.add_argument(
        "--crawl-delay",
        type=int,
        help="Delay between web requests in seconds (overrides PYCONTEXTIFY_CRAWL_DELAY_SECONDS)",
    )

    # Server configuration
    parser.add_argument(
        "--no-auto-persist",
        action="store_true",
        help="Disable automatic index persistence (overrides PYCONTEXTIFY_AUTO_PERSIST)",
    )
    parser.add_argument(
        "--no-auto-load",
        action="store_true",
        help="Disable automatic index loading on startup (overrides PYCONTEXTIFY_AUTO_LOAD)",
    )

    # Embedding configuration
    parser.add_argument(
        "--embedding-provider",
        choices=["sentence_transformers", "ollama", "openai"],
        help="Embedding provider to use (overrides PYCONTEXTIFY_EMBEDDING_PROVIDER)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model name (overrides PYCONTEXTIFY_EMBEDDING_MODEL)",
    )

    # Verbose logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize logging output (WARNING level only)",
    )

    return parser.parse_args()


def initialize_manager(config_overrides: Optional[Dict[str, Any]] = None) -> IndexManager:
    """Initialize the global IndexManager with optional configuration overrides.
    
    Args:
        config_overrides: Optional dictionary of configuration overrides from CLI args
    """
    global manager
    if manager is None:
        try:
            config = Config(config_overrides=config_overrides)
            manager = IndexManager(config)
            logger.info("IndexManager initialized successfully")
            if config_overrides:
                logger.info(f"Applied CLI configuration overrides: {list(config_overrides.keys())}")
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


def perform_initial_indexing(args: argparse.Namespace, mgr: IndexManager) -> None:
    """Perform initial document and codebase indexing if specified in CLI args.
    
    Args:
        args: Parsed command-line arguments
        mgr: Initialized IndexManager instance
    """
    total_indexed = 0
    
    # Index initial documents
    if args.initial_documents:
        logger.info(f"Indexing {len(args.initial_documents)} initial documents...")
        for doc_path in args.initial_documents:
            try:
                path_obj = Path(doc_path).resolve()
                if not path_obj.exists():
                    logger.warning(f"Document not found, skipping: {doc_path}")
                    continue
                    
                if not path_obj.is_file():
                    logger.warning(f"Path is not a file, skipping: {doc_path}")
                    continue
                    
                # Check supported extensions
                supported_extensions = {".pdf", ".md", ".txt"}
                if path_obj.suffix.lower() not in supported_extensions:
                    logger.warning(
                        f"Unsupported file type {path_obj.suffix}, skipping: {doc_path}"
                    )
                    continue
                    
                result = mgr.index_document(str(path_obj))
                if "error" not in result:
                    chunks_added = result.get("chunks_added", 0)
                    total_indexed += chunks_added
                    logger.info(f"Successfully indexed document {doc_path}: {chunks_added} chunks")
                else:
                    logger.error(f"Failed to index document {doc_path}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error indexing document {doc_path}: {e}")
    
    # Index initial codebases
    if args.initial_codebase:
        logger.info(f"Indexing {len(args.initial_codebase)} initial codebases...")
        for codebase_path in args.initial_codebase:
            try:
                path_obj = Path(codebase_path).resolve()
                if not path_obj.exists():
                    logger.warning(f"Codebase directory not found, skipping: {codebase_path}")
                    continue
                    
                if not path_obj.is_dir():
                    logger.warning(f"Path is not a directory, skipping: {codebase_path}")
                    continue
                    
                result = mgr.index_codebase(str(path_obj))
                if "error" not in result:
                    files_processed = result.get("files_processed", 0)
                    chunks_added = result.get("chunks_added", 0)
                    total_indexed += chunks_added
                    logger.info(
                        f"Successfully indexed codebase {codebase_path}: {files_processed} files, {chunks_added} chunks"
                    )
                else:
                    logger.error(f"Failed to index codebase {codebase_path}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error indexing codebase {codebase_path}: {e}")
    
    # Index initial webpages
    if args.initial_webpages:
        logger.info(f"Indexing {len(args.initial_webpages)} initial webpages...")
        for webpage_url in args.initial_webpages:
            try:
                # Validate URL
                if not webpage_url.startswith(("http://", "https://")):
                    logger.warning(f"Invalid URL (must start with http:// or https://), skipping: {webpage_url}")
                    continue
                
                # Apply crawling settings
                recursive = args.recursive_crawling if hasattr(args, 'recursive_crawling') else False
                max_depth = getattr(args, 'max_crawl_depth', 1)
                
                # Validate and limit max_depth
                if max_depth < 1 or max_depth > 3:
                    max_depth = min(max(max_depth, 1), 3)
                    logger.warning(f"Adjusted max_crawl_depth to {max_depth} (valid range: 1-3)")
                
                result = mgr.index_webpage(webpage_url, recursive=recursive, max_depth=max_depth)
                if "error" not in result:
                    pages_processed = result.get("pages_processed", 1)
                    chunks_added = result.get("chunks_added", 0)
                    total_indexed += chunks_added
                    logger.info(
                        f"Successfully indexed webpage {webpage_url}: {pages_processed} pages, {chunks_added} chunks "
                        f"(recursive={recursive}, max_depth={max_depth})"
                    )
                else:
                    logger.error(f"Failed to index webpage {webpage_url}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error indexing webpage {webpage_url}: {e}")
    
    if total_indexed > 0:
        logger.info(f"Initial indexing completed: {total_indexed} total chunks indexed")
        
        # Save the index if auto-persist is enabled
        try:
            if mgr.config.auto_persist:
                mgr.save_index()
                logger.info("Initial index saved to disk")
        except Exception as e:
            logger.warning(f"Failed to save initial index: {e}")
    else:
        logger.info("No initial indexing performed")


def args_to_config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert CLI arguments to configuration overrides dictionary.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary of configuration overrides
    """
    overrides = {}
    
    # Index configuration
    if args.index_path:
        overrides["index_dir"] = args.index_path
    if args.index_name:
        overrides["index_name"] = args.index_name
        
    # Server configuration
    if args.no_auto_persist:
        overrides["auto_persist"] = False
    if args.no_auto_load:
        overrides["auto_load"] = False
        
    # Embedding configuration
    if args.embedding_provider:
        overrides["embedding_provider"] = args.embedding_provider
    if args.embedding_model:
        overrides["embedding_model"] = args.embedding_model
        
    # Webpage crawling configuration
    if hasattr(args, 'crawl_delay') and args.crawl_delay is not None:
        overrides["crawl_delay_seconds"] = args.crawl_delay
        
    return overrides


def setup_logging(args: argparse.Namespace) -> None:
    """Setup logging level based on CLI arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
        
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True  # Reconfigure existing loggers
    )


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
        # Parse command-line arguments
        args = parse_args()
        
        # Setup logging based on CLI args
        setup_logging(args)
        
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
        
        # Convert CLI args to config overrides
        config_overrides = args_to_config_overrides(args)
        if config_overrides:
            logger.info(f"Using CLI configuration overrides: {list(config_overrides.keys())}")

        # Initialize manager with CLI overrides
        mgr = initialize_manager(config_overrides)
        
        # Perform initial indexing if specified
        if args.initial_documents or args.initial_codebase:
            logger.info("Performing initial indexing...")
            perform_initial_indexing(args, mgr)
            logger.info("Initial indexing completed, starting MCP server")
        else:
            logger.info("No initial indexing requested")

        # Run the MCP server
        logger.info("MCP server ready and listening for requests...")
        mcp.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        cleanup_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise


if __name__ == "__main__":
    main()
