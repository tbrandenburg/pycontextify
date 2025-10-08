"""FastMCP server for PyContextify.

This module implements the MCP server with simplified interface providing
6 essential functions for semantic search over codebases, documents, and webpages.
"""

import argparse
import logging
import signal
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from .config import Config
from .indexer import IndexManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe singleton manager
_manager_lock = Lock()
_manager_instance: Optional[IndexManager] = None
_manager_config_overrides: Optional[Dict[str, Any]] = None


# Common MCP validation and error handling utilities
def validate_string_param(
    value: Any, param_name: str, allow_empty: bool = False
) -> str:
    """Validate string parameter for MCP functions.

    Args:
        value: Value to validate
        param_name: Name of the parameter for error messages
        allow_empty: Whether to allow empty strings

    Returns:
        Validated string value

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, str):
        raise ValueError(f"{param_name} must be a string")

    if not allow_empty and not value.strip():
        raise ValueError(f"{param_name} cannot be empty")

    return value.strip() if not allow_empty else value


def validate_int_param(
    value: Any, param_name: str, min_val: int = None, max_val: int = None
) -> int:
    """Validate integer parameter for MCP functions.

    Args:
        value: Value to validate
        param_name: Name of the parameter for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Validated integer value

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, int):
        raise ValueError(f"{param_name} must be an integer")

    if min_val is not None and value < min_val:
        raise ValueError(f"{param_name} must be at least {min_val}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{param_name} must be at most {max_val}")

    return value


def validate_bool_param(value: Any, param_name: str) -> bool:
    """Validate boolean parameter for MCP functions.

    Args:
        value: Value to validate
        param_name: Name of the parameter for error messages

    Returns:
        Validated boolean value

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, bool):
        raise ValueError(f"{param_name} must be a boolean")
    return value


def validate_choice_param(
    value: Any, param_name: str, valid_choices: List[str], default: str = None
) -> str:
    """Validate choice parameter for MCP functions.

    Args:
        value: Value to validate
        param_name: Name of the parameter for error messages
        valid_choices: List of valid choices
        default: Default value if invalid (optional)

    Returns:
        Validated choice value

    Raises:
        ValueError: If validation fails and no default provided
    """
    if not isinstance(value, str) or value not in valid_choices:
        if default is not None:
            logger.warning(f"Invalid {param_name} '{value}', using default '{default}'")
            return default
        raise ValueError(f"{param_name} must be one of: {', '.join(valid_choices)}")
    return value


def handle_mcp_errors(operation_name: str, func_impl, *args):
    """Common error handling for MCP functions.

    Args:
        operation_name: Name of the operation for error messages
        func_impl: The actual implementation function to call
        *args: Arguments to pass to the function

    Returns:
        Result from function or structured error dict
    """
    try:
        return func_impl(*args)
    except ValueError as e:
        # Validation errors - return structured error
        error_msg = str(e)
        logger.warning(f"{operation_name} validation error: {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        # Unexpected errors - log and return structured error
        error_msg = f"{operation_name} failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


async def handle_mcp_errors_async(operation_name: str, func_impl, *args):
    """Common error handling for async MCP functions.

    Args:
        operation_name: Name of the operation for error messages
        func_impl: The actual async implementation function to call
        *args: Arguments to pass to the function

    Returns:
        Result from function or structured error dict
    """
    try:
        return await func_impl(*args)
    except ValueError as e:
        # Validation errors - return structured error
        error_msg = str(e)
        logger.warning(f"{operation_name} validation error: {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        # Unexpected errors - log and return structured error
        error_msg = f"{operation_name} failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the MCP server."""
    parser = argparse.ArgumentParser(
        prog="pycontextify",
        description=(
            "PyContextify MCP Server - Semantic search over codebases, "
            "documents, and webpages"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Start server with custom index path
  pycontextify --index-path ./my_index

  # Start with initial documents
  pycontextify --initial-documents ./docs/readme.md ./docs/api.md

  # Start with initial codebase and custom index
  pycontextify --index-path ./project_index --initial-codebase ./src

  # Start with initial webpages
  pycontextify --initial-webpages https://docs.python.org

  # Recursive webpage crawling with depth limit
  pycontextify --initial-webpages https://docs.example.com \
               --recursive-crawling --max-crawl-depth 2 --crawl-delay 2

  # Full example with all content types
  pycontextify --index-path ./my_index --index-name project_search \
               --initial-documents ./README.md --initial-codebase ./src \
               --initial-webpages https://api-docs.com --recursive-crawling

Configuration priority: CLI arguments > Environment variables > Defaults

Environment variables can still be used for all settings. Use --help for details.
        """,
    )

    # Index configuration
    parser.add_argument(
        "--index-path",
        type=str,
        help=(
            "Directory path for vector storage and index files "
            "(overrides PYCONTEXTIFY_INDEX_DIR)"
        ),
    )
    parser.add_argument(
        "--index-name",
        type=str,
        help=(
            "Custom index name "
            "(overrides PYCONTEXTIFY_INDEX_NAME, default: semantic_index)"
        ),
    )
    parser.add_argument(
        "--index-bootstrap-archive-url",
        type=str,
        help=(
            "Optional HTTPS or file URL to an index bootstrap archive "
            "(overrides PYCONTEXTIFY_INDEX_BOOTSTRAP_ARCHIVE_URL)"
        ),
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
        help=(
            "Delay between web requests in seconds "
            "(overrides PYCONTEXTIFY_CRAWL_DELAY_SECONDS)"
        ),
    )

    # Server configuration
    parser.add_argument(
        "--no-auto-persist",
        action="store_true",
        help=(
            "Disable automatic index persistence "
            "(overrides PYCONTEXTIFY_AUTO_PERSIST)"
        ),
    )
    parser.add_argument(
        "--no-auto-load",
        action="store_true",
        help=(
            "Disable automatic index loading on startup "
            "(overrides PYCONTEXTIFY_AUTO_LOAD)"
        ),
    )

    # Embedding configuration
    parser.add_argument(
        "--embedding-provider",
        choices=["sentence_transformers", "ollama", "openai"],
        help=(
            "Embedding provider to use " "(overrides PYCONTEXTIFY_EMBEDDING_PROVIDER)"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help=("Embedding model name (overrides PYCONTEXTIFY_EMBEDDING_MODEL)"),
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


def initialize_manager(
    config_overrides: Optional[Dict[str, Any]] = None,
) -> IndexManager:
    """Initialize the IndexManager with thread-safe singleton pattern.

    Args:
        config_overrides: Optional dictionary of configuration overrides from CLI args

    Returns:
        IndexManager instance

    Raises:
        Exception: If manager initialization fails
    """
    global _manager_instance, _manager_config_overrides

    with _manager_lock:
        if _manager_instance is None:
            try:
                # Store config for cleanup handler
                _manager_config_overrides = config_overrides
                config = Config(config_overrides=config_overrides)
                _manager_instance = IndexManager(config)
                logger.info("IndexManager initialized successfully")
                if config_overrides:
                    override_keys = list(config_overrides.keys())
                    logger.info(f"Applied CLI configuration overrides: {override_keys}")
            except Exception as e:
                logger.error(f"Failed to initialize IndexManager: {e}")
                _manager_instance = None  # Reset on failure
                _manager_config_overrides = None
                raise

    return _manager_instance


def get_manager() -> Optional[IndexManager]:
    """Get the current manager instance without initialization.

    Returns:
        Current IndexManager instance or None if not initialized
    """
    with _manager_lock:
        return _manager_instance


def reset_manager() -> None:
    """Reset the manager singleton - primarily for testing.

    This function should only be used in test environments to ensure
    clean state between test runs.
    """
    global _manager_instance, _manager_config_overrides
    with _manager_lock:
        if _manager_instance:
            try:
                _manager_instance.__exit__(None, None, None)
            except Exception:
                pass  # Ignore cleanup errors during test reset
        _manager_instance = None
        _manager_config_overrides = None


# Create FastMCP application
mcp = FastMCP("PyContextify")


def _index_code_impl(path: str) -> Dict[str, Any]:
    """Implementation for index_code with validation and business logic."""
    # Validate parameters
    path = validate_string_param(path, "path")

    path_obj = Path(path).resolve()
    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")

    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Initialize manager and index
    mgr = initialize_manager()
    result = mgr.index_codebase(str(path_obj))

    logger.info(f"Code indexing completed for {path}: {result}")
    return result


@mcp.tool
def index_code(path: str) -> Dict[str, Any]:
    """Index a codebase directory for semantic search.

    This function recursively scans a directory for code files and indexes them
    for semantic search. It supports various programming languages.

    Args:
        path: Path to the codebase directory to index

    Returns:
        Dictionary with indexing statistics including files processed and chunks added
    """
    return handle_mcp_errors("Code indexing", _index_code_impl, path)


def _index_document_impl(path: str) -> Dict[str, Any]:
    """Implementation for index_document with validation and business logic."""
    # Validate parameters
    path = validate_string_param(path, "path")

    path_obj = Path(path).resolve()
    if not path_obj.exists():
        raise ValueError(f"File does not exist: {path}")

    if not path_obj.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Check supported extensions
    supported_extensions = {".pdf", ".md", ".txt"}
    if path_obj.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported file type. Supported: {', '.join(supported_extensions)}"
        )

    # Initialize manager and index
    mgr = initialize_manager()
    result = mgr.index_document(str(path_obj))

    logger.info(f"Document indexing completed for {path}: {result}")
    return result


@mcp.tool
def index_document(path: str) -> Dict[str, Any]:
    """Index a single document (PDF, Markdown, or text file) for semantic search.

    This function indexes individual documents and extracts their structure
    for semantic search capabilities.

    Args:
        path: Path to the document file to index

    Returns:
        Dictionary with indexing statistics including chunks added
    """
    return handle_mcp_errors("Document indexing", _index_document_impl, path)


async def _index_webpage_impl(
    url: str, recursive: bool, max_depth: int
) -> Dict[str, Any]:
    """Implementation for index_webpage with validation and business logic."""
    # Validate parameters
    url = validate_string_param(url, "url")

    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    recursive = validate_bool_param(recursive, "recursive")

    # Validate and limit max_depth
    original_depth = max_depth
    max_depth = validate_int_param(max_depth, "max_depth", min_val=1, max_val=3)

    if original_depth > 3:
        logger.warning(f"Limited max_depth from {original_depth} to 3 for safety")

    # Initialize manager and index
    mgr = initialize_manager()
    result = await mgr.index_webpage(
        url,
        recursive=recursive,
        max_depth=max_depth,
    )

    logger.info(f"Webpage indexing completed for {url}: {result}")
    return result


@mcp.tool
async def index_webpage(
    url: str,
    recursive: bool = False,
    max_depth: int = 1,
) -> Dict[str, Any]:
    """Index web content for semantic search with optional recursive crawling.

    This function indexes web pages and can optionally follow links to index
    related pages. It extracts web content structure.

    Args:
        url: URL of the webpage to index
        recursive: Whether to follow links and index linked pages
        max_depth: Maximum depth for recursive crawling. 0 = unlimited,
                  1 = starting URL + direct children, 2 includes grandchildren, etc.
                  Ignored if recursive=False. Max allowed: 3

    Returns:
        Dictionary with indexing statistics including pages processed and chunks added
    """
    return await handle_mcp_errors_async(
        "Webpage indexing",
        _index_webpage_impl,
        url,
        recursive,
        max_depth,
    )


@mcp.tool
def search(query: str, top_k: int = 5, display_format: str = "structured") -> Any:
    """Perform semantic search across all indexed content.

    This function searches for content similar to the provided query across
    all indexed codebases, documents, and webpages using vector similarity.
    The default output format is structured data for programmatic use.

    Args:
        query: Search query text
        top_k: Maximum number of results to return (default: 5)
        display_format: Output format - 'structured' (default), 'readable',
            or 'summary'

    Returns:
        Structured search results as list of dictionaries, or formatted text
        for readable/summary
    """
    try:
        # Validate query
        if not query or not isinstance(query, str):
            if display_format == "structured":
                return []
            return "❌ Error: Query must be a non-empty string"

        query = query.strip()
        if not query:
            if display_format == "structured":
                return []
            return "❌ Error: Query cannot be empty"

        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1:
            top_k = 5

        # Validate display format
        valid_formats = ["readable", "structured", "summary"]
        if display_format not in valid_formats:
            display_format = "readable"

        # Limit top_k to reasonable range
        top_k = min(max(top_k, 1), 50)

        # Initialize manager and search
        mgr = initialize_manager()
        response = mgr.search(query, top_k, display_format)

        # Handle SearchResponse format
        if hasattr(response, "success"):
            if not response.success:
                if display_format == "structured":
                    return []
                error_msg = f"❌ Search failed: {response.error or 'Unknown error'}"
                logger.error(f"Search failed: {response.error}")
                return error_msg

            # Return formatted output or structured data
            if display_format == "structured":
                # Return structured list of result dictionaries for programmatic use
                structured_results = []
                for result in response.results:
                    result_dict = {
                        "chunk_id": result.chunk_id,
                        "chunk_text": result.text,
                        "similarity_score": result.relevance_score,
                        "source_path": result.source_path,
                        "source_type": result.source_type,
                        "metadata": (
                            result.metadata.to_dict()
                            if result.metadata and hasattr(result.metadata, "to_dict")
                            else (result.metadata if result.metadata else {})
                        ),
                        "scores": (
                            result.scores.to_dict()
                            if result.scores and hasattr(result.scores, "to_dict")
                            else (
                                result.scores
                                if result.scores
                                else {"vector_score": result.relevance_score}
                            )
                        ),
                    }
                    structured_results.append(result_dict)

                logger.info(
                    f"Search completed for '{query}': {len(structured_results)} results"
                )
                return structured_results
            else:
                # Return human-readable formatted output
                formatted_output = (
                    response.formatted_output
                    or response.format_for_display(display_format)
                )
                logger.info(
                    f"Search completed for '{query}': {len(response.results)} results"
                )
                return formatted_output
        else:
            # Legacy fallback (shouldn't happen)
            logger.info(f"Search completed for '{query}': fallback response")
            if display_format == "structured":
                return []
            return str(response)

    except Exception as e:
        error_msg = f"Search failed for query '{query}': {str(e)}"
        logger.error(error_msg)
        if display_format == "structured":
            return []
        return error_msg


@mcp.tool
def reset_index(remove_files: bool = True, confirm: bool = False) -> Dict[str, Any]:
    """Reset the entire knowledge index, clearing all indexed content.

    This function clears all indexed data from memory and optionally removes
    saved index files from disk. This is a destructive operation that cannot
    be undone without re-indexing all content.

    Args:
        remove_files: Whether to remove saved index files from disk (default: True)
        confirm: Safety confirmation - must be True to proceed (default: False)

    Returns:
        Dictionary with reset operation results and before/after statistics
    """
    try:
        # Safety check - require explicit confirmation
        if not confirm:
            return {
                "success": False,
                "error": (
                    "Reset operation requires explicit confirmation. "
                    "Set confirm=True to proceed."
                ),
                "warning": (
                    "This operation will permanently delete all indexed content."
                ),
                "help": "Use reset_index(confirm=True) to proceed with the reset.",
            }

        # Validate parameters
        if not isinstance(remove_files, bool):
            return {"success": False, "error": "remove_files must be a boolean"}

        if not isinstance(confirm, bool):
            return {"success": False, "error": "confirm must be a boolean"}

        # Initialize manager to get before-reset stats
        mgr = initialize_manager()

        # Capture before-reset statistics
        try:
            before_status = mgr.get_status()
            before_stats = {
                "total_chunks": before_status.get("index_stats", {}).get(
                    "total_chunks", 0
                ),
                "total_documents": before_status.get("index_stats", {}).get(
                    "total_documents", 0
                ),
                "memory_usage_mb": before_status.get("performance", {}).get(
                    "memory_usage_mb", 0
                ),
            }
        except Exception:
            # If we can't get stats, provide defaults
            before_stats = {
                "total_chunks": 0,
                "total_documents": 0,
                "memory_usage_mb": 0,
            }

        # Perform the reset operation
        reset_result = mgr.clear_index(remove_files=remove_files)

        if not reset_result.get("success", False):
            error_detail = reset_result.get("error", "Unknown error")
            return {
                "success": False,
                "error": f"Reset operation failed: {error_detail}",
                "before_reset": before_stats,
            }

        # Capture after-reset statistics
        try:
            after_status = mgr.get_status()
            after_stats = {
                "total_chunks": after_status.get("index_stats", {}).get(
                    "total_chunks", 0
                ),
                "memory_usage_mb": after_status.get("performance", {}).get(
                    "memory_usage_mb", 0
                ),
            }
        except Exception:
            after_stats = {"total_chunks": 0, "memory_usage_mb": 0}

        success_message = "Index reset completed successfully. "
        if remove_files:
            success_message += "All indexed data and files have been removed."
        else:
            success_message += "All indexed data cleared from memory."

        result = {
            "success": True,
            "message": success_message,
            "before_reset": before_stats,
            "after_reset": after_stats,
            "files_removed": remove_files,
        }

        chunks_removed = before_stats["total_chunks"]
        logger.info(
            f"Index reset completed: {chunks_removed} chunks removed, "
            f"files_removed={remove_files}"
        )
        return result

    except Exception as e:
        error_msg = f"Failed to reset index: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}


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
        current_manager = get_manager()
        if current_manager is not None:
            result = current_manager.get_status()
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
        result["manager_initialized"] = get_manager() is not None
        result["mcp_server"] = {
            "name": "PyContextify",
            "version": "0.1.0",
            "mcp_functions": [
                "index_code",
                "index_document",
                "index_webpage",
                "search",
                "reset_index",
                "status",
            ],
            "interface": "simplified",  # Simplified vector search interface
        }

        logger.info("Status request completed successfully")
        return result

    except Exception as e:
        error_msg = f"Failed to get system status: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "manager_initialized": get_manager() is not None,
        }


async def perform_initial_indexing(args: argparse.Namespace, mgr: IndexManager) -> None:
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
                    logger.info(
                        f"Successfully indexed document {doc_path}: "
                        f"{chunks_added} chunks"
                    )
                else:
                    logger.error(
                        f"Failed to index document {doc_path}: {result['error']}"
                    )

            except Exception as e:
                logger.error(f"Error indexing document {doc_path}: {e}")

    # Index initial codebases
    if args.initial_codebase:
        logger.info(f"Indexing {len(args.initial_codebase)} initial codebases...")
        for codebase_path in args.initial_codebase:
            try:
                path_obj = Path(codebase_path).resolve()
                if not path_obj.exists():
                    logger.warning(
                        f"Codebase directory not found, skipping: {codebase_path}"
                    )
                    continue

                if not path_obj.is_dir():
                    logger.warning(
                        f"Path is not a directory, skipping: {codebase_path}"
                    )
                    continue

                result = mgr.index_codebase(str(path_obj))
                if "error" not in result:
                    files_processed = result.get("files_processed", 0)
                    chunks_added = result.get("chunks_added", 0)
                    total_indexed += chunks_added
                    logger.info(
                        f"Successfully indexed codebase {codebase_path}: "
                        f"{files_processed} files, {chunks_added} chunks"
                    )
                else:
                    logger.error(
                        f"Failed to index codebase {codebase_path}: {result['error']}"
                    )

            except Exception as e:
                logger.error(f"Error indexing codebase {codebase_path}: {e}")

    # Index initial webpages
    if args.initial_webpages:
        logger.info(f"Indexing {len(args.initial_webpages)} initial webpages...")
        for webpage_url in args.initial_webpages:
            try:
                # Validate URL
                if not webpage_url.startswith(("http://", "https://")):
                    logger.warning(
                        "Invalid URL (must start with http:// or https://), "
                        f"skipping: {webpage_url}"
                    )
                    continue

                # Apply crawling settings
                recursive = (
                    args.recursive_crawling
                    if hasattr(args, "recursive_crawling")
                    else False
                )
                max_depth = getattr(args, "max_crawl_depth", 1)

                # Validate and limit max_depth
                if max_depth < 1 or max_depth > 3:
                    max_depth = min(max(max_depth, 1), 3)
                    logger.warning(
                        f"Adjusted max_crawl_depth to {max_depth} (valid range: 1-3)"
                    )

                result = await mgr.index_webpage(
                    webpage_url, recursive=recursive, max_depth=max_depth
                )
                if "error" not in result:
                    pages_processed = result.get("pages_processed", 1)
                    chunks_added = result.get("chunks_added", 0)
                    total_indexed += chunks_added
                    logger.info(
                        f"Successfully indexed webpage {webpage_url}: "
                        f"{pages_processed} pages, {chunks_added} chunks "
                        f"(recursive={recursive}, max_depth={max_depth})"
                    )
                else:
                    logger.error(
                        f"Failed to index webpage {webpage_url}: {result['error']}"
                    )

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
    if getattr(args, "index_bootstrap_archive_url", None):
        overrides["bootstrap_archive_url"] = args.index_bootstrap_archive_url

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
    if hasattr(args, "crawl_delay") and args.crawl_delay is not None:
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
        force=True,  # Reconfigure existing loggers
    )


def cleanup_handler(signum, frame):
    """Handle graceful shutdown signals with enhanced resource management."""
    import sys

    logger.info("Received shutdown signal, cleaning up...")

    current_manager = get_manager()
    if current_manager:
        try:
            # Use context manager for proper cleanup sequence
            # This will handle auto-save, embedder cleanup, and other resources
            logger.info("Initiating comprehensive cleanup sequence")
            current_manager.__exit__(None, None, None)
            logger.info("Manager cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during manager cleanup: {e}")
            # Try individual cleanup steps as fallback
            try:
                if hasattr(current_manager, "embedder") and current_manager.embedder:
                    current_manager.embedder.cleanup()
                    logger.info("Fallback embedder cleanup completed")
            except Exception as fallback_e:
                logger.error(f"Fallback cleanup also failed: {fallback_e}")
    else:
        logger.info("No manager instance found, skipping cleanup")

    logger.info("Cleanup completed, shutting down gracefully")
    sys.exit(0)


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
        logger.info("Server provides 6 essential MCP functions:")
        logger.info("  - index_code(path): Index codebase directory")
        logger.info("  - index_document(path): Index document")
        logger.info("  - index_webpage(url, recursive, max_depth): Index web content")
        logger.info("  - search(query, top_k): Basic semantic search")
        logger.info("  - reset_index(confirm=True): Clear all indexed content")
        logger.info("  - status(): Get system status and statistics")

        # Convert CLI args to config overrides
        config_overrides = args_to_config_overrides(args)
        if config_overrides:
            logger.info(
                f"Using CLI configuration overrides: {list(config_overrides.keys())}"
            )

        # Initialize manager with CLI overrides
        mgr = initialize_manager(config_overrides)

        # Perform initial indexing if specified
        if args.initial_documents or args.initial_codebase or args.initial_webpages:
            logger.info("Performing initial indexing...")
            import asyncio

            try:
                asyncio.run(perform_initial_indexing(args, mgr))
            except RuntimeError as e:
                if "event loop is already running" in str(e).lower():
                    # If there's already a running event loop, use it
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(perform_initial_indexing(args, mgr))
                    finally:
                        loop.close()
                else:
                    raise
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
