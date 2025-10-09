"""FastMCP server for PyContextify.

This module implements the MCP server with simplified interface providing
5 essential functions for semantic search over codebases and documents.
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
    import asyncio
    import time

    start_time = time.time()

    try:
        logger.info(f"{operation_name} started")

        # Add timeout protection to prevent hanging
        try:
            result = await asyncio.wait_for(
                func_impl(*args), timeout=120.0
            )  # 2 minute timeout
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            timeout_msg = (
                f"{operation_name} timed out after {elapsed:.2f}s (exceeded 120s limit)"
            )
            logger.error(timeout_msg)
            return {
                "error": timeout_msg,
                "operation": operation_name,
                "elapsed_seconds": elapsed,
            }

        elapsed = time.time() - start_time
        logger.info(f"{operation_name} completed successfully in {elapsed:.2f}s")

        # Ensure result is JSON serializable and properly formatted
        if result is None:
            logger.warning(
                f"{operation_name} returned None, converting to structured response"
            )
            result = {
                "warning": "Operation completed but returned no data",
                "operation": operation_name,
                "elapsed_seconds": elapsed,
            }
        elif not isinstance(result, dict):
            logger.warning(
                f"{operation_name} returned non-dict result: {type(result)}, converting to dict"
            )
            result = {
                "result": result,
                "operation": operation_name,
                "elapsed_seconds": elapsed,
            }
        # Result is valid dict, ready to return

        # Result ready for return

        return result
    except ValueError as e:
        # Validation errors - return structured error
        elapsed = time.time() - start_time
        error_msg = str(e)
        logger.warning(
            f"{operation_name} validation error after {elapsed:.2f}s: {error_msg}"
        )
        return {
            "error": error_msg,
            "operation": operation_name,
            "elapsed_seconds": elapsed,
        }
    except Exception as e:
        # Unexpected errors - log and return structured error
        elapsed = time.time() - start_time
        error_msg = f"{operation_name} failed: {str(e)}"
        logger.error(f"{operation_name} failed after {elapsed:.2f}s: {error_msg}")
        return {
            "error": error_msg,
            "operation": operation_name,
            "elapsed_seconds": elapsed,
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the MCP server."""
    parser = argparse.ArgumentParser(
        prog="pycontextify",
        description=(
            "PyContextify MCP Server - Semantic search over codebases and documents"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Start server with custom index path
  pycontextify --index-path ./my_index

  # Start with initial filebase indexing
  pycontextify --initial-filebase ./src --topic my-project

  # Index with custom index path and topic
  pycontextify --index-path ./project_index --initial-filebase ./src --topic codebase

  # Index documentation with topic
  pycontextify --initial-filebase ./docs --topic documentation

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
        "--initial-filebase",
        type=str,
        help="Directory path to index at startup using unified filebase pipeline",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Topic name for initial indexing (required with --initial-filebase)",
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


def _index_filebase_impl(
    base_path: str,
    topic: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Implementation for index_filebase with validation and business logic."""
    # Validate parameters
    base_path = validate_string_param(base_path, "base_path")
    topic = validate_string_param(topic, "topic")

    # Initialize manager and index
    mgr = initialize_manager()
    result = mgr.index_filebase(
        base_path=base_path,
        topic=topic,
        include=include,
        exclude=exclude,
        exclude_dirs=exclude_dirs,
    )

    logger.info(
        f"Filebase indexing completed for {base_path} (topic: {topic}): {result}"
    )
    return result


@mcp.tool
def index_filebase(
    base_path: str,
    topic: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Index a filebase (directory tree) for semantic search.

    This is the unified indexing function that handles all file types
    (code, documents, PDFs) with a single consistent pipeline.

    Args:
        base_path: Root directory path to index
        topic: Topic name for organizing indexed content (required)
        include: Optional list of fnmatch patterns to include (e.g., ["*.py", "*.md"])
        exclude: Optional list of fnmatch patterns to exclude (e.g., ["*_test.py"])
        exclude_dirs: Optional list of directory names to exclude (e.g., ["node_modules", ".git"])

    Returns:
        Dictionary with indexing statistics including:
        - topic: The topic name
        - base_path: Root directory indexed
        - files_crawled: Total files discovered
        - files_loaded: Files successfully loaded
        - chunks_created: Total chunks created
        - vectors_embedded: Vectors generated
        - errors: Number of errors encountered
        - duration_seconds: Time taken
    """
    return handle_mcp_errors(
        "Filebase indexing",
        _index_filebase_impl,
        base_path,
        topic,
        include,
        exclude,
        exclude_dirs,
    )


@mcp.tool
def discover() -> Dict[str, Any]:
    """Discover all indexed topics.

    Returns a list of unique topic names from all indexed content,
    useful for browsing and filtering indexed material.

    Returns:
        Dictionary with:
        - topics: Sorted list of unique topic names
        - count: Number of unique topics
    """
    try:
        mgr = initialize_manager()
        topics = mgr.metadata_store.discover_topics()

        result = {
            "topics": topics,
            "count": len(topics),
        }

        logger.info(f"Discovered {len(topics)} topics")
        return result

    except Exception as e:
        error_msg = f"Failed to discover topics: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "topics": [],
            "count": 0,
        }


@mcp.tool
def search(query: str, top_k: int = 5, display_format: str = "structured") -> Any:
    """Perform semantic search across all indexed content.

    This function searches for content similar to the provided query across
    all indexed codebases and documents using vector similarity.
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
                "index_filebase",
                "discover",
                "search",
                "reset_index",
                "status",
            ],
            "interface": "unified_filebase",  # Unified filebase indexing
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
    """Perform initial filebase indexing if specified in CLI args.

    Args:
        args: Parsed command-line arguments
        mgr: Initialized IndexManager instance
    """
    # Index initial filebase (safe attribute access)
    if getattr(args, "initial_filebase", None):
        # Validate that topic is provided
        if not getattr(args, "topic", None):
            logger.error(
                "ERROR: --topic is required when using --initial-filebase\n"
                "Example: pycontextify --initial-filebase ./src --topic my-project"
            )
            import sys

            sys.exit(1)

        logger.info(
            f"Indexing initial filebase: {args.initial_filebase} (topic: {args.topic})"
        )

        try:
            result = mgr.index_filebase(
                base_path=args.initial_filebase,
                topic=args.topic,
            )

            if "error" not in result:
                logger.info(
                    f"Successfully indexed filebase: "
                    f"{result.get('chunks_created', 0)} chunks from "
                    f"{result.get('files_loaded', 0)} files in "
                    f"{result.get('duration_seconds', 0)}s"
                )

                # Print stats to stdout for user visibility
                print("\n=== Indexing Complete ===")
                print(f"Topic: {result.get('topic')}")
                print(f"Files crawled: {result.get('files_crawled', 0)}")
                print(f"Files loaded: {result.get('files_loaded', 0)}")
                print(f"Chunks created: {result.get('chunks_created', 0)}")
                print(f"Vectors embedded: {result.get('vectors_embedded', 0)}")
                print(f"Errors: {result.get('errors', 0)}")
                print(f"Duration: {result.get('duration_seconds', 0)}s")
                print("========================\n")
            else:
                logger.error(f"Failed to index filebase: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error indexing filebase: {e}")
    else:
        logger.info("No initial indexing requested")


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
        logger.info("Server provides 5 essential MCP functions:")
        logger.info("  - index_filebase(path, topic): Unified filebase indexing")
        logger.info("  - discover(): List indexed topics")
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
        if getattr(args, "initial_filebase", None):
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
