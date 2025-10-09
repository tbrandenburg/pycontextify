"""Consolidated tests for PyContextify MCP server functionality.

This module combines the essential test cases from test_mcp_simple.py,
test_mcp_server_simple.py, and test_mcp_server_enhanced.py, focusing
on the most portable and essential test coverage.
"""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

import pycontextify.mcp as mcp_module

# Import MCP functions and utilities
from pycontextify import mcp
from pycontextify.config import Config
from pycontextify.mcp import (
    args_to_config_overrides,
    get_manager,
    handle_mcp_errors,
    initialize_manager,
    perform_initial_indexing,
    reset_manager,
    setup_logging,
    validate_bool_param,
    validate_choice_param,
    validate_int_param,
    validate_string_param,
)


class TestMCPValidationFunctions:
    """Test parameter validation utility functions."""

    def test_validate_string_param_valid(self):
        """Test string validation with valid inputs."""
        assert validate_string_param("test", "param") == "test"
        assert validate_string_param("  test  ", "param") == "test"

    def test_validate_string_param_empty_allowed(self):
        """Test string validation allowing empty strings."""
        assert validate_string_param("", "param", allow_empty=True) == ""
        assert validate_string_param("  ", "param", allow_empty=True) == "  "

    def test_validate_string_param_invalid(self):
        """Test string validation with invalid inputs."""
        with pytest.raises(ValueError, match="param must be a string"):
            validate_string_param(123, "param")

        with pytest.raises(ValueError, match="param cannot be empty"):
            validate_string_param("", "param", allow_empty=False)

    def test_validate_int_param_valid(self):
        """Test integer validation with valid inputs."""
        assert validate_int_param(42, "param") == 42
        assert validate_int_param(0, "param") == 0

    def test_validate_int_param_with_bounds(self):
        """Test integer validation with min/max bounds."""
        assert validate_int_param(5, "param", min_val=1, max_val=10) == 5
        assert validate_int_param(1, "param", min_val=1) == 1
        assert validate_int_param(10, "param", max_val=10) == 10

    def test_validate_int_param_invalid(self):
        """Test integer validation with invalid inputs."""
        with pytest.raises(ValueError, match="param must be an integer"):
            validate_int_param("not_int", "param")

        with pytest.raises(ValueError, match="param must be at least 5"):
            validate_int_param(3, "param", min_val=5)

    def test_validate_bool_param_valid(self):
        """Test boolean validation with valid inputs."""
        assert validate_bool_param(True, "param") is True
        assert validate_bool_param(False, "param") is False

    def test_validate_bool_param_invalid(self):
        """Test boolean validation with invalid inputs."""
        with pytest.raises(ValueError, match="param must be a boolean"):
            validate_bool_param("true", "param")

    def test_validate_choice_param_valid(self):
        """Test choice validation with valid inputs."""
        choices = ["option1", "option2", "option3"]
        assert validate_choice_param("option1", "param", choices) == "option1"
        assert validate_choice_param("option3", "param", choices) == "option3"

    def test_validate_choice_param_invalid_with_default(self):
        """Test choice validation with invalid input and default."""
        choices = ["option1", "option2"]
        result = validate_choice_param("invalid", "param", choices, default="option1")
        assert result == "option1"


class TestMCPErrorHandling:
    """Test error handling utility functions."""

    def test_handle_mcp_errors_success(self):
        """Test successful function execution."""

        def success_func(x):
            return {"result": x * 2}

        result = handle_mcp_errors("test_op", success_func, 5)
        assert result == {"result": 10}

    def test_handle_mcp_errors_value_error(self):
        """Test handling of ValueError."""

        def error_func():
            raise ValueError("Invalid parameter")

        result = handle_mcp_errors("test_op", error_func)
        assert result == {"error": "Invalid parameter"}

    def test_handle_mcp_errors_generic_exception(self):
        """Test handling of generic exceptions."""

        def error_func():
            raise RuntimeError("Something went wrong")

        result = handle_mcp_errors("test_op", error_func)
        assert result == {"error": "test_op failed: Something went wrong"}


class TestManagerSingleton:
    """Test manager singleton functionality."""

    def setup_method(self):
        """Reset manager before each test."""
        reset_manager()

    def teardown_method(self):
        """Clean up after each test."""
        reset_manager()

    def test_get_manager_uninitialized(self):
        """Test getting manager when not initialized."""
        assert get_manager() is None

    @patch("pycontextify.mcp.IndexManager")
    @patch("pycontextify.mcp.Config")
    def test_initialize_manager_success(self, mock_config_class, mock_manager_class):
        """Test successful manager initialization."""
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager

        overrides = {"index_dir": "/custom/path"}
        manager = initialize_manager(overrides)

        assert manager == mock_manager
        assert get_manager() == mock_manager
        mock_config_class.assert_called_once_with(config_overrides=overrides)
        mock_manager_class.assert_called_once_with(mock_config)

    def test_reset_manager(self):
        """Test manager reset functionality."""
        with patch("pycontextify.mcp.IndexManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            # Initialize manager
            initialize_manager()
            assert get_manager() is not None

            # Reset should clear it
            reset_manager()
            assert get_manager() is None


class TestMCPServerFunctions:
    """Test MCP server function endpoints."""

    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        # Reset manager singleton for each test
        reset_manager()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        # Clean up manager singleton
        reset_manager()

    @patch("pycontextify.mcp.IndexManager")
    def test_index_filebase_success(self, mock_manager_class):
        """Test successful filebase indexing with unified API."""
        # Create test directory with files
        test_dir = self.test_dir / "test_files"
        test_dir.mkdir()
        test_file = test_dir / "test.py"
        test_file.write_text("def hello(): return 'world'")

        # Mock IndexManager
        mock_manager = Mock()
        mock_manager.index_filebase.return_value = {
            "topic": "test_code",
            "base_path": str(test_dir),
            "files_crawled": 1,
            "files_loaded": 1,
            "chunks_created": 5,
            "vectors_embedded": 5,
            "errors": 0,
        }
        mock_manager_class.return_value = mock_manager

        # Access the actual function through FastMCP
        index_filebase_fn = mcp_module.mcp._tool_manager._tools["index_filebase"].fn
        result = index_filebase_fn(
            base_path=str(test_dir),
            topic="test_code"
        )

        assert "error" not in result
        assert result["files_loaded"] == 1
        assert result["chunks_created"] == 5
        assert result["topic"] == "test_code"
        mock_manager.index_filebase.assert_called_once()

    def test_index_filebase_nonexistent_path(self):
        """Test indexing nonexistent path."""
        index_filebase_fn = mcp_module.mcp._tool_manager._tools["index_filebase"].fn
        result = index_filebase_fn(
            base_path="/nonexistent/path",
            topic="test"
        )
        assert "error" in result
        assert "does not exist" in result["error"].lower()

    def test_index_filebase_missing_topic(self):
        """Test indexing without required topic."""
        test_dir = self.test_dir / "test_files"
        test_dir.mkdir()
        
        index_filebase_fn = mcp_module.mcp._tool_manager._tools["index_filebase"].fn
        result = index_filebase_fn(
            base_path=str(test_dir),
            topic=""  # Empty topic
        )
        assert "error" in result
        assert "topic" in result["error"].lower()

    @patch("pycontextify.mcp.IndexManager")
    def test_discover_topics(self, mock_manager_class):
        """Test discover function returns indexed topics."""
        mock_manager = Mock()
        
        # Mock storage with discover_topics method
        mock_storage = Mock()
        mock_storage.discover_topics.return_value = ["code", "docs", "guides"]
        mock_manager.metadata_store = mock_storage
        mock_manager_class.return_value = mock_manager

        # Access discover function
        discover_fn = mcp_module.mcp._tool_manager._tools["discover"].fn
        result = discover_fn()

        assert "topics" in result
        assert len(result["topics"]) == 3
        assert "code" in result["topics"]
        assert "docs" in result["topics"]
        assert "guides" in result["topics"]

    @patch("pycontextify.mcp.IndexManager")
    def test_search_success(self, mock_manager_class):
        """Test successful search."""
        mock_manager = Mock()

        # Create mock SearchResponse with proper structure
        mock_response = Mock()
        mock_response.success = True
        mock_response.results = [Mock()]
        mock_response.results[0].chunk_id = "test_chunk"
        mock_response.results[0].text = "test content"
        mock_response.results[0].relevance_score = 0.9
        mock_response.results[0].source_path = "/test.py"
        mock_response.results[0].source_type = "code"
        mock_response.results[0].metadata = None
        mock_response.results[0].scores = None

        mock_manager.search.return_value = mock_response
        mock_manager_class.return_value = mock_manager

        search_fn = mcp_module.mcp._tool_manager._tools["search"].fn
        result = search_fn("test query", top_k=5)

        assert len(result) == 1
        assert result[0]["chunk_id"] == "test_chunk"
        assert result[0]["similarity_score"] == 0.9
        mock_manager.search.assert_called_once_with("test query", 5, "structured")

    def test_search_empty_query(self):
        """Test search with empty query."""
        search_fn = mcp_module.mcp._tool_manager._tools["search"].fn
        result = search_fn("")
        assert result == []

    @patch("pycontextify.mcp.IndexManager")
    def test_status_function(self, mock_manager_class):
        """Test the status MCP function."""
        mock_manager = Mock()
        mock_status = {
            "status": "healthy",
            "metadata": {"total_chunks": 0},
            "vector_store": {"total_vectors": 0},
            "embedding": {"provider": "sentence_transformers"},
        }
        mock_manager.get_status.return_value = mock_status
        mock_manager_class.return_value = mock_manager

        status_fn = mcp_module.mcp._tool_manager._tools["status"].fn
        status = status_fn()

        # Verify basic structure
        assert "metadata" in status
        assert "vector_store" in status
        assert "embedding" in status
        assert "mcp_server" in status

        # MCP server info should be added
        assert status["mcp_server"]["name"] == "PyContextify"
        assert len(status["mcp_server"]["mcp_functions"]) == 5

    @patch("pycontextify.mcp.IndexManager")
    def test_reset_index_success(self, mock_manager_class):
        """Test successful index reset."""
        mock_manager = Mock()
        mock_manager.get_status.return_value = {
            "index_stats": {"total_chunks": 5, "total_documents": 2},
            "performance": {"memory_usage_mb": 100},
        }
        mock_manager.clear_index.return_value = {"success": True}
        mock_manager_class.return_value = mock_manager

        reset_fn = mcp_module.mcp._tool_manager._tools["reset_index"].fn
        result = reset_fn(remove_files=True, confirm=True)

        assert result["success"] is True
        assert "before_reset" in result
        assert result["before_reset"]["total_chunks"] == 5
        mock_manager.clear_index.assert_called_once_with(remove_files=True)

    def test_reset_index_requires_confirmation(self):
        """Test that reset requires explicit confirmation."""
        reset_fn = mcp_module.mcp._tool_manager._tools["reset_index"].fn
        result = reset_fn(remove_files=True, confirm=False)

        assert result["success"] is False
        assert "confirmation" in result["error"]


@pytest.fixture
def mcp_isolated():
    """Setup isolated MCP environment for testing."""
    # Reset manager singleton
    reset_manager()

    try:
        # Use isolated temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test config
            config = Config()
            config.index_dir = Path(temp_dir)
            config.auto_persist = False
            config.auto_load = False
            config.embedding_model = "all-MiniLM-L6-v2"  # Faster for testing

            # Initialize manager with config overrides
            mcp.initialize_manager(
                {
                    "index_dir": str(Path(temp_dir)),
                    "auto_persist": False,
                    "auto_load": False,
                    "embedding_model": "all-MiniLM-L6-v2",
                }
            )

            yield mcp

    finally:
        # Cleanup
        try:
            current_manager = mcp.get_manager()
            if current_manager:
                current_manager.clear_index()
                if hasattr(current_manager, "embedder") and current_manager.embedder:
                    current_manager.embedder.cleanup()
        except Exception:
            pass
        finally:
            # Reset manager singleton
            reset_manager()


class TestMCPIntegration:
    """Integration tests using isolated MCP environment."""

    def test_full_workflow_integration(self, mcp_isolated):
        """Test complete workflow with multiple document types."""
        print("\n🚀 Testing full MCP workflow...")

        # Step 1: Check initial status
        status = mcp_isolated.status.fn()
        assert status["metadata"]["total_chunks"] == 0
        print("✅ Initial status verified")

        # Step 2: Index a document using unified API
        test_content = """# API Reference Guide
        
## REST Endpoints
- GET /api/v1/users
- POST /api/v1/users
- PUT /api/v1/users/{id}

## Authentication
Use Bearer tokens in the Authorization header.
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            doc_dir = Path(temp_dir) / "docs"
            doc_dir.mkdir()
            doc_file = doc_dir / "api_guide.md"
            doc_file.write_text(test_content, encoding="utf-8")

            try:
                result = mcp_isolated.index_filebase.fn(
                    base_path=str(doc_dir),
                    topic="api_docs"
                )
                assert "error" not in result
                assert result["chunks_created"] > 0
                print(f"✅ Document indexed: {result['chunks_created']} chunks")

                # Step 3: Verify search works
                results = mcp_isolated.search.fn("API authentication", top_k=3)
                assert isinstance(results, list)
                assert len(results) > 0
                print(f"✅ Search working: {len(results)} results found")

                # Check result structure
                result = results[0]
                assert "chunk_text" in result
                assert "similarity_score" in result or "score" in result

                score = result.get("similarity_score", result.get("score", 0))
                # Accept numpy types for score validation
                assert isinstance(
                    score, (int, float, np.integer, np.floating)
                ), f"Score should be numeric: {score}"

            except Exception as e:
                print(f"Test failed: {e}")
                raise

    def test_error_handling_integration(self, mcp_isolated):
        """Test MCP function error handling."""
        # Test non-existent directory
        result = mcp_isolated.index_filebase.fn(
            base_path="/non/existent/directory",
            topic="test"
        )
        assert "error" in result
        print("✅ Error handling for missing directories")

        # Test empty topic
        with tempfile.TemporaryDirectory() as temp_dir:
            result = mcp_isolated.index_filebase.fn(
                base_path=temp_dir,
                topic=""  # Empty topic should fail
            )
            assert "error" in result
            print("✅ Error handling for empty topic")

        # Test invalid search
        results = mcp_isolated.search.fn("", top_k=5)
        assert results == []
        print("✅ Error handling for empty queries")


class TestMCPUtilityFunctions:
    """Test MCP utility functions."""

    def test_args_to_config_overrides(self):
        """Test CLI args to config overrides conversion."""
        # Create mock args
        args = argparse.Namespace()
        args.index_path = "/custom/index"
        args.index_name = "custom_index"
        args.no_auto_persist = True
        args.no_auto_load = True
        args.embedding_provider = "openai"
        args.embedding_model = "text-embedding-3-small"

        overrides = args_to_config_overrides(args)

        assert overrides["index_dir"] == "/custom/index"
        assert overrides["index_name"] == "custom_index"
        assert overrides["auto_persist"] is False
        assert overrides["auto_load"] is False
        assert overrides["embedding_provider"] == "openai"
        assert overrides["embedding_model"] == "text-embedding-3-small"

    @patch("pycontextify.mcp.IndexManager")
    async def test_perform_initial_indexing(self, mock_manager_class):
        """Test initial indexing functionality with new unified API."""
        mock_manager = Mock()
        mock_manager.index_filebase.return_value = {
            "files_crawled": 2,
            "files_loaded": 2,
            "chunks_created": 5,
            "vectors_embedded": 5,
        }
        mock_manager_class.return_value = mock_manager

        # Create mock args for new API
        args = argparse.Namespace()
        args.initial_filebase = None
        args.topic = None

        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "test.py").write_text("print('hello')")
            
            # Set up args for filebase indexing
            args.initial_filebase = temp_dir
            args.topic = "test_code"

            await perform_initial_indexing(args, mock_manager)

            # Should have called index_filebase
            assert mock_manager.index_filebase.called
            call_args = mock_manager.index_filebase.call_args
            assert call_args[1]["topic"] == "test_code"
