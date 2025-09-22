"""Simplified comprehensive tests for MCP server functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import pycontextify.mcp_server as mcp_server_module
from pycontextify.mcp_server import (
    args_to_config_overrides,
    initialize_manager,
    perform_initial_indexing,
    reset_manager,
)


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

    @patch("pycontextify.mcp_server.IndexManager")
    def test_index_code_success(self, mock_manager_class):
        """Test successful code indexing."""
        # Create test directory
        test_code_dir = self.test_dir / "test_code"
        test_code_dir.mkdir()
        test_file = test_code_dir / "test.py"
        test_file.write_text("def hello(): return 'world'")

        # Mock IndexManager
        mock_manager = Mock()
        mock_manager.index_codebase.return_value = {
            "files_processed": 1,
            "chunks_added": 5,
            "status": "success",
        }
        mock_manager_class.return_value = mock_manager

        # Access the actual function through FastMCP
        index_code_fn = mcp_server_module.mcp._tool_manager._tools["index_code"].fn
        result = index_code_fn(str(test_code_dir))

        assert "error" not in result
        assert result["files_processed"] == 1
        assert result["chunks_added"] == 5
        mock_manager.index_codebase.assert_called_once_with(
            str(test_code_dir.resolve())
        )

    def test_index_code_nonexistent_path(self):
        """Test indexing nonexistent path."""
        index_code_fn = mcp_server_module.mcp._tool_manager._tools["index_code"].fn
        result = index_code_fn("/nonexistent/path")
        assert "error" in result
        assert "does not exist" in result["error"]

    def test_index_code_empty_path(self):
        """Test indexing with empty path."""
        index_code_fn = mcp_server_module.mcp._tool_manager._tools["index_code"].fn
        result = index_code_fn("")
        assert "error" in result
        assert "cannot be empty" in result["error"]

    @patch("pycontextify.mcp_server.IndexManager")
    def test_index_document_success(self, mock_manager_class):
        """Test successful document indexing."""
        test_doc = self.test_dir / "test.pdf"
        test_doc.write_text("dummy pdf content")

        mock_manager = Mock()
        mock_manager.index_document.return_value = {
            "chunks_added": 3,
            "status": "success",
        }
        mock_manager_class.return_value = mock_manager

        index_document_fn = mcp_server_module.mcp._tool_manager._tools[
            "index_document"
        ].fn
        result = index_document_fn(str(test_doc))

        assert "error" not in result
        assert result["chunks_added"] == 3
        mock_manager.index_document.assert_called_once_with(str(test_doc.resolve()))

    def test_index_document_nonexistent_file(self):
        """Test indexing nonexistent document."""
        index_document_fn = mcp_server_module.mcp._tool_manager._tools[
            "index_document"
        ].fn
        result = index_document_fn("/nonexistent/file.pdf")
        assert "error" in result
        assert "does not exist" in result["error"]

    def test_index_document_unsupported_extension(self):
        """Test indexing unsupported file type."""
        test_file = self.test_dir / "test.exe"
        test_file.write_text("binary content")

        index_document_fn = mcp_server_module.mcp._tool_manager._tools[
            "index_document"
        ].fn
        result = index_document_fn(str(test_file))
        assert "error" in result
        assert "Unsupported file type" in result["error"]

    @patch("pycontextify.mcp_server.IndexManager")
    def test_index_webpage_success(self, mock_manager_class):
        """Test successful webpage indexing."""
        mock_manager = Mock()
        mock_manager.index_webpage.return_value = {
            "pages_processed": 1,
            "chunks_added": 8,
            "status": "success",
        }
        mock_manager_class.return_value = mock_manager

        index_webpage_fn = mcp_server_module.mcp._tool_manager._tools[
            "index_webpage"
        ].fn
        result = index_webpage_fn("https://example.com", recursive=False, max_depth=1)

        assert "error" not in result
        assert result["pages_processed"] == 1
        assert result["chunks_added"] == 8
        mock_manager.index_webpage.assert_called_once_with(
            "https://example.com", recursive=False, max_depth=1
        )

    def test_index_webpage_invalid_url(self):
        """Test indexing invalid URL."""
        index_webpage_fn = mcp_server_module.mcp._tool_manager._tools[
            "index_webpage"
        ].fn
        result = index_webpage_fn("not-a-url")
        assert "error" in result
        assert "must start with http://" in result["error"]

    @patch("pycontextify.mcp_server.IndexManager")
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

        search_fn = mcp_server_module.mcp._tool_manager._tools["search"].fn
        result = search_fn("test query", top_k=5)

        assert len(result) == 1
        assert result[0]["chunk_id"] == "test_chunk"
        assert result[0]["similarity_score"] == 0.9
        mock_manager.search.assert_called_once_with("test query", 5, "structured")

    def test_search_empty_query(self):
        """Test search with empty query."""
        search_fn = mcp_server_module.mcp._tool_manager._tools["search"].fn
        result = search_fn("")
        assert result == []

    @patch("pycontextify.mcp_server.IndexManager")
    def test_search_with_context_success(self, mock_manager_class):
        """Test successful context search."""
        mock_manager = Mock()

        # Create mock SearchResponse with proper structure
        mock_response = Mock()
        mock_response.success = True
        mock_response.results = [Mock()]
        mock_response.results[0].chunk_id = "context_chunk"
        mock_response.results[0].text = "context content"
        mock_response.results[0].relevance_score = 0.85
        mock_response.results[0].source_path = "/test.py"
        mock_response.results[0].source_type = "code"
        mock_response.results[0].metadata = None
        mock_response.results[0].scores = None
        mock_response.results[0].context = {"relationships": ["related_chunk_1"]}

        mock_manager.search_with_context.return_value = mock_response
        mock_manager_class.return_value = mock_manager

        search_with_context_fn = mcp_server_module.mcp._tool_manager._tools[
            "search_with_context"
        ].fn
        result = search_with_context_fn("test query", top_k=3, include_related=True)

        assert len(result) == 1
        assert result[0]["chunk_id"] == "context_chunk"
        assert "relationships" in result[0]
        mock_manager.search_with_context.assert_called_once_with(
            "test query", 3, True, "structured"
        )

    @patch("pycontextify.mcp_server.IndexManager")
    def test_status_success(self, mock_manager_class):
        """Test successful status retrieval."""
        mock_manager = Mock()
        mock_manager.get_status.return_value = {
            "total_chunks": 100,
            "total_files": 15,
            "index_size_mb": 5.2,
            "embedding_provider": "sentence_transformers",
        }
        mock_manager_class.return_value = mock_manager

        status_fn = mcp_server_module.mcp._tool_manager._tools["status"].fn
        result = status_fn()

        assert result["total_chunks"] == 100
        assert result["total_files"] == 15
        assert result["manager_initialized"] is True
        assert "mcp_server" in result
        assert result["mcp_server"]["name"] == "PyContextify"
        assert "index_code" in result["mcp_server"]["mcp_functions"]


class TestMCPServerUtilities:
    """Test utility functions."""

    def test_args_to_config_overrides(self):
        """Test conversion of CLI args to config overrides."""
        from argparse import Namespace

        args = Namespace(
            index_path="/custom/index",
            index_name="custom_index",
            no_auto_persist=True,
            no_auto_load=False,
            embedding_provider="openai",
            embedding_model="text-embedding-ada-002",
            crawl_delay=3,
        )

        overrides = args_to_config_overrides(args)

        assert overrides["index_dir"] == "/custom/index"
        assert overrides["index_name"] == "custom_index"
        assert overrides["auto_persist"] is False
        assert "auto_load" not in overrides  # Should not be set when False
        assert overrides["embedding_provider"] == "openai"
        assert overrides["embedding_model"] == "text-embedding-ada-002"
        assert overrides["crawl_delay_seconds"] == 3

    @patch("pycontextify.mcp_server.IndexManager")
    def test_initialize_manager_with_overrides(self, mock_manager_class):
        """Test manager initialization with config overrides."""
        with patch("pycontextify.mcp_server.Config") as mock_config_class:
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            overrides = {"index_dir": "/custom/path", "embedding_provider": "openai"}
            manager = initialize_manager(overrides)

            assert manager == mock_manager
            mock_config_class.assert_called_once_with(config_overrides=overrides)
            mock_manager_class.assert_called_once_with(mock_config)

    @patch("pycontextify.mcp_server.IndexManager")
    def test_perform_initial_indexing_documents(self, mock_manager_class):
        """Test initial document indexing."""
        from argparse import Namespace

        # Create test documents
        test_dir = Path(tempfile.mkdtemp())
        test_doc = test_dir / "test.md"
        test_doc.write_text("# Test Document\nThis is a test document.")

        mock_manager = Mock()
        mock_manager.index_document.return_value = {"chunks_added": 2}
        mock_manager.config.auto_persist = True
        mock_manager.save_index = Mock()
        mock_manager_class.return_value = mock_manager

        args = Namespace(
            initial_documents=[str(test_doc)],
            initial_codebase=None,
            initial_webpages=None,
        )

        perform_initial_indexing(args, mock_manager)

        mock_manager.index_document.assert_called_once_with(str(test_doc.resolve()))
        mock_manager.save_index.assert_called_once()

        # Cleanup
        import shutil

        shutil.rmtree(test_dir)

    def test_error_handling_in_functions(self):
        """Test error handling in MCP functions."""
        # Test search error handling
        with patch(
            "pycontextify.mcp_server.initialize_manager",
            side_effect=Exception("Manager error"),
        ):
            search_fn = mcp_server_module.mcp._tool_manager._tools["search"].fn
            result = search_fn("test query")
            assert result == []  # Should return empty list on error

        # Test search_with_context error handling
        with patch(
            "pycontextify.mcp_server.initialize_manager",
            side_effect=Exception("Manager error"),
        ):
            search_with_context_fn = mcp_server_module.mcp._tool_manager._tools[
                "search_with_context"
            ].fn
            result = search_with_context_fn("test query")
            assert result == []  # Should return empty list on error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
