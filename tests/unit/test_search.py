"""Unit tests for SearchService."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from pycontextify.search import SearchService
from pycontextify.search_models import SearchErrorCode


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.use_hybrid_search = False
    config.embedding_provider = "test_provider"
    config.embedding_model = "test_model"
    config.keyword_weight = 0.3
    return config


@pytest.fixture
def mock_embedder_service():
    """Create mock embedder service."""
    service = Mock()
    mock_embedder = Mock()
    mock_embedder.embed_single.return_value = np.array([0.1] * 768)
    service.get_embedder.return_value = mock_embedder
    return service


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = Mock()
    store.is_empty.return_value = False
    # Mock search results: (distances, indices)
    store.search.return_value = (
        np.array([0.2, 0.4, 0.6]),
        np.array([0, 1, 2]),
    )
    return store


@pytest.fixture
def mock_metadata_store():
    """Create mock metadata store."""
    store = Mock()
    store.get_stats.return_value = {"total_chunks": 3}

    # Create mock chunks
    mock_chunk = Mock()
    mock_chunk.chunk_id = "chunk_1"
    mock_chunk.source_path = "/path/to/file.py"
    mock_chunk.source_type = Mock(value="code")
    mock_chunk.chunk_text = "test content"
    mock_chunk.metadata = {"key": "value"}
    mock_chunk.parent_section = None

    store.get_chunk.return_value = mock_chunk
    store.get_all_chunks.return_value = [mock_chunk] * 3
    return store


@pytest.fixture
def search_service(
    mock_config, mock_embedder_service, mock_vector_store, mock_metadata_store
):
    """Create SearchService instance."""
    return SearchService(
        mock_config, mock_embedder_service, mock_vector_store, mock_metadata_store
    )


class TestSearchServiceEmptyIndex:
    """Tests for empty index handling."""

    def test_returns_error_for_empty_index(
        self, mock_config, mock_embedder_service, mock_metadata_store
    ):
        """Should return error response when index is empty."""
        empty_vector_store = Mock()
        empty_vector_store.is_empty.return_value = True

        service = SearchService(
            mock_config,
            mock_embedder_service,
            empty_vector_store,
            mock_metadata_store,
        )

        response = service.search("test query")

        assert response.success is False
        assert response.error_code == SearchErrorCode.NO_CONTENT.value
        assert "No indexed content" in response.error
        # Recovery suggestions are stored in performance dict
        assert response.performance is not None
        assert "error_recovery" in response.performance
        assert len(response.performance["error_recovery"]["suggestions"]) > 0

    def test_returns_error_for_none_vector_store(
        self, mock_config, mock_embedder_service, mock_metadata_store
    ):
        """Should return error when vector store is None."""
        service = SearchService(
            mock_config, mock_embedder_service, None, mock_metadata_store
        )

        response = service.search("test query")

        assert response.success is False
        assert response.error_code == SearchErrorCode.NO_CONTENT.value


class TestSearchServiceVectorSearch:
    """Tests for vector-only search."""

    def test_successful_vector_search(self, search_service, mock_embedder_service):
        """Should perform successful vector search."""
        response = search_service.search("test query", top_k=3)

        assert response.success is True
        assert len(response.results) <= 3
        assert response.query == "test query"
        assert response.search_config["hybrid_search"] is False

        # Verify embedder was called
        embedder = mock_embedder_service.get_embedder.return_value
        embedder.embed_single.assert_called_once_with("test query")

    def test_vector_search_returns_results(self, search_service):
        """Should return properly formatted results."""
        response = search_service.search("test query")

        assert len(response.results) > 0
        result = response.results[0]
        assert hasattr(result, "chunk_id")
        assert hasattr(result, "text")
        assert hasattr(result, "relevance_score")
        assert hasattr(result, "scores")
        assert hasattr(result, "source_info")

    def test_vector_search_respects_top_k(self, search_service, mock_vector_store):
        """Should respect top_k parameter."""
        # Return more results than requested
        mock_vector_store.search.return_value = (
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([0, 1, 2, 3, 4]),
        )

        response = search_service.search("test query", top_k=2)

        assert len(response.results) <= 2

    def test_vector_search_converts_distance_to_similarity(self, search_service):
        """Should convert distance to similarity score."""
        response = search_service.search("test query")

        # Distance 0.2 should convert to similarity 0.8
        result = response.results[0]
        assert 0.0 <= result.relevance_score <= 1.0


class TestSearchServiceHybridSearch:
    """Tests for hybrid search."""

    def test_hybrid_search_when_enabled(
        self, mock_config, mock_embedder_service, mock_vector_store, mock_metadata_store
    ):
        """Should use hybrid search when enabled."""
        mock_config.use_hybrid_search = True

        mock_hybrid = Mock()
        mock_hybrid.get_stats.return_value = {"indexed_documents": 3}
        mock_hybrid_result = Mock()
        mock_hybrid_result.chunk_id = "chunk_1"
        mock_hybrid_result.source_path = "/path/to/file.py"
        mock_hybrid_result.source_type = "code"
        mock_hybrid_result.text = "test"
        mock_hybrid_result.vector_score = 0.8
        mock_hybrid_result.keyword_score = 0.6
        mock_hybrid_result.combined_score = 0.7
        mock_hybrid_result.metadata = {"key": "value"}
        mock_hybrid.search.return_value = [mock_hybrid_result]

        service = SearchService(
            mock_config,
            mock_embedder_service,
            mock_vector_store,
            mock_metadata_store,
            hybrid_search=mock_hybrid,
        )

        response = service.search("test query")

        assert response.success is True
        mock_hybrid.search.assert_called_once()
        assert response.search_config["hybrid_search"] is True

    def test_hybrid_search_builds_index_if_needed(
        self, mock_config, mock_embedder_service, mock_vector_store, mock_metadata_store
    ):
        """Should build hybrid index if document count doesn't match."""
        mock_config.use_hybrid_search = True

        mock_hybrid = Mock()
        # Stats don't match - should trigger rebuild
        mock_hybrid.get_stats.return_value = {"indexed_documents": 0}
        mock_hybrid.search.return_value = []

        service = SearchService(
            mock_config,
            mock_embedder_service,
            mock_vector_store,
            mock_metadata_store,
            hybrid_search=mock_hybrid,
        )

        service.search("test query")

        # Should have called add_documents to build index
        mock_hybrid.add_documents.assert_called_once()

    def test_hybrid_search_fallback_on_error(
        self, mock_config, mock_embedder_service, mock_vector_store, mock_metadata_store
    ):
        """Should fall back to vector search if hybrid fails."""
        mock_config.use_hybrid_search = True

        mock_hybrid = Mock()
        mock_hybrid.get_stats.return_value = {"indexed_documents": 3}
        mock_hybrid.search.side_effect = RuntimeError("Hybrid failed")

        service = SearchService(
            mock_config,
            mock_embedder_service,
            mock_vector_store,
            mock_metadata_store,
            hybrid_search=mock_hybrid,
        )

        response = service.search("test query")

        # Should still succeed with vector-only results
        assert response.success is True
        assert len(response.results) > 0


class TestSearchServiceErrorHandling:
    """Tests for error handling."""

    def test_handles_embedder_error(self, search_service, mock_embedder_service):
        """Should handle embedder errors gracefully."""
        embedder = mock_embedder_service.get_embedder.return_value
        embedder.embed_single.side_effect = RuntimeError("Embedding failed")

        response = search_service.search("test query")

        assert response.success is False
        assert response.error_code == SearchErrorCode.SEARCH_ERROR.value
        assert "failed" in response.error.lower()

    def test_handles_vector_store_error(self, search_service, mock_vector_store):
        """Should handle vector store errors."""
        mock_vector_store.search.side_effect = RuntimeError("Search failed")

        response = search_service.search("test query")

        assert response.success is False
        assert response.error_code == SearchErrorCode.SEARCH_ERROR.value

    def test_handles_missing_chunks(self, search_service, mock_metadata_store):
        """Should handle missing chunks gracefully."""
        mock_metadata_store.get_chunk.return_value = None

        response = search_service.search("test query")

        # Should still succeed but with no results
        assert response.success is True
        assert len(response.results) == 0


class TestSearchServiceDisplayFormats:
    """Tests for different display formats."""

    def test_readable_format(self, search_service):
        """Should format results as readable text."""
        response = search_service.search("test query", display_format="readable")

        assert response.display_format == "readable"
        assert response.formatted_output is not None

    def test_structured_format(self, search_service):
        """Should return structured format."""
        response = search_service.search("test query", display_format="structured")

        assert response.display_format == "structured"
        # Structured format shouldn't have formatted_output
        assert response.formatted_output is None

    def test_summary_format(self, search_service):
        """Should format results as summary."""
        response = search_service.search("test query", display_format="summary")

        assert response.display_format == "summary"
        assert response.formatted_output is not None


class TestSearchServiceResultEnhancement:
    """Tests for result enhancement."""

    @patch("pycontextify.search.enhance_search_results_with_ranking")
    def test_enhances_results_with_ranking(self, mock_enhance, search_service):
        """Should enhance results with ranking information."""
        mock_enhance.return_value = []

        search_service.search("test query")

        mock_enhance.assert_called_once()
        call_args = mock_enhance.call_args[1]
        assert call_args["query"] == "test query"
        assert call_args["include_explanations"] is True
        assert call_args["include_confidence"] is True


class TestSearchServicePerformance:
    """Tests for performance tracking."""

    def test_includes_performance_info(self, search_service):
        """Should include performance information in response."""
        response = search_service.search("test query")

        assert response.performance is not None
        # Performance is a dict, not an object
        assert isinstance(response.performance, dict)

    def test_tracks_search_mode(self, search_service, mock_config):
        """Should track whether search used hybrid or vector mode."""
        response = search_service.search("test query")

        # Default is vector-only
        assert "vector" in str(response.performance).lower()


class TestSearchServiceSourceInfo:
    """Tests for source information extraction."""

    def test_creates_source_info_for_valid_path(
        self, search_service, mock_metadata_store
    ):
        """Should create source info with file metadata."""
        response = search_service.search("test query")

        result = response.results[0]
        assert "file_path" in result.source_info
        assert "source_type" in result.source_info

    def test_handles_invalid_path_gracefully(self, search_service, mock_metadata_store):
        """Should handle invalid paths without crashing."""
        mock_chunk = mock_metadata_store.get_chunk.return_value
        mock_chunk.source_path = "/nonexistent/path.py"

        response = search_service.search("test query")

        # Should still succeed
        assert response.success is True
        result = response.results[0]
        assert "file_path" in result.source_info

    def test_includes_section_title_when_available(
        self, search_service, mock_metadata_store
    ):
        """Should include section title in source info."""
        mock_chunk = mock_metadata_store.get_chunk.return_value
        mock_chunk.parent_section = "Test Section"

        response = search_service.search("test query")

        result = response.results[0]
        assert "section_title" in result.source_info
        assert result.source_info["section_title"] == "Test Section"


class TestSearchServiceConfiguration:
    """Tests for search configuration."""

    def test_includes_search_config_in_response(self, search_service):
        """Should include search configuration in response."""
        response = search_service.search("test query")

        assert response.search_config is not None
        assert "hybrid_search" in response.search_config
        assert "embedding_provider" in response.search_config
        assert "embedding_model" in response.search_config

    def test_config_reflects_hybrid_setting(
        self, mock_config, mock_embedder_service, mock_vector_store, mock_metadata_store
    ):
        """Should reflect hybrid search setting in config."""
        mock_config.use_hybrid_search = True

        service = SearchService(
            mock_config, mock_embedder_service, mock_vector_store, mock_metadata_store
        )

        response = service.search("test query")

        assert response.search_config["hybrid_search"] is True
