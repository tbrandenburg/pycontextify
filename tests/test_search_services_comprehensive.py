"""Comprehensive tests for search_services.py module.

This module focuses on testing the search service classes and their interactions
to improve code coverage.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

# Disable sentence transformer mocking for all tests in this module
pytestmark = pytest.mark.no_mock_st

from pycontextify.index.search_services import (
    HybridSearchService,
    RerankingService,
    SearchCandidate,
    SearchOrchestrator,
    VectorSearchService,
)


class TestSearchCandidate:
    """Test SearchCandidate class methods."""

    def test_search_candidate_creation(self):
        """Test basic SearchCandidate creation."""
        candidate = SearchCandidate(
            chunk_id="test-chunk-1",
            source_path="/test/file.txt",
            source_type="document",
            text="Test content",
            relevance_score=0.85,
        )

        assert candidate.chunk_id == "test-chunk-1"
        assert candidate.source_path == "/test/file.txt"
        assert candidate.source_type == "document"
        assert candidate.text == "Test content"
        assert candidate.relevance_score == 0.85
        assert candidate.metadata == {}
        assert candidate.final_score == 0.85

        # Check default scoring fields
        assert candidate.vector_score is None
        assert candidate.keyword_score is None
        assert candidate.combined_score is None
        assert candidate.rerank_score is None
        assert candidate.original_score is None

    def test_search_candidate_with_metadata(self):
        """Test SearchCandidate creation with metadata."""
        metadata = {"word_count": 50, "section": "introduction"}
        candidate = SearchCandidate(
            chunk_id="test-chunk-1",
            source_path="/test/file.txt",
            source_type="document",
            text="Test content",
            relevance_score=0.85,
            metadata=metadata,
        )

        assert candidate.metadata == metadata
        assert candidate.metadata["word_count"] == 50
        assert candidate.metadata["section"] == "introduction"


class TestVectorSearchService:
    """Test VectorSearchService methods."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for VectorSearchService."""
        vector_store = Mock()
        metadata_store = Mock()
        embedder = Mock()
        config = Mock()
        return vector_store, metadata_store, embedder, config

    @pytest.fixture
    def vector_service(self, mock_components):
        """Create VectorSearchService instance with mocked components."""
        vector_store, metadata_store, embedder, config = mock_components
        return VectorSearchService(vector_store, metadata_store, embedder, config)

    def test_vector_service_initialization(self, mock_components):
        """Test VectorSearchService initialization."""
        vector_store, metadata_store, embedder, config = mock_components
        service = VectorSearchService(vector_store, metadata_store, embedder, config)

        assert service.vector_store == vector_store
        assert service.metadata_store == metadata_store
        assert service.embedder == embedder
        assert service.config == config

    def test_search_success(self, vector_service, mock_components):
        """Test successful vector search."""
        vector_store, metadata_store, embedder, config = mock_components

        # Setup mock responses
        embedder.embed_single.return_value = [0.1, 0.2, 0.3]  # Mock embedding
        vector_store.search.return_value = ([0.2, 0.4], [1, 2])  # distances, indices

        # Mock chunk objects
        mock_chunk1 = Mock()
        mock_chunk1.chunk_id = "chunk-1"
        mock_chunk1.source_path = "/test/file1.txt"
        mock_chunk1.source_type = "document"
        mock_chunk1.chunk_text = "Test content 1"
        mock_chunk1.metadata = {"section": "intro"}

        mock_chunk2 = Mock()
        mock_chunk2.chunk_id = "chunk-2"
        mock_chunk2.source_path = "/test/file2.txt"
        mock_chunk2.source_type = "document"
        mock_chunk2.chunk_text = "Test content 2"
        mock_chunk2.metadata = {"section": "body"}

        # Setup _get_chunk_by_faiss_id behavior
        def get_chunk_side_effect(faiss_id):
            if faiss_id == 1:
                return mock_chunk1
            elif faiss_id == 2:
                return mock_chunk2
            return None

        vector_service._get_chunk_by_faiss_id = Mock(side_effect=get_chunk_side_effect)

        # Perform search
        candidates, timing_info = vector_service.search("test query", top_k=2)

        # Verify results
        assert len(candidates) == 2
        assert isinstance(timing_info, dict)
        assert "embedding_time" in timing_info
        assert "vector_time" in timing_info

        # Check first candidate
        assert candidates[0].chunk_id == "chunk-1"
        assert candidates[0].source_path == "/test/file1.txt"
        assert candidates[0].text == "Test content 1"
        assert candidates[0].relevance_score == 0.8  # 1.0 - 0.2
        assert candidates[0].vector_score == 0.8

        # Check second candidate
        assert candidates[1].chunk_id == "chunk-2"
        assert candidates[1].source_path == "/test/file2.txt"
        assert candidates[1].text == "Test content 2"
        assert candidates[1].relevance_score == 0.6  # 1.0 - 0.4
        assert candidates[1].vector_score == 0.6

        # Verify method calls
        embedder.embed_single.assert_called_once_with("test query")
        vector_store.search.assert_called_once()

    def test_get_chunk_by_faiss_id_success(self, vector_service, mock_components):
        """Test successful chunk retrieval by FAISS ID."""
        vector_store, metadata_store, embedder, config = mock_components

        mock_chunk = Mock()
        mock_chunk.chunk_id = "chunk-1"
        metadata_store.get_all_chunks.return_value = [mock_chunk]

        result = vector_service._get_chunk_by_faiss_id(0)

        assert result == mock_chunk
        metadata_store.get_all_chunks.assert_called_once()

    def test_get_chunk_by_faiss_id_fallback(self, vector_service, mock_components):
        """Test chunk retrieval fallback to chunk_id lookup."""
        vector_store, metadata_store, embedder, config = mock_components

        mock_chunk = Mock()
        mock_chunk.chunk_id = "5"

        metadata_store.get_all_chunks.return_value = None
        metadata_store.get_chunk_by_chunk_id.return_value = mock_chunk

        result = vector_service._get_chunk_by_faiss_id(5)

        assert result == mock_chunk
        metadata_store.get_chunk_by_chunk_id.assert_called_once_with("5")

    def test_get_chunk_by_faiss_id_error_handling(
        self, vector_service, mock_components
    ):
        """Test chunk retrieval error handling."""
        vector_store, metadata_store, embedder, config = mock_components

        metadata_store.get_all_chunks.side_effect = AttributeError("Test error")

        result = vector_service._get_chunk_by_faiss_id(0)

        assert result is None


class TestHybridSearchService:
    """Test HybridSearchService methods."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for HybridSearchService."""
        hybrid_search = Mock()
        metadata_store = Mock()
        config = Mock()
        config.use_hybrid_search = True
        return hybrid_search, metadata_store, config

    @pytest.fixture
    def hybrid_service(self, mock_components):
        """Create HybridSearchService instance with mocked components."""
        hybrid_search, metadata_store, config = mock_components
        return HybridSearchService(hybrid_search, metadata_store, config)

    def test_hybrid_service_initialization(self, mock_components):
        """Test HybridSearchService initialization."""
        hybrid_search, metadata_store, config = mock_components
        service = HybridSearchService(hybrid_search, metadata_store, config)

        assert service.hybrid_search == hybrid_search
        assert service.metadata_store == metadata_store
        assert service.config == config

    def test_enhance_disabled(self, hybrid_service, mock_components):
        """Test enhancement when hybrid search is disabled."""
        hybrid_search, metadata_store, config = mock_components
        config.use_hybrid_search = False

        vector_candidates = [SearchCandidate("1", "/test.txt", "doc", "text", 0.9)]

        result_candidates, timing_info = hybrid_service.enhance(
            "query", vector_candidates, top_k=1
        )

        assert result_candidates == vector_candidates
        assert timing_info == {}

    def test_enhance_no_hybrid_search(self):
        """Test enhancement when hybrid_search is None."""
        service = HybridSearchService(None, Mock(), Mock())

        vector_candidates = [SearchCandidate("1", "/test.txt", "doc", "text", 0.9)]

        result_candidates, timing_info = service.enhance(
            "query", vector_candidates, top_k=1
        )

        assert result_candidates == vector_candidates
        assert timing_info == {}

    def test_enhance_success(self, hybrid_service, mock_components):
        """Test successful hybrid enhancement."""
        hybrid_search, metadata_store, config = mock_components

        # Mock _ensure_hybrid_search_index
        hybrid_service._ensure_hybrid_search_index = Mock()

        # Mock hybrid search results
        mock_result = Mock()
        mock_result.chunk_id = "enhanced-1"
        mock_result.source_path = "/test/enhanced.txt"
        mock_result.source_type = "document"
        mock_result.text = "Enhanced content"
        mock_result.combined_score = 0.95
        mock_result.vector_score = 0.8
        mock_result.keyword_score = 0.9
        mock_result.metadata = {"enhanced": True}

        hybrid_search.search.return_value = [mock_result]

        vector_candidates = [SearchCandidate("1", "/test.txt", "doc", "text", 0.8)]

        result_candidates, timing_info = hybrid_service.enhance(
            "test query", vector_candidates, top_k=1
        )

        # Verify results
        assert len(result_candidates) == 1
        assert "keyword_time" in timing_info

        enhanced = result_candidates[0]
        assert enhanced.chunk_id == "enhanced-1"
        assert enhanced.source_path == "/test/enhanced.txt"
        assert enhanced.relevance_score == 0.95
        assert enhanced.vector_score == 0.8
        assert enhanced.keyword_score == 0.9
        assert enhanced.combined_score == 0.95
        assert enhanced.final_score == 0.95

        hybrid_service._ensure_hybrid_search_index.assert_called_once()
        hybrid_search.search.assert_called_once()

    def test_enhance_no_results_fallback(self, hybrid_service, mock_components):
        """Test fallback when hybrid search returns no results."""
        hybrid_search, metadata_store, config = mock_components

        hybrid_service._ensure_hybrid_search_index = Mock()
        hybrid_search.search.return_value = []

        vector_candidates = [SearchCandidate("1", "/test.txt", "doc", "text", 0.8)]

        result_candidates, timing_info = hybrid_service.enhance(
            "query", vector_candidates, top_k=1
        )

        assert result_candidates == vector_candidates
        assert "keyword_time" in timing_info

    def test_enhance_exception_handling(self, hybrid_service, mock_components):
        """Test exception handling during enhancement."""
        hybrid_search, metadata_store, config = mock_components

        hybrid_service._ensure_hybrid_search_index = Mock()
        hybrid_search.search.side_effect = Exception("Search error")

        vector_candidates = [SearchCandidate("1", "/test.txt", "doc", "text", 0.8)]

        result_candidates, timing_info = hybrid_service.enhance(
            "query", vector_candidates, top_k=1
        )

        assert result_candidates == vector_candidates
        assert "hybrid_search_error" in timing_info
        assert timing_info["hybrid_search_error"] == "Search error"


class TestRerankingService:
    """Test RerankingService methods."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for RerankingService."""
        reranker = Mock()
        config = Mock()
        config.use_reranking = True
        return reranker, config

    @pytest.fixture
    def reranking_service(self, mock_components):
        """Create RerankingService instance with mocked components."""
        reranker, config = mock_components
        return RerankingService(reranker, config)

    def test_reranking_service_initialization(self, mock_components):
        """Test RerankingService initialization."""
        reranker, config = mock_components
        service = RerankingService(reranker, config)

        assert service.reranker == reranker
        assert service.config == config

    def test_rerank_disabled(self):
        """Test reranking when disabled."""
        config = Mock()
        config.use_reranking = False
        service = RerankingService(Mock(), config)

        candidates = [SearchCandidate("1", "/test.txt", "doc", "text", 0.8)]

        result_candidates, timing_info = service.rerank("query", candidates, top_k=5)

        assert result_candidates == candidates
        assert timing_info == {}

    def test_rerank_no_reranker(self):
        """Test reranking when reranker is None."""
        service = RerankingService(None, Mock())

        candidates = [SearchCandidate("1", "/test.txt", "doc", "text", 0.8)]

        result_candidates, timing_info = service.rerank("query", candidates, top_k=5)

        assert result_candidates == candidates
        assert timing_info == {}


class TestSearchOrchestrator:
    """Test SearchOrchestrator methods."""

    @pytest.fixture
    def mock_services(self):
        """Create mock search services."""
        vector_service = Mock()
        hybrid_service = Mock()
        reranking_service = Mock()
        config = Mock()
        return vector_service, hybrid_service, reranking_service, config

    @pytest.fixture
    def orchestrator(self, mock_services):
        """Create SearchOrchestrator instance with mocked services."""
        vector_service, hybrid_service, reranking_service, config = mock_services
        return SearchOrchestrator(
            vector_service, hybrid_service, reranking_service, config
        )

    def test_orchestrator_initialization(self, mock_services):
        """Test SearchOrchestrator initialization."""
        vector_service, hybrid_service, reranking_service, config = mock_services
        orchestrator = SearchOrchestrator(
            vector_service, hybrid_service, reranking_service, config
        )

        assert orchestrator.vector_service == vector_service
        assert orchestrator.hybrid_service == hybrid_service
        assert orchestrator.reranking_service == reranking_service
        assert orchestrator.config == config

    def test_orchestrate_search_full_pipeline(self, orchestrator, mock_services):
        """Test full search orchestration pipeline."""
        vector_service, hybrid_service, reranking_service, config = mock_services

        # Configure for full pipeline
        config.use_hybrid_search = True
        config.use_reranking = True

        # Setup mock responses
        vector_candidates = [
            SearchCandidate("1", "/test.txt", "doc", "vector text", 0.8)
        ]
        hybrid_candidates = [
            SearchCandidate("2", "/test.txt", "doc", "hybrid text", 0.9)
        ]
        final_candidates = [
            SearchCandidate("3", "/test.txt", "doc", "reranked text", 0.95)
        ]

        vector_service.search.return_value = (vector_candidates, {"vector_time": 0.1})
        hybrid_service.enhance.return_value = (hybrid_candidates, {"hybrid_time": 0.2})
        reranking_service.rerank.return_value = (final_candidates, {"rerank_time": 0.3})

        # Perform orchestration
        results, total_timing = orchestrator.search("test query", top_k=10)

        # Verify results
        assert results == final_candidates
        assert "vector_time" in total_timing
        assert "hybrid_time" in total_timing
        assert "rerank_time" in total_timing
        assert "total_time" in total_timing

        # Verify service calls
        vector_service.search.assert_called_once_with(
            "test query", 30
        )  # top_k * 3 for reranking
        hybrid_service.enhance.assert_called_once_with(
            "test query", vector_candidates, 10
        )
        reranking_service.rerank.assert_called_once_with(
            "test query", hybrid_candidates, 10
        )

    def test_orchestrate_search_minimal_pipeline(self, orchestrator, mock_services):
        """Test search orchestration with minimal pipeline (vector only)."""
        vector_service, hybrid_service, reranking_service, config = mock_services

        # Configure for minimal pipeline
        config.use_hybrid_search = False
        config.use_reranking = False

        # Setup vector-only response
        vector_candidates = [
            SearchCandidate("1", "/test.txt", "doc", "vector text", 0.8)
        ]
        vector_service.search.return_value = (vector_candidates, {"vector_time": 0.1})

        # Other services return unchanged results
        hybrid_service.enhance.return_value = (vector_candidates, {})
        reranking_service.rerank.return_value = (vector_candidates, {})

        # Perform orchestration
        results, total_timing = orchestrator.search("test query", top_k=5)

        # Verify results
        assert results == vector_candidates
        assert "vector_time" in total_timing
        assert "total_time" in total_timing

        vector_service.search.assert_called_once_with("test query", 5)
