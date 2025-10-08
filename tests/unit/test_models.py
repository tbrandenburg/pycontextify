"""Comprehensive tests for models.py utility functions and methods.

This module focuses on testing utility functions, constructors, formatters,
and other easily testable functionality in models.py to improve code coverage.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Disable sentence transformer mocking for all tests in this module
pytestmark = pytest.mark.no_mock_st

from pycontextify.search_models import (
    SearchErrorCode,
    SearchResponse,
    SearchResult,
    create_search_performance_info,
    create_search_provenance,
    create_structured_metadata,
    create_structured_position,
    create_structured_scores,
)


class TestSearchErrorCode:
    """Test SearchErrorCode enum values."""

    def test_error_code_values(self):
        """Test that error codes have expected string values."""
        assert SearchErrorCode.NO_CONTENT.value == "NO_CONTENT"
        assert SearchErrorCode.INVALID_QUERY.value == "INVALID_QUERY"
        assert SearchErrorCode.EMBEDDER_UNAVAILABLE.value == "EMBEDDER_UNAVAILABLE"
        assert SearchErrorCode.TIMEOUT.value == "TIMEOUT"
        assert SearchErrorCode.SEARCH_ERROR.value == "SEARCH_ERROR"


class TestSearchResult:
    """Test SearchResult data class methods."""

    def test_search_result_creation(self):
        """Test basic SearchResult creation."""
        result = SearchResult(
            chunk_id="test-chunk-1",
            source_path="/test/file.txt",
            source_type="document",
            text="Test content",
            relevance_score=0.85,
        )
        assert result.chunk_id == "test-chunk-1"
        assert result.source_path == "/test/file.txt"
        assert result.source_type == "document"
        assert result.text == "Test content"
        assert result.relevance_score == 0.85
        assert result.position is None
        assert result.scores is None

    def test_search_result_to_dict(self):
        """Test SearchResult.to_dict() method."""
        result = SearchResult(
            chunk_id="test-chunk-1",
            source_path="/test/file.txt",
            source_type="document",
            text="Test content",
            relevance_score=0.85,
            scores={"vector": 0.8, "keyword": 0.9},
            metadata={"word_count": 50},
        )
        result_dict = result.to_dict()

        # Check required fields are present
        assert result_dict["chunk_id"] == "test-chunk-1"
        assert result_dict["source_path"] == "/test/file.txt"
        assert result_dict["source_type"] == "document"
        assert result_dict["text"] == "Test content"
        assert result_dict["relevance_score"] == 0.85
        assert result_dict["scores"] == {"vector": 0.8, "keyword": 0.9}
        assert result_dict["metadata"] == {"word_count": 50}

        # Check None fields are not present
        assert "position" not in result_dict
        assert "context" not in result_dict

    def test_search_result_with_all_fields(self):
        """Test SearchResult with all optional fields filled."""
        result = SearchResult(
            chunk_id="test-chunk-1",
            source_path="/test/file.txt",
            source_type="document",
            text="Test content",
            relevance_score=0.85,
            position={"start_char": 0, "end_char": 100},
            scores={"vector": 0.8, "keyword": 0.9},
            metadata={"word_count": 50},
            context={"relationships": {"references": ["Entity1"]}},
            provenance={"features_used": ["vector"]},
            rank=1,
            relevance_explanation={"match_type": "exact"},
            confidence_scores={"overall": 0.9},
            source_info={"document_title": "Test Document"},
        )
        result_dict = result.to_dict()

        # All fields should be present
        assert len(result_dict) == 14  # All non-None fields


class TestSearchResponse:
    """Test SearchResponse methods and class methods."""

    def test_search_response_creation(self):
        """Test basic SearchResponse creation."""
        results = [
            SearchResult(
                chunk_id="test-1",
                source_path="/test.txt",
                source_type="document",
                text="content",
                relevance_score=0.85,
            )
        ]

        response = SearchResponse(
            success=True,
            query="test query",
            search_config={"top_k": 5},
            results=results,
            total_results=1,
        )

        assert response.success is True
        assert response.query == "test query"
        assert response.search_config == {"top_k": 5}
        assert len(response.results) == 1
        assert response.total_results == 1
        assert response.api_version == "3.2"
        assert response.display_format == "readable"

    def test_search_response_to_dict(self):
        """Test SearchResponse.to_dict() method."""
        results = [
            SearchResult(
                chunk_id="test-1",
                source_path="/test.txt",
                source_type="document",
                text="content",
                relevance_score=0.85,
            )
        ]

        response = SearchResponse(
            success=True,
            query="test query",
            search_config={"top_k": 5},
            results=results,
            total_results=1,
            performance={"search_time_ms": 100},
            error="test error",
            error_code="TEST_ERROR",
        )

        response_dict = response.to_dict()

        assert response_dict["success"] is True
        assert response_dict["query"] == "test query"
        assert response_dict["search_config"] == {"top_k": 5}
        assert len(response_dict["results"]) == 1
        assert response_dict["total_results"] == 1
        assert response_dict["api_version"] == "3.2"
        assert response_dict["performance"] == {"search_time_ms": 100}
        assert response_dict["error"] == "test error"
        assert response_dict["error_code"] == "TEST_ERROR"
        assert response_dict["display_format"] == "readable"

    def test_create_success_class_method(self):
        """Test SearchResponse.create_success() class method."""
        results = [
            SearchResult(
                chunk_id="test-1",
                source_path="/test.txt",
                source_type="document",
                text="content",
                relevance_score=0.85,
            )
        ]

        response = SearchResponse.create_success(
            query="test query",
            results=results,
            search_config={"top_k": 5},
            performance={"search_time_ms": 100},
            query_analysis={"intent": "lookup"},
        )

        assert response.success is True
        assert response.query == "test query"
        assert response.results == results
        assert response.total_results == 1
        assert response.performance == {"search_time_ms": 100}
        assert response.query_analysis == {"intent": "lookup"}
        assert response.error is None

    def test_create_error_class_method(self):
        """Test SearchResponse.create_error() class method."""
        response = SearchResponse.create_error(
            query="test query",
            error="Search failed",
            error_code="SEARCH_ERROR",
            search_config={"top_k": 5},
            recovery_suggestions=["Try simpler query", "Check index"],
        )

        assert response.success is False
        assert response.query == "test query"
        assert response.error == "Search failed"
        assert response.error_code == "SEARCH_ERROR"
        assert response.search_config == {"top_k": 5}
        assert len(response.results) == 0
        assert response.total_results == 0

        # Check recovery suggestions in performance
        assert "error_recovery" in response.performance
        assert response.performance["error_recovery"]["suggestions"] == [
            "Try simpler query",
            "Check index",
        ]

    def test_create_error_with_partial_results(self):
        """Test create_error with partial results."""
        partial_results = [
            SearchResult(
                chunk_id="test-1",
                source_path="/test.txt",
                source_type="document",
                text="content",
                relevance_score=0.85,
            )
        ]

        response = SearchResponse.create_error(
            query="test query", error="Partial failure", partial_results=partial_results
        )

        assert response.success is False
        assert len(response.results) == 1
        assert response.total_results == 1

    def test_create_degraded_class_method(self):
        """Test SearchResponse.create_degraded() class method."""
        results = [
            SearchResult(
                chunk_id="test-1",
                source_path="/test.txt",
                source_type="document",
                text="content",
                relevance_score=0.85,
            )
        ]

        response = SearchResponse.create_degraded(
            query="test query",
            results=results,
            search_config={"top_k": 5},
            performance={"search_time_ms": 100},
            degradation_reason="Keyword search failed",
            failed_components=["keyword_search"],
        )

        assert response.success is True  # Still successful with partial results
        assert response.query == "test query"
        assert response.results == results
        assert response.error == "Degraded search: Keyword search failed"
        assert response.error_code == "DEGRADED_SEARCH"

        # Check degradation info in performance
        assert "degradation" in response.performance
        assert response.performance["degradation"]["reason"] == "Keyword search failed"
        assert response.performance["degradation"]["failed_components"] == [
            "keyword_search"
        ]


class TestSearchResponseFormatting:
    """Test SearchResponse formatting methods."""

    def test_format_for_display_readable(self):
        """Test format_for_display with readable format."""
        results = [
            SearchResult(
                chunk_id="test-1",
                source_path="/test/file.txt",
                source_type="document",
                text="This is test content for formatting",
                relevance_score=0.85,
            )
        ]

        response = SearchResponse.create_success(
            query="test query", results=results, search_config={"top_k": 5}
        )

        formatted = response.format_for_display("readable")
        assert isinstance(formatted, str)
        assert "Found 1 result for 'test query'" in formatted
        assert "**Result 1**" in formatted
        assert "This is test content for formatting" in formatted

    def test_format_for_display_summary(self):
        """Test format_for_display with summary format."""
        results = [
            SearchResult(
                chunk_id="test-1",
                source_path="/test/file.txt",
                source_type="document",
                text="This is test content for summary",
                relevance_score=0.85,
            )
        ]

        response = SearchResponse.create_success(
            query="test query", results=results, search_config={"top_k": 5}
        )

        formatted = response.format_for_display("summary")
        assert isinstance(formatted, str)
        assert "1 result for 'test query'" in formatted
        assert "Top result from file.txt" in formatted

    def test_format_for_display_structured(self):
        """Test format_for_display with structured format."""
        results = [
            SearchResult(
                chunk_id="test-1",
                source_path="/test/file.txt",
                source_type="document",
                text="content",
                relevance_score=0.85,
            )
        ]

        response = SearchResponse.create_success(
            query="test query", results=results, search_config={"top_k": 5}
        )

        formatted = response.format_for_display("structured")
        assert isinstance(formatted, dict)
        assert formatted["query"] == "test query"
        assert formatted["success"] is True

    def test_format_for_display_unknown_format(self):
        """Test format_for_display with unknown format defaults to readable."""
        results = [
            SearchResult(
                chunk_id="test-1",
                source_path="/test/file.txt",
                source_type="document",
                text="content",
                relevance_score=0.85,
            )
        ]

        response = SearchResponse.create_success(
            query="test query", results=results, search_config={"top_k": 5}
        )

        formatted = response.format_for_display("unknown")
        assert isinstance(formatted, str)
        assert "Found 1 result for 'test query'" in formatted

    def test_format_readable_error(self):
        """Test readable formatting for error responses."""
        response = SearchResponse.create_error(
            query="test query", error="Something went wrong"
        )

        formatted = response.format_for_display("readable")
        assert "❌ Search failed: Something went wrong" in formatted

    def test_format_readable_no_results(self):
        """Test readable formatting for no results."""
        response = SearchResponse.create_success(
            query="test query", results=[], search_config={"top_k": 5}
        )

        formatted = response.format_for_display("readable")
        assert "🔍 No results found for 'test query'" in formatted

    def test_get_source_icon(self):
        """Test _get_source_icon method."""
        response = SearchResponse.create_success("query", [], {})

        assert response._get_source_icon("document") == "📄"
        assert response._get_source_icon("codebase") == "💻"
        assert response._get_source_icon("pdf") == "📄"
        assert response._get_source_icon("unknown") == "📄"  # default

    def test_get_display_source_name(self):
        """Test _get_display_source_name method."""
        response = SearchResponse.create_success("query", [], {})

        # Test with source_info containing title
        source_info = {"document_title": "My Document"}
        name = response._get_display_source_name("/path/file.txt", source_info)
        assert name == "My Document"

        # Test without source_info - should use filename
        name = response._get_display_source_name("/path/to/file.txt")
        assert name == "file.txt"

        # Test with source_info but no title
        source_info = {"other_field": "value"}
        name = response._get_display_source_name("/path/file.txt", source_info)
        assert name == "file.txt"

    def test_create_content_preview(self):
        """Test _create_content_preview method."""
        response = SearchResponse.create_success("query", [], {})

        # Test short content
        preview = response._create_content_preview("Short text")
        assert preview == "Short text"

        # Test empty content
        preview = response._create_content_preview("")
        assert preview == "[No content available]"

        # Test long content truncation
        long_text = "This is a very long text. " * 50
        preview = response._create_content_preview(long_text, max_length=100)
        assert len(preview) <= 103  # 100 + "..."
        assert preview.endswith("...")

        # Test content with good sentence break
        text_with_sentences = (
            "First sentence. Second sentence. Third sentence continues for a while."
        )
        preview = response._create_content_preview(text_with_sentences, max_length=50)
        # Should break at sentence if possible
        assert "First sentence." in preview


class TestUtilityFunctions:
    """Test standalone utility functions."""

    def test_create_search_performance_info(self):
        """Test create_search_performance_info function."""
        start_time = time.time() - 0.5  # 500ms ago

        perf_info = create_search_performance_info(
            start_time=start_time,
            search_mode="hybrid",
            total_candidates=100,
            vector_time=0.2,
            keyword_time=0.15,
            embedding_time=0.05,
        )

        assert "search_time_ms" in perf_info
        assert perf_info["search_mode"] == "hybrid"
        assert perf_info["total_candidates"] == 100
        assert "timestamp" in perf_info
        assert "components_used" in perf_info
        assert "timing_breakdown" in perf_info

        # Check timing breakdown
        timing = perf_info["timing_breakdown"]
        assert timing["vector_search_ms"] == 200  # 0.2 * 1000
        assert timing["keyword_search_ms"] == 150  # 0.15 * 1000
        assert timing["embedding_ms"] == 50  # 0.05 * 1000

        # Check components used
        components = perf_info["components_used"]
        assert "embedding" in components
        assert "vector_search" in components
        assert "keyword_search" in components

    def test_create_search_performance_info_minimal(self):
        """Test create_search_performance_info with minimal parameters."""
        start_time = time.time() - 0.1  # 100ms ago

        perf_info = create_search_performance_info(
            start_time=start_time, search_mode="vector"
        )

        assert "search_time_ms" in perf_info
        assert perf_info["search_mode"] == "vector"
        assert perf_info["total_candidates"] == 0
        assert perf_info["components_used"] == []
        assert "timing_breakdown" not in perf_info  # No timing data provided

    def test_create_structured_position(self):
        """Test create_structured_position function."""
        # Test with all parameters
        position = create_structured_position(
            start_char=10, end_char=50, line_start=1, line_end=3
        )
        assert position["start_char"] == 10
        assert position["end_char"] == 50
        assert position["line_start"] == 1
        assert position["line_end"] == 3
        assert position["char_length"] == 40  # 50 - 10
        assert position["line_count"] == 3  # max(1, 3 - 1 + 1)

        # Test with partial parameters
        position = create_structured_position(start_char=10)
        assert position["start_char"] == 10
        assert "char_length" not in position  # No end_char provided

        # Test with no parameters
        position = create_structured_position()
        assert position is None

    def test_create_structured_scores(self):
        """Test create_structured_scores function."""
        # Test with all scores
        scores = create_structured_scores(
            vector_score=0.8,
            keyword_score=0.7,
            original_score=0.85,
            combined_score=0.87,
        )
        assert scores["vector"] == 0.8
        assert scores["keyword"] == 0.7
        assert scores["original"] == 0.85
        assert scores["combined"] == 0.87

        # Test with partial scores
        scores = create_structured_scores(vector_score=0.8)
        assert scores["vector"] == 0.8
        assert len(scores) == 1

        # Test with no scores
        scores = create_structured_scores()
        assert scores is None

    def test_create_structured_metadata(self):
        """Test create_structured_metadata function."""
        metadata = create_structured_metadata(
            created_at="2024-01-01T00:00:00Z",
            file_extension=".py",
            word_count=100,
            embedding_provider="openai",
            custom_field="custom_value",
        )

        # Check categorization
        assert "content" in metadata
        assert "source" in metadata
        assert "processing" in metadata

        assert metadata["content"]["word_count"] == 100
        assert metadata["source"]["created_at"] == "2024-01-01T00:00:00Z"
        assert metadata["source"]["file_extension"] == ".py"
        assert metadata["processing"]["embedding_provider"] == "openai"
        assert (
            metadata["custom_field"] == "custom_value"
        )  # Uncategorized fields go to root

        # Test with minimal metadata
        metadata = create_structured_metadata()
        # Should return empty dict if no categories have content
        assert metadata == {}

    def test_create_search_provenance(self):
        """Test create_search_provenance function."""
        provenance = create_search_provenance(
            search_features=["vector", "keyword"],
            ranking_factors={"boost": 1.2},
            search_stage="final",
            confidence=0.85,
        )

        assert provenance["features_used"] == ["vector", "keyword"]
        assert provenance["search_stage"] == "final"
        assert provenance["ranking_factors"] == {"boost": 1.2}
        assert provenance["confidence"] == 0.85
        assert "timestamp" in provenance

        # Test minimal provenance
        provenance = create_search_provenance(search_features=["vector"])
        assert provenance["features_used"] == ["vector"]
        assert provenance["search_stage"] == "final"  # default
        assert "timestamp" in provenance
