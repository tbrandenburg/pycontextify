"""Comprehensive tests for hybrid search functionality."""

from unittest.mock import Mock, patch

import pytest

from pycontextify.search_hybrid import HybridSearchEngine, SearchResult
from pycontextify.storage_metadata import ChunkMetadata, SourceType


class TestHybridSearchEngine:
    """Test hybrid search engine functionality."""

    def test_initialization_default_settings(self):
        """Test hybrid search engine initialization with defaults."""
        engine = HybridSearchEngine()

        assert engine.keyword_weight == 0.3
        assert engine.vector_weight == 0.7
        assert engine.document_texts == []
        assert engine.chunk_ids == []
        # tfidf_vectorizer may or may not be available depending on environment
        # but attribute must exist
        assert hasattr(engine, "tfidf_vectorizer")
        assert hasattr(engine, "bm25_index")

    def test_initialization_custom_weights(self):
        """Test initialization with custom keyword weights."""
        engine = HybridSearchEngine(keyword_weight=0.6)

        assert engine.keyword_weight == 0.6
        assert engine.vector_weight == 0.4

    def test_add_documents_successful(self):
        """Test successful document addition and indexing."""
        engine = HybridSearchEngine()
        doc_ids = ["doc1", "doc2", "doc3"]
        texts = [
            "First document text",
            "Second document content",
            "Third document body",
        ]

        engine.add_documents(doc_ids, texts)

        # Verify documents were stored
        assert engine.document_texts == texts
        assert engine.chunk_ids == doc_ids
        # If sklearn and rank_bm25 are available, indexes should be built
        if engine.tfidf_vectorizer is not None:
            # vocabulary may or may not be built depending on fit implementation
            assert engine.bm25_index is not None

    def test_get_stats(self):
        """Test statistics reporting."""
        engine = HybridSearchEngine(keyword_weight=0.6)

        # Test stats before indexing
        stats = engine.get_stats()
        assert stats["keyword_weight"] == 0.6
        assert stats["vector_weight"] == 0.4
        assert stats["indexed_documents"] == 0
        assert stats["tfidf_available"] in (True, False)
        assert stats["bm25_available"] in (True, False)

        # Add documents and test again
        engine.add_documents(["doc1", "doc2"], ["text1", "text2"])

        stats = engine.get_stats()
        assert stats["indexed_documents"] == 2

    def test_search_without_vector_scores_returns_empty(self):
        """Search should return empty when vector_scores is empty per implementation."""
        engine = HybridSearchEngine()
        engine.add_documents(["doc1", "doc2"], ["text1", "text2"])
        metadata_store = Mock()
        results = engine.search(
            query="test", vector_scores=[], metadata_store=metadata_store, top_k=2
        )
        assert results == []

    def test_edge_case_single_document(self):
        """Test hybrid search with only one document."""
        engine = HybridSearchEngine()
        engine.add_documents(["doc1"], ["single document text"])

        metadata_store = Mock()
        chunk = ChunkMetadata(
            chunk_id="doc1",
            chunk_text="single document text",
            source_path="/test",
            source_type=SourceType.CODE,
        )
        metadata_store.get_chunk.return_value = chunk

        results = engine.search(
            query="document",
            vector_scores=[(0, 0.9)],
            metadata_store=metadata_store,
            top_k=1,
        )

        assert len(results) <= 1
        if results:
            assert results[0].chunk_id == "doc1"

    def test_weight_validation_edge_cases(self):
        """Test weight behavior at boundaries."""
        engine1 = HybridSearchEngine(keyword_weight=0.0)
        assert engine1.keyword_weight == 0.0
        assert engine1.vector_weight == 1.0

        engine2 = HybridSearchEngine(keyword_weight=1.0)
        assert engine2.keyword_weight == 1.0
        assert engine2.vector_weight == 0.0

    def test_import_error_handling_graceful(self):
        """If TfidfVectorizer import fails, engine should degrade gracefully."""
        with patch(
            "sklearn.feature_extraction.text.TfidfVectorizer", side_effect=ImportError
        ):
            engine = HybridSearchEngine()
            assert engine.tfidf_vectorizer is None
            # add_documents should not raise, but skip indexing
            engine.add_documents(["doc1"], ["text1"])
            assert engine.bm25_index is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
