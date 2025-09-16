"""Streamlined reranker tests - core functionality with proper mocking."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from pycontextify.index.reranker import CrossEncoderReranker, RerankResult


@dataclass 
class MockSearchResult:
    """Mock search result for testing."""
    text: str
    score: float = 0.8
    chunk_id: str = "test_chunk"
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestCrossEncoderRerankerStreamlined:
    """Streamlined tests for CrossEncoderReranker functionality."""

    def test_initialization_scenarios(self):
        """Test initialization with different scenarios."""
        # Test successful initialization
        mock_sentence_transformers = MagicMock()
        mock_cross_encoder = MagicMock()
        mock_sentence_transformers.CrossEncoder.return_value = mock_cross_encoder
        
        with patch.dict('sys.modules', {'sentence_transformers': mock_sentence_transformers}):
            reranker = CrossEncoderReranker()
            assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
            assert reranker.model == mock_cross_encoder
            assert reranker.is_available is True

        # Test custom model name
        with patch.dict('sys.modules', {'sentence_transformers': mock_sentence_transformers}):
            custom_model = "cross-encoder/ms-marco-TinyBERT-L-6"
            reranker = CrossEncoderReranker(model_name=custom_model)
            assert reranker.model_name == custom_model

        # Test import error handling
        import sys
        with patch.dict('sys.modules', clear=True):
            if 'sentence_transformers' in sys.modules:
                del sys.modules['sentence_transformers']
            
            reranker = CrossEncoderReranker()
            assert reranker.model is None
            assert reranker.is_available is False

        # Test model load error
        mock_sentence_transformers.CrossEncoder.side_effect = Exception("Model load failed")
        with patch.dict('sys.modules', {'sentence_transformers': mock_sentence_transformers}):
            reranker = CrossEncoderReranker()
            assert reranker.model is None
            assert reranker.is_available is False

    def test_reranking_when_unavailable(self):
        """Test reranking behavior when model is unavailable."""
        reranker = CrossEncoderReranker()
        reranker.model = None
        reranker.is_available = False
        
        results = [MockSearchResult("text 1"), MockSearchResult("text 2")]
        reranked = reranker.rerank("query", results)
        
        # Should return original results converted to RerankResult format
        assert len(reranked) == 2
        assert all(isinstance(r, RerankResult) for r in reranked)
        assert reranked[0].text == "text 1"
        assert reranked[1].text == "text 2"
        
        # Test empty results
        reranked_empty = reranker.rerank("query", [])
        assert len(reranked_empty) == 0

    def test_successful_reranking_operations(self):
        """Test successful reranking operations with proper mocking."""
        # Create a proper mock reranker that appears available
        reranker = CrossEncoderReranker()
        reranker.is_available = True
        
        # Mock the model properly
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.85]
        reranker.model = mock_model
        
        results = [
            MockSearchResult("text 1", score=0.5),
            MockSearchResult("text 2", score=0.8), 
            MockSearchResult("text 3", score=0.6)
        ]
        
        # Test reranking with score combination
        reranked = reranker.rerank("test query", results, combine_scores=True)
        
        assert len(reranked) == 3
        assert all(isinstance(r, RerankResult) for r in reranked)
        
        # Results should be sorted by final score (descending)
        for i in range(len(reranked) - 1):
            assert reranked[i].final_score >= reranked[i + 1].final_score
        
        # Verify predict was called with correct format
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args[0][0]
        expected_pairs = [
            ["test query", "text 1"],
            ["test query", "text 2"], 
            ["test query", "text 3"]
        ]
        assert call_args == expected_pairs
        
        # Test standalone reranker scores (no combination)
        mock_model.reset_mock()
        reranked_standalone = reranker.rerank("test query", results, combine_scores=False)
        
        assert len(reranked_standalone) == 3
        # Should be sorted by reranker score only
        for i in range(len(reranked_standalone) - 1):
            assert reranked_standalone[i].final_score >= reranked_standalone[i + 1].final_score

    def test_top_k_limiting_and_custom_scoring(self):
        """Test top-k limiting and custom score weights."""
        reranker = CrossEncoderReranker()
        reranker.is_available = True
        
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.85, 0.6]
        reranker.model = mock_model
        
        results = [MockSearchResult(f"text {i}", score=0.5) for i in range(4)]
        
        # Test top_k limiting
        reranked = reranker.rerank("test query", results, top_k=2)
        assert len(reranked) == 2
        
        # Test custom score weights
        mock_model.reset_mock()
        mock_model.predict.return_value = [0.9, 0.7]
        
        results_custom = [
            MockSearchResult("text 1", score=0.5),
            MockSearchResult("text 2", score=0.8)
        ]
        
        reranked_custom = reranker.rerank(
            "query",
            results_custom, 
            combine_scores=True,
            score_weight=0.3  # Lower weight for rerank score
        )
        
        assert len(reranked_custom) == 2
        # With score_weight=0.3, original scores should have more influence

    def test_text_and_metadata_extraction(self):
        """Test text and metadata extraction from different result formats."""
        reranker = CrossEncoderReranker()
        
        # Test text extraction from objects
        obj1 = MockSearchResult("content from text attr")
        extracted = reranker._extract_text_from_result(obj1)
        assert extracted == "content from text attr"
        
        obj2 = type('TestObj', (), {'content': 'content from content attr'})()
        extracted = reranker._extract_text_from_result(obj2)
        assert extracted == "content from content attr"
        
        obj3 = type('TestObj', (), {'chunk_text': 'content from chunk_text'})()
        extracted = reranker._extract_text_from_result(obj3)
        assert extracted == "content from chunk_text"
        
        # Test with string
        extracted = reranker._extract_text_from_result("direct string")
        assert extracted == "direct string"
        
        # Test text extraction from dicts
        dict1 = {"text": "text from dict"}
        extracted = reranker._extract_text_from_result(dict1)
        assert extracted == "text from dict"
        
        dict2 = {"content": "content from dict"}
        extracted = reranker._extract_text_from_result(dict2)
        assert extracted == "content from dict"
        
        dict3 = {"chunk_text": "chunk_text from dict"}
        extracted = reranker._extract_text_from_result(dict3)
        assert extracted == "chunk_text from dict"
        
        # Test original score extraction
        obj_score = MockSearchResult("text", score=0.75)
        score = reranker._extract_original_score(obj_score)
        assert score == 0.75
        
        obj_combined = type('TestObj', (), {'combined_score': 0.85})()
        score = reranker._extract_original_score(obj_combined)
        assert score == 0.85
        
        dict_score = {"score": 0.65}
        score = reranker._extract_original_score(dict_score)
        assert score == 0.65

    def test_chunk_id_and_metadata_extraction(self):
        """Test chunk ID and metadata extraction."""
        reranker = CrossEncoderReranker()
        
        # Test chunk ID extraction
        obj_chunk = MockSearchResult("text", chunk_id="chunk123")
        chunk_id = reranker._extract_chunk_id(obj_chunk)
        assert chunk_id == "chunk123"
        
        dict_chunk = {"chunk_id": "dict_chunk"}
        chunk_id = reranker._extract_chunk_id(dict_chunk)
        assert chunk_id == "dict_chunk"
        
        # Test fallback behavior
        obj_no_chunk = type('TestObj', (), {})()
        chunk_id = reranker._extract_chunk_id(obj_no_chunk)
        assert chunk_id == "unknown"
        
        # Test metadata extraction
        obj_meta = MockSearchResult("text", metadata={"key": "value"})
        metadata = reranker._extract_metadata(obj_meta)
        assert metadata == {"key": "value"}
        
        # Test dict with metadata fields
        dict_meta = {
            "text": "content", 
            "embedding_provider": "openai", 
            "file_extension": ".txt"
        }
        metadata = reranker._extract_metadata(dict_meta)
        expected_metadata = {"embedding_provider": "openai", "file_extension": ".txt"}
        assert metadata == expected_metadata

    def test_error_handling(self):
        """Test error handling during reranking."""
        reranker = CrossEncoderReranker()
        reranker.is_available = True
        
        # Mock model that fails during prediction
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        reranker.model = mock_model
        
        results = [MockSearchResult("text 1")]
        reranked = reranker.rerank("query", results)
        
        # Should fall back to original results
        assert len(reranked) == 1
        assert reranked[0].text == "text 1"

    def test_rerank_result_dataclass(self):
        """Test the RerankResult dataclass."""
        result = RerankResult(
            chunk_id="test_chunk",
            text="test text",
            original_score=0.75,
            rerank_score=0.85,
            final_score=0.80,
            source_path="test_source.txt",
            source_type="text",
            metadata={"key": "value"}
        )
        
        assert result.chunk_id == "test_chunk"
        assert result.text == "test text"
        assert result.original_score == 0.75
        assert result.rerank_score == 0.85
        assert result.final_score == 0.80
        assert result.source_path == "test_source.txt"
        assert result.source_type == "text"
        assert result.metadata == {"key": "value"}

    def test_mixed_result_types(self):
        """Test reranking with different types of results."""
        reranker = CrossEncoderReranker()
        reranker.is_available = True
        
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.85]
        reranker.model = mock_model
        
        # Mix of objects, dicts, and strings
        results = [
            MockSearchResult("object text", score=0.5),
            {"text": "dict text", "score": 0.6},
            "plain string"
        ]
        
        reranked = reranker.rerank("query", results)
        
        assert len(reranked) == 3
        assert all(isinstance(r, RerankResult) for r in reranked)
        
        # Check that all different result types were processed
        texts = [r.text for r in reranked]
        assert "object text" in texts
        assert "dict text" in texts 
        assert "plain string" in texts

    def test_utility_methods(self):
        """Test utility and information methods."""
        reranker = CrossEncoderReranker()
        
        # Test model info
        info = reranker.get_model_info()
        expected_keys = {"model_name", "is_available", "model_loaded"}
        assert all(key in info for key in expected_keys)
        assert info["model_name"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        # Test stats
        stats = reranker.get_stats()
        expected_stats_keys = {"model_name", "is_available", "model_loaded"}
        assert all(key in stats for key in expected_stats_keys)
        
        # Test warmup (may succeed or fail depending on model availability)
        result = reranker.warmup()
        assert isinstance(result, bool)  # Should return a boolean
