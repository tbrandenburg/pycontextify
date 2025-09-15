"""Tests for the embedding system."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from pycontextify.index.embedders import (
    BaseEmbedder,
    EmbedderFactory,
    SentenceTransformersEmbedder,
)
from pycontextify.index.embedders.base import (
    EmbeddingError,
    ModelNotFoundError,
    ProviderNotAvailableError,
)


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing."""

    provider_name = "mock"
    supported_models = ["mock-model"]

    def __init__(self, model_name: str, dimension: int = 384, **kwargs):
        super().__init__(model_name, **kwargs)
        self.dimension = dimension
        self._is_initialized = True

    def embed_texts(self, texts):
        # Return random embeddings
        return np.random.rand(len(texts), self.dimension).astype(np.float32)

    def embed_single(self, text):
        return np.random.rand(self.dimension).astype(np.float32)

    def get_dimension(self):
        return self.dimension

    def get_model_info(self):
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "dimension": self.dimension,
        }

    def is_available(self):
        return True

    def cleanup(self):
        pass


class TestBaseEmbedder:
    """Test the BaseEmbedder abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseEmbedder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbedder("test-model")

    def test_mock_embedder_initialization(self):
        """Test mock embedder initialization."""
        embedder = MockEmbedder("mock-model", dimension=256)

        assert embedder.get_provider_name() == "mock"
        assert embedder.get_model_name() == "mock-model"
        assert embedder.get_dimension() == 256
        assert embedder.is_available()

    def test_embed_texts_validation(self):
        """Test text validation in embed_texts."""
        embedder = MockEmbedder("mock-model")

        # Valid texts should work
        texts = ["hello", "world"]
        embeddings = embedder.embed_texts(texts)
        assert embeddings.shape == (2, 384)

        # Empty list should raise error
        with pytest.raises(EmbeddingError, match="No texts provided"):
            embedder._validate_texts([])

        # Non-string items should raise error
        with pytest.raises(EmbeddingError, match="not a string"):
            embedder._validate_texts(["hello", 123])

        # Empty strings should raise error
        with pytest.raises(EmbeddingError, match="empty or whitespace-only"):
            embedder._validate_texts(["hello", "   "])

    def test_normalize_embeddings(self):
        """Test embedding normalization."""
        embedder = MockEmbedder("mock-model")

        # Create non-normalized embeddings
        embeddings = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        normalized = embedder._normalize_embeddings(embeddings)

        # Check L2 norms are 1.0 (or close due to floating point)
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0)

        # First vector should be [0.6, 0.8]
        assert np.allclose(normalized[0], [0.6, 0.8])

    def test_ensure_float32(self):
        """Test conversion to float32."""
        embedder = MockEmbedder("mock-model")

        # Test different input types
        embeddings_int = np.array([[1, 2], [3, 4]], dtype=np.int32)
        embeddings_float64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        result_int = embedder._ensure_float32(embeddings_int)
        result_float = embedder._ensure_float32(embeddings_float64)

        assert result_int.dtype == np.float32
        assert result_float.dtype == np.float32

    def test_config_summary(self):
        """Test configuration summary generation."""
        embedder = MockEmbedder("mock-model", api_key="secret", batch_size=32)

        summary = embedder.get_config_summary()

        assert summary["provider"] == "mock"
        assert summary["model"] == "mock-model"
        assert "config" in summary

        # Sensitive info should be hidden
        assert "api_key" not in summary["config"]
        assert "batch_size" in summary["config"]

    def test_context_manager(self):
        """Test context manager functionality."""
        with MockEmbedder("mock-model") as embedder:
            assert embedder is not None
            assert embedder.is_available()


class TestEmbedderFactory:
    """Test the EmbedderFactory class."""

    def test_register_custom_provider(self):
        """Test registering custom embedding provider."""
        # Register mock provider
        EmbedderFactory.register_provider("mock", MockEmbedder)

        # Should be able to create it
        embedder = EmbedderFactory.create_embedder("mock", "mock-model", dimension=256)
        assert isinstance(embedder, MockEmbedder)
        assert embedder.get_dimension() == 256

    def test_create_nonexistent_provider(self):
        """Test creating embedder for non-existent provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            EmbedderFactory.create_embedder("nonexistent", "model")

    def test_get_available_providers(self):
        """Test getting available providers."""
        # Register mock provider
        EmbedderFactory.register_provider("mock", MockEmbedder)

        available = EmbedderFactory.get_available_providers()

        # Mock should be available
        assert "mock" in available

        # sentence_transformers might or might not be available depending on installation
        assert isinstance(available, list)

    def test_get_supported_models(self):
        """Test getting supported models for a provider."""
        EmbedderFactory.register_provider("mock", MockEmbedder)

        models = EmbedderFactory.get_supported_models("mock")
        assert "mock-model" in models

        # Test non-existent provider
        with pytest.raises(ValueError, match="Unknown provider"):
            EmbedderFactory.get_supported_models("nonexistent")

    def test_validate_provider_config(self):
        """Test provider configuration validation."""
        # Test sentence_transformers config
        assert EmbedderFactory.validate_provider_config(
            "sentence_transformers", device="cpu", batch_size=32
        )

        with pytest.raises(ValueError, match="Invalid device"):
            EmbedderFactory.validate_provider_config(
                "sentence_transformers", device="invalid"
            )

        with pytest.raises(ValueError, match="Invalid batch_size"):
            EmbedderFactory.validate_provider_config(
                "sentence_transformers", batch_size=-1
            )

        # Test OpenAI config
        assert EmbedderFactory.validate_provider_config("openai", api_key="valid-key")

        with pytest.raises(ValueError, match="api_key is required"):
            EmbedderFactory.validate_provider_config("openai")

        with pytest.raises(ValueError, match="appears to be invalid"):
            EmbedderFactory.validate_provider_config("openai", api_key="x")

    def test_get_provider_info(self):
        """Test getting provider information."""
        EmbedderFactory.register_provider("mock", MockEmbedder)

        info = EmbedderFactory.get_provider_info("mock")

        assert info["name"] == "mock"
        assert info["class"] == "MockEmbedder"
        assert "mock-model" in info["supported_models"]
        assert "available" in info

    def test_list_all_providers(self):
        """Test listing all providers."""
        EmbedderFactory.register_provider("mock", MockEmbedder)

        providers = EmbedderFactory.list_all_providers()

        assert "mock" in providers
        assert isinstance(providers["mock"], dict)

    def test_get_recommendations(self):
        """Test getting provider recommendations."""
        recommendations = EmbedderFactory.get_recommendations()

        assert "development" in recommendations
        assert "production_quality" in recommendations
        assert "local_privacy" in recommendations

        # Each recommendation should have provider, model, and reason
        for rec in recommendations.values():
            assert "provider" in rec
            assert "model" in rec
            assert "reason" in rec


class TestSentenceTransformersEmbedder:
    """Test the SentenceTransformersEmbedder class."""

    @pytest.mark.skipif(
        not pytest.importorskip(
            "sentence_transformers", reason="sentence-transformers not installed"
        ),
        reason="Requires sentence-transformers",
    )
    def test_initialization(self):
        """Test sentence transformers embedder initialization."""
        embedder = SentenceTransformersEmbedder("all-MiniLM-L6-v2")

        assert embedder.provider_name == "sentence_transformers"
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.device == "auto"
        assert embedder.batch_size == 32

    @pytest.mark.skipif(
        not pytest.importorskip(
            "sentence_transformers", reason="sentence-transformers not installed"
        ),
        reason="Requires sentence-transformers",
    )
    def test_custom_configuration(self):
        """Test custom configuration options."""
        embedder = SentenceTransformersEmbedder(
            "all-MiniLM-L6-v2", device="cpu", batch_size=16, normalize_embeddings=False
        )

        assert embedder.device == "cpu"
        assert embedder.batch_size == 16
        assert embedder.normalize_embeddings is False

    def test_provider_not_available(self):
        """Test behavior when sentence-transformers is not available."""
        with patch("importlib.import_module", side_effect=ImportError):
            with pytest.raises(
                ProviderNotAvailableError, match="sentence-transformers not installed"
            ):
                embedder = SentenceTransformersEmbedder("all-MiniLM-L6-v2")
                embedder._ensure_model_loaded()

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_loading_error(self, mock_sentence_transformer):
        """Test handling of model loading errors."""
        mock_sentence_transformer.side_effect = Exception("Model not found")

        embedder = SentenceTransformersEmbedder("invalid-model")

        with pytest.raises(EmbeddingError, match="Failed to load model"):
            embedder._ensure_model_loaded()

    @patch("sentence_transformers.SentenceTransformer")
    def test_embedding_operations(self, mock_sentence_transformer):
        """Test embedding operations with mocked model."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        mock_sentence_transformer.return_value = mock_model

        embedder = SentenceTransformersEmbedder("all-MiniLM-L6-v2")
        embedder._embedding_dimension = 384

        # Test batch embedding
        texts = ["hello", "world"]
        embeddings = embedder.embed_texts(texts)

        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32
        mock_model.encode.assert_called()

        # Test single embedding
        mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        embedding = embedder.embed_single("hello")

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    @patch("sentence_transformers.SentenceTransformer")
    def test_get_dimension(self, mock_sentence_transformer):
        """Test getting embedding dimension."""
        # Setup mock to return specific dimension
        mock_model = MagicMock()
        test_embedding = np.random.rand(1, 256).astype(np.float32)
        mock_model.encode.return_value = test_embedding
        mock_sentence_transformer.return_value = mock_model

        embedder = SentenceTransformersEmbedder("test-model")

        dimension = embedder.get_dimension()
        assert dimension == 256

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_info(self, mock_sentence_transformer):
        """Test getting model information."""
        mock_model = MagicMock()
        mock_model.max_seq_length = 512
        mock_sentence_transformer.return_value = mock_model

        embedder = SentenceTransformersEmbedder("test-model", device="cpu")
        embedder._embedding_dimension = 384

        info = embedder.get_model_info()

        assert info["provider"] == "sentence_transformers"
        assert info["model_name"] == "test-model"
        assert info["device"] == "cpu"
        assert info["embedding_dimension"] == 384
        assert info["max_seq_length"] == 512

    def test_is_available_without_import(self):
        """Test availability check when sentence-transformers not installed."""
        with patch("importlib.import_module", side_effect=ImportError):
            embedder = SentenceTransformersEmbedder("test-model")
            assert not embedder.is_available()

    @patch("sentence_transformers.SentenceTransformer")
    def test_cleanup(self, mock_sentence_transformer):
        """Test cleanup functionality."""
        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_sentence_transformer.return_value = mock_model

        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            embedder = SentenceTransformersEmbedder("test-model")
            embedder._ensure_model_loaded()

            # Cleanup should clear model and GPU cache
            embedder.cleanup()

            assert embedder._model is None
            assert not embedder._is_initialized

    def test_get_recommended_models(self):
        """Test getting recommended models."""
        recommendations = SentenceTransformersEmbedder.get_recommended_models()

        assert "best_quality" in recommendations
        assert "fastest" in recommendations
        assert "balanced" in recommendations

        # Should return actual model names
        assert "all-mpnet-base-v2" in recommendations.values()
        assert "all-MiniLM-L6-v2" in recommendations.values()

    def test_validate_model_name(self):
        """Test model name validation."""
        # Known supported models should be valid
        assert SentenceTransformersEmbedder.validate_model_name("all-mpnet-base-v2")
        assert SentenceTransformersEmbedder.validate_model_name("all-MiniLM-L6-v2")

        # Models following sentence-transformers patterns should be valid
        assert SentenceTransformersEmbedder.validate_model_name(
            "sentence-transformers/all-mpnet-base-v2"
        )
        assert SentenceTransformersEmbedder.validate_model_name(
            "multi-qa-mpnet-base-dot-v1"
        )
        assert SentenceTransformersEmbedder.validate_model_name(
            "paraphrase-MiniLM-L6-v2"
        )

        # Random strings might not be valid (depends on implementation)
        # This test might need adjustment based on actual validation logic

    @patch("sentence_transformers.SentenceTransformer")
    def test_memory_usage_estimation(self, mock_sentence_transformer):
        """Test memory usage estimation."""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_sentence_transformer.return_value = mock_model

        embedder = SentenceTransformersEmbedder("all-mpnet-base-v2")
        memory_info = embedder.get_memory_usage()

        assert "model_loaded" in memory_info
        assert "estimated_memory_mb" in memory_info

        if memory_info["model_loaded"]:
            # Should have reasonable memory estimate for mpnet
            assert memory_info["estimated_memory_mb"] > 0
