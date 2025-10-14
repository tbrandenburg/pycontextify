"""Tests for the embedding system."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from pycontextify.embedder import (
    BaseEmbedder,
    EmbeddingError,
    ModelNotFoundError,
    ProviderNotAvailableError,
)
from pycontextify.embedder_factory import EmbedderFactory
from pycontextify.embedder_sentence_transformers_embedder import (
    SentenceTransformersEmbedder,
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

        # Test unknown provider
        with pytest.raises(ValueError, match="Unknown provider"):
            EmbedderFactory.validate_provider_config("nonexistent_provider")

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
        # Device could be auto or cpu depending on system configuration
        assert embedder.device in ["auto", "cpu", "cuda"]
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
        # Create embedder first
        embedder = SentenceTransformersEmbedder("all-MiniLM-L6-v2")

        # Mock the import at the exact location where it's used
        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("sentence-transformers not installed")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(
                ProviderNotAvailableError, match="sentence-transformers not installed"
            ):
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
        # max_seq_length might not always be available
        if "max_seq_length" in info:
            assert info["max_seq_length"] == 512

    @pytest.mark.no_mock_st
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


"""Unit tests for EmbedderService."""

import threading
import time
from unittest.mock import Mock, patch

import pytest

from pycontextify.embedder import EmbedderService


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.get_embedding_config.return_value = {
        "provider": "sentence_transformers",
        "model": "all-mpnet-base-v2",
        "device": "cpu",
    }
    return config


@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    embedder = Mock()
    embedder.get_provider_name.return_value = "sentence_transformers"
    embedder.get_model_name.return_value = "all-mpnet-base-v2"
    embedder.get_dimension.return_value = 768
    embedder.cleanup.return_value = None
    return embedder


@pytest.fixture
def embedder_service(mock_config):
    """Create EmbedderService instance."""
    return EmbedderService(mock_config)


class TestEmbedderServiceInitialization:
    """Tests for EmbedderService initialization."""

    def test_initializes_without_loading_embedder(self, mock_config):
        """Should initialize without loading embedder."""
        service = EmbedderService(mock_config)

        assert service.config == mock_config
        assert service._embedder is None
        assert service._initialized is False
        # Config should not be called yet (lazy loading)
        mock_config.get_embedding_config.assert_not_called()

    def test_is_not_loaded_initially(self, embedder_service):
        """Should report as not loaded initially."""
        assert embedder_service.is_loaded() is False


class TestEmbedderServiceLazyLoading:
    """Tests for lazy loading functionality."""

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_loads_embedder_on_first_access(
        self, mock_factory, embedder_service, mock_embedder, mock_config
    ):
        """Should load embedder on first get_embedder() call."""
        mock_factory.create_embedder.return_value = mock_embedder

        result = embedder_service.get_embedder()

        assert result == mock_embedder
        assert embedder_service.is_loaded() is True
        mock_config.get_embedding_config.assert_called_once()
        mock_factory.create_embedder.assert_called_once_with(
            provider="sentence_transformers",
            model_name="all-mpnet-base-v2",
            device="cpu",
        )

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_reuses_embedder_on_subsequent_calls(
        self, mock_factory, embedder_service, mock_embedder
    ):
        """Should reuse embedder on subsequent calls without recreating."""
        mock_factory.create_embedder.return_value = mock_embedder

        # First call
        result1 = embedder_service.get_embedder()
        # Second call
        result2 = embedder_service.get_embedder()

        assert result1 == result2 == mock_embedder
        # Factory should only be called once
        mock_factory.create_embedder.assert_called_once()

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_loads_embedder_when_getting_dimension(
        self, mock_factory, embedder_service, mock_embedder
    ):
        """Should lazy load embedder when getting dimension."""
        mock_factory.create_embedder.return_value = mock_embedder

        dimension = embedder_service.get_dimension()

        assert dimension == 768
        assert embedder_service.is_loaded() is True
        mock_factory.create_embedder.assert_called_once()

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_loads_embedder_when_getting_provider_name(
        self, mock_factory, embedder_service, mock_embedder
    ):
        """Should lazy load embedder when getting provider name."""
        mock_factory.create_embedder.return_value = mock_embedder

        provider = embedder_service.get_provider_name()

        assert provider == "sentence_transformers"
        assert embedder_service.is_loaded() is True

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_loads_embedder_when_getting_model_name(
        self, mock_factory, embedder_service, mock_embedder
    ):
        """Should lazy load embedder when getting model name."""
        mock_factory.create_embedder.return_value = mock_embedder

        model = embedder_service.get_model_name()

        assert model == "all-mpnet-base-v2"
        assert embedder_service.is_loaded() is True


class TestEmbedderServiceErrorHandling:
    """Tests for error handling."""

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_raises_exception_on_load_failure(self, mock_factory, embedder_service):
        """Should raise exception if embedder loading fails."""
        mock_factory.create_embedder.side_effect = RuntimeError("Load failed")

        with pytest.raises(RuntimeError, match="Load failed"):
            embedder_service.get_embedder()

        assert embedder_service.is_loaded() is False

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_allows_retry_after_failed_load(
        self, mock_factory, embedder_service, mock_embedder
    ):
        """Should allow retry after failed load attempt."""
        # First call fails
        mock_factory.create_embedder.side_effect = [
            RuntimeError("Load failed"),
            mock_embedder,
        ]

        # First attempt fails
        with pytest.raises(RuntimeError):
            embedder_service.get_embedder()

        assert embedder_service.is_loaded() is False

        # Second attempt succeeds
        result = embedder_service.get_embedder()

        assert result == mock_embedder
        assert embedder_service.is_loaded() is True


class TestEmbedderServiceThreadSafety:
    """Tests for thread-safe lazy loading."""

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_concurrent_access_initializes_once(
        self, mock_factory, embedder_service, mock_embedder
    ):
        """Should initialize embedder only once under concurrent access."""
        # Track how many times create_embedder is called
        call_count = [0]
        lock = threading.Lock()

        def create_embedder_with_delay(*args, **kwargs):
            """Simulates slow embedder initialization."""
            with lock:
                call_count[0] += 1
            # Simulate loading time
            time.sleep(0.1)
            return mock_embedder

        mock_factory.create_embedder.side_effect = create_embedder_with_delay

        # Launch 10 concurrent threads trying to get the embedder
        threads = []
        results = []

        def get_embedder_thread():
            result = embedder_service.get_embedder()
            results.append(result)

        for _ in range(10):
            t = threading.Thread(target=get_embedder_thread)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify embedder was only created once
        assert call_count[0] == 1
        # All threads got the same embedder instance
        assert all(r == mock_embedder for r in results)
        assert len(results) == 10

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_fast_path_no_lock_contention(
        self, mock_factory, embedder_service, mock_embedder
    ):
        """Should use fast path without lock after initialization."""
        mock_factory.create_embedder.return_value = mock_embedder

        # Initialize embedder
        embedder_service.get_embedder()

        # Subsequent calls should use fast path (no lock)
        # We can't directly verify lock is not acquired, but we can
        # ensure performance is good by making many calls quickly
        start = time.time()
        for _ in range(1000):
            embedder_service.get_embedder()
        duration = time.time() - start

        # Should be very fast (< 0.1s for 1000 calls)
        assert duration < 0.1
        # Factory called only once
        mock_factory.create_embedder.assert_called_once()


class TestEmbedderServiceCleanup:
    """Tests for cleanup functionality."""

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_cleanup_without_loading(self, mock_factory, embedder_service):
        """Should handle cleanup when embedder was never loaded."""
        # Should not raise
        embedder_service.cleanup()

        assert embedder_service.is_loaded() is False

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_cleanup_after_loading(self, mock_factory, embedder_service, mock_embedder):
        """Should clean up embedder after loading."""
        mock_factory.create_embedder.return_value = mock_embedder

        # Load embedder
        embedder_service.get_embedder()
        assert embedder_service.is_loaded() is True

        # Cleanup
        embedder_service.cleanup()

        assert embedder_service.is_loaded() is False
        mock_embedder.cleanup.assert_called_once()
        assert embedder_service._embedder is None

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_cleanup_handles_embedder_error(
        self, mock_factory, embedder_service, mock_embedder
    ):
        """Should handle errors during embedder cleanup gracefully."""
        mock_factory.create_embedder.return_value = mock_embedder
        mock_embedder.cleanup.side_effect = RuntimeError("Cleanup failed")

        # Load embedder
        embedder_service.get_embedder()

        # Cleanup should not raise, but log warning
        embedder_service.cleanup()

        # Should still reset state
        assert embedder_service.is_loaded() is False
        assert embedder_service._embedder is None

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_can_reload_after_cleanup(
        self, mock_factory, embedder_service, mock_embedder
    ):
        """Should be able to reload embedder after cleanup."""
        mock_factory.create_embedder.return_value = mock_embedder

        # Load, cleanup, load again
        embedder_service.get_embedder()
        embedder_service.cleanup()
        result = embedder_service.get_embedder()

        assert result == mock_embedder
        assert embedder_service.is_loaded() is True
        # Should be called twice (initial load + reload)
        assert mock_factory.create_embedder.call_count == 2


class TestEmbedderServiceConfiguration:
    """Tests for configuration handling."""

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_passes_config_to_factory(self, mock_factory, mock_embedder):
        """Should pass all config parameters to factory."""
        config = Mock()
        config.get_embedding_config.return_value = {
            "provider": "custom_provider",
            "model": "custom_model",
            "batch_size": 32,
            "max_length": 512,
            "device": "cuda",
        }
        mock_factory.create_embedder.return_value = mock_embedder

        service = EmbedderService(config)
        service.get_embedder()

        mock_factory.create_embedder.assert_called_once_with(
            provider="custom_provider",
            model_name="custom_model",
            batch_size=32,
            max_length=512,
            device="cuda",
        )

    @patch("pycontextify.embedder_factory.EmbedderFactory")
    def test_filters_out_provider_and_model_from_kwargs(
        self, mock_factory, mock_embedder
    ):
        """Should not pass provider and model as kwargs."""
        config = Mock()
        config.get_embedding_config.return_value = {
            "provider": "test_provider",
            "model": "test_model",
            "custom_param": "value",
        }
        mock_factory.create_embedder.return_value = mock_embedder

        service = EmbedderService(config)
        service.get_embedder()

        # Verify call
        call_args = mock_factory.create_embedder.call_args
        assert call_args[1]["provider"] == "test_provider"
        assert call_args[1]["model_name"] == "test_model"
        assert call_args[1]["custom_param"] == "value"
        # Should not have 'model' in kwargs (only as model_name)
        assert "model" not in call_args[1]
