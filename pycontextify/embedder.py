"""Base classes for embedding providers in PyContextify.

This module defines the abstract base class that all embedding providers
must implement, along with common utilities and error handling. Also includes
the EmbedderService for embedder lifecycle management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class EmbeddingError(Exception):
    """Base exception for embedding operations."""

    pass


class ModelNotFoundError(EmbeddingError):
    """Raised when requested model is not available."""

    pass


class ProviderNotAvailableError(EmbeddingError):
    """Raised when provider dependencies are missing."""

    pass


class BaseEmbedder(ABC):
    """Abstract base class for all embedding providers.

    This class defines the interface that all embedding providers must
    implement to be compatible with PyContextify's indexing system.
    """

    # Provider identification - must be set by subclasses
    provider_name: str = ""
    supported_models: List[str] = []

    def __init__(self, model_name: str, **kwargs) -> None:
        """Initialize embedder with model name and provider-specific parameters.

        Args:
            model_name: Name of the model to use
            **kwargs: Provider-specific configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
        self._is_initialized = False

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Batch embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)

        Raises:
            EmbeddingError: If embedding fails
        """
        pass

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text for queries.

        Args:
            text: Text to embed

        Returns:
            numpy array with shape (embedding_dim,)

        Raises:
            EmbeddingError: If embedding fails
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension for FAISS index initialization.

        Returns:
            Dimension of embeddings produced by this model
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information for status reporting.

        Returns:
            Dictionary with model information
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured.

        Returns:
            True if provider can be used, False otherwise
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by the embedder."""
        pass

    def _validate_texts(self, texts: List[str]) -> None:
        """Validate input texts.

        Args:
            texts: List of texts to validate

        Raises:
            EmbeddingError: If texts are invalid
        """
        if not texts:
            raise EmbeddingError("No texts provided for embedding")

        if not isinstance(texts, list):
            raise EmbeddingError("Texts must be provided as a list")

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingError(f"Text at index {i} is not a string")

            if not text.strip():
                raise EmbeddingError(f"Text at index {i} is empty or whitespace-only")

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors for cosine similarity.

        Args:
            embeddings: Raw embeddings array

        Returns:
            Normalized embeddings array
        """
        # Compute L2 norm along the last axis
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)

        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)

        return embeddings / norms

    def _ensure_float32(self, embeddings: np.ndarray) -> np.ndarray:
        """Ensure embeddings are float32 for FAISS compatibility.

        Args:
            embeddings: Input embeddings

        Returns:
            Embeddings as float32 array
        """
        return embeddings.astype(np.float32)

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.provider_name

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "config": {
                k: v
                for k, v in self.config.items()
                if k not in ["api_key", "token", "password"]
            },  # Hide sensitive info
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


class EmbedderService:
    """Manages embedder lifecycle with thread-safety.

    This service uses double-checked locking to ensure thread-safe
    initialization of the embedder. The embedder is typically pre-loaded
    during MCP server startup for fast first-request performance.
    """

    def __init__(self, config):
        """Initialize embedder service with configuration.

        Args:
            config: Configuration object with embedding settings
        """
        self.config = config
        self._embedder: Optional[BaseEmbedder] = None
        self._initialized = False
        self._lock = __import__("threading").Lock()

    def get_embedder(self) -> BaseEmbedder:
        """Get embedder, initializing if needed (thread-safe).

        Uses double-checked locking pattern to ensure the embedder
        is only initialized once even under concurrent access.
        Embedders are typically pre-loaded during server startup.

        Returns:
            Initialized embedder instance

        Raises:
            Exception: If embedder initialization fails
        """
        # Fast path: already initialized (no lock needed)
        if self._initialized:
            return self._embedder

        # Slow path: need to initialize (acquire lock)
        with self._lock:
            # Double-check after acquiring lock
            if self._initialized:
                return self._embedder

            try:
                logger = __import__("logging").getLogger(__name__)
                logger.info("Initializing embedder...")
                embedding_config = self.config.get_embedding_config()

                logger.info(
                    f"Initializing embedder: {embedding_config['provider']} "
                    f"with model {embedding_config['model']}"
                )

                from .embedder_factory import EmbedderFactory

                self._embedder = EmbedderFactory.create_embedder(
                    provider=embedding_config["provider"],
                    model_name=embedding_config["model"],
                    **{
                        k: v
                        for k, v in embedding_config.items()
                        if k not in ["provider", "model"]
                    },
                )

                self._initialized = True

                logger.info(
                    f"Successfully loaded embedder: "
                    f"{self._embedder.get_provider_name()}"
                )

                return self._embedder

            except Exception as e:
                logger = __import__("logging").getLogger(__name__)
                logger.error(f"Failed to initialize embedder: {e}")
                raise

    def is_loaded(self) -> bool:
        """Check if embedder is loaded.

        Returns:
            True if embedder has been initialized, False otherwise
        """
        return self._initialized

    def get_dimension(self) -> Optional[int]:
        """Get embedding dimension (loads embedder if needed).

        Returns:
            Embedding dimension, or None if not loaded

        Raises:
            Exception: If embedder loading fails
        """
        return self.get_embedder().get_dimension()

    def get_provider_name(self) -> str:
        """Get embedder provider name (loads if needed).

        Returns:
            Provider name

        Raises:
            Exception: If embedder loading fails
        """
        return self.get_embedder().get_provider_name()

    def get_model_name(self) -> str:
        """Get embedder model name (loads if needed).

        Returns:
            Model name

        Raises:
            Exception: If embedder loading fails
        """
        return self.get_embedder().get_model_name()

    def cleanup(self) -> None:
        """Clean up embedder resources.

        This method should be called when the embedder is no longer needed
        to free up resources (model memory, caches, etc.).
        """
        with self._lock:
            if self._embedder:
                try:
                    logger = __import__("logging").getLogger(__name__)
                    logger.info("Cleaning up embedder resources")
                    self._embedder.cleanup()
                except Exception as e:
                    logger = __import__("logging").getLogger(__name__)
                    logger.warning(f"Error during embedder cleanup: {e}")
                finally:
                    self._embedder = None
                    self._initialized = False


__all__ = [
    "BaseEmbedder",
    "EmbeddingError",
    "EmbedderService",
    "ModelNotFoundError",
    "ProviderNotAvailableError",
]
