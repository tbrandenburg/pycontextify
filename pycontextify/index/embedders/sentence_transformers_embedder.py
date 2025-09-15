"""Sentence Transformers embedding provider for PyContextify.

This module implements the sentence-transformers embedding provider,
which is the default provider for the system.
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .base import (
    BaseEmbedder,
    EmbeddingError,
    ModelNotFoundError,
    ProviderNotAvailableError,
)

logger = logging.getLogger(__name__)


class SentenceTransformersEmbedder(BaseEmbedder):
    """Sentence Transformers embedding provider.

    This class provides embedding functionality using the sentence-transformers
    library, which offers a wide variety of pre-trained models optimized
    for semantic similarity tasks.
    """

    provider_name = "sentence_transformers"
    supported_models = [
        "all-mpnet-base-v2",  # High quality, 768 dim
        "all-MiniLM-L6-v2",  # Fast and good quality, 384 dim
        "all-distilroberta-v1",  # Good balance, 768 dim
        "multi-qa-mpnet-base-dot-v1",  # Question-answering optimized, 768 dim
        "paraphrase-mpnet-base-v2",  # Paraphrase detection, 768 dim
        "paraphrase-MiniLM-L6-v2",  # Fast paraphrase detection, 384 dim
    ]

    def __init__(self, model_name: str, **kwargs) -> None:
        """Initialize Sentence Transformers embedder.

        Args:
            model_name: Name of the sentence-transformers model
            **kwargs: Additional configuration options:
                - device: Device to use ('cpu', 'cuda', 'auto')
                - batch_size: Batch size for processing (default: 32)
                - show_progress: Show progress bars (default: False)
                - normalize_embeddings: Normalize embeddings (default: True)
        """
        super().__init__(model_name, **kwargs)

        # Handle device selection - convert "auto" to appropriate device
        device_param = kwargs.get("device", "auto")
        if device_param == "auto":
            # Auto-detect best available device
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"  # Fallback if torch not available
        else:
            self.device = device_param
        self.batch_size = kwargs.get("batch_size", 32)
        self.show_progress = kwargs.get("show_progress", False)
        self.normalize_embeddings = kwargs.get("normalize_embeddings", True)

        self._model = None
        self._embedding_dimension = None

    def _ensure_model_loaded(self) -> None:
        """Lazy load the sentence transformer model."""
        if self._model is not None:
            return

        try:
            import sentence_transformers
        except ImportError:
            raise ProviderNotAvailableError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )

        try:
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = sentence_transformers.SentenceTransformer(
                self.model_name, device=self.device
            )

            # Get embedding dimension
            test_text = "test"
            test_embedding = self._model.encode([test_text], show_progress_bar=False)
            self._embedding_dimension = test_embedding.shape[1]

            self._is_initialized = True
            logger.info(
                f"Model loaded successfully. Embedding dimension: {self._embedding_dimension}"
            )

        except Exception as e:
            error_msg = f"Failed to load model '{self.model_name}': {str(e)}"
            logger.error(error_msg)

            if "not found" in str(e).lower():
                raise ModelNotFoundError(error_msg)
            else:
                raise EmbeddingError(error_msg)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Batch embed multiple texts using sentence-transformers.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        self._validate_texts(texts)
        self._ensure_model_loaded()

        try:
            logger.debug(
                f"Embedding {len(texts)} texts with batch size {self.batch_size}"
            )

            # Process in batches to manage memory
            all_embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]

                batch_embeddings = self._model.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress and len(texts) > 100,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                )

                all_embeddings.append(batch_embeddings)

            # Concatenate all batches
            embeddings = np.concatenate(all_embeddings, axis=0)

            # Ensure float32 for FAISS compatibility
            embeddings = self._ensure_float32(embeddings)

            logger.debug(f"Successfully embedded {len(texts)} texts")
            return embeddings

        except Exception as e:
            error_msg = f"Failed to embed texts: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text for queries.

        Args:
            text: Text to embed

        Returns:
            numpy array with shape (embedding_dim,)
        """
        if not text or not text.strip():
            raise EmbeddingError("Empty text provided for embedding")

        self._ensure_model_loaded()

        try:
            embedding = self._model.encode(
                [text.strip()],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )

            # Return single embedding (remove batch dimension)
            result = embedding[0]

            # Ensure float32 for FAISS compatibility
            return self._ensure_float32(result)

        except Exception as e:
            error_msg = f"Failed to embed single text: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)

    def get_dimension(self) -> int:
        """Return embedding dimension for FAISS index initialization."""
        if self._embedding_dimension is None:
            self._ensure_model_loaded()
        return self._embedding_dimension

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information for status reporting."""
        info = {
            "provider": self.provider_name,
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "is_loaded": self._model is not None,
        }

        if self._embedding_dimension is not None:
            info["embedding_dimension"] = self._embedding_dimension

        # Add model-specific information if loaded
        if self._model is not None:
            try:
                info["max_seq_length"] = getattr(
                    self._model, "max_seq_length", "unknown"
                )
                info["model_card"] = getattr(self._model, "model_card", {})
            except Exception:
                pass

        return info

    def is_available(self) -> bool:
        """Check if sentence-transformers is available and model can be loaded."""
        try:
            # Check if sentence-transformers is available
            import sentence_transformers  # noqa: F401

            # Try to load the model if not already loaded
            if self._model is None:
                try:
                    self._ensure_model_loaded()
                except Exception:
                    return False

            return True

        except ImportError:
            return False

    def cleanup(self) -> None:
        """Clean up resources used by the embedder."""
        if self._model is not None:
            # Free GPU memory if using CUDA
            if hasattr(self._model, "device") and "cuda" in str(self._model.device):
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass

            # Clear model reference
            self._model = None
            self._is_initialized = False
            logger.info("Sentence Transformers model cleaned up")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        memory_info = {
            "model_loaded": self._model is not None,
            "estimated_memory_mb": 0,
        }

        if self._model is not None:
            try:
                # Rough estimation based on model size
                if "mpnet" in self.model_name.lower():
                    memory_info["estimated_memory_mb"] = 420  # ~420MB for MPNet models
                elif "minilm" in self.model_name.lower():
                    memory_info["estimated_memory_mb"] = 90  # ~90MB for MiniLM models
                elif "distilroberta" in self.model_name.lower():
                    memory_info["estimated_memory_mb"] = 290  # ~290MB for DistilRoBERTa
                else:
                    memory_info["estimated_memory_mb"] = 300  # Default estimate

                # Add GPU memory info if available
                if hasattr(self._model, "device") and "cuda" in str(self._model.device):
                    try:
                        import torch

                        memory_info["gpu_memory_allocated_mb"] = (
                            torch.cuda.memory_allocated() / (1024 * 1024)
                        )
                        memory_info["gpu_memory_cached_mb"] = (
                            torch.cuda.memory_reserved() / (1024 * 1024)
                        )
                    except ImportError:
                        pass
            except Exception:
                pass

        return memory_info

    @classmethod
    def get_recommended_models(cls) -> Dict[str, str]:
        """Get recommended models for different use cases."""
        return {
            "best_quality": "all-mpnet-base-v2",
            "fastest": "all-MiniLM-L6-v2",
            "balanced": "all-distilroberta-v1",
            "qa_optimized": "multi-qa-mpnet-base-dot-v1",
            "paraphrases": "paraphrase-mpnet-base-v2",
            "small_fast": "paraphrase-MiniLM-L6-v2",
        }

    @classmethod
    def validate_model_name(cls, model_name: str) -> bool:
        """Validate if a model name is supported."""
        # For sentence-transformers, we can't easily check all available models
        # without loading them, so we'll be permissive and only check our known list
        if model_name in cls.supported_models:
            return True

        # Allow any valid sentence-transformers model name
        # They typically follow these patterns:
        patterns = [
            "sentence-transformers/",
            "all-",
            "multi-qa-",
            "paraphrase-",
            "msmarco-",
        ]

        return any(model_name.startswith(pattern) for pattern in patterns)
