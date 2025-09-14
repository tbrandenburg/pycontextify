"""Base classes for embedding providers in PyContextify.

This module defines the abstract base class that all embedding providers
must implement, along with common utilities and error handling.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
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
            "config": {k: v for k, v in self.config.items() 
                      if k not in ['api_key', 'token', 'password']},  # Hide sensitive info
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()