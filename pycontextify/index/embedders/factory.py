"""Embedder factory for PyContextify.

This module implements the factory pattern for creating embedding providers,
making it easy to switch between different providers and add new ones.
"""

from typing import Dict, List, Type, Any
import logging

from .base import BaseEmbedder, ProviderNotAvailableError, EmbeddingError
from .sentence_transformers_embedder import SentenceTransformersEmbedder

logger = logging.getLogger(__name__)


class EmbedderFactory:
    """Factory class for creating embedding providers.
    
    This class manages the creation and configuration of different
    embedding providers, supporting both built-in and custom providers.
    """
    
    # Registry of available providers
    _providers: Dict[str, Type[BaseEmbedder]] = {
        "sentence_transformers": SentenceTransformersEmbedder,
    }
    
    @classmethod
    def register_provider(cls, name: str, embedder_class: Type[BaseEmbedder]) -> None:
        """Register a new embedding provider.
        
        Args:
            name: Provider name
            embedder_class: Class implementing BaseEmbedder interface
        """
        if not issubclass(embedder_class, BaseEmbedder):
            raise ValueError(f"Provider class must inherit from BaseEmbedder")
        
        cls._providers[name] = embedder_class
        logger.info(f"Registered embedding provider: {name}")
    
    @classmethod
    def create_embedder(cls, provider: str, model_name: str, **kwargs) -> BaseEmbedder:
        """Create an embedding provider instance.
        
        Args:
            provider: Name of the provider ('sentence_transformers', 'ollama', 'openai')
            model_name: Model name for the provider
            **kwargs: Provider-specific configuration parameters
            
        Returns:
            Configured embedder instance
            
        Raises:
            ValueError: If provider is not supported
            ProviderNotAvailableError: If provider dependencies are missing
            EmbeddingError: If embedder creation fails
        """
        if provider not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"Unsupported provider '{provider}'. Available: {available}")
        
        embedder_class = cls._providers[provider]
        
        # Check if provider is available before creating
        try:
            # Create a temporary instance to check availability
            temp_embedder = embedder_class(model_name, **kwargs)
            if not temp_embedder.is_available():
                raise ProviderNotAvailableError(f"Provider '{provider}' is not available")
            temp_embedder.cleanup()
        except Exception as e:
            if isinstance(e, (ProviderNotAvailableError, EmbeddingError)):
                raise
            raise EmbeddingError(f"Failed to validate provider '{provider}': {str(e)}")
        
        try:
            embedder = embedder_class(model_name, **kwargs)
            logger.info(f"Created embedder: {provider} with model {model_name}")
            return embedder
        except Exception as e:
            error_msg = f"Failed to create embedder for provider '{provider}': {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available embedding providers.
        
        Returns:
            List of provider names that are available for use
        """
        available = []
        
        for provider_name, embedder_class in cls._providers.items():
            try:
                # Try to create a temporary instance to check availability
                # Use a default model name for testing
                default_models = {
                    "sentence_transformers": "all-MiniLM-L6-v2",
                    "ollama": "nomic-embed-text",
                    "openai": "text-embedding-3-small"
                }
                
                model_name = default_models.get(provider_name, "default")
                temp_embedder = embedder_class(model_name)
                
                if temp_embedder.is_available():
                    available.append(provider_name)
                
                temp_embedder.cleanup()
                
            except Exception as e:
                logger.debug(f"Provider {provider_name} not available: {str(e)}")
        
        return available
    
    @classmethod
    def get_supported_models(cls, provider: str) -> List[str]:
        """Get supported models for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of supported model names
            
        Raises:
            ValueError: If provider is not registered
        """
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        embedder_class = cls._providers[provider]
        return getattr(embedder_class, 'supported_models', [])
    
    @classmethod
    def validate_provider_config(cls, provider: str, **kwargs) -> bool:
        """Validate provider-specific configuration.
        
        Args:
            provider: Provider name
            **kwargs: Configuration parameters to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If provider is not registered or config is invalid
        """
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        if provider == "sentence_transformers":
            return cls._validate_sentence_transformers_config(**kwargs)
        elif provider == "ollama":
            return cls._validate_ollama_config(**kwargs)
        elif provider == "openai":
            return cls._validate_openai_config(**kwargs)
        
        return True
    
    @classmethod
    def _validate_sentence_transformers_config(cls, **kwargs) -> bool:
        """Validate sentence-transformers specific configuration."""
        device = kwargs.get('device', 'auto')
        batch_size = kwargs.get('batch_size', 32)
        
        if device not in ['cpu', 'cuda', 'auto']:
            raise ValueError(f"Invalid device '{device}' for sentence_transformers")
        
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"Invalid batch_size '{batch_size}' for sentence_transformers")
        
        return True
    
    @classmethod
    def _validate_ollama_config(cls, **kwargs) -> bool:
        """Validate Ollama specific configuration."""
        base_url = kwargs.get('base_url', 'http://localhost:11434')
        
        if not base_url or not isinstance(base_url, str):
            raise ValueError("Ollama base_url is required and must be a string")
        
        if not base_url.startswith(('http://', 'https://')):
            raise ValueError("Ollama base_url must start with http:// or https://")
        
        return True
    
    @classmethod
    def _validate_openai_config(cls, **kwargs) -> bool:
        """Validate OpenAI specific configuration."""
        api_key = kwargs.get('api_key')
        
        if not api_key or not isinstance(api_key, str):
            raise ValueError("OpenAI api_key is required and must be a string")
        
        if len(api_key.strip()) < 10:  # Basic sanity check
            raise ValueError("OpenAI api_key appears to be invalid (too short)")
        
        return True
    
    @classmethod
    def get_provider_info(cls, provider: str) -> Dict[str, Any]:
        """Get information about a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary with provider information
            
        Raises:
            ValueError: If provider is not registered
        """
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        embedder_class = cls._providers[provider]
        
        info = {
            "name": provider,
            "class": embedder_class.__name__,
            "supported_models": getattr(embedder_class, 'supported_models', []),
            "provider_name": getattr(embedder_class, 'provider_name', provider),
        }
        
        # Add availability check
        try:
            default_models = {
                "sentence_transformers": "all-MiniLM-L6-v2",
                "ollama": "nomic-embed-text", 
                "openai": "text-embedding-3-small"
            }
            model_name = default_models.get(provider, "default")
            temp_embedder = embedder_class(model_name)
            info["available"] = temp_embedder.is_available()
            temp_embedder.cleanup()
        except Exception as e:
            info["available"] = False
            info["error"] = str(e)
        
        return info
    
    @classmethod
    def list_all_providers(cls) -> Dict[str, Any]:
        """List all registered providers with their information.
        
        Returns:
            Dictionary mapping provider names to their information
        """
        return {name: cls.get_provider_info(name) for name in cls._providers.keys()}
    
    # Future provider stubs - these will raise NotImplementedError until implemented
    
    @classmethod
    def _create_ollama_embedder(cls, model_name: str, base_url: str = "http://localhost:11434", **kwargs) -> BaseEmbedder:
        """Create Ollama embedder (future implementation).
        
        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
            **kwargs: Additional configuration
            
        Returns:
            OllamaEmbedder instance
            
        Raises:
            NotImplementedError: Until Ollama provider is implemented
        """
        raise NotImplementedError("Ollama provider not yet implemented")
    
    @classmethod
    def _create_openai_embedder(cls, model_name: str, api_key: str, **kwargs) -> BaseEmbedder:
        """Create OpenAI embedder (future implementation).
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key
            **kwargs: Additional configuration
            
        Returns:
            OpenAIEmbedder instance
            
        Raises:
            NotImplementedError: Until OpenAI provider is implemented
        """
        raise NotImplementedError("OpenAI provider not yet implemented")
    
    @classmethod
    def get_recommendations(cls) -> Dict[str, Dict[str, str]]:
        """Get provider recommendations for different use cases.
        
        Returns:
            Dictionary mapping use cases to provider recommendations
        """
        return {
            "development": {
                "provider": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
                "reason": "Fast and lightweight for development"
            },
            "production_quality": {
                "provider": "sentence_transformers", 
                "model": "all-mpnet-base-v2",
                "reason": "Best quality embeddings for production use"
            },
            "production_speed": {
                "provider": "sentence_transformers",
                "model": "all-MiniLM-L6-v2", 
                "reason": "Good balance of speed and quality"
            },
            "local_privacy": {
                "provider": "sentence_transformers",
                "model": "all-mpnet-base-v2",
                "reason": "Fully local processing, no data leaves your machine"
            },
            "question_answering": {
                "provider": "sentence_transformers",
                "model": "multi-qa-mpnet-base-dot-v1",
                "reason": "Optimized for question-answering tasks"
            }
        }