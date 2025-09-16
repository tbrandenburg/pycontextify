"""Configuration management for PyContextify.

This module handles all environment variable configuration and validation
for the MCP server, including embedding providers, persistence settings,
chunking parameters, and relationship extraction settings.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


class Config:
    """Configuration manager for PyContextify MCP server.

    Loads and validates environment variables for all system components
    including embedding providers, persistence settings, and relationship
    extraction configuration.
    """

    def __init__(self, env_file: Optional[str] = None, config_overrides: Optional[Dict[str, Any]] = None) -> None:
        """Initialize configuration from environment variables with optional CLI overrides.

        Args:
            env_file: Optional path to .env file to load
            config_overrides: Optional dictionary of configuration overrides from CLI arguments
        """
        self.config_overrides = config_overrides or {}
        # Load environment variables from .env file if it exists
        if env_file:
            load_dotenv(env_file)
        else:
            # Look for .env in current directory
            env_path = Path(".env")
            if env_path.exists():
                load_dotenv(env_path)

        # Index storage configuration (CLI overrides have priority)
        self.index_dir = self._get_path_config(
            "PYCONTEXTIFY_INDEX_DIR", 
            "./index_data", 
            override_key="index_dir"
        )
        self.index_name = self._get_config(
            "PYCONTEXTIFY_INDEX_NAME", 
            "semantic_index", 
            override_key="index_name"
        )

        # Persistence settings (CLI overrides have priority)
        self.auto_persist = self._get_bool_config(
            "PYCONTEXTIFY_AUTO_PERSIST", 
            True, 
            override_key="auto_persist"
        )
        self.auto_load = self._get_bool_config(
            "PYCONTEXTIFY_AUTO_LOAD", 
            True, 
            override_key="auto_load"
        )
        self.compress_metadata = self._get_bool_config(
            "PYCONTEXTIFY_COMPRESS_METADATA", True
        )
        self.backup_indices = self._get_bool_config(
            "PYCONTEXTIFY_BACKUP_INDICES", False
        )
        self.max_backups = self._get_int_config("PYCONTEXTIFY_MAX_BACKUPS", 3)

        # Embedding provider configuration (CLI overrides have priority)
        self.embedding_provider = self._get_config(
            "PYCONTEXTIFY_EMBEDDING_PROVIDER", 
            "sentence_transformers",
            override_key="embedding_provider"
        )
        self.embedding_model = self._get_config(
            "PYCONTEXTIFY_EMBEDDING_MODEL", 
            "all-MiniLM-L6-v2",
            override_key="embedding_model"
        )

        # Provider-specific settings
        self.ollama_base_url = os.getenv(
            "PYCONTEXTIFY_OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.openai_api_key = os.getenv("PYCONTEXTIFY_OPENAI_API_KEY")

        # Text chunking configuration
        self.chunk_size = self._get_int_config("PYCONTEXTIFY_CHUNK_SIZE", 512)
        self.chunk_overlap = self._get_int_config("PYCONTEXTIFY_CHUNK_OVERLAP", 50)

        # Relationship extraction settings
        self.enable_relationships = self._get_bool_config(
            "PYCONTEXTIFY_ENABLE_RELATIONSHIPS", True
        )
        self.max_relationships_per_chunk = self._get_int_config(
            "PYCONTEXTIFY_MAX_RELATIONSHIPS_PER_CHUNK", 10
        )
        
        # Advanced search settings
        self.use_hybrid_search = self._get_bool_config(
            "PYCONTEXTIFY_USE_HYBRID_SEARCH", True
        )
        self.use_reranking = self._get_bool_config(
            "PYCONTEXTIFY_USE_RERANKING", True
        )
        self.keyword_weight = self._get_float_config(
            "PYCONTEXTIFY_KEYWORD_WEIGHT", 0.3
        )
        self.reranking_model = os.getenv(
            "PYCONTEXTIFY_RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        # PDF processing settings
        self.pdf_engine = os.getenv("PYCONTEXTIFY_PDF_ENGINE", "pymupdf")

        # Performance settings (optional)
        self.max_file_size_mb = self._get_int_config(
            "PYCONTEXTIFY_MAX_FILE_SIZE_MB", 10
        )
        self.batch_size = self._get_int_config("PYCONTEXTIFY_BATCH_SIZE", 32)
        self.crawl_delay_seconds = self._get_int_config(
            "PYCONTEXTIFY_CRAWL_DELAY_SECONDS", 
            1,
            override_key="crawl_delay_seconds"
        )

        # Validate configuration
        self._validate_config()

    def _get_config(self, env_key: str, default: Any, override_key: Optional[str] = None) -> Any:
        """Get configuration value with CLI override priority.
        
        Priority: CLI override > Environment variable > Default
        
        Args:
            env_key: Environment variable key
            default: Default value if neither override nor env var is set
            override_key: Key in config_overrides dictionary
            
        Returns:
            Configuration value with proper priority
        """
        # Check CLI override first
        if override_key and override_key in self.config_overrides:
            return self.config_overrides[override_key]
        
        # Fall back to environment variable or default
        return os.getenv(env_key, default)

    def _get_bool_config(self, key: str, default: bool, override_key: Optional[str] = None) -> bool:
        """Get boolean configuration value with CLI override priority."""
        # Check CLI override first
        if override_key and override_key in self.config_overrides:
            return bool(self.config_overrides[override_key])
        
        # Fall back to environment variable
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def _get_int_config(self, key: str, default: int, override_key: Optional[str] = None) -> int:
        """Get integer configuration value with CLI override priority."""
        # Check CLI override first
        if override_key and override_key in self.config_overrides:
            try:
                return int(self.config_overrides[override_key])
            except (ValueError, TypeError):
                return default
        
        # Fall back to environment variable
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def _get_float_config(self, key: str, default: float, override_key: Optional[str] = None) -> float:
        """Get float configuration value with CLI override priority."""
        # Check CLI override first
        if override_key and override_key in self.config_overrides:
            try:
                return float(self.config_overrides[override_key])
            except (ValueError, TypeError):
                return default
        
        # Fall back to environment variable
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default
    
    def _get_path_config(self, key: str, default: str, override_key: Optional[str] = None) -> Path:
        """Get path configuration value with CLI override priority."""
        # Check CLI override first
        if override_key and override_key in self.config_overrides:
            value = str(self.config_overrides[override_key])
        else:
            # Fall back to environment variable or default
            value = os.getenv(key, default)
        
        path = Path(value)
        # Convert to absolute path
        return path.resolve()

    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate embedding provider
        supported_providers = ["sentence_transformers", "ollama", "openai"]
        if self.embedding_provider not in supported_providers:
            raise ValueError(
                f"Unsupported embedding provider: {self.embedding_provider}"
            )

        # Validate provider-specific settings
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI provider")

        # Validate chunking parameters
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        # Validate relationship settings
        if self.max_relationships_per_chunk <= 0:
            raise ValueError("Max relationships per chunk must be positive")

        # Validate advanced search settings
        if not (0.0 <= self.keyword_weight <= 1.0):
            raise ValueError("Keyword weight must be between 0.0 and 1.0")
        
        # Validate PDF engine
        supported_pdf_engines = ["pymupdf", "pypdf2", "pdfplumber"]
        if self.pdf_engine not in supported_pdf_engines:
            raise ValueError(f"Unsupported PDF engine: {self.pdf_engine}")

        # Validate performance settings
        if self.max_file_size_mb <= 0:
            raise ValueError("Max file size must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.crawl_delay_seconds < 1:
            raise ValueError("Crawl delay must be at least 1 second for respectful crawling")

    def validate_provider_config(self) -> bool:
        """Validate provider-specific configuration."""
        if self.embedding_provider == "sentence_transformers":
            # Sentence transformers doesn't need additional validation
            return True
        elif self.embedding_provider == "ollama":
            # Could add Ollama connectivity check here
            return True
        elif self.embedding_provider == "openai":
            return bool(self.openai_api_key)
        return False

    def get_index_paths(self) -> Dict[str, Path]:
        """Get all index-related file paths."""
        base_path = self.index_dir / self.index_name
        return {
            "index": base_path.with_suffix(".faiss"),
            "metadata": base_path.with_suffix(".pkl"),
            "relationships": base_path.with_name(f"{base_path.name}_relationships.pkl"),
        }

    def ensure_index_directory(self) -> None:
        """Ensure index directory exists."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging and status reporting."""
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "index_dir": str(self.index_dir),
            "auto_persist": self.auto_persist,
            "auto_load": self.auto_load,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "enable_relationships": self.enable_relationships,
            "max_relationships_per_chunk": self.max_relationships_per_chunk,
            "backup_indices": self.backup_indices,
            "max_backups": self.max_backups,
            "use_hybrid_search": self.use_hybrid_search,
            "use_reranking": self.use_reranking,
            "keyword_weight": self.keyword_weight,
            "reranking_model": self.reranking_model,
            "pdf_engine": self.pdf_engine,
        }

    def is_provider_available(self, provider: str) -> bool:
        """Check if a specific embedding provider is available."""
        if provider == "sentence_transformers":
            try:
                import sentence_transformers  # noqa: F401

                return True
            except ImportError:
                return False
        elif provider == "ollama":
            try:
                import ollama  # noqa: F401

                return True
            except ImportError:
                return False
        elif provider == "openai":
            try:
                import openai  # noqa: F401

                return bool(self.openai_api_key)
            except ImportError:
                return False
        return False

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding-specific configuration."""
        config = {
            "provider": self.embedding_provider,
            "model": self.embedding_model,
        }

        if self.embedding_provider == "ollama":
            config["base_url"] = self.ollama_base_url
        elif self.embedding_provider == "openai":
            config["api_key"] = self.openai_api_key

        return config

    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking-specific configuration."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "enable_relationships": self.enable_relationships,
            "max_relationships_per_chunk": self.max_relationships_per_chunk,
        }

    def get_persistence_config(self) -> Dict[str, Any]:
        """Get persistence-specific configuration."""
        return {
            "auto_persist": self.auto_persist,
            "auto_load": self.auto_load,
            "compress_metadata": self.compress_metadata,
            "backup_indices": self.backup_indices,
            "max_backups": self.max_backups,
            "index_dir": str(self.index_dir),
            "index_name": self.index_name,
        }
