"""Tests for the configuration system."""

import os
import tempfile
from pathlib import Path

import pytest

from pycontextify.index.config import Config


class TestConfig:
    """Test the configuration management system."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = Config()

        assert config.embedding_provider == "sentence_transformers"
        assert config.embedding_model == "all-mpnet-base-v2"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.auto_persist is True
        assert config.enable_relationships is True
        assert config.max_relationships_per_chunk == 10

    def test_environment_variable_loading(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("PYCONTEXTIFY_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("PYCONTEXTIFY_EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("PYCONTEXTIFY_OPENAI_API_KEY", "test-api-key")
        monkeypatch.setenv("PYCONTEXTIFY_CHUNK_SIZE", "256")
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        monkeypatch.setenv("PYCONTEXTIFY_ENABLE_RELATIONSHIPS", "false")

        config = Config()

        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.chunk_size == 256
        assert config.auto_persist is False
        assert config.enable_relationships is False

    def test_boolean_configuration_parsing(self, monkeypatch):
        """Test boolean value parsing from environment variables."""
        test_cases = [
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            ("on", True),
            ("off", False),
            ("TRUE", True),
            ("FALSE", False),
        ]

        for env_value, expected in test_cases:
            monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", env_value)
            config = Config()
            assert config.auto_persist == expected

    def test_integer_configuration_parsing(self, monkeypatch):
        """Test integer value parsing from environment variables."""
        monkeypatch.setenv("PYCONTEXTIFY_CHUNK_SIZE", "1024")
        monkeypatch.setenv("PYCONTEXTIFY_MAX_BACKUPS", "5")

        config = Config()

        assert config.chunk_size == 1024
        assert config.max_backups == 5

    def test_invalid_integer_fallback(self, monkeypatch):
        """Test fallback to defaults for invalid integer values."""
        monkeypatch.setenv("PYCONTEXTIFY_CHUNK_SIZE", "invalid")

        config = Config()

        assert config.chunk_size == 512  # Default value

    def test_path_configuration(self, monkeypatch):
        """Test path configuration and resolution."""
        test_dir = "/tmp/test_index"
        monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", test_dir)

        config = Config()

        assert str(config.index_dir) == str(Path(test_dir).resolve())

    def test_configuration_validation(self):
        """Test configuration validation."""
        config = Config()

        # Should not raise any exceptions with default config
        assert config is not None

    def test_invalid_embedding_provider(self, monkeypatch):
        """Test validation of embedding provider."""
        monkeypatch.setenv("PYCONTEXTIFY_EMBEDDING_PROVIDER", "invalid_provider")

        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            Config()

    def test_openai_validation_without_api_key(self, monkeypatch):
        """Test OpenAI provider requires API key."""
        monkeypatch.setenv("PYCONTEXTIFY_EMBEDDING_PROVIDER", "openai")
        # Don't set API key

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            Config()

    def test_openai_validation_with_api_key(self, monkeypatch):
        """Test OpenAI provider with API key."""
        monkeypatch.setenv("PYCONTEXTIFY_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("PYCONTEXTIFY_OPENAI_API_KEY", "test-api-key")

        config = Config()
        assert config.embedding_provider == "openai"
        assert config.openai_api_key == "test-api-key"

    def test_chunk_size_validation(self, monkeypatch):
        """Test chunk size validation."""
        monkeypatch.setenv("PYCONTEXTIFY_CHUNK_SIZE", "0")

        with pytest.raises(ValueError, match="Chunk size must be positive"):
            Config()

    def test_chunk_overlap_validation(self, monkeypatch):
        """Test chunk overlap validation."""
        monkeypatch.setenv("PYCONTEXTIFY_CHUNK_OVERLAP", "-1")

        with pytest.raises(ValueError, match="Chunk overlap cannot be negative"):
            Config()

    def test_chunk_overlap_size_relationship(self, monkeypatch):
        """Test chunk overlap must be less than chunk size."""
        monkeypatch.setenv("PYCONTEXTIFY_CHUNK_SIZE", "100")
        monkeypatch.setenv("PYCONTEXTIFY_CHUNK_OVERLAP", "150")

        with pytest.raises(
            ValueError, match="Chunk overlap must be less than chunk size"
        ):
            Config()

    def test_provider_availability_check(self):
        """Test provider availability checking."""
        config = Config()

        # Sentence transformers should be available (or at least attempt to check)
        availability = config.is_provider_available("sentence_transformers")
        assert isinstance(availability, bool)

        # Non-existent provider should be False
        availability = config.is_provider_available("nonexistent")
        assert availability is False

    def test_get_index_paths(self):
        """Test index path generation."""
        config = Config()
        paths = config.get_index_paths()

        assert "index" in paths
        assert "metadata" in paths
        assert "relationships" in paths

        assert paths["index"].suffix == ".faiss"
        assert paths["metadata"].suffix == ".pkl"
        assert paths["relationships"].suffix == ".pkl"

    def test_ensure_index_directory(self):
        """Test index directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_index"

            # Mock the config to use our test directory
            config = Config()
            config.index_dir = test_dir

            # Directory shouldn't exist initially
            assert not test_dir.exists()

            # Ensure directory
            config.ensure_index_directory()

            # Directory should now exist
            assert test_dir.exists()
            assert test_dir.is_dir()

    def test_config_summary(self):
        """Test configuration summary generation."""
        config = Config()
        summary = config.get_config_summary()

        required_keys = [
            "embedding_provider",
            "embedding_model",
            "index_dir",
            "auto_persist",
            "chunk_size",
            "enable_relationships",
        ]

        for key in required_keys:
            assert key in summary

    def test_embedding_config(self):
        """Test embedding-specific configuration."""
        config = Config()
        embedding_config = config.get_embedding_config()

        assert "provider" in embedding_config
        assert "model" in embedding_config
        assert embedding_config["provider"] == config.embedding_provider
        assert embedding_config["model"] == config.embedding_model

    def test_chunking_config(self):
        """Test chunking-specific configuration."""
        config = Config()
        chunking_config = config.get_chunking_config()

        assert "chunk_size" in chunking_config
        assert "chunk_overlap" in chunking_config
        assert "enable_relationships" in chunking_config
        assert chunking_config["chunk_size"] == config.chunk_size

    def test_persistence_config(self):
        """Test persistence-specific configuration."""
        config = Config()
        persistence_config = config.get_persistence_config()

        assert "auto_persist" in persistence_config
        assert "auto_load" in persistence_config
        assert "index_dir" in persistence_config
        assert persistence_config["auto_persist"] == config.auto_persist

    def test_dot_env_file_loading(self):
        """Test loading from .env file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("PYCONTEXTIFY_EMBEDDING_MODEL=test-model\n")
            f.write("PYCONTEXTIFY_CHUNK_SIZE=1000\n")
            env_file = f.name

        try:
            config = Config(env_file=env_file)
            assert config.embedding_model == "test-model"
            assert config.chunk_size == 1000
        finally:
            os.unlink(env_file)
