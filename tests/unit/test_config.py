"""Tests for the configuration system."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from pycontextify.index.config import Config


class TestConfig(unittest.TestCase):
    """Test the configuration management system."""

    def test_default_configuration(self):
        """Test default configuration values."""
        # Clear environment variables and .env file loading for pure defaults
        with patch.dict(os.environ, {}, clear=True):
            with patch("pycontextify.index.config.load_dotenv"):
                config = Config()

                self.assertEqual(config.embedding_provider, "sentence_transformers")
                self.assertEqual(config.embedding_model, "all-MiniLM-L6-v2")
                self.assertIsNone(config.bootstrap_archive_url)
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.chunk_overlap, 50)
        self.assertTrue(config.auto_persist)
        self.assertTrue(config.enable_relationships)
        self.assertEqual(config.max_relationships_per_chunk, 10)

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "PYCONTEXTIFY_EMBEDDING_PROVIDER": "openai",
            "PYCONTEXTIFY_EMBEDDING_MODEL": "text-embedding-3-small",
            "PYCONTEXTIFY_OPENAI_API_KEY": "test-api-key",
            "PYCONTEXTIFY_CHUNK_SIZE": "256",
            "PYCONTEXTIFY_AUTO_PERSIST": "false",
            "PYCONTEXTIFY_ENABLE_RELATIONSHIPS": "false",
        }

        with patch.dict(os.environ, env_vars):
            config = Config()

            self.assertEqual(config.embedding_provider, "openai")
            self.assertEqual(config.embedding_model, "text-embedding-3-small")
            self.assertEqual(config.chunk_size, 256)
            self.assertFalse(config.auto_persist)
            self.assertFalse(config.enable_relationships)

    def test_bootstrap_configuration_validation(self):
        """Test validation of bootstrap archive configuration."""
        env_vars = {
            "PYCONTEXTIFY_INDEX_BOOTSTRAP_ARCHIVE_URL": "ftp://example.com/index.zip"
        }
        with patch.dict(os.environ, env_vars):
            with self.assertRaisesRegex(ValueError, "http://, https://, or file://"):
                Config()

        valid_env = {
            "PYCONTEXTIFY_INDEX_BOOTSTRAP_ARCHIVE_URL": "https://example.com/index.tar.gz"
        }
        with patch.dict(os.environ, valid_env):
            config = Config()
            self.assertEqual(
                config.bootstrap_archive_url, "https://example.com/index.tar.gz"
            )
            self.assertEqual(
                config.bootstrap_checksum_url,
                "https://example.com/index.tar.gz.sha256",
            )

    def test_cli_override_priority(self):
        """Test that CLI overrides take priority over environment variables."""
        env_vars = {
            "PYCONTEXTIFY_INDEX_NAME": "env_index",
            "PYCONTEXTIFY_AUTO_PERSIST": "true",
            "PYCONTEXTIFY_EMBEDDING_PROVIDER": "sentence_transformers",
        }

        cli_overrides = {
            "index_name": "cli_index",
            "auto_persist": False,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-ada-002",
        }

        with patch.dict(os.environ, env_vars):
            # Patch openai api key to avoid validation error
            with patch.dict(os.environ, {"PYCONTEXTIFY_OPENAI_API_KEY": "test-key"}):
                config = Config(config_overrides=cli_overrides)

                # CLI overrides should take priority
                self.assertEqual(config.index_name, "cli_index")
                self.assertFalse(config.auto_persist)
                self.assertEqual(config.embedding_provider, "openai")
                self.assertEqual(config.embedding_model, "text-embedding-ada-002")

    def test_boolean_configuration_parsing(self):
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
            with patch.dict(os.environ, {"PYCONTEXTIFY_AUTO_PERSIST": env_value}):
                config = Config()
                self.assertEqual(config.auto_persist, expected)

    def test_integer_configuration_parsing(self):
        """Test integer value parsing from environment variables."""
        env_vars = {"PYCONTEXTIFY_CHUNK_SIZE": "1024", "PYCONTEXTIFY_MAX_BACKUPS": "5"}

        with patch.dict(os.environ, env_vars):
            config = Config()

            self.assertEqual(config.chunk_size, 1024)
            self.assertEqual(config.max_backups, 5)

    def test_invalid_integer_fallback(self):
        """Test fallback to defaults for invalid integer values."""
        with patch.dict(os.environ, {"PYCONTEXTIFY_CHUNK_SIZE": "invalid"}):
            config = Config()

            self.assertEqual(config.chunk_size, 512)  # Default value

    def test_path_configuration(self):
        """Test path configuration and resolution."""
        test_dir = "/tmp/test_index"
        with patch.dict(os.environ, {"PYCONTEXTIFY_INDEX_DIR": test_dir}):
            config = Config()

            self.assertEqual(str(config.index_dir), str(Path(test_dir).resolve()))

    def test_configuration_validation(self):
        """Test configuration validation."""
        config = Config()

        # Should not raise any exceptions with default config
        assert config is not None

    def test_invalid_embedding_provider(self):
        """Test validation of embedding provider."""
        with patch.dict(
            os.environ, {"PYCONTEXTIFY_EMBEDDING_PROVIDER": "invalid_provider"}
        ):
            with self.assertRaisesRegex(
                ValueError, "Unsupported embedding provider|Unknown provider"
            ):
                Config()

    def test_openai_validation_without_api_key(self):
        """Test OpenAI provider requires API key."""
        with patch.dict(
            os.environ, {"PYCONTEXTIFY_EMBEDDING_PROVIDER": "openai"}, clear=True
        ):
            # Remove any existing API key
            env_dict = os.environ.copy()
            if "PYCONTEXTIFY_OPENAI_API_KEY" in env_dict:
                del env_dict["PYCONTEXTIFY_OPENAI_API_KEY"]
            with patch.dict(os.environ, env_dict, clear=True):
                with self.assertRaisesRegex(
                    ValueError, "OpenAI API key is required|Unknown provider"
                ):
                    Config()

    def test_openai_validation_with_api_key(self):
        """Test OpenAI provider with API key."""
        env_vars = {
            "PYCONTEXTIFY_EMBEDDING_PROVIDER": "openai",
            "PYCONTEXTIFY_OPENAI_API_KEY": "test-api-key",
        }
        with patch.dict(os.environ, env_vars):
            try:
                config = Config()
                self.assertEqual(config.embedding_provider, "openai")
                self.assertEqual(config.openai_api_key, "test-api-key")
            except ValueError as e:
                # Skip test if OpenAI provider is not available
                if "Unknown provider" in str(e):
                    self.skipTest("OpenAI provider not available")
                else:
                    raise

    def test_chunk_size_validation(self):
        """Test chunk size validation."""
        with patch.dict(os.environ, {"PYCONTEXTIFY_CHUNK_SIZE": "0"}):
            with self.assertRaisesRegex(ValueError, "Chunk size must be positive"):
                Config()

    def test_chunk_overlap_validation(self):
        """Test chunk overlap validation."""
        with patch.dict(os.environ, {"PYCONTEXTIFY_CHUNK_OVERLAP": "-1"}):
            with self.assertRaisesRegex(ValueError, "Chunk overlap cannot be negative"):
                Config()

    def test_chunk_overlap_size_relationship(self):
        """Test chunk overlap must be less than chunk size."""
        env_vars = {
            "PYCONTEXTIFY_CHUNK_SIZE": "100",
            "PYCONTEXTIFY_CHUNK_OVERLAP": "150",
        }
        with patch.dict(os.environ, env_vars):
            with self.assertRaisesRegex(
                ValueError, "Chunk overlap must be less than chunk size"
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
            # Clear any existing PyContextify environment variables to ensure clean test
            env_backup = {}
            pycontextify_keys = [
                k for k in os.environ.keys() if k.startswith("PYCONTEXTIFY_")
            ]
            for key in pycontextify_keys:
                env_backup[key] = os.environ.pop(key)

            try:
                config = Config(env_file=env_file)
                # The chunk size should be loaded from the .env file
                self.assertEqual(config.chunk_size, 1000)
            finally:
                # Restore original environment
                for key, value in env_backup.items():
                    os.environ[key] = value
        finally:
            os.unlink(env_file)


class TestEnhancedConfig:
    """Test enhanced configuration functionality with advanced search features."""

    def test_default_advanced_search_settings(self):
        """Test default values for advanced search settings."""
        config = Config()

        assert config.use_hybrid_search is True
        assert config.keyword_weight == 0.3
        assert config.pdf_engine == "pymupdf"

    def test_custom_advanced_search_settings(self, monkeypatch):
        """Test custom advanced search settings from environment."""
        monkeypatch.setenv("PYCONTEXTIFY_USE_HYBRID_SEARCH", "false")
        monkeypatch.setenv("PYCONTEXTIFY_KEYWORD_WEIGHT", "0.5")
        monkeypatch.setenv("PYCONTEXTIFY_PDF_ENGINE", "pypdf2")

        config = Config()

        assert config.use_hybrid_search is False
        assert config.keyword_weight == 0.5
        assert config.pdf_engine == "pypdf2"

    def test_keyword_weight_validation(self, monkeypatch):
        """Test validation of keyword weight parameter."""
        # Test valid weights
        for weight in [0.0, 0.3, 0.5, 1.0]:
            monkeypatch.setenv("PYCONTEXTIFY_KEYWORD_WEIGHT", str(weight))
            config = Config()
            assert config.keyword_weight == weight

        # Test invalid weights
        monkeypatch.setenv("PYCONTEXTIFY_KEYWORD_WEIGHT", "1.5")
        with pytest.raises(
            ValueError, match="Keyword weight must be between 0.0 and 1.0"
        ):
            Config()

        monkeypatch.setenv("PYCONTEXTIFY_KEYWORD_WEIGHT", "-0.1")
        with pytest.raises(
            ValueError, match="Keyword weight must be between 0.0 and 1.0"
        ):
            Config()

    def test_pdf_engine_validation(self, monkeypatch):
        """Test validation of PDF engine parameter."""
        # Test valid engines
        for engine in ["pymupdf", "pypdf2", "pdfplumber"]:
            monkeypatch.setenv("PYCONTEXTIFY_PDF_ENGINE", engine)
            config = Config()
            assert config.pdf_engine == engine

        # Test invalid engine
        monkeypatch.setenv("PYCONTEXTIFY_PDF_ENGINE", "invalid_engine")
        with pytest.raises(ValueError, match="Unsupported PDF engine: invalid_engine"):
            Config()

    def test_float_config_parsing(self, monkeypatch):
        """Test float configuration parsing."""
        # Test valid float
        monkeypatch.setenv("PYCONTEXTIFY_KEYWORD_WEIGHT", "0.75")
        config = Config()
        assert config.keyword_weight == 0.75

        # Test invalid float falls back to default
        monkeypatch.setenv("PYCONTEXTIFY_KEYWORD_WEIGHT", "invalid")
        config = Config()
        assert config.keyword_weight == 0.3  # Default value

    def test_config_summary_includes_new_settings(self):
        """Test that config summary includes new settings."""
        config = Config()
        summary = config.get_config_summary()

        # Check that new settings are included
        assert "use_hybrid_search" in summary
        assert "keyword_weight" in summary
        assert "pdf_engine" in summary

        # Verify values
        assert summary["use_hybrid_search"] is True
        assert summary["keyword_weight"] == 0.3
        assert summary["pdf_engine"] == "pymupdf"

    def test_env_file_loading_with_advanced_settings(self, monkeypatch):
        """Test loading advanced settings from .env file."""
        # Clear any existing environment variables first to ensure clean test
        pycontextify_keys = [
            k for k in os.environ.keys() if k.startswith("PYCONTEXTIFY_")
        ]
        for key in pycontextify_keys:
            monkeypatch.delenv(key, raising=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("PYCONTEXTIFY_USE_HYBRID_SEARCH=false\n")
            f.write("PYCONTEXTIFY_KEYWORD_WEIGHT=0.4\n")
            f.write("PYCONTEXTIFY_PDF_ENGINE=pdfplumber\n")
            env_file = f.name

        try:
            config = Config(env_file=env_file)
            assert config.use_hybrid_search is False
            assert config.keyword_weight == 0.4
            assert config.pdf_engine == "pdfplumber"
        finally:
            os.unlink(env_file)

    def test_backward_compatibility(self):
        """Test that existing configuration still works."""
        config = Config()

        # Test that all existing settings are still available
        assert hasattr(config, "embedding_provider")
        assert hasattr(config, "embedding_model")
        assert hasattr(config, "chunk_size")
        assert hasattr(config, "chunk_overlap")
        assert hasattr(config, "enable_relationships")
        assert hasattr(config, "auto_persist")

        # Test existing methods still work
        assert config.get_config_summary() is not None
        assert config.get_embedding_config() is not None
        assert config.get_chunking_config() is not None
        assert config.get_persistence_config() is not None


if __name__ == "__main__":
    unittest.main()
