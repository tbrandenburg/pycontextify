"""Tests for enhanced configuration with advanced search features."""

import os
import tempfile
import pytest
from pathlib import Path

from pycontextify.index.config import Config


class TestEnhancedConfig:
    """Test enhanced configuration functionality."""

    def test_default_advanced_search_settings(self):
        """Test default values for advanced search settings."""
        config = Config()
        
        assert config.use_hybrid_search is True
        assert config.use_reranking is True
        assert config.keyword_weight == 0.3
        assert config.reranking_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.pdf_engine == "pymupdf"
    
    def test_custom_advanced_search_settings(self, monkeypatch):
        """Test custom advanced search settings from environment."""
        monkeypatch.setenv("PYCONTEXTIFY_USE_HYBRID_SEARCH", "false")
        monkeypatch.setenv("PYCONTEXTIFY_USE_RERANKING", "false")
        monkeypatch.setenv("PYCONTEXTIFY_KEYWORD_WEIGHT", "0.5")
        monkeypatch.setenv("PYCONTEXTIFY_RERANKING_MODEL", "custom-model")
        monkeypatch.setenv("PYCONTEXTIFY_PDF_ENGINE", "pypdf2")
        
        config = Config()
        
        assert config.use_hybrid_search is False
        assert config.use_reranking is False
        assert config.keyword_weight == 0.5
        assert config.reranking_model == "custom-model"
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
        with pytest.raises(ValueError, match="Keyword weight must be between 0.0 and 1.0"):
            Config()
        
        monkeypatch.setenv("PYCONTEXTIFY_KEYWORD_WEIGHT", "-0.1")
        with pytest.raises(ValueError, match="Keyword weight must be between 0.0 and 1.0"):
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
        assert "use_reranking" in summary
        assert "keyword_weight" in summary
        assert "reranking_model" in summary
        assert "pdf_engine" in summary
        
        # Verify values
        assert summary["use_hybrid_search"] is True
        assert summary["use_reranking"] is True
        assert summary["keyword_weight"] == 0.3
        assert summary["reranking_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert summary["pdf_engine"] == "pymupdf"
    
    def test_env_file_loading_with_advanced_settings(self, monkeypatch):
        """Test loading advanced settings from .env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
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
        assert hasattr(config, 'embedding_provider')
        assert hasattr(config, 'embedding_model')
        assert hasattr(config, 'chunk_size')
        assert hasattr(config, 'chunk_overlap')
        assert hasattr(config, 'enable_relationships')
        assert hasattr(config, 'auto_persist')
        
        # Test existing methods still work
        assert config.get_config_summary() is not None
        assert config.get_embedding_config() is not None
        assert config.get_chunking_config() is not None
        assert config.get_persistence_config() is not None