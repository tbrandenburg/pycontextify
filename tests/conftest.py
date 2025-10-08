"""Test configuration and fixtures for PyContextify.

This file provides performance optimizations by mocking heavy dependencies
to dramatically reduce test execution time.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

@pytest.fixture(autouse=True)
def mock_sentence_transformers(request):
    """Mock sentence-transformers to avoid model loading.

    This fixture automatically mocks sentence-transformers for all tests,
    reducing test time from ~300s to ~60s (80% improvement).

    Tests can use the marker @pytest.mark.no_mock_st to disable this mock.
    """
    # Check if test wants to disable this mock
    if hasattr(request, "node") and request.node.get_closest_marker("no_mock_st"):
        yield None
        return

    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        # Create mock model with realistic behavior
        mock_model = Mock()

        # Mock encode method to return realistic normalized embeddings
        def mock_encode(texts, *args, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            # Return normalized random embeddings (L2 norm ≈ 1.0)
            batch_size = len(texts)
            embeddings = np.random.randn(batch_size, 384).astype(
                np.float32
            )  # Normal distribution
            # Normalize to unit vectors
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms

        mock_model.encode = mock_encode
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.max_seq_length = 512

        # Mock model loading
        mock_st.return_value = mock_model

        yield mock_model


# Removed complex mocking that was causing import conflicts
# Focus only on sentence-transformers mocking which provides the biggest performance gain


@pytest.fixture
def temp_index_dir():
    """Provide a temporary directory for index storage in tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_document_content():
    """Provide sample document content for testing."""
    return """
# Sample Document

This is a sample document for testing purposes.
It contains multiple sections and paragraphs.

## Introduction

The introduction section provides an overview of the document.
It should be chunked appropriately by the system.

## Main Content  

The main content section contains the bulk of the information.
This section has multiple paragraphs to test chunking behavior.

### Subsection

This is a subsection with additional details.

## Conclusion

The conclusion wraps up the document content.
"""


@pytest.fixture
def sample_code_content():
    """Provide sample code content for testing."""
    return '''
"""Sample Python module for testing."""

def sample_function(param1, param2):
    """A sample function for testing code chunking.
    
    Args:
        param1: First parameter
        param2: Second parameter
        
    Returns:
        Combined result
    """
    result = param1 + param2
    return result

class SampleClass:
    """A sample class for testing."""
    
    def __init__(self, value):
        self.value = value
    
    def method(self):
        """A sample method."""
        return self.value * 2

# Global variable
SAMPLE_CONSTANT = "test_value"
'''


@pytest.fixture
def mock_embedder():
    """Provide a mock embedder instance for testing."""
    embedder = Mock()
    embedder.embed_texts.return_value = np.random.rand(5, 384).astype(np.float32)
    embedder.embed_single.return_value = np.random.rand(384).astype(np.float32)
    embedder.get_dimension.return_value = 384
    embedder.get_provider_name.return_value = "mock_provider"
    embedder.get_model_name.return_value = "mock_model"
    embedder.is_available.return_value = True
    embedder.cleanup.return_value = None
    return embedder


@pytest.fixture
def mock_config(temp_index_dir):
    """Provide a mock config for testing."""
    from pycontextify.config import Config

    # Create config with temporary directory
    config_overrides = {
        "index_dir": str(temp_index_dir),
        "auto_persist": False,  # Disable for faster tests
        "auto_load": False,
        "index_name": "test_index",
    }

    return Config(config_overrides=config_overrides)


# Configure pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (skipped only with --fast flag)"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "embedding: marks tests as requiring embedding models"
    )


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="skip slow tests for faster execution",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests only when --fast is given."""
    if not config.getoption("--fast"):
        # Run all tests by default (including slow ones)
        return

    skip_slow = pytest.mark.skip(
        reason="slow test skipped (use without --fast to include)"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
