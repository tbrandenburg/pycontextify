"""Basic tests for PyContextify package."""

# import pytest  # Not currently used

from pycontextify import Config, IndexManager, __version__


def test_version():
    """Test that version is defined."""
    assert __version__ == "0.1.0"


def test_imports():
    """Test that main classes can be imported."""
    assert IndexManager is not None
    assert Config is not None


def test_config_creation():
    """Test that Config can be instantiated."""
    # This will test basic config creation once implemented
    # config = Config()
    # assert config is not None
    pass


def test_index_manager_creation():
    """Test that IndexManager can be instantiated."""
    # This will test basic manager creation once implemented
    # config = Config()
    # manager = IndexManager(config)
    # assert manager is not None
    pass


def test_relationship_store_import():
    """Test that RelationshipStore can be imported."""
    from pycontextify.index import RelationshipStore

    assert RelationshipStore is not None


def test_chunker_imports():
    """Test that all chunker types can be imported."""
    # This will test chunker imports once implemented
    # from pycontextify.index.chunker import CodeChunker, DocumentChunker, WebPageChunker
    # assert CodeChunker is not None
    # assert DocumentChunker is not None
    # assert WebPageChunker is not None
    pass


def test_simplified_mcp_interface():
    """Test that the simplified MCP interface is maintained."""
    # This will test that only the essential MCP functions are exposed
    # Expected functions: index_code, index_document, index_webpage, search, status, search_with_context
    # Removed functions: find_related, get_relationships (for simplicity)
    pass
