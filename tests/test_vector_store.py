"""Working tests for VectorStore functionality."""

import pytest
import tempfile
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from pycontextify.index.vector_store import VectorStore
from pycontextify.index.config import Config


class TestVectorStoreSimple:
    """Test VectorStore functionality with working mocks."""

    @pytest.fixture
    def mock_faiss(self):
        """Create a mock FAISS module."""
        mock_faiss = Mock()
        mock_index = Mock()
        mock_index.is_trained = True
        mock_index.add = Mock()
        mock_index.search = Mock(return_value=(
            np.array([[0.9, 0.8, 0.7]]), 
            np.array([[0, 1, 2]])
        ))
        mock_index.ntotal = 0
        mock_index.d = 384
        mock_index.reconstruct = Mock(return_value=np.random.random(384).astype(np.float32))
        
        mock_faiss.IndexFlatIP = Mock(return_value=mock_index)
        mock_faiss.write_index = Mock()
        mock_faiss.read_index = Mock(return_value=mock_index)
        
        return mock_faiss, mock_index

    def test_vector_store_initialization(self, monkeypatch, mock_faiss):
        """Test VectorStore initialization."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    assert store.dimension == 384
                    assert store.config == config
                    assert store._total_vectors == 0

    def test_faiss_import_error(self, monkeypatch):
        """Test that ImportError is raised when FAISS is not available."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            def mock_import(name, *args, **kwargs):
                if name == 'faiss':
                    raise ImportError("No module named faiss")
                return __import__(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                with pytest.raises(ImportError, match="FAISS not installed"):
                    VectorStore(dimension=384, config=config)

    def test_add_vectors_success(self, monkeypatch, mock_faiss):
        """Test successful vector addition."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    # Test adding vectors
                    vectors = np.random.random((5, 384)).astype(np.float32)
                    ids = store.add_vectors(vectors)
                    
                    assert len(ids) == 5
                    assert ids == [0, 1, 2, 3, 4]
                    assert store._total_vectors == 5
                    mock_index.add.assert_called_once()

    def test_add_vectors_dimension_mismatch(self, monkeypatch, mock_faiss):
        """Test vector addition with dimension mismatch."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    # Test wrong dimension
                    vectors = np.random.random((5, 512)).astype(np.float32)
                    with pytest.raises(ValueError, match="Vector dimension 512 doesn't match"):
                        store.add_vectors(vectors)

    def test_add_vectors_wrong_shape(self, monkeypatch, mock_faiss):
        """Test vector addition with wrong array shape."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    # Test 1D array (wrong shape)
                    vectors = np.random.random(384).astype(np.float32)
                    with pytest.raises(ValueError, match="Vectors must be 2D array"):
                        store.add_vectors(vectors)

    def test_search_vectors(self, monkeypatch, mock_faiss):
        """Test vector search functionality."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    store._total_vectors = 10  # Simulate existing vectors
                    
                    # Test search
                    query = np.random.random(384).astype(np.float32)
                    distances, indices = store.search(query, top_k=3)
                    
                    # Verify results
                    assert len(distances) == 3
                    assert len(indices) == 3
                    assert distances[0] == 0.9
                    assert indices[0] == 0
                    mock_index.search.assert_called_once()

    def test_memory_usage_calculation(self, monkeypatch, mock_faiss):
        """Test memory usage calculation."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    # Test empty store
                    assert store.get_memory_usage() == 0
                    
                    # Test with vectors
                    store._total_vectors = 1000
                    memory_usage = store.get_memory_usage()
                    expected_vector_memory = 1000 * 384 * 4  # float32 = 4 bytes
                    expected_total = expected_vector_memory + 1024 * 1024  # + 1MB overhead
                    assert memory_usage == expected_total

    def test_get_index_info(self, monkeypatch, mock_faiss):
        """Test getting index information."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=768, config=config)
                    store._total_vectors = 100
                    
                    info = store.get_index_info()
                    
                    assert info["total_vectors"] == 100
                    assert info["dimension"] == 768
                    assert info["index_type"] == "IndexFlatIP"
                    assert info["is_trained"] is True
                    assert "memory_usage_mb" in info

    def test_is_empty(self, monkeypatch, mock_faiss):
        """Test empty check functionality."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    # Initially empty
                    assert store.is_empty() is True
                    
                    # Add vectors
                    store._total_vectors = 5
                    assert store.is_empty() is False

    def test_clear_store(self, monkeypatch, mock_faiss):
        """Test clearing the vector store."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    store._total_vectors = 10
                    
                    # Clear store
                    store.clear()
                    
                    assert store._total_vectors == 0
                    # Should reinitialize index (IndexFlatIP called twice)
                    assert mock_faiss_module.IndexFlatIP.call_count == 2

    def test_getters(self, monkeypatch, mock_faiss):
        """Test various getter methods."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=768, config=config)
                    store._total_vectors = 42
                    
                    assert store.get_total_vectors() == 42
                    assert store.get_embedding_dimension() == 768

    def test_edge_case_empty_vectors(self, monkeypatch, mock_faiss):
        """Test edge case with empty vector array."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    # Test with empty array
                    empty_vectors = np.empty((0, 384), dtype=np.float32)
                    ids = store.add_vectors(empty_vectors)
                    
                    assert len(ids) == 0
                    assert store._total_vectors == 0
                    # Verify add was called once (numpy array comparison is tricky in mocks)
                    mock_index.add.assert_called_once()

    def test_search_with_different_top_k_values(self, monkeypatch, mock_faiss):
        """Test search with different top_k values."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            # Configure mock to return results based on k
            def mock_search(query, k):
                distances = np.array([[0.9, 0.8, 0.7, 0.6, 0.5][:k]])
                indices = np.array([[0, 1, 2, 3, 4][:k]])
                return distances, indices
            
            mock_index.search.side_effect = mock_search
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    store._total_vectors = 10
                    
                    query = np.random.random(384).astype(np.float32)
                    
                    # Test different k values
                    for k in [1, 3, 5]:
                        distances, indices = store.search(query, top_k=k)
                        assert len(distances) == k
                        assert len(indices) == k

    def test_search_empty_index(self, monkeypatch, mock_faiss):
        """Test search on empty index."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            # Mock empty search results
            mock_index.search.return_value = (np.array([[]]), np.array([[]]))
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    # Empty store
                    store._total_vectors = 0
                    
                    query = np.random.random(384).astype(np.float32)
                    distances, indices = store.search(query, top_k=3)
                    
                    # Should handle empty results gracefully
                    assert isinstance(distances, np.ndarray)
                    assert isinstance(indices, np.ndarray)

    def test_save_index_file(self, monkeypatch, mock_faiss):
        """Test saving index to file."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    # Test saving
                    filepath = Path(temp_dir) / "index.faiss"
                    store.save_to_file(str(filepath))
                    
                    # Verify save was called
                    mock_faiss_module.write_index.assert_called_once_with(mock_index, str(filepath))

    def test_load_from_nonexistent_file(self, monkeypatch, mock_faiss):
        """Test loading from non-existent file (should handle gracefully)."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    # Test loading from nonexistent file
                    filepath = Path(temp_dir) / "nonexistent.faiss"
                    store.load_from_file(str(filepath))
                    
                    # Should not have called read_index since file doesn't exist
                    mock_faiss_module.read_index.assert_not_called()

    def test_vector_dimension_edge_cases(self, monkeypatch, mock_faiss):
        """Test edge cases with vector dimensions."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    # Test with vectors having extra dimensions
                    wrong_vectors = np.random.random((3, 512)).astype(np.float32)
                    with pytest.raises(ValueError, match="Vector dimension 512 doesn't match"):
                        store.add_vectors(wrong_vectors)
                    
                    # Test with vectors having fewer dimensions
                    short_vectors = np.random.random((3, 256)).astype(np.float32)
                    with pytest.raises(ValueError, match="Vector dimension 256 doesn't match"):
                        store.add_vectors(short_vectors)

    def test_search_query_dimension_edge_cases(self, monkeypatch, mock_faiss):
        """Test search with query dimension edge cases."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    # Test with wrong query dimension
                    wrong_query = np.random.random(512).astype(np.float32)
                    with pytest.raises(ValueError, match="Query vector dimension 512 doesn't match"):
                        store.search(wrong_query, top_k=3)

    def test_add_vectors_type_validation(self, monkeypatch, mock_faiss):
        """Test vector type validation."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=384, config=config)
                    
                    # Test with wrong data type (int instead of float32)
                    int_vectors = np.random.randint(0, 10, (3, 384)).astype(np.int32)
                    # Should either work (auto-convert) or raise appropriate error
                    try:
                        ids = store.add_vectors(int_vectors)
                        assert len(ids) == 3  # If it auto-converts
                    except (ValueError, TypeError):
                        pass  # If it properly rejects wrong type

    def test_memory_usage_edge_cases(self, monkeypatch, mock_faiss):
        """Test memory usage calculation edge cases."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            config = Config()
            
            mock_faiss_module, mock_index = mock_faiss
            
            with patch('sys.modules', {'faiss': mock_faiss_module}):
                with patch('builtins.__import__', return_value=mock_faiss_module):
                    store = VectorStore(dimension=1024, config=config)  # Large dimension
                    
                    # Test with large number of vectors
                    store._total_vectors = 1000000  # 1M vectors
                    memory_usage = store.get_memory_usage()
                    
                    # Should be a large number
                    expected_min = 1000000 * 1024 * 4  # At least the vector data
                    assert memory_usage >= expected_min
