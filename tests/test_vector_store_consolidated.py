"""Consolidated vector store tests - combining original and enhanced test approaches."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from pycontextify.index.config import Config
from pycontextify.index.vector_store import VectorStore


class TestVectorStoreConsolidated:
    """Consolidated tests for vector store functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = Config()
        self.config.backup_indices = True

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization_and_faiss_import_error(self):
        """Test vector store initialization and FAISS import error handling."""
        # Test successful initialization
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.is_trained = True
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            store = VectorStore(dimension=384, config=self.config)
            assert store.dimension == 384
            assert store.get_embedding_dimension() == 384
            assert store._total_vectors == 0
            mock_faiss.IndexFlatIP.assert_called_once_with(384)

        # Test FAISS import error
        with patch(
            "pycontextify.index.vector_store.VectorStore._initialize_index"
        ) as mock_init:
            mock_init.side_effect = ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

            with pytest.raises(ImportError, match="FAISS not installed"):
                VectorStore(dimension=128, config=self.config)

    def test_vector_operations_and_validation(self):
        """Test vector addition, search, and dimension validation."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),  # distances
            np.array([[0, 1, 2]]),  # indices
        )
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            store = VectorStore(dimension=128, config=self.config)

            # Test successful vector addition
            vectors = np.random.random((5, 128)).astype(np.float32)
            ids = store.add_vectors(vectors)
            assert len(ids) == 5
            assert ids == [0, 1, 2, 3, 4]
            assert store._total_vectors == 5
            mock_index.add.assert_called_once()

            # Test dimension mismatch
            wrong_vectors = np.random.random((3, 64)).astype(np.float32)
            with pytest.raises(ValueError, match="doesn't match index dimension"):
                store.add_vectors(wrong_vectors)

            # Test wrong array shape
            vectors_1d = np.random.random(128).astype(np.float32)
            with pytest.raises(ValueError, match="Vectors must be 2D array"):
                store.add_vectors(vectors_1d)

            # Test search functionality
            store._total_vectors = 10  # Simulate existing vectors
            query = np.random.random(128).astype(np.float32)
            distances, indices = store.search(query, top_k=3)

            assert len(distances) == 3
            assert len(indices) == 3
            assert distances[0] == 0.9
            assert indices[0] == 0
            mock_index.search.assert_called_once()

            # Test search dimension validation
            wrong_query = np.random.random(64).astype(np.float32)
            with pytest.raises(ValueError, match="doesn't match index dimension"):
                store.search(wrong_query, top_k=5)

    def test_memory_usage_and_index_info(self):
        """Test memory usage calculation and index information."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.is_trained = True
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            store = VectorStore(dimension=384, config=self.config)

            # Test empty store
            assert store.get_memory_usage() == 0
            assert store.is_empty()
            assert store.get_total_vectors() == 0

            # Test with vectors
            store._total_vectors = 1000
            memory_usage = store.get_memory_usage()
            expected_vector_memory = 1000 * 384 * 4  # float32 = 4 bytes
            expected_total = expected_vector_memory + 1024 * 1024  # + 1MB overhead
            assert memory_usage == expected_total
            assert not store.is_empty()

            # Test index info
            info = store.get_index_info()
            assert info["total_vectors"] == 1000
            assert info["dimension"] == 384
            assert info["index_type"] == "IndexFlatIP"
            assert info["is_trained"] is True
            assert "memory_usage_mb" in info

    def test_clear_and_context_manager(self):
        """Test clearing functionality and context manager support."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            store = VectorStore(dimension=128, config=self.config)
            store._total_vectors = 100

            assert not store.is_empty()
            store.clear()
            assert store.is_empty()
            assert store._total_vectors == 0

            # Test context manager
            with VectorStore(dimension=128, config=self.config) as context_store:
                assert context_store.dimension == 128

    def test_save_and_load_operations(self):
        """Test save/load operations with backup support."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_loaded_index = MagicMock()
        mock_loaded_index.ntotal = 50
        mock_loaded_index.d = 128
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.read_index.return_value = mock_loaded_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            store = VectorStore(dimension=128, config=self.config)
            save_path = self.test_dir / "test_index.faiss"

            # Test basic save
            store.save_to_file(str(save_path))
            mock_faiss.write_index.assert_called_once_with(mock_index, str(save_path))

            # Test save with backup (create existing file first)
            save_path.write_text("existing index")
            with patch.object(store, "_create_backup") as mock_backup:
                store.save_to_file(str(save_path))
                mock_backup.assert_called_once_with(save_path)

            # Test save without backup when disabled
            config_no_backup = Config()
            config_no_backup.backup_indices = False
            store_no_backup = VectorStore(dimension=128, config=config_no_backup)
            with patch.object(store_no_backup, "_create_backup") as mock_backup:
                store_no_backup.save_to_file(str(save_path))
                mock_backup.assert_not_called()

            # Test successful load
            load_path = self.test_dir / "load_test.faiss"
            load_path.write_text("dummy index")

            store.load_from_file(str(load_path))
            mock_faiss.read_index.assert_called_once_with(str(load_path))
            assert store._total_vectors == 50

            # Test load from non-existent file
            nonexistent_path = self.test_dir / "nonexistent.faiss"
            store.load_from_file(str(nonexistent_path))  # Should not raise error

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            store = VectorStore(dimension=128, config=self.config)

            # Test save error handling
            mock_faiss.write_index.side_effect = Exception("Save failed")
            save_path = self.test_dir / "error_test.faiss"

            with pytest.raises(Exception, match="Save failed"):
                store.save_to_file(str(save_path))

            # Test load error handling (should raise the error)
            mock_faiss.read_index.side_effect = Exception("Load failed")
            load_path = self.test_dir / "error_load.faiss"
            load_path.write_text("corrupted index")

            with pytest.raises(Exception, match="Load failed"):
                store.load_from_file(str(load_path))

            # Test dimension mismatch on load
            mock_faiss.read_index.side_effect = None  # Reset side effect
            mock_loaded_index = MagicMock()
            mock_loaded_index.ntotal = 50
            mock_loaded_index.d = 256  # Wrong dimension
            mock_faiss.read_index.return_value = mock_loaded_index

            with pytest.raises(ValueError, match="doesn't match expected"):
                store.load_from_file(str(load_path))

    def test_backup_functionality(self):
        """Test backup creation and cleanup."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            store = VectorStore(dimension=128, config=self.config)
            original_path = self.test_dir / "test_index.faiss"

            # Create original file
            original_path.write_text("original index")

            with patch("shutil.copy2") as mock_copy:
                store._create_backup(original_path)

                mock_copy.assert_called_once()
                backup_path = mock_copy.call_args[0][1]
                assert str(original_path.stem) in str(backup_path)
                assert "backup" in str(backup_path)

    def test_vector_retrieval_and_advanced_operations(self):
        """Test individual vector retrieval and advanced operations."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.reconstruct.return_value = np.array([1.0, 2.0, 3.0])
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            store = VectorStore(dimension=3, config=self.config)
            store._total_vectors = 10

            # Test successful vector retrieval
            vector = store.get_vector(5)
            assert len(vector) == 3
            assert vector[0] == 1.0
            mock_index.reconstruct.assert_called_once_with(5)

            # Test invalid vector retrieval (returns None, doesn't raise)
            result = store.get_vector(15)  # Index >= total_vectors
            assert result is None

            result = store.get_vector(-1)  # Negative index
            assert result is None

    def test_search_edge_cases_and_large_operations(self):
        """Test search edge cases and large batch operations."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            store = VectorStore(dimension=128, config=self.config)

            # Test large batch operations
            large_batch = np.random.random((10000, 128)).astype(np.float32)
            ids = store.add_vectors(large_batch)
            assert len(ids) == 10000
            assert ids == list(range(0, 10000))
            mock_index.add.assert_called_once()

            # Test search with limited vectors
            mock_index.search.return_value = (
                np.array([[0.9, 0.8]]),  # Only 2 vectors available
                np.array([[0, 1]]),
            )
            store._total_vectors = 2  # Only 2 vectors in index

            query = np.random.random(128).astype(np.float32)
            distances, indices = store.search(
                query, top_k=10
            )  # Request more than available

            # Should only return available vectors
            assert len(distances) == 2
            assert len(indices) == 2

            # Verify search was called with limited k
            args = mock_index.search.call_args[0]
            assert args[1] == 2  # min(top_k=10, total_vectors=2)

    def test_dimension_validation_comprehensive(self):
        """Test comprehensive dimension validation."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            store = VectorStore(dimension=128, config=self.config)

            # Test validation method (if it exists)
            if hasattr(store, "validate_dimension"):
                # Test 1D vector
                vector_1d = np.random.random(128)
                assert store.validate_dimension(vector_1d) is True

                wrong_1d = np.random.random(64)
                assert store.validate_dimension(wrong_1d) is False

                # Test 2D vectors
                vectors_2d = np.random.random((5, 128))
                assert store.validate_dimension(vectors_2d) is True

                wrong_2d = np.random.random((5, 64))
                assert store.validate_dimension(wrong_2d) is False
