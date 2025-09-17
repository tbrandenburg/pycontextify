"""Comprehensive tests for auto-persist and auto-load functionality."""

import tempfile
import pytest
from pathlib import Path
from pycontextify.index.config import Config
from pycontextify.index.manager import IndexManager


class TestPersistenceFeatures:
    """Test auto-persist and auto-load functionality."""
    
    def test_complete_persistence_cycle(self):
        """Test that indexing, saving, and loading works correctly."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_index_dir = temp_path / "test_index"
            
            # Test content
            test_content = """
# Test Document

This is a test document for persistence testing.
It contains multiple sections and should be chunked.

## Section 1
Content about testing persistence.

## Section 2  
More content to ensure we have multiple chunks.
"""
            
            # Create test file
            test_file = temp_path / "test_doc.md"
            test_file.write_text(test_content)
            
            # Phase 1: Index and save
            config_overrides = {
                "index_dir": str(test_index_dir),
                "auto_persist": True,
                "auto_load": True,
                "index_name": "test_index"
            }
            
            config1 = Config(config_overrides=config_overrides)
            manager1 = IndexManager(config1)
            
            # Index the document
            result = manager1.index_document(str(test_file))
            
            # Verify indexing succeeded
            assert result["chunks_added"] > 0
            assert result["source_type"] == "document"
            
            # Get stats before "shutdown"
            stats_before = manager1.get_status()
            chunks_before = stats_before["index_stats"]["total_chunks"]
            vectors_before = stats_before["index_stats"]["total_vectors"]
            
            assert chunks_before > 0
            assert vectors_before > 0
            assert chunks_before == vectors_before
            
            # Verify files were created
            index_files = list(test_index_dir.glob("*"))
            expected_files = ["test_index.faiss", "test_index.pkl", "test_index_relationships.pkl"]
            actual_files = [f.name for f in index_files]
            
            for expected_file in expected_files:
                assert expected_file in actual_files
            
            # Phase 2: Create new manager (simulating restart)
            config2 = Config(config_overrides=config_overrides)
            manager2 = IndexManager(config2)
            
            # Verify data was loaded
            stats_after = manager2.get_status()
            chunks_after = stats_after["index_stats"]["total_chunks"]
            vectors_after = stats_after["index_stats"]["total_vectors"]
            
            # Should have same amount of data
            assert chunks_after == chunks_before
            assert vectors_after == vectors_before
            assert chunks_after > 0
            
            # Verify search works
            search_results = manager2.search("test", top_k=3)
            assert len(search_results) > 0
            assert "chunk_text" in search_results[0]
    
    def test_embedding_model_compatibility(self):
        """Test that index loading handles embedding model compatibility."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_index_dir = temp_path / "test_index"
            
            test_content = "# Test\nSome content for testing embedding compatibility."
            test_file = temp_path / "test.md"
            test_file.write_text(test_content)
            
            # Create index with one model
            config_overrides1 = {
                "index_dir": str(test_index_dir),
                "auto_persist": True,
                "auto_load": True,
                "index_name": "test_index",
                "embedding_model": "all-MiniLM-L6-v2"
            }
            
            config1 = Config(config_overrides=config_overrides1)
            manager1 = IndexManager(config1)
            result = manager1.index_document(str(test_file))
            
            assert result["chunks_added"] > 0
            stats1 = manager1.get_status()
            original_chunks = stats1["index_stats"]["total_chunks"]
            
            # Try to load with different model (should auto-adjust)
            config_overrides2 = {
                "index_dir": str(test_index_dir),
                "auto_persist": True,
                "auto_load": True,
                "index_name": "test_index",
                "embedding_model": "all-mpnet-base-v2"  # Different model
            }
            
            config2 = Config(config_overrides=config_overrides2)
            manager2 = IndexManager(config2)
            
            stats2 = manager2.get_status()
            loaded_chunks = stats2["index_stats"]["total_chunks"]
            
            # Should load with original model settings
            assert loaded_chunks == original_chunks
            assert stats2["embedding"]["model"] == "all-MiniLM-L6-v2"  # Original model
    
    def test_auto_load_disabled(self):
        """Test that auto-load can be disabled."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_index_dir = temp_path / "test_index"
            
            test_content = "# Test\nContent for testing auto-load disable."
            test_file = temp_path / "test.md"
            test_file.write_text(test_content)
            
            # Create and save index
            config_overrides1 = {
                "index_dir": str(test_index_dir),
                "auto_persist": True,
                "auto_load": True,
                "index_name": "test_index"
            }
            
            config1 = Config(config_overrides=config_overrides1)
            manager1 = IndexManager(config1)
            manager1.index_document(str(test_file))
            
            # Verify files exist
            assert (test_index_dir / "test_index.faiss").exists()
            
            # Create new manager with auto-load disabled
            config_overrides2 = {
                "index_dir": str(test_index_dir),
                "auto_persist": True,
                "auto_load": False,  # Disabled
                "index_name": "test_index"
            }
            
            config2 = Config(config_overrides=config_overrides2)
            manager2 = IndexManager(config2)
            
            # Should start empty
            stats = manager2.get_status()
            assert stats["index_stats"]["total_chunks"] == 0
            assert stats["index_stats"]["total_vectors"] == 0
    
    def test_partial_file_loading(self):
        """Test behavior when some index files are missing."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_index_dir = temp_path / "test_index"
            
            test_content = "# Test\nContent for testing partial loading."
            test_file = temp_path / "test.md"
            test_file.write_text(test_content)
            
            # Create complete index
            config_overrides = {
                "index_dir": str(test_index_dir),
                "auto_persist": True,
                "auto_load": True,
                "index_name": "test_index"
            }
            
            config1 = Config(config_overrides=config_overrides)
            manager1 = IndexManager(config1)
            manager1.index_document(str(test_file))
            
            # Remove one file
            (test_index_dir / "test_index_relationships.pkl").unlink()
            
            # Try to load - should handle gracefully
            config2 = Config(config_overrides=config_overrides)
            manager2 = IndexManager(config2)
            
            # Should start fresh since files are incomplete
            stats = manager2.get_status()
            assert stats["index_stats"]["total_chunks"] == 0
    
    def test_large_index_performance(self):
        """Test persistence with a larger index."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_index_dir = temp_path / "test_index"
            
            # Create multiple test documents
            docs = []
            for i in range(10):
                content = f"""
# Document {i}

This is document number {i} for testing persistence performance.
It contains multiple sections to generate several chunks.

## Section A-{i}
Content section A for document {i}.

## Section B-{i}  
Content section B for document {i}.

## Section C-{i}
Content section C for document {i}.
"""
                doc_file = temp_path / f"doc_{i}.md"
                doc_file.write_text(content)
                docs.append(doc_file)
            
            config_overrides = {
                "index_dir": str(test_index_dir),
                "auto_persist": True,
                "auto_load": True,
                "index_name": "test_index"
            }
            
            config1 = Config(config_overrides=config_overrides)
            manager1 = IndexManager(config1)
            
            # Index all documents
            total_chunks = 0
            for doc_file in docs:
                result = manager1.index_document(str(doc_file))
                total_chunks += result["chunks_added"]
            
            assert total_chunks >= 20  # Should have many chunks
            
            stats_before = manager1.get_status()
            chunks_before = stats_before["index_stats"]["total_chunks"]
            
            # Test loading
            config2 = Config(config_overrides=config_overrides)
            manager2 = IndexManager(config2)
            
            stats_after = manager2.get_status()
            chunks_after = stats_after["index_stats"]["total_chunks"]
            
            assert chunks_after == chunks_before
            assert chunks_after >= 20
            
            # Test search still works
            search_results = manager2.search("document", top_k=5)
            assert len(search_results) == 5