"""Integration tests for enhanced IndexManager with hybrid search and reranking."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from pycontextify.index.config import Config
from pycontextify.index.manager import IndexManager


class TestEnhancedIntegration:
    """Test enhanced IndexManager integration."""

    def test_index_manager_initialization_with_enhancements(self, monkeypatch):
        """Test IndexManager initialization with enhanced features."""
        # Configure enhanced features
        monkeypatch.setenv("PYCONTEXTIFY_USE_HYBRID_SEARCH", "true")
        monkeypatch.setenv("PYCONTEXTIFY_USE_RERANKING", "true")
        monkeypatch.setenv("PYCONTEXTIFY_KEYWORD_WEIGHT", "0.4")
        monkeypatch.setenv("PYCONTEXTIFY_PDF_ENGINE", "pymupdf")
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")  # Disable persistence for tests

        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            
            config = Config()
            
            # Mock embedding components to avoid model loading
            with patch('pycontextify.index.embedders.EmbedderFactory.create_embedder') as mock_embedder_factory, \
                 patch('pycontextify.index.manager.VectorStore') as mock_vector_store_class:
                mock_embedder = Mock()
                mock_embedder.get_provider_name.return_value = "sentence_transformers"
                mock_embedder.get_model_name.return_value = "all-mpnet-base-v2"
                mock_embedder.get_dimension.return_value = 768
                mock_embedder.is_available.return_value = True
                mock_embedder_factory.return_value = mock_embedder
                
                # Mock VectorStore to avoid FAISS dependency
                mock_vector_store = Mock()
                mock_vector_store.is_empty.return_value = True
                mock_vector_store_class.return_value = mock_vector_store
                
                # Mock hybrid search to avoid sklearn/rank_bm25 dependencies in test
                with patch('pycontextify.index.hybrid_search.HybridSearchEngine') as mock_hybrid_search_class:
                    mock_hybrid_search = Mock()
                    mock_hybrid_search.get_stats.return_value = {
                        "keyword_weight": 0.4,
                        "vector_weight": 0.6,
                        "indexed_documents": 0
                    }
                    mock_hybrid_search_class.return_value = mock_hybrid_search
                    
                    # Mock reranker to avoid model loading
                    with patch('pycontextify.index.reranker.CrossEncoderReranker') as mock_reranker_class:
                        mock_reranker = Mock()
                        mock_reranker.is_available = True
                        mock_reranker.warmup.return_value = True
                        mock_reranker.get_stats.return_value = {
                            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                            "is_available": True,
                            "model_loaded": True
                        }
                        mock_reranker_class.return_value = mock_reranker
                        
                        # Initialize IndexManager
                        manager = IndexManager(config)
                        
                        # Verify components are initialized
                        assert manager.config.use_hybrid_search is True
                        assert manager.config.use_reranking is True
                        assert manager.config.keyword_weight == 0.4
                        assert manager.config.pdf_engine == "pymupdf"
                        
                        # With lazy loading, embedder and vector_store are None until needed
                        assert manager.embedder is None  # Not loaded yet due to lazy loading
                        assert manager.vector_store is None  # Not loaded yet due to lazy loading
                        
                        # These components are initialized eagerly
                        assert manager.hybrid_search is not None
                        assert manager.reranker is not None
                        
                        # Trigger lazy loading by calling _ensure_embedder_loaded
                        manager._ensure_embedder_loaded()
                        
                        # Now embedder and vector_store should be loaded
                        assert manager.embedder is not None
                        assert manager.vector_store is not None

    def test_enhanced_status_function(self, monkeypatch):
        """Test enhanced status function includes new metrics."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            
            config = Config()
            
            # Mock all components
            with patch('pycontextify.index.embedders.EmbedderFactory.create_embedder') as mock_embedder_factory, \
                 patch('pycontextify.index.manager.VectorStore') as mock_vector_store_class:
                mock_embedder = Mock()
                mock_embedder.get_provider_name.return_value = "sentence_transformers"
                mock_embedder.get_model_name.return_value = "all-mpnet-base-v2"
                mock_embedder.get_dimension.return_value = 768
                mock_embedder.is_available.return_value = True
                mock_embedder_factory.return_value = mock_embedder
                
                # Mock VectorStore
                mock_vector_store = Mock()
                mock_vector_store.is_empty.return_value = True
                mock_vector_store_class.return_value = mock_vector_store
                
                with patch('pycontextify.index.hybrid_search.HybridSearchEngine') as mock_hybrid_search_class:
                    mock_hybrid_search = Mock()
                    mock_hybrid_search.get_stats.return_value = {
                        "keyword_weight": 0.3,
                        "vector_weight": 0.7,
                        "indexed_documents": 0,
                        "tfidf_available": True,
                        "bm25_available": True
                    }
                    mock_hybrid_search_class.return_value = mock_hybrid_search
                    
                    with patch('pycontextify.index.reranker.CrossEncoderReranker') as mock_reranker_class:
                        mock_reranker = Mock()
                        mock_reranker.is_available = True
                        mock_reranker.warmup.return_value = True
                        mock_reranker.get_stats.return_value = {
                            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                            "is_available": True,
                            "model_loaded": True
                        }
                        mock_reranker_class.return_value = mock_reranker
                        
                        manager = IndexManager(config)
                        status = manager.get_status()
                        
                        # Verify enhanced status fields
                        assert "status" in status
                        assert status["status"] == "healthy"
                        
                        # Check traditional fields
                        assert "metadata" in status
                        assert "relationships" in status
                        assert "vector_store" in status
                        assert "embedding" in status
                        assert "persistence" in status
                        assert "configuration" in status
                        
                        # Check new enhanced fields
                        assert "hybrid_search" in status
                        assert "reranking" in status
                        assert "performance" in status
                        
                        # Verify performance metrics
                        performance = status["performance"]
                        assert "cpu_usage_percent" in performance
                        assert "memory_usage_mb" in performance
                        assert "memory_total_mb" in performance
                        assert "memory_available_mb" in performance
                        assert "memory_usage_percent" in performance
                        assert "disk_usage_percent" in performance
                        assert "disk_free_gb" in performance
                        
                        # Verify enhanced components stats
                        hybrid_search_stats = status["hybrid_search"]
                        if hybrid_search_stats:  # Only check if hybrid search is initialized
                            assert "keyword_weight" in hybrid_search_stats
                            assert "vector_weight" in hybrid_search_stats
                        
                        reranking_stats = status["reranking"]
                        if reranking_stats:  # Only check if reranker is initialized
                            assert "model_name" in reranking_stats
                            assert "is_available" in reranking_stats

    def test_pdf_document_loading_integration(self, monkeypatch):
        """Test PDF document loading with enhanced loader."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        monkeypatch.setenv("PYCONTEXTIFY_PDF_ENGINE", "pymupdf")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            
            config = Config()
            
            # Create a fake PDF file
            fake_pdf_path = os.path.join(temp_dir, "test.pdf")
            with open(fake_pdf_path, "w") as f:
                f.write("This is a fake PDF content for testing")
            
            # Mock embedder and PDF loader
            with patch('pycontextify.index.embedders.EmbedderFactory.create_embedder') as mock_embedder_factory, \
                 patch('pycontextify.index.manager.VectorStore') as mock_vector_store_class:
                mock_embedder = Mock()
                mock_embedder.get_provider_name.return_value = "sentence_transformers"
                mock_embedder.get_model_name.return_value = "all-mpnet-base-v2" 
                mock_embedder.get_dimension.return_value = 768
                mock_embedder.is_available.return_value = True
                mock_embedder.embed_texts.return_value = [[0.1] * 768]  # Mock embedding
                mock_embedder_factory.return_value = mock_embedder
                
                # Mock VectorStore
                mock_vector_store = Mock()
                mock_vector_store.is_empty.return_value = True
                mock_vector_store_class.return_value = mock_vector_store
                
                # Mock the DocumentLoader's load method and chunking pipeline
                with patch('pycontextify.index.loaders.DocumentLoader.load') as mock_document_load, \
                     patch('pycontextify.index.manager.MetadataStore') as mock_metadata_store_class, \
                     patch('pycontextify.index.manager.ChunkerFactory') as mock_chunker_factory, \
                     patch('pycontextify.index.manager.RelationshipStore') as mock_relationship_store_class:
                    
                    mock_document_load.return_value = [(fake_pdf_path, "This is extracted PDF text content for testing.")]
                    
                    # Mock metadata store
                    mock_metadata_store = Mock()
                    mock_metadata_store.add_chunk.return_value = None
                    mock_metadata_store.get_chunks_by_source_path.return_value = []  # No existing chunks for reindexing
                    mock_metadata_store_class.return_value = mock_metadata_store
                    
                    # Mock relationship store
                    mock_relationship_store = Mock()
                    mock_relationship_store.build_relationships_from_chunks.return_value = None
                    mock_relationship_store_class.return_value = mock_relationship_store
                    
                    # Mock chunker - create a proper ChunkMetadata mock
                    mock_chunker = Mock()
                    from pycontextify.index.metadata import ChunkMetadata, SourceType
                    
                    # Create a real ChunkMetadata object instead of a mock
                    real_chunk = ChunkMetadata(
                        chunk_id="test-chunk-1",
                        source_path=fake_pdf_path,
                        source_type=SourceType.DOCUMENT,
                        chunk_text="Test chunk text",
                        start_char=0,
                        end_char=15,
                        embedding_provider="sentence_transformers",
                        embedding_model="all-mpnet-base-v2"
                    )
                    
                    mock_chunker.chunk_text.return_value = [real_chunk]
                    mock_chunker_factory.get_chunker.return_value = mock_chunker
                    
                    # Mock vector store add_vectors method
                    mock_vector_store.add_vectors.return_value = [0]  # Return list of faiss IDs
                    
                    with patch('pycontextify.index.hybrid_search.HybridSearchEngine'), \
                         patch('pycontextify.index.reranker.CrossEncoderReranker'):
                        
                        manager = IndexManager(config)
                        
                        # Test document indexing
                        result = manager.index_document(fake_pdf_path)
                        
                        # Verify successful indexing
                        assert "error" not in result
                        assert "file_processed" in result
                        assert "chunks_added" in result
                        assert "source_type" in result
                        assert result["source_type"] == "document"
                        assert result["embedding_provider"] == "sentence_transformers"

    def test_hybrid_search_integration(self, monkeypatch):
        """Test hybrid search integration in the search flow."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        monkeypatch.setenv("PYCONTEXTIFY_USE_HYBRID_SEARCH", "true")
        monkeypatch.setenv("PYCONTEXTIFY_USE_RERANKING", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            
            config = Config()
            
            # Use the same comprehensive mocking approach that works
            with patch('pycontextify.index.embedders.EmbedderFactory.create_embedder') as mock_embedder_factory, \
                 patch('pycontextify.index.manager.VectorStore') as mock_vector_store_class, \
                 patch('pycontextify.index.manager.MetadataStore') as mock_metadata_store_class, \
                 patch('pycontextify.index.hybrid_search.HybridSearchEngine') as mock_hybrid_search_class, \
                 patch('pycontextify.index.reranker.CrossEncoderReranker'):
                 
                # Mock embedder
                mock_embedder = Mock()
                mock_embedder.get_provider_name.return_value = "sentence_transformers"
                mock_embedder.get_model_name.return_value = "all-mpnet-base-v2"
                mock_embedder.get_dimension.return_value = 768
                mock_embedder.is_available.return_value = True
                mock_embedder.embed_single.return_value = [0.1] * 768
                mock_embedder_factory.return_value = mock_embedder
                
                # Mock VectorStore
                mock_vector_store = Mock()
                mock_vector_store.is_empty.return_value = False
                mock_vector_store.search.return_value = ([0.8], [0])
                mock_vector_store_class.return_value = mock_vector_store
                
                # Mock MetadataStore
                mock_metadata_store = Mock()
                mock_chunk = Mock()
                mock_chunk.source_path = "/test/path"
                mock_chunk.source_type.value = "document"
                mock_chunk.chunk_text = "Test content"
                mock_chunk.chunk_id = "test-chunk-1"
                mock_chunk.start_char = 0
                mock_chunk.end_char = 100
                mock_chunk.created_at.isoformat.return_value = "2023-01-01T00:00:00"
                
                mock_metadata_store.get_chunk.return_value = mock_chunk
                mock_metadata_store.get_stats.return_value = {"total_chunks": 1}
                mock_metadata_store.get_all_chunks.return_value = [mock_chunk]
                mock_metadata_store_class.return_value = mock_metadata_store
                
                # Mock hybrid search
                mock_hybrid_search = Mock()
                mock_hybrid_search.get_stats.return_value = {
                    "indexed_documents": 1,
                    "keyword_weight": 0.3,
                    "vector_weight": 0.7
                }
                
                # Mock search results
                mock_search_result = Mock()
                mock_search_result.combined_score = 0.85
                mock_search_result.vector_score = 0.9
                mock_search_result.keyword_score = 0.7
                mock_search_result.source_path = "/test/path"
                mock_search_result.source_type = "document"
                mock_search_result.text = "Test content"
                mock_search_result.chunk_id = "test-chunk-1"
                mock_search_result.metadata = {"test": "metadata"}
                
                mock_hybrid_search.search.return_value = [mock_search_result]
                mock_hybrid_search.add_documents.return_value = None
                mock_hybrid_search_class.return_value = mock_hybrid_search
                
                manager = IndexManager(config)
                
                # Manually set the mocked components to simulate post-lazy-loading state
                manager.vector_store = mock_vector_store
                manager.metadata_store = mock_metadata_store
                manager.hybrid_search = mock_hybrid_search
                manager._embedder_initialized = True
                manager.embedder = mock_embedder
                
                # Perform search
                response = manager.search("test query", top_k=5)
                
                # Verify result format includes hybrid search scores
                assert response.success, f"Search should succeed: {response.error}"
                assert len(response.results) > 0
                result = response.results[0]
                assert result.relevance_score > 0  # combined score
                assert result.scores is not None
                assert "vector" in result.scores
                assert "keyword" in result.scores
                assert result.chunk_id == "test-chunk-1"
                
                # Verify hybrid search was called
                assert mock_hybrid_search.search.called

    def test_search_with_reranking(self, monkeypatch):
        """Test search with reranking integration."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        monkeypatch.setenv("PYCONTEXTIFY_USE_RERANKING", "true")
        monkeypatch.setenv("PYCONTEXTIFY_USE_HYBRID_SEARCH", "false")  # Disable hybrid for this test
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            
            config = Config()
            
            # Mock embedder
            with patch('pycontextify.index.embedders.EmbedderFactory.create_embedder') as mock_embedder_factory, \
                 patch('pycontextify.index.manager.VectorStore') as mock_vector_store_class:
                mock_embedder = Mock()
                mock_embedder.get_provider_name.return_value = "sentence_transformers"
                mock_embedder.get_model_name.return_value = "all-mpnet-base-v2"
                mock_embedder.get_dimension.return_value = 768
                mock_embedder.is_available.return_value = True
                mock_embedder.embed_single.return_value = [0.1] * 768
                mock_embedder_factory.return_value = mock_embedder
                
                # Mock VectorStore
                mock_vector_store = Mock()
                mock_vector_store.is_empty.return_value = False
                mock_vector_store.search.return_value = ([0.85], [0])
                mock_vector_store_class.return_value = mock_vector_store
                
                # Mock reranker
                with patch('pycontextify.index.reranker.CrossEncoderReranker') as mock_reranker_class:
                    mock_reranker = Mock()
                    mock_reranker.is_available = True
                    mock_reranker.warmup.return_value = True
                    
                    # Mock rerank results
                    mock_rerank_result = Mock()
                    mock_rerank_result.final_score = 0.92
                    mock_rerank_result.original_score = 0.85
                    mock_rerank_result.rerank_score = 0.95
                    mock_rerank_result.source_path = "/test/path"
                    mock_rerank_result.source_type = "document"
                    mock_rerank_result.text = "Reranked content"
                    mock_rerank_result.chunk_id = "reranked-chunk-1"
                    mock_rerank_result.metadata = {"reranked": True}
                    
                    mock_reranker.rerank.return_value = [mock_rerank_result]
                    mock_reranker_class.return_value = mock_reranker
                    
                    with patch('pycontextify.index.hybrid_search.HybridSearchEngine'):
                        manager = IndexManager(config)
                        
                        # Mock metadata store
                        mock_chunk = Mock()
                        mock_chunk.source_path = "/test/path"
                        mock_chunk.source_type.value = "document"
                        mock_chunk.chunk_text = "Original content"
                        mock_chunk.chunk_id = "original-chunk-1"
                        mock_chunk.start_char = 0
                        mock_chunk.end_char = 100
                        mock_chunk.created_at.isoformat.return_value = "2023-01-01T00:00:00"
                        
                        # Set up components for lazy loading
                        manager.vector_store = mock_vector_store
                        manager.metadata_store.get_chunk = Mock(return_value=mock_chunk)
                        manager.reranker = mock_reranker
                        manager._embedder_initialized = True
                        manager.embedder = mock_embedder
                        
                        # Perform search
                        response = manager.search("test query", top_k=3)
                        
                        # Verify reranker was called
                        assert mock_reranker.rerank.called
                        
                        # Verify result format includes reranking scores
                        assert response.success, f"Search should succeed: {response.error}"
                        assert len(response.results) > 0
                        result = response.results[0]
                        assert result.relevance_score == 0.92  # final score
                        assert result.scores is not None
                        assert "original" in result.scores
                        assert "rerank" in result.scores
                        assert result.chunk_id == "reranked-chunk-1"

    def test_configuration_integration(self, monkeypatch):
        """Test that configuration settings are properly integrated."""
        # Set specific configuration
        monkeypatch.setenv("PYCONTEXTIFY_USE_HYBRID_SEARCH", "true")
        monkeypatch.setenv("PYCONTEXTIFY_USE_RERANKING", "false")
        monkeypatch.setenv("PYCONTEXTIFY_KEYWORD_WEIGHT", "0.4")
        monkeypatch.setenv("PYCONTEXTIFY_RERANKING_MODEL", "custom-model")
        monkeypatch.setenv("PYCONTEXTIFY_PDF_ENGINE", "pdfplumber")
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            
            config = Config()
            
            # Mock components
            with patch('pycontextify.index.embedders.EmbedderFactory.create_embedder') as mock_embedder_factory, \
                 patch('pycontextify.index.manager.VectorStore') as mock_vector_store_class:
                mock_embedder = Mock()
                mock_embedder.get_provider_name.return_value = "sentence_transformers"
                mock_embedder.get_model_name.return_value = "all-mpnet-base-v2"
                mock_embedder.get_dimension.return_value = 768
                mock_embedder.is_available.return_value = True
                mock_embedder_factory.return_value = mock_embedder
                
                # Mock VectorStore
                mock_vector_store = Mock()
                mock_vector_store.is_empty.return_value = True
                mock_vector_store_class.return_value = mock_vector_store
                
                with patch('pycontextify.index.hybrid_search.HybridSearchEngine') as mock_hybrid_search_class:
                    mock_hybrid_search = Mock()
                    mock_hybrid_search_class.return_value = mock_hybrid_search
                    
                    # Test IndexManager respects configuration
                    manager = IndexManager(config)
                    
                    # Verify configuration is used
                    assert manager.config.use_hybrid_search is True
                    assert manager.config.use_reranking is False
                    assert manager.config.keyword_weight == 0.4
                    assert manager.config.reranking_model == "custom-model"
                    assert manager.config.pdf_engine == "pdfplumber"
                    
                    # Verify hybrid search was initialized with correct weight
                    mock_hybrid_search_class.assert_called_once_with(keyword_weight=0.4)
                    
                    # Verify reranker was NOT initialized (disabled)
                    assert manager.reranker is None

    def test_error_handling_in_enhanced_components(self, monkeypatch):
        """Test error handling when enhanced components fail to initialize."""
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", temp_dir)
            
            config = Config()
            
            # Mock embedder
            with patch('pycontextify.index.embedders.EmbedderFactory.create_embedder') as mock_embedder_factory, \
                 patch('pycontextify.index.manager.VectorStore') as mock_vector_store_class:
                mock_embedder = Mock()
                mock_embedder.get_provider_name.return_value = "sentence_transformers"
                mock_embedder.get_model_name.return_value = "all-mpnet-base-v2"
                mock_embedder.get_dimension.return_value = 768
                mock_embedder.is_available.return_value = True
                mock_embedder_factory.return_value = mock_embedder
                
                # Mock VectorStore
                mock_vector_store = Mock()
                mock_vector_store.is_empty.return_value = True
                mock_vector_store_class.return_value = mock_vector_store
                
                # Test hybrid search initialization failure
                with patch('pycontextify.index.hybrid_search.HybridSearchEngine') as mock_hybrid_search_class:
                    mock_hybrid_search_class.side_effect = ImportError("sklearn not available")
                    
                    with patch('pycontextify.index.reranker.CrossEncoderReranker') as mock_reranker_class:
                        mock_reranker_class.side_effect = ImportError("sentence-transformers not available")
                        
                        # Should still initialize without crashing
                        manager = IndexManager(config)
                        
                        # Components should be None due to import errors
                        assert manager.hybrid_search is None
                        assert manager.reranker is None
                        
                        # With lazy loading, these are initially None until needed
                        assert manager.embedder is None  # Not loaded yet
                        assert manager.vector_store is None  # Not loaded yet
                        
                        # Trigger lazy loading
                        manager._ensure_embedder_loaded()
                        
                        # Now basic components should be initialized
                        assert manager.embedder is not None
                        assert manager.vector_store is not None
                        
                        # Status should still work
                        status = manager.get_status()
                        assert status["status"] == "healthy"