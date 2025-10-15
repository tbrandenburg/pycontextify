"""Unit tests for IndexingPipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from pycontextify.indexer import IndexingPipeline


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return Mock()


@pytest.fixture
def mock_embedder_service():
    """Create mock embedder service."""
    service = Mock()
    mock_embedder = Mock()
    mock_embedder.get_provider_name.return_value = "test_provider"
    mock_embedder.get_model_name.return_value = "test_model"
    mock_embedder.embed_texts.return_value = [[0.1] * 768, [0.2] * 768]
    service.get_embedder.return_value = mock_embedder
    return service


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = Mock()
    store.add_vectors.return_value = [0, 1]
    return store


@pytest.fixture
def mock_metadata_store():
    """Create mock metadata store."""
    return Mock()


@pytest.fixture
def pipeline(
    mock_config, mock_embedder_service, mock_vector_store, mock_metadata_store
):
    """Create IndexingPipeline instance."""
    return IndexingPipeline(
        mock_config, mock_embedder_service, mock_vector_store, mock_metadata_store
    )


class TestIndexingPipelineValidation:
    """Tests for input validation."""

    def test_validates_topic_required(self, pipeline):
        """Should raise ValueError if topic is None."""
        with TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="topic is required"):
                pipeline.index_filebase(tmpdir, None)

    def test_validates_topic_not_empty(self, pipeline):
        """Should raise ValueError if topic is empty."""
        with TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="topic is required"):
                pipeline.index_filebase(tmpdir, "")

    def test_validates_topic_not_whitespace(self, pipeline):
        """Should raise ValueError if topic is only whitespace."""
        with TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="topic is required"):
                pipeline.index_filebase(tmpdir, "   ")

    def test_validates_path_exists(self, pipeline):
        """Should raise FileNotFoundError if path doesn't exist."""
        with pytest.raises(FileNotFoundError):
            pipeline.index_filebase("/nonexistent/path", "test")

    @patch("pycontextify.indexer.FileLoaderFactory")
    @patch("pycontextify.indexer.ChunkerFactory")
    @patch("pycontextify.indexer.postprocess_file")
    @patch("pycontextify.indexer.postprocess_filebase")
    def test_accepts_single_file_path(
        self,
        mock_postprocess_fb,
        mock_postprocess_file,
        mock_chunker_factory,
        mock_loader_factory,
        pipeline,
    ):
        """Should successfully index when a single file path is provided."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "file.txt"
            file_path.write_text("hello")

            mock_loader = Mock()
            mock_loader.load.return_value = [
                {
                    "text": "hello",
                    "metadata": {
                        "full_path": str(file_path),
                        "file_extension": "txt",
                    },
                }
            ]
            mock_loader_factory.return_value = mock_loader

            mock_chunker_factory.chunk_normalized_docs.return_value = [
                {
                    "text": "hello chunk",
                    "metadata": {
                        "full_path": str(file_path),
                        "file_extension": "txt",
                    },
                }
            ]

            mock_postprocess_file.side_effect = lambda x: x
            mock_postprocess_fb.side_effect = lambda x: x

            # Ensure mock components line up with single chunk expectations
            embedder = pipeline.embedder_service.get_embedder.return_value
            embedder.embed_texts.return_value = [[0.1] * 768]
            pipeline.vector_store.add_vectors.return_value = [0]

            stats = pipeline.index_filebase(str(file_path), "test")

            assert stats["files_crawled"] == 1
            assert stats["files_loaded"] == 1
            assert stats["chunks_created"] == 1
            assert stats["vectors_embedded"] == 1


class TestIndexingPipelineExecution:
    """Tests for successful pipeline execution."""

    @patch("pycontextify.indexer.FileCrawler")
    @patch("pycontextify.indexer.FileLoaderFactory")
    @patch("pycontextify.indexer.ChunkerFactory")
    @patch("pycontextify.indexer.postprocess_file")
    @patch("pycontextify.indexer.postprocess_filebase")
    def test_successful_indexing(
        self,
        mock_postprocess_fb,
        mock_postprocess_f,
        mock_chunker_factory,
        mock_loader_factory,
        mock_crawler_class,
        pipeline,
        mock_vector_store,
        mock_metadata_store,
    ):
        """Should execute complete pipeline successfully."""
        with TemporaryDirectory() as tmpdir:
            # Setup mocks
            mock_crawler = Mock()
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def test(): pass")
            mock_crawler.crawl.return_value = [test_file]
            mock_crawler_class.return_value = mock_crawler

            mock_loader = Mock()
            mock_loader.load.return_value = [
                {
                    "text": "test content",
                    "metadata": {
                        "full_path": str(test_file),
                        "file_extension": "py",
                    },
                }
            ]
            mock_loader_factory.return_value = mock_loader

            mock_chunker_factory.chunk_normalized_docs.return_value = [
                {
                    "text": "chunk 1",
                    "metadata": {
                        "full_path": str(test_file),
                        "file_extension": "py",
                    },
                },
                {
                    "text": "chunk 2",
                    "metadata": {
                        "full_path": str(test_file),
                        "file_extension": "py",
                    },
                },
            ]

            mock_postprocess_f.side_effect = lambda x: x
            mock_postprocess_fb.side_effect = lambda x: x

            # Execute
            stats = pipeline.index_filebase(tmpdir, "test_topic")

            # Verify
            assert stats["topic"] == "test_topic"
            assert stats["files_crawled"] == 1
            assert stats["files_loaded"] == 1
            assert stats["chunks_created"] == 2
            assert stats["vectors_embedded"] == 2
            assert stats["errors"] == 0
            assert "duration_seconds" in stats

            # Verify calls
            mock_crawler.crawl.assert_called_once()
            mock_loader.load.assert_called_once()
            mock_vector_store.add_vectors.assert_called_once()
            assert mock_metadata_store.add_chunk.call_count == 2

    @patch("pycontextify.indexer.FileCrawler")
    def test_handles_no_files_found(self, mock_crawler_class, pipeline):
        """Should handle case when no files are found."""
        with TemporaryDirectory() as tmpdir:
            mock_crawler = Mock()
            mock_crawler.crawl.return_value = []
            mock_crawler_class.return_value = mock_crawler

            stats = pipeline.index_filebase(tmpdir, "test")

            assert stats["files_crawled"] == 0
            assert stats["files_loaded"] == 0
            assert stats["chunks_created"] == 0
            assert stats["errors"] == 0

    @patch("pycontextify.indexer.FileCrawler")
    @patch("pycontextify.indexer.FileLoaderFactory")
    def test_handles_no_chunks_created(
        self, mock_loader_factory, mock_crawler_class, pipeline
    ):
        """Should handle case when no chunks are created."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")

            mock_crawler = Mock()
            mock_crawler.crawl.return_value = [test_file]
            mock_crawler_class.return_value = mock_crawler

            mock_loader = Mock()
            mock_loader.load.return_value = []  # No documents loaded
            mock_loader_factory.return_value = mock_loader

            stats = pipeline.index_filebase(tmpdir, "test")

            assert stats["files_crawled"] == 1
            assert stats["files_loaded"] == 0
            assert stats["chunks_created"] == 0


class TestIndexingPipelineErrorHandling:
    """Tests for error handling at each step."""

    @patch("pycontextify.indexer.FileCrawler")
    def test_handles_crawl_error(self, mock_crawler_class, pipeline):
        """Should handle crawl errors gracefully."""
        with TemporaryDirectory() as tmpdir:
            mock_crawler = Mock()
            mock_crawler.crawl.side_effect = RuntimeError("Crawl failed")
            mock_crawler_class.return_value = mock_crawler

            stats = pipeline.index_filebase(tmpdir, "test")

            assert stats["errors"] == 1
            assert any("crawl" in e["stage"] for e in stats["error_samples"])
            assert stats["files_crawled"] == 0

    @patch("pycontextify.indexer.FileCrawler")
    @patch("pycontextify.indexer.FileLoaderFactory")
    def test_handles_load_error_per_file(
        self, mock_loader_factory, mock_crawler_class, pipeline
    ):
        """Should handle per-file load errors and continue."""
        with TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.txt"
            file2 = Path(tmpdir) / "file2.txt"
            file1.write_text("test1")
            file2.write_text("test2")

            mock_crawler = Mock()
            mock_crawler.crawl.return_value = [file1, file2]
            mock_crawler_class.return_value = mock_crawler

            mock_loader = Mock()
            # First file fails, second succeeds
            mock_loader.load.side_effect = [
                RuntimeError("Load failed"),
                [
                    {
                        "text": "content",
                        "metadata": {
                            "full_path": str(file2),
                            "file_extension": "txt",
                        },
                    }
                ],
            ]
            mock_loader_factory.return_value = mock_loader

            # Need to mock chunker too
            with patch("pycontextify.indexer.ChunkerFactory") as mock_chunker:
                mock_chunker.chunk_normalized_docs.return_value = [
                    {
                        "text": "chunk",
                        "metadata": {
                            "full_path": str(file2),
                            "file_extension": "txt",
                        },
                    }
                ]

                with patch("pycontextify.indexer.postprocess_file") as mock_pf:
                    mock_pf.side_effect = lambda x: x
                    with patch("pycontextify.indexer.postprocess_filebase") as mock_pfb:
                        mock_pfb.side_effect = lambda x: x

                        stats = pipeline.index_filebase(tmpdir, "test")

            # Should process second file despite first failing
            assert stats["errors"] == 1
            assert stats["files_loaded"] == 1
            assert stats["chunks_created"] == 1

    @patch("pycontextify.indexer.FileCrawler")
    @patch("pycontextify.indexer.FileLoaderFactory")
    @patch("pycontextify.indexer.ChunkerFactory")
    @patch("pycontextify.indexer.postprocess_file")
    @patch("pycontextify.indexer.postprocess_filebase")
    def test_handles_embedding_error(
        self,
        mock_postprocess_fb,
        mock_postprocess_f,
        mock_chunker_factory,
        mock_loader_factory,
        mock_crawler_class,
        pipeline,
        mock_embedder_service,
    ):
        """Should handle embedding errors."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")

            mock_crawler = Mock()
            mock_crawler.crawl.return_value = [test_file]
            mock_crawler_class.return_value = mock_crawler

            mock_loader = Mock()
            mock_loader.load.return_value = [
                {
                    "text": "content",
                    "metadata": {"full_path": str(test_file), "file_extension": "txt"},
                }
            ]
            mock_loader_factory.return_value = mock_loader

            mock_chunker_factory.chunk_normalized_docs.return_value = [
                {
                    "text": "chunk",
                    "metadata": {"full_path": str(test_file), "file_extension": "txt"},
                }
            ]

            mock_postprocess_f.side_effect = lambda x: x
            mock_postprocess_fb.side_effect = lambda x: x

            # Make embedder fail
            mock_embedder = mock_embedder_service.get_embedder.return_value
            mock_embedder.embed_texts.side_effect = RuntimeError("Embedding failed")

            stats = pipeline.index_filebase(tmpdir, "test")

            assert stats["errors"] == 1
            assert any("embed" in e["stage"] for e in stats["error_samples"])
            assert stats["vectors_embedded"] == 0

    @patch("pycontextify.indexer.FileCrawler")
    @patch("pycontextify.indexer.FileLoaderFactory")
    @patch("pycontextify.indexer.ChunkerFactory")
    @patch("pycontextify.indexer.postprocess_file")
    @patch("pycontextify.indexer.postprocess_filebase")
    def test_handles_storage_error(
        self,
        mock_postprocess_fb,
        mock_postprocess_f,
        mock_chunker_factory,
        mock_loader_factory,
        mock_crawler_class,
        pipeline,
        mock_vector_store,
    ):
        """Should handle storage errors."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")

            mock_crawler = Mock()
            mock_crawler.crawl.return_value = [test_file]
            mock_crawler_class.return_value = mock_crawler

            mock_loader = Mock()
            mock_loader.load.return_value = [
                {
                    "text": "content",
                    "metadata": {"full_path": str(test_file), "file_extension": "txt"},
                }
            ]
            mock_loader_factory.return_value = mock_loader

            mock_chunker_factory.chunk_normalized_docs.return_value = [
                {
                    "text": "chunk",
                    "metadata": {"full_path": str(test_file), "file_extension": "txt"},
                }
            ]

            mock_postprocess_f.side_effect = lambda x: x
            mock_postprocess_fb.side_effect = lambda x: x

            # Make storage fail
            mock_vector_store.add_vectors.side_effect = RuntimeError("Storage failed")

            stats = pipeline.index_filebase(tmpdir, "test")

            assert stats["errors"] == 1
            assert any("store" in e["stage"] for e in stats["error_samples"])


class TestIndexingPipelineStatsTracking:
    """Tests for statistics tracking."""

    @patch("pycontextify.indexer.FileCrawler")
    @patch("pycontextify.indexer.FileLoaderFactory")
    @patch("pycontextify.indexer.ChunkerFactory")
    @patch("pycontextify.indexer.postprocess_file")
    @patch("pycontextify.indexer.postprocess_filebase")
    def test_tracks_timing(
        self,
        mock_postprocess_fb,
        mock_postprocess_f,
        mock_chunker_factory,
        mock_loader_factory,
        mock_crawler_class,
        pipeline,
    ):
        """Should track timing information."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")

            mock_crawler = Mock()
            mock_crawler.crawl.return_value = [test_file]
            mock_crawler_class.return_value = mock_crawler

            mock_loader = Mock()
            mock_loader.load.return_value = [
                {
                    "text": "content",
                    "metadata": {"full_path": str(test_file), "file_extension": "txt"},
                }
            ]
            mock_loader_factory.return_value = mock_loader

            mock_chunker_factory.chunk_normalized_docs.return_value = [
                {
                    "text": "chunk",
                    "metadata": {"full_path": str(test_file), "file_extension": "txt"},
                }
            ]

            mock_postprocess_f.side_effect = lambda x: x
            mock_postprocess_fb.side_effect = lambda x: x

            stats = pipeline.index_filebase(tmpdir, "test")

            assert "started_at" in stats
            assert "finished_at" in stats
            assert "duration_seconds" in stats
            assert stats["duration_seconds"] >= 0

    def test_limits_error_samples(self, pipeline):
        """Should limit error samples to 10."""
        stats = {
            "topic": "test",
            "chunks_created": 0,
            "files_loaded": 0,
            "error_samples": [
                {"stage": "test", "error": f"error {i}"} for i in range(20)
            ],
        }

        finalized = pipeline._finalize_stats(stats, 0)

        assert len(finalized["error_samples"]) == 10

    @patch("pycontextify.indexer.FileCrawler")
    def test_includes_base_path_in_stats(self, mock_crawler_class, pipeline):
        """Should include base_path in stats."""
        with TemporaryDirectory() as tmpdir:
            mock_crawler = Mock()
            mock_crawler.crawl.return_value = []
            mock_crawler_class.return_value = mock_crawler

            stats = pipeline.index_filebase(tmpdir, "test")

            assert "base_path" in stats
            assert Path(stats["base_path"]).is_absolute()


class TestChunkMetadataCreation:
    """Tests for chunk metadata creation."""

    def test_creates_metadata_for_code_file(self, pipeline):
        """Should correctly identify code files."""
        chunk_dict = {
            "text": "def test(): pass",
            "metadata": {
                "full_path": "/path/to/file.py",
                "file_extension": "py",
                "embedding_provider": "test",
                "embedding_model": "model",
            },
        }

        metadata = pipeline._create_chunk_metadata(chunk_dict)

        assert metadata.source_type.value == "code"
        assert metadata.file_extension == "py"
        assert metadata.chunk_text == "def test(): pass"

    def test_creates_metadata_for_document_file(self, pipeline):
        """Should correctly identify document files."""
        chunk_dict = {
            "text": "Document content",
            "metadata": {
                "full_path": "/path/to/doc.md",
                "file_extension": "md",
                "embedding_provider": "test",
                "embedding_model": "model",
            },
        }

        metadata = pipeline._create_chunk_metadata(chunk_dict)

        assert metadata.source_type.value == "document"
        assert metadata.file_extension == "md"

    def test_uses_default_embedding_info(self, pipeline):
        """Should use defaults if embedding info missing."""
        chunk_dict = {
            "text": "content",
            "metadata": {
                "full_path": "/path/to/file.txt",
                "file_extension": "txt",
            },
        }

        metadata = pipeline._create_chunk_metadata(chunk_dict)

        assert metadata.embedding_provider == "sentence_transformers"
        assert metadata.embedding_model == "all-mpnet-base-v2"
