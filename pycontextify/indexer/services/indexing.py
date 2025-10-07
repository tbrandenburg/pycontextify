"""Content ingestion workflows for the index."""

from __future__ import annotations

import logging
from typing import Any, Dict

from ...chunker import ChunkerFactory
from ...storage.metadata import MetadataStore, SourceType
from ..loaders import LoaderFactory
from .embedding import EmbeddingService
from ...orchestrator.config import Config


class IndexingService:
    """Coordinate loaders, chunkers and embeddings for ingestion."""

    def __init__(
        self,
        config: Config,
        metadata_store: MetadataStore,
        embedding_service: EmbeddingService,
    ) -> None:
        self._config = config
        self._metadata_store = metadata_store
        self._embedding_service = embedding_service
        self._logger = logging.getLogger(__name__)

    def index_codebase(self, path: str) -> Dict[str, Any]:
        """Index a source tree of code files."""
        self._logger.info("Starting codebase indexing: %s", path)
        try:
            loader = LoaderFactory.get_loader(
                SourceType.CODE, max_file_size_mb=self._config.max_file_size_mb
            )
            files = loader.load(path)
            if not files:
                return {"error": "No files found to index"}

            chunks_added = 0
            for file_path, content in files:
                chunks_added += self._process_content(content, file_path, SourceType.CODE)

            return {
                "files_processed": len(files),
                "chunks_added": chunks_added,
                "source_type": "code",
            }
        except Exception as exc:
            error_msg = f"Failed to index codebase {path}: {exc}"
            self._logger.error(error_msg)
            return {"error": error_msg}

    def index_document(self, path: str) -> Dict[str, Any]:
        """Index a single document file."""
        self._logger.info("Starting document indexing: %s", path)
        try:
            loader = LoaderFactory.get_loader(
                SourceType.DOCUMENT, pdf_engine=self._config.pdf_engine
            )
            files = loader.load(path)
            if not files:
                return {"error": "Could not load document"}

            file_path, content = files[0]
            chunks_added = self._process_content(
                content, file_path, SourceType.DOCUMENT
            )
            return {
                "file_processed": file_path,
                "chunks_added": chunks_added,
                "source_type": "document",
            }
        except Exception as exc:
            error_msg = f"Failed to index document {path}: {exc}"
            self._logger.error(error_msg)
            return {"error": error_msg}

    def index_webpage(
        self, url: str, recursive: bool = False, max_depth: int = 1
    ) -> Dict[str, Any]:
        """Index the content of a web page or crawl."""
        self._logger.info(
            "Starting webpage indexing: %s (recursive=%s, max_depth=%s)",
            url,
            recursive,
            max_depth,
        )
        try:
            loader = LoaderFactory.get_loader(
                SourceType.WEBPAGE, delay_seconds=self._config.crawl_delay_seconds
            )
            pages = loader.load(url, recursive=recursive, max_depth=max_depth)
            if not pages:
                return {"error": "Could not load any web pages"}

            chunks_added = 0
            for page_url, content in pages:
                chunks_added += self._process_content(
                    content, page_url, SourceType.WEBPAGE
                )

            return {
                "pages_processed": len(pages),
                "chunks_added": chunks_added,
                "source_type": "webpage",
                "recursive": recursive,
                "max_depth": max_depth,
            }
        except Exception as exc:
            error_msg = f"Failed to index webpage {url}: {exc}"
            self._logger.error(error_msg)
            return {"error": error_msg}

    def _process_content(
        self, content: str, source_path: str, source_type: SourceType
    ) -> int:
        """Chunk and embed the provided content."""
        existing_chunks = self._metadata_store.get_chunks_by_source_path(source_path)
        if existing_chunks:
            self._logger.info(
                "Found %s existing chunks for %s, removing for re-indexing",
                len(existing_chunks),
                source_path,
            )
            vector_store = self._embedding_service.vector_store
            if vector_store is not None:
                faiss_ids_to_remove = []
                for chunk in existing_chunks:
                    faiss_id = self._metadata_store.get_faiss_id(chunk.chunk_id)
                    if faiss_id is not None:
                        faiss_ids_to_remove.append(faiss_id)
                if faiss_ids_to_remove:
                    vector_store.remove_vectors(faiss_ids_to_remove)

            for chunk in existing_chunks:
                faiss_id = self._metadata_store.get_faiss_id(chunk.chunk_id)
                if faiss_id is not None:
                    self._metadata_store.remove_chunk(faiss_id)
            self._logger.info(
                "Removed %s existing chunks for re-indexing", len(existing_chunks)
            )

        chunker = ChunkerFactory.get_chunker(source_type, self._config)
        self._embedding_service.ensure_loaded()
        embedder = self._embedding_service.embedder

        chunks = chunker.chunk_text(
            content,
            source_path,
            embedder.get_provider_name(),
            embedder.get_model_name(),
        )
        if not chunks:
            return 0

        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store = self._embedding_service.ensure_vector_store()
        faiss_ids = vector_store.add_vectors(embeddings)

        for chunk, faiss_id in zip(chunks, faiss_ids):
            self._metadata_store.add_chunk(chunk)

        return len(chunks)
