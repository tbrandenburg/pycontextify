"""Central IndexManager orchestrating indexing services."""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..orchestrator.config import Config
from ..search.models import SearchResponse
from ..storage.metadata import MetadataStore
from .services import (
    BootstrapService,
    EmbeddingService,
    IndexingService,
    PersistenceService,
    SearchService,
    SystemStatusService,
)

logger = logging.getLogger(__name__)


class IndexManager:
    """High-level façade coordinating indexing, persistence and search."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.metadata_store = MetadataStore()

        self._embedding_service = EmbeddingService(config)
        self._persistence_service = PersistenceService(config, self._embedding_service)
        self._search_service = SearchService(config, self.metadata_store, self._embedding_service)
        self._indexing_service = IndexingService(
            config, self.metadata_store, self._embedding_service
        )
        self._bootstrap_service = BootstrapService(
            config, self.metadata_store, self._embedding_service
        )
        self._system_status_service = SystemStatusService(
            config,
            self.metadata_store,
            self._embedding_service,
            self._search_service,
        )

        self.performance_logger = self._search_service.performance_logger

        if self.config.auto_load:
            self._bootstrap_service.auto_load()

    # ------------------------------------------------------------------
    # Properties exposing underlying services for backwards compatibility
    # ------------------------------------------------------------------
    @property
    def embedder(self):
        """Expose the lazily loaded embedder instance."""
        return self._embedding_service.embedder

    @property
    def vector_store(self):
        """Expose the managed vector store."""
        return self._embedding_service.vector_store

    @property
    def hybrid_search(self):
        """Expose the hybrid search engine when available."""
        return self._search_service.get_hybrid_engine()

    # ------------------------------------------------------------------
    # Delegated behaviour
    # ------------------------------------------------------------------
    def _ensure_embedder_loaded(self) -> None:
        """Compatibility helper for scripts invoking the old private API."""
        self._embedding_service.ensure_loaded()

    def index_codebase(self, path: str) -> Dict[str, Any]:
        """Index a codebase directory."""
        stats = self._indexing_service.index_codebase(path)
        if "error" in stats:
            return stats
        self._persistence_service.auto_save(self.metadata_store)
        return self._augment_with_embedding_info(stats)

    def index_document(self, path: str) -> Dict[str, Any]:
        """Index a single document."""
        stats = self._indexing_service.index_document(path)
        if "error" in stats:
            return stats
        self._persistence_service.auto_save(self.metadata_store)
        return self._augment_with_embedding_info(stats)

    def index_webpage(
        self, url: str, recursive: bool = False, max_depth: int = 1
    ) -> Dict[str, Any]:
        """Index web content."""
        stats = self._indexing_service.index_webpage(url, recursive, max_depth)
        if "error" in stats:
            return stats
        self._persistence_service.auto_save(self.metadata_store)
        return self._augment_with_embedding_info(stats)

    def search(
        self, query: str, top_k: int = 5, display_format: str = "readable"
    ) -> SearchResponse:
        """Perform semantic search with optional hybrid enhancement."""
        return self._search_service.search(query, top_k, display_format)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get search performance statistics summary."""
        return self._search_service.get_performance_summary()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return self._system_status_service.get_status()

    def clear_index(self, remove_files: bool = False) -> Dict[str, Any]:
        """Clear all indexed data."""
        try:
            self.metadata_store.clear()
            self._embedding_service.clear_vector_store()

            removed_files: list[str] = []
            if remove_files:
                removed_files = self._persistence_service.remove_index_files()

            logger.info("Cleared index data")
            return {"success": True, "files_removed": remove_files}
        except Exception as exc:  # pragma: no cover - defensive guard
            error_msg = f"Failed to clear index: {exc}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def save_index(self) -> Dict[str, Any]:
        """Manually save index to disk."""
        try:
            self._persistence_service.auto_save(self.metadata_store)
            return {"success": True}
        except Exception as exc:  # pragma: no cover - defensive guard
            error_msg = f"Failed to save index: {exc}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "IndexManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.config.auto_persist and exc_type is None:
            try:
                logger.info("Auto-saving index during cleanup")
                self.save_index()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Auto-save failed during cleanup: %s", exc)

        self._embedding_service.cleanup()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _augment_with_embedding_info(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Attach embedding provider/model details to indexing statistics."""
        self._embedding_service.ensure_loaded()
        embedder = self._embedding_service.embedder
        stats.update(
            {
                "embedding_provider": embedder.get_provider_name(),
                "embedding_model": embedder.get_model_name(),
            }
        )
        return stats
