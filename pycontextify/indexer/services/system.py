"""System and health status reporting."""

from __future__ import annotations

import logging
from typing import Any, Dict

import psutil

from ...orchestrator.config import Config
from ...storage.metadata import MetadataStore
from .embedding import EmbeddingService
from .search import SearchService


class SystemStatusService:
    """Aggregate runtime information for monitoring."""

    def __init__(
        self,
        config: Config,
        metadata_store: MetadataStore,
        embedding_service: EmbeddingService,
        search_service: SearchService,
    ) -> None:
        self._config = config
        self._metadata_store = metadata_store
        self._embedding_service = embedding_service
        self._search_service = search_service
        self._logger = logging.getLogger(__name__)

    def get_status(self) -> Dict[str, Any]:
        """Return a structured snapshot of the system state."""
        try:
            metadata_stats = self._metadata_store.get_stats()
            vector_store = self._embedding_service.vector_store
            vector_stats = (
                vector_store.get_index_info() if vector_store else {"total_vectors": 0, "dimension": None}
            )

            index_stats = {
                "total_chunks": metadata_stats.get("total_chunks", 0),
                "total_documents": len(
                    {chunk.source_path for chunk in self._metadata_store.get_all_chunks()}
                ),
                "total_vectors": vector_stats.get("total_vectors", 0),
                "source_types": metadata_stats.get("source_types", {}),
            }

            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            embedder = self._embedding_service.get_loaded_embedder()
            if not embedder:
                embedding_info = {
                    "provider": self._config.embedding_provider,
                    "model": self._config.embedding_model,
                    "dimension": None,
                    "is_available": False,
                }
            else:
                embedding_info = {
                    "provider": embedder.get_provider_name(),
                    "model": embedder.get_model_name(),
                    "dimension": embedder.get_dimension(),
                    "is_available": embedder.is_available(),
                }

            paths = self._config.get_index_paths()
            essential_paths = {k: v for k, v in paths.items() if k in {"metadata", "index"}}
            persistence_info = {
                "auto_persist": self._config.auto_persist,
                "index_dir": str(self._config.index_dir),
                "index_files_exist": all(path.exists() for path in essential_paths.values()),
                "last_modified": {},
            }
            for name, path in paths.items():
                if path.exists():
                    persistence_info["last_modified"][name] = path.stat().st_mtime

            hybrid_info = self._search_service.get_hybrid_stats()

            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage("/")
            performance = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_mb": memory_mb,
                "memory_total_mb": memory_info.total / (1024 * 1024),
                "memory_available_mb": memory_info.available / (1024 * 1024),
                "memory_usage_percent": memory_info.percent,
                "disk_usage_percent": disk_usage.percent,
                "disk_free_gb": disk_usage.free / (1024 * 1024 * 1024),
            }

            return {
                "status": "healthy",
                "index_stats": index_stats,
                "metadata": metadata_stats,
                "relationships": {},
                "vector_store": vector_stats,
                "embedding": embedding_info,
                "hybrid_search": hybrid_info,
                "performance": performance,
                "persistence": persistence_info,
                "configuration": self._config.get_config_summary(),
            }
        except Exception as exc:  # pragma: no cover - defensive path
            self._logger.error("Failed to gather system status: %s", exc)
            return {
                "status": "error",
                "error": str(exc),
                "metadata": {},
                "relationships": {},
                "vector_store": {},
                "embedding": {
                    "provider": getattr(self._config, "embedding_provider", "unknown"),
                    "model": getattr(self._config, "embedding_model", "unknown"),
                    "dimension": None,
                    "is_available": False,
                },
                "hybrid_search": {},
                "performance": {},
                "persistence": {},
                "configuration": {},
            }
