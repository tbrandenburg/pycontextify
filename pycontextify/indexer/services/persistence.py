"""Persistence utilities for index artefacts."""

from __future__ import annotations

import logging
from typing import List

from ...orchestrator.config import Config
from ...storage.metadata import MetadataStore
from .embedding import EmbeddingService


class PersistenceService:
    """Handle persistence concerns such as auto-saving and cleanup."""

    def __init__(self, config: Config, embedding_service: EmbeddingService) -> None:
        self._config = config
        self._embedding_service = embedding_service
        self._logger = logging.getLogger(__name__)

    def auto_save(self, metadata_store: MetadataStore) -> None:
        """Persist metadata and vector store when auto-persist is enabled."""
        if not self._config.auto_persist:
            return

        self._config.ensure_index_directory()
        paths = self._config.get_index_paths()

        index_path = str(paths["index"])
        metadata_path = str(paths["metadata"])

        self._embedding_service.save_vector_store(index_path)
        metadata_store.save_to_file(metadata_path, self._config.compress_metadata)
        self._logger.info("Auto-saved index to disk")

    def remove_index_files(self) -> List[str]:
        """Delete persisted index artefacts and return the removed paths."""
        removed: List[str] = []
        for path in self._config.get_index_paths().values():
            if path.exists():
                path.unlink()
                removed.append(str(path))
                self._logger.info("Removed %s", path)
        return removed
