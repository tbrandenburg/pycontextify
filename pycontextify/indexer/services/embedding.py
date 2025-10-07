"""Embedding and vector store lifecycle management."""

from __future__ import annotations

import logging
from typing import Optional

from ...embedder import EmbedderFactory
from ...orchestrator.config import Config
from ...storage.metadata import MetadataStore
from ...storage.vector import VectorStore


class EmbeddingService:
    """Coordinates lazy embedder and vector store initialization."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._embedder = None
        self._vector_store: Optional[VectorStore] = None
        self._initialized = False
        self._logger = logging.getLogger(__name__)

    @property
    def embedder(self):
        """Return the active embedder, loading it on demand."""
        self.ensure_loaded()
        return self._embedder

    @property
    def vector_store(self) -> Optional[VectorStore]:
        """Return the managed vector store instance if available."""
        return self._vector_store

    def ensure_loaded(self) -> None:
        """Lazily load the embedder and initialise the vector store."""
        if self._initialized:
            return

        embedding_config = self._config.get_embedding_config()
        self._logger.info(
            "Lazy loading embedder: %s with model %s",
            embedding_config["provider"],
            embedding_config["model"],
        )

        self._embedder = EmbedderFactory.create_embedder(
            provider=embedding_config["provider"],
            model_name=embedding_config["model"],
            **{
                key: value
                for key, value in embedding_config.items()
                if key not in {"provider", "model"}
            },
        )
        self._initialized = True
        self._initialize_vector_store()
        self._logger.info(
            "Successfully loaded embedder: %s",
            self._embedder.get_provider_name(),
        )

    def _initialize_vector_store(self) -> None:
        """Create the vector store once the embedder dimension is known."""
        if self._embedder is None:
            return
        if self._vector_store is not None:
            return

        dimension = self._embedder.get_dimension()
        self._vector_store = VectorStore(dimension, self._config)
        self._logger.info("Initialized vector store with dimension %s", dimension)

    def ensure_vector_store(self) -> VectorStore:
        """Return an initialised vector store, creating it on demand."""
        self.ensure_loaded()
        self._initialize_vector_store()
        if self._vector_store is None:
            raise RuntimeError("Vector store could not be initialised")
        return self._vector_store

    def is_initialized(self) -> bool:
        """Return True when the embedder has been initialised."""
        return self._initialized

    def get_loaded_embedder(self):
        """Return the embedder if already initialised without triggering loads."""
        if not self._initialized:
            return None
        return self._embedder

    def is_vector_store_empty(self) -> bool:
        """Return True if the vector store has no vectors."""
        if self._vector_store is None:
            return True
        return self._vector_store.is_empty()

    def validate_embedding(self, metadata_store: MetadataStore) -> bool:
        """Validate compatibility between stored metadata and the embedder."""
        if metadata_store.get_stats().get("total_chunks", 0) == 0:
            return True

        self.ensure_loaded()
        if self._embedder is None:
            return False
        return metadata_store.validate_embedding_compatibility(
            self._embedder.get_provider_name(),
            self._embedder.get_model_name(),
        )

    def save_vector_store(self, path: str) -> None:
        """Persist the vector store to disk when present."""
        if self._vector_store is None:
            return
        self._vector_store.save_to_file(path)

    def load_vector_store(self, path: str) -> None:
        """Load vector store contents from disk, creating the store if needed."""
        vector_store = self.ensure_vector_store()
        vector_store.load_from_file(path)

    def clear_vector_store(self) -> None:
        """Remove all vectors from the managed store."""
        if self._vector_store is not None:
            self._vector_store.clear()

    def cleanup(self) -> None:
        """Release embedder resources and reset state."""
        if self._embedder is not None:
            try:
                self._embedder.cleanup()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                self._logger.warning("Embedder cleanup failed: %s", exc)
        self._embedder = None
        self._vector_store = None
        self._initialized = False
