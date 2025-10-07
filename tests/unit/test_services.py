"""Unit tests for the refactored indexing services."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from pycontextify.indexer.services.embedding import EmbeddingService
from pycontextify.indexer.services.persistence import PersistenceService
from pycontextify.storage.metadata import ChunkMetadata, MetadataStore, SourceType


class DummyEmbedder:
    """Simple stand-in for an embedding backend used in service tests."""

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self._cleaned = False

    def get_provider_name(self) -> str:  # pragma: no cover - trivial
        return "dummy"

    def get_model_name(self) -> str:  # pragma: no cover - trivial
        return "dummy-model"

    def get_dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts):  # pragma: no cover - not used in tests
        return np.zeros((len(texts), self._dimension), dtype=np.float32)

    def embed_single(self, text):  # pragma: no cover - not used in tests
        return np.zeros(self._dimension, dtype=np.float32)

    def is_available(self) -> bool:  # pragma: no cover - trivial
        return True

    def cleanup(self) -> None:
        self._cleaned = True


class DummyVectorStore:
    """Minimal vector store spy capturing persistence calls."""

    def __init__(self, dimension: int, config) -> None:  # pragma: no cover - simple init
        self.dimension = dimension
        self.config = config
        self.saved_path: str | None = None

    def save_to_file(self, path: str) -> None:
        self.saved_path = path

    def clear(self) -> None:  # pragma: no cover - not used
        pass

    def is_empty(self) -> bool:  # pragma: no cover - not used
        return False

    def get_index_info(self) -> Dict[str, int]:  # pragma: no cover - not used
        return {"total_vectors": 0, "dimension": self.dimension}


class DummyConfig:
    """Lightweight configuration stub for service testing."""

    def __init__(self, base_dir: Path) -> None:
        self.auto_persist = True
        self.compress_metadata = False
        self.index_dir = base_dir
        self.index_name = "test_index"
        self.embedding_provider = "dummy"
        self.embedding_model = "dummy-model"

    def get_embedding_config(self) -> Dict[str, str]:
        return {"provider": self.embedding_provider, "model": self.embedding_model}

    def ensure_index_directory(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def get_index_paths(self) -> Dict[str, Path]:
        base_path = self.index_dir / self.index_name
        return {
            "index": base_path.with_suffix(".faiss"),
            "metadata": base_path.with_suffix(".pkl"),
            "relationships": base_path.with_name(f"{base_path.name}_relationships.pkl"),
        }

    def get_config_summary(self) -> Dict[str, str]:  # pragma: no cover - diagnostic
        return {}


@pytest.fixture()
def dummy_config(tmp_path: Path) -> DummyConfig:
    return DummyConfig(tmp_path)


def test_embedding_service_lazy_initialisation(monkeypatch, dummy_config):
    """Embedder and vector store should only initialise on demand."""

    created = {}

    def fake_create_embedder(*_, **__):
        created["called"] = True
        return DummyEmbedder()

    monkeypatch.setattr(
        "pycontextify.indexer.services.embedding.EmbedderFactory.create_embedder",
        fake_create_embedder,
    )
    monkeypatch.setattr(
        "pycontextify.indexer.services.embedding.VectorStore",
        DummyVectorStore,
    )

    service = EmbeddingService(dummy_config)

    assert not service.is_initialized()
    service.ensure_loaded()

    assert service.is_initialized()
    assert created["called"] is True
    assert isinstance(service.embedder, DummyEmbedder)
    assert isinstance(service.vector_store, DummyVectorStore)


def test_persistence_service_auto_save_creates_files(tmp_path: Path):
    """Auto-save should persist metadata and vector store artefacts."""

    config = DummyConfig(tmp_path)
    embedding_service = EmbeddingService(config)
    embedding_service._vector_store = DummyVectorStore(4, config)

    metadata_store = MetadataStore()
    metadata_store.add_chunk(
        ChunkMetadata(
            source_path="demo.txt",
            source_type=SourceType.DOCUMENT,
            chunk_text="demo",
        )
    )

    persistence = PersistenceService(config, embedding_service)
    persistence.auto_save(metadata_store)

    paths = config.get_index_paths()
    assert paths["metadata"].exists()
    assert embedding_service.vector_store.saved_path == str(paths["index"])
