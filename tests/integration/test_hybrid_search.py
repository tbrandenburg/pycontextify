"""Integration tests for hybrid search backed by the metadata store."""

from pycontextify.search_hybrid import HybridSearchEngine
from pycontextify.storage_metadata import ChunkMetadata, MetadataStore, SourceType


def test_hybrid_search_with_metadata_store():
    """Hybrid search returns metadata-backed results when vector scores are provided."""
    metadata_store = MetadataStore()
    engine = HybridSearchEngine()

    chunk1 = ChunkMetadata(
        chunk_id="chunk1",
        source_path="/test.py",
        chunk_text="Python programming language",
        source_type=SourceType.CODE,
    )
    chunk2 = ChunkMetadata(
        chunk_id="chunk2",
        source_path="/test.py",
        chunk_text="JavaScript web development",
        source_type=SourceType.CODE,
    )

    faiss_id1 = metadata_store.add_chunk(chunk1)
    faiss_id2 = metadata_store.add_chunk(chunk2)

    engine.add_documents([chunk1.chunk_id, chunk2.chunk_id], [chunk1.chunk_text, chunk2.chunk_text])
    results = engine.search(
        query="python",
        vector_scores=[(faiss_id1, 0.9), (faiss_id2, 0.1)],
        metadata_store=metadata_store,
        top_k=2,
    )

    assert results
    assert results[0].chunk_id == chunk1.chunk_id
    assert results[0].source_path == chunk1.source_path
