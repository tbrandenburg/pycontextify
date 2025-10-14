"""Search components and service for PyContextify.

This module provides the SearchService for orchestrating search operations
and exports search-related models and utilities.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from .search_hybrid import HybridSearchEngine
from .search_models import (
    SearchErrorCode,
    SearchPerformanceLogger,
    SearchResponse,
    SearchResult,
    create_search_performance_info,
    create_search_provenance,
    create_structured_metadata,
    create_structured_scores,
    enhance_search_results_with_ranking,
)

logger = logging.getLogger(__name__)


class SearchService:
    """Orchestrates search operations with vector and hybrid strategies.

    This service provides a unified search interface that automatically
    handles vector similarity search and optionally enhances results with
    keyword-based hybrid search.
    """

    def __init__(
        self, config, embedder_service, vector_store, metadata_store, hybrid_search=None
    ):
        """Initialize search service.

        Args:
            config: Configuration object
            embedder_service: EmbedderService instance
            vector_store: VectorStore instance
            metadata_store: MetadataStore instance
            hybrid_search: HybridSearchEngine instance (optional)
        """
        self.config = config
        self.embedder_service = embedder_service
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.hybrid_search = hybrid_search

    def search(
        self, query: str, top_k: int = 5, display_format: str = "readable"
    ) -> SearchResponse:
        """Perform search with optional hybrid enhancement.

        Args:
            query: Search query text
            top_k: Number of results to return
            display_format: Output format ('readable', 'structured', 'summary')

        Returns:
            SearchResponse with results and metadata
        """
        start_time = time.time()

        try:
            # Check if index is populated
            if self.vector_store is None or self.vector_store.is_empty():
                return SearchResponse.create_error(
                    query=query,
                    error="No indexed content available. Please index content first.",
                    error_code=SearchErrorCode.NO_CONTENT.value,
                    search_config=self._get_search_config(),
                    recovery_suggestions=[
                        "Use index_filebase() to add content",
                        "Check if auto_load is enabled and index files exist",
                        "Verify the vector store was initialized properly",
                    ],
                )

            # Get embedder
            embedder = self.embedder_service.get_embedder()

            # Generate query embedding
            query_embedding = embedder.embed_single(query)

            # Perform vector search (get extra candidates for hybrid filtering)
            distances, indices = self.vector_store.search(query_embedding, top_k * 2)

            # Create initial vector results
            vector_results = self._create_vector_results(distances, indices)

            # Apply hybrid search if enabled
            if self.config.use_hybrid_search and self.hybrid_search:
                try:
                    self._ensure_hybrid_index()
                    final_results = self.hybrid_search.search(
                        query=query,
                        vector_scores=list(
                            zip(indices, [1 - d for d in distances])
                        ),  # Convert distance to similarity
                        metadata_store=self.metadata_store,
                        top_k=top_k,
                    )
                    # Convert HybridSearchEngine results to SearchResult
                    final_results = self._convert_hybrid_results(final_results)
                except Exception as e:
                    logger.warning(f"Hybrid search failed, using vector only: {e}")
                    final_results = vector_results[:top_k]
            else:
                final_results = vector_results[:top_k]

            # Enhance with ranking information
            enhanced_results = enhance_search_results_with_ranking(
                results=final_results,
                query=query,
                include_explanations=True,
                include_confidence=True,
            )

            # Create performance info
            performance = create_search_performance_info(
                start_time=start_time,
                search_mode="hybrid" if self.config.use_hybrid_search else "vector",
                total_candidates=len(vector_results),
            )

            # Create successful response
            response = SearchResponse.create_success(
                query=query,
                results=enhanced_results,
                search_config=self._get_search_config(),
                performance=performance,
            )

            response.display_format = display_format
            if display_format != "structured":
                response.formatted_output = response.format_for_display(display_format)

            return response

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResponse.create_error(
                query=query,
                error=f"Search operation failed: {str(e)}",
                error_code=SearchErrorCode.SEARCH_ERROR.value,
                search_config=self._get_search_config(),
                recovery_suggestions=[
                    "Check if the index is properly loaded",
                    "Verify the embedder is available and functional",
                    "Try a simpler query if the current one is complex",
                ],
            )

    def _create_vector_results(self, distances, indices) -> List[SearchResult]:
        """Convert FAISS vector search results to SearchResult objects."""
        results = []
        for distance, idx in zip(distances, indices):
            if idx >= 0:  # Valid index
                chunk = self.metadata_store.get_chunk(idx)
                if chunk:
                    # Convert distance to similarity score (cosine distance -> similarity)
                    relevance_score = max(0.0, 1.0 - distance)
                    source_info = self._create_source_info(chunk)

                    result = SearchResult(
                        chunk_id=chunk.chunk_id,
                        source_path=chunk.source_path,
                        source_type=chunk.source_type.value,
                        text=chunk.chunk_text,
                        relevance_score=relevance_score,
                        scores=create_structured_scores(
                            vector_score=relevance_score,
                            original_score=relevance_score,
                        ),
                        metadata=(
                            create_structured_metadata(**chunk.metadata)
                            if chunk.metadata
                            else None
                        ),
                        provenance=create_search_provenance(
                            search_features=["vector_search"],
                            search_stage="vector_only",
                        ),
                        source_info=source_info,
                    )
                    results.append(result)
        return results

    def _convert_hybrid_results(
        self, hybrid_results
    ) -> List[SearchResult]:
        """Convert HybridSearchEngine results to search_models.SearchResult."""
        converted = []
        for hr in hybrid_results:
            result = SearchResult(
                chunk_id=hr.chunk_id,
                source_path=hr.source_path,
                source_type=hr.source_type,
                text=hr.text,
                relevance_score=hr.combined_score,
                scores=create_structured_scores(
                    vector_score=hr.vector_score,
                    keyword_score=hr.keyword_score,
                    combined_score=hr.combined_score,
                ),
                metadata=create_structured_metadata(**hr.metadata),
                provenance=create_search_provenance(
                    search_features=["hybrid_search"],
                    search_stage="hybrid",
                ),
                source_info=hr.metadata,  # Already contains source info
            )
            converted.append(result)
        return converted

    def _ensure_hybrid_index(self) -> None:
        """Ensure hybrid search index is built from current chunks."""
        if not self.hybrid_search:
            return

        current_chunk_count = self.metadata_store.get_stats()["total_chunks"]
        hybrid_stats = self.hybrid_search.get_stats()

        if hybrid_stats["indexed_documents"] != current_chunk_count:
            logger.info("Building hybrid search index...")
            all_chunks = self.metadata_store.get_all_chunks()
            if all_chunks:
                chunk_ids = [c.chunk_id for c in all_chunks]
                texts = [c.chunk_text for c in all_chunks]
                self.hybrid_search.add_documents(chunk_ids, texts)
                logger.info(f"Built hybrid search index with {len(texts)} documents")

    def _create_source_info(self, chunk) -> Dict[str, Any]:
        """Extract source information from chunk metadata."""
        source_path = getattr(chunk, "source_path", "/unknown")
        source_type = getattr(chunk, "source_type", None)

        source_info = {
            "file_path": str(source_path),
            "source_type": (
                source_type.value if hasattr(source_type, "value") else str(source_type)
            ),
        }

        # Add file metadata if path is valid
        if isinstance(source_path, str) and source_path != "/unknown":
            try:
                file_path = Path(source_path)
                if file_path.exists():
                    stat = file_path.stat()
                    source_info.update(
                        {
                            "filename": file_path.name,
                            "file_extension": file_path.suffix.lower(),
                            "file_size_bytes": stat.st_size,
                            "modified_at": stat.st_mtime,
                        }
                    )
            except Exception:
                pass

        # Add chunk metadata
        if hasattr(chunk, "parent_section") and chunk.parent_section:
            source_info["section_title"] = str(chunk.parent_section)

        return source_info

    def _get_search_config(self) -> Dict[str, Any]:
        """Get current search configuration for response metadata."""
        return {
            "hybrid_search": self.config.use_hybrid_search,
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
            "keyword_weight": getattr(self.config, "keyword_weight", 0.3),
        }


__all__ = [
    "HybridSearchEngine",
    "SearchErrorCode",
    "SearchPerformanceLogger",
    "SearchResponse",
    "SearchResult",
    "SearchService",
    "create_search_performance_info",
    "create_search_provenance",
    "create_structured_metadata",
    "create_structured_scores",
    "enhance_search_results_with_ranking",
]
