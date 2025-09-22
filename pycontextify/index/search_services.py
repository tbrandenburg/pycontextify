"""Focused search services to replace the monolithic search implementation.

This module contains focused services that implement the Single Responsibility Principle:
- VectorSearchService: Handles pure vector similarity search
- HybridSearchService: Enhances vector search with keyword matching
- RerankingService: Applies neural reranking to improve result quality
- SearchOrchestrator: Coordinates the search pipeline
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SearchCandidate:
    """Represents a search result candidate in the pipeline."""

    def __init__(
        self,
        chunk_id: str,
        source_path: str,
        source_type: str,
        text: str,
        relevance_score: float,
        metadata: Optional[Dict] = None,
    ):
        self.chunk_id = chunk_id
        self.source_path = source_path
        self.source_type = source_type
        self.text = text
        self.relevance_score = relevance_score
        self.metadata = metadata or {}

        # Additional scoring information
        self.vector_score = None
        self.keyword_score = None
        self.combined_score = None
        self.rerank_score = None
        self.original_score = None  # Score before reranking
        self.final_score = relevance_score


class VectorSearchService:
    """Focused service for vector similarity search operations."""

    def __init__(self, vector_store, metadata_store, embedder, config):
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.embedder = embedder
        self.config = config

    def search(
        self, query: str, top_k: int
    ) -> Tuple[List[SearchCandidate], Dict[str, Any]]:
        """Perform vector similarity search.

        Args:
            query: Search query text
            top_k: Number of candidates to return

        Returns:
            Tuple of (candidates, timing_info)
        """
        timing_info = {}

        # Embed query
        embedding_start = time.time()
        query_vector = self.embedder.embed_single(query)
        timing_info["embedding_time"] = time.time() - embedding_start

        # Search vector store
        vector_start = time.time()
        distances, indices = self.vector_store.search(query_vector, top_k)
        timing_info["vector_time"] = time.time() - vector_start

        # Convert to candidates
        candidates = []
        for distance, faiss_id in zip(distances, indices):
            # Get chunk metadata
            chunk = self._get_chunk_by_faiss_id(int(faiss_id))
            if chunk:
                candidate = SearchCandidate(
                    chunk_id=chunk.chunk_id,
                    source_path=chunk.source_path,
                    source_type=(
                        chunk.source_type.value
                        if hasattr(chunk.source_type, "value")
                        else str(chunk.source_type)
                    ),
                    text=chunk.chunk_text,
                    relevance_score=float(
                        1.0 - distance
                    ),  # Convert distance to similarity
                    metadata=chunk.metadata if hasattr(chunk, "metadata") else {},
                )
                candidate.vector_score = candidate.relevance_score
                candidates.append(candidate)

        logger.debug(f"Vector search returned {len(candidates)} candidates")
        return candidates, timing_info

    def _get_chunk_by_faiss_id(self, faiss_id: int):
        """Get chunk by FAISS ID. This is a simplified lookup."""
        try:
            # This would need to be implemented based on how FAISS IDs map to chunks
            # For now, assume metadata_store has a method to get chunk by index
            all_chunks = self.metadata_store.get_all_chunks()
            if all_chunks and 0 <= faiss_id < len(all_chunks):
                return all_chunks[faiss_id]

            # Fallback: try to get chunk by chunk_id if FAISS ID maps to chunk_id
            return self.metadata_store.get_chunk_by_chunk_id(str(faiss_id))
        except (AttributeError, IndexError, TypeError) as e:
            logger.debug(f"Could not get chunk for FAISS ID {faiss_id}: {e}")
            return None


class HybridSearchService:
    """Focused service for hybrid search enhancement (vector + keyword)."""

    def __init__(self, hybrid_search, metadata_store, config):
        self.hybrid_search = hybrid_search
        self.metadata_store = metadata_store
        self.config = config

    def enhance(
        self, query: str, vector_candidates: List[SearchCandidate], top_k: int
    ) -> Tuple[List[SearchCandidate], Dict[str, Any]]:
        """Enhance vector candidates with keyword search.

        Args:
            query: Search query text
            vector_candidates: Candidates from vector search
            top_k: Number of enhanced candidates to return

        Returns:
            Tuple of (enhanced_candidates, timing_info)
        """
        timing_info = {}

        if not self.hybrid_search or not self.config.use_hybrid_search:
            logger.debug("Hybrid search not enabled, returning vector candidates")
            return vector_candidates[:top_k], timing_info

        try:
            # Ensure hybrid search index is built
            self._ensure_hybrid_search_index()

            # Prepare vector scores for hybrid search
            vector_scores = [
                (candidate.chunk_id, candidate.vector_score)
                for candidate in vector_candidates
            ]

            # Perform hybrid search
            keyword_start = time.time()
            hybrid_results = self.hybrid_search.search(
                query=query,
                vector_scores=vector_scores,
                metadata_store=self.metadata_store,
                top_k=top_k,
            )
            timing_info["keyword_time"] = time.time() - keyword_start

            if not hybrid_results:
                logger.warning(
                    "Hybrid search returned no results, using vector candidates"
                )
                return vector_candidates[:top_k], timing_info

            # Convert hybrid results to enhanced candidates
            enhanced_candidates = []
            for result in hybrid_results:
                candidate = SearchCandidate(
                    chunk_id=result.chunk_id,
                    source_path=result.source_path,
                    source_type=result.source_type,
                    text=result.text,
                    relevance_score=result.combined_score,
                    metadata=result.metadata if hasattr(result, "metadata") else {},
                )
                candidate.vector_score = (
                    result.vector_score if hasattr(result, "vector_score") else None
                )
                candidate.keyword_score = (
                    result.keyword_score if hasattr(result, "keyword_score") else None
                )
                candidate.combined_score = result.combined_score
                candidate.final_score = result.combined_score
                enhanced_candidates.append(candidate)

            logger.debug(
                f"Hybrid search enhanced to {len(enhanced_candidates)} candidates"
            )
            return enhanced_candidates, timing_info

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}, falling back to vector search")
            timing_info["hybrid_search_error"] = str(e)
            return vector_candidates[:top_k], timing_info

    def _ensure_hybrid_search_index(self):
        """Ensure hybrid search index is built from current chunks."""
        current_chunk_count = self.metadata_store.get_stats().get("total_chunks", 0)
        hybrid_stats = (
            self.hybrid_search.get_stats()
            if hasattr(self.hybrid_search, "get_stats")
            else {"indexed_documents": 0}
        )

        if hybrid_stats.get("indexed_documents", 0) != current_chunk_count:
            logger.info("Building hybrid search index...")
            all_chunks = self.metadata_store.get_all_chunks()
            if all_chunks:
                chunk_ids = [chunk.chunk_id for chunk in all_chunks]
                texts = [chunk.chunk_text for chunk in all_chunks]
                self.hybrid_search.add_documents(chunk_ids, texts)
                logger.info(f"Built hybrid search index with {len(texts)} documents")


class RerankingService:
    """Focused service for neural reranking operations."""

    def __init__(self, reranker, config):
        self.reranker = reranker
        self.config = config

    def rerank(
        self, query: str, candidates: List[SearchCandidate], top_k: int
    ) -> Tuple[List[SearchCandidate], Dict[str, Any]]:
        """Apply neural reranking to improve result quality.

        Args:
            query: Search query text
            candidates: Candidates to rerank
            top_k: Number of reranked candidates to return

        Returns:
            Tuple of (reranked_candidates, timing_info)
        """
        timing_info = {}

        if not self.reranker or not self.config.use_reranking or not candidates:
            logger.debug("Reranking not enabled or no candidates, returning original")
            return candidates[:top_k], timing_info

        try:
            rerank_start = time.time()

            # Convert candidates to legacy format for reranker
            legacy_results = []
            for candidate in candidates:
                legacy_results.append(
                    {
                        "score": candidate.relevance_score,
                        "source_path": candidate.source_path,
                        "source_type": candidate.source_type,
                        "chunk_text": candidate.text,
                        "chunk_id": candidate.chunk_id,
                        "metadata": candidate.metadata,
                    }
                )

            # Apply reranking
            reranked_results = self.reranker.rerank(
                query=query, search_results=legacy_results, top_k=top_k
            )
            timing_info["rerank_time"] = time.time() - rerank_start

            # Convert back to candidates
            reranked_candidates = []
            for result in reranked_results:
                candidate = SearchCandidate(
                    chunk_id=result.chunk_id,
                    source_path=result.source_path,
                    source_type=result.source_type,
                    text=result.text,
                    relevance_score=result.final_score,
                    metadata=result.metadata if hasattr(result, "metadata") else {},
                )
                # Preserve original scoring information
                original_candidate = next(
                    (c for c in candidates if c.chunk_id == candidate.chunk_id), None
                )
                if original_candidate:
                    candidate.vector_score = original_candidate.vector_score
                    candidate.keyword_score = original_candidate.keyword_score
                    candidate.combined_score = original_candidate.combined_score
                    # Preserve the original relevance score before reranking
                    candidate.original_score = original_candidate.relevance_score
                else:
                    # Fallback: use the result's original score if available
                    candidate.original_score = (
                        result.original_score
                        if hasattr(result, "original_score")
                        else None
                    )

                candidate.rerank_score = (
                    result.rerank_score
                    if hasattr(result, "rerank_score")
                    else candidate.relevance_score
                )
                candidate.final_score = result.final_score
                reranked_candidates.append(candidate)

            logger.debug(
                f"Reranking completed, returned {len(reranked_candidates)} candidates"
            )
            return reranked_candidates, timing_info

        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original candidates")
            timing_info["rerank_error"] = str(e)
            return candidates[:top_k], timing_info


class SearchOrchestrator:
    """Orchestrates the search pipeline with focused services."""

    def __init__(
        self,
        vector_service: VectorSearchService,
        hybrid_service: HybridSearchService,
        reranking_service: RerankingService,
        config,
    ):
        self.vector_service = vector_service
        self.hybrid_service = hybrid_service
        self.reranking_service = reranking_service
        self.config = config

    def search(
        self, query: str, top_k: int
    ) -> Tuple[List[SearchCandidate], Dict[str, Any]]:
        """Execute the complete search pipeline.

        Args:
            query: Search query text
            top_k: Number of final results to return

        Returns:
            Tuple of (final_candidates, performance_info)
        """
        start_time = time.time()
        performance_info = {"start_time": start_time}
        search_mode = "vector"
        failed_components = []

        # Step 1: Vector similarity search
        search_top_k = top_k * 3 if self.config.use_reranking else top_k
        candidates, vector_timing = self.vector_service.search(query, search_top_k)
        performance_info.update(vector_timing)

        if not candidates:
            logger.warning("Vector search returned no candidates")
            performance_info.update(
                {
                    "total_time": time.time() - start_time,
                    "search_mode": "vector_empty",
                    "failed_components": failed_components,
                }
            )
            return [], performance_info

        # Step 2: Hybrid search enhancement (if enabled)
        if self.hybrid_service and self.config.use_hybrid_search:
            search_mode = "hybrid"
            candidates, hybrid_timing = self.hybrid_service.enhance(
                query, candidates, top_k
            )
            performance_info.update(hybrid_timing)

            if "hybrid_search_error" in hybrid_timing:
                failed_components.append("hybrid_search")
                search_mode = "vector_fallback"

        # Step 3: Neural reranking (if enabled)
        if self.reranking_service and self.config.use_reranking:
            search_mode = "reranked" if search_mode == "vector" else "hybrid_reranked"
            candidates, rerank_timing = self.reranking_service.rerank(
                query, candidates, top_k
            )
            performance_info.update(rerank_timing)

            if "rerank_error" in rerank_timing:
                failed_components.append("reranking")
                search_mode = search_mode.replace("_reranked", "_fallback")

        # Final results
        final_candidates = candidates[:top_k]
        performance_info.update(
            {
                "total_time": time.time() - start_time,
                "search_mode": search_mode,
                "total_candidates": len(candidates),
                "final_results": len(final_candidates),
                "failed_components": failed_components,
            }
        )

        logger.info(
            f"Search completed: {len(final_candidates)} results in {performance_info['total_time']:.2f}s (mode: {search_mode})"
        )
        return final_candidates, performance_info
