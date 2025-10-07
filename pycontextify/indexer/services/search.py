"""Search coordination across vector and hybrid engines."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from ...storage.metadata import MetadataStore, SourceType
from ...orchestrator.config import Config
from ...search.models import (
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
from ..pdf_loader import PDFLoader
from .embedding import EmbeddingService


class SearchService:
    """Encapsulate all search related behaviours."""

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
        self._performance_logger = SearchPerformanceLogger()
        self._hybrid_search = None
        self._initialize_hybrid_search()

    @property
    def performance_logger(self) -> SearchPerformanceLogger:
        """Expose the performance logger for reporting."""
        return self._performance_logger

    def search(
        self, query: str, top_k: int = 5, display_format: str = "readable"
    ) -> SearchResponse:
        """Perform semantic search with optional hybrid enhancement."""
        start_time = time.time()
        try:
            vector_store = self._embedding_service.vector_store
            if vector_store is None or vector_store.is_empty():
                return SearchResponse.create_error(
                    query=query,
                    error="No indexed content available. Please index some documents, code, or webpages first.",
                    error_code=SearchErrorCode.NO_CONTENT.value,
                    search_config=self._get_search_config(),
                    recovery_suggestions=[
                        "Use index_document(), index_codebase(), or index_webpage() to add content",
                        "Check if auto_load is enabled and index files exist",
                        "Verify the vector store was initialized properly",
                    ],
                )

            self._embedding_service.ensure_loaded()
            embedder = self._embedding_service.embedder
            query_embedding = embedder.embed_single(query)

            distances, indices = vector_store.search(query_embedding, top_k * 2)
            search_results = self._create_vector_search_results(distances, indices)

            if self._config.use_hybrid_search and self._hybrid_search:
                try:
                    self._ensure_hybrid_search_index()
                    hybrid_results = self._hybrid_search.search(query, top_k * 2)
                    combined = self._combine_hybrid_results(search_results, hybrid_results)
                    search_results = combined[:top_k]
                except Exception as exc:
                    self._logger.warning("Hybrid search failed, using vector-only: %s", exc)
                    search_results = search_results[:top_k]
            else:
                search_results = search_results[:top_k]

            enhanced_results = enhance_search_results_with_ranking(
                results=search_results,
                query=query,
                include_explanations=True,
                include_confidence=True,
            )
            performance = create_search_performance_info(
                start_time=start_time,
                search_mode="hybrid" if self._config.use_hybrid_search else "vector",
                total_candidates=len(search_results),
            )

            response = SearchResponse.create_success(
                query=query,
                results=enhanced_results,
                search_config=self._get_search_config(),
                performance=performance,
            )
            response.display_format = display_format
            if display_format != "structured":
                response.formatted_output = response.format_for_display(display_format)

            self._performance_logger.log_search_performance(response)
            return response
        except Exception as exc:
            self._logger.error("Search failed: %s", exc)
            return SearchResponse.create_error(
                query=query,
                error=f"Search operation failed: {exc}",
                error_code=SearchErrorCode.SEARCH_ERROR.value,
                search_config=self._get_search_config(),
                recovery_suggestions=[
                    "Check if the index is properly loaded",
                    "Verify the embedder is available and functional",
                    "Try a simpler query if the current one is complex",
                ],
            )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return aggregated performance statistics."""
        return self._performance_logger.get_performance_summary()

    def get_hybrid_stats(self) -> Dict[str, Any]:
        """Expose hybrid search statistics for monitoring."""
        if self._hybrid_search:
            return self._hybrid_search.get_stats()
        return {}

    def get_hybrid_engine(self):
        """Return the underlying hybrid engine for compatibility."""
        return self._hybrid_search

    def _initialize_hybrid_search(self) -> None:
        """Initialize hybrid search engine if enabled."""
        if not self._config.use_hybrid_search:
            self._logger.info("Hybrid search disabled by configuration")
            return
        try:
            from ...search.hybrid import HybridSearchEngine

            self._hybrid_search = HybridSearchEngine(
                keyword_weight=self._config.keyword_weight
            )
            self._logger.info(
                "Initialized hybrid search with keyword weight: %s",
                self._config.keyword_weight,
            )
        except ImportError as exc:
            self._logger.warning("Could not initialize hybrid search: %s", exc)
            self._hybrid_search = None

    def _ensure_hybrid_search_index(self) -> None:
        """Ensure hybrid search index is built from current chunks."""
        if not self._hybrid_search:
            return
        current_chunk_count = self._metadata_store.get_stats()["total_chunks"]
        hybrid_stats = self._hybrid_search.get_stats()
        if hybrid_stats["indexed_documents"] == current_chunk_count:
            return

        self._logger.info("Building hybrid search index...")
        all_chunks = self._metadata_store.get_all_chunks()
        if not all_chunks:
            return

        chunk_ids = [chunk.chunk_id for chunk in all_chunks]
        texts = [chunk.chunk_text for chunk in all_chunks]
        self._hybrid_search.add_documents(chunk_ids, texts)
        self._logger.info(
            "Built hybrid search index with %s documents", len(texts)
        )

    def _get_search_config(self) -> Dict[str, Any]:
        """Summarise search configuration for responses."""
        return {
            "hybrid_search": self._config.use_hybrid_search,
            "embedding_provider": self._config.embedding_provider,
            "embedding_model": self._config.embedding_model,
            "keyword_weight": getattr(self._config, "keyword_weight", 0.3),
        }

    def _create_vector_search_results(self, distances, indices) -> List[SearchResult]:
        """Convert raw vector search results into SearchResult objects."""
        results: List[SearchResult] = []
        for distance, idx in zip(distances, indices):
            if idx < 0:
                continue
            chunk = self._metadata_store.get_chunk(idx)
            if not chunk:
                continue

            source_info = self._create_source_info_from_chunk(chunk)
            relevance_score = max(0.0, 1.0 - distance)
            metadata = (
                create_structured_metadata(**chunk.metadata)
                if getattr(chunk, "metadata", None)
                else None
            )
            result = SearchResult(
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                source_type=chunk.source_type.value
                if isinstance(chunk.source_type, SourceType)
                else chunk.source_type,
                text=chunk.chunk_text,
                relevance_score=relevance_score,
                scores=create_structured_scores(
                    vector_score=relevance_score,
                    original_score=relevance_score,
                ),
                metadata=metadata,
                provenance=create_search_provenance(
                    search_features=["vector_search"],
                    search_stage="vector_only",
                ),
                source_info=source_info,
            )
            results.append(result)
        return results

    def _combine_hybrid_results(self, vector_results, hybrid_results):
        """Combine vector and keyword scores."""
        combined = {result.chunk_id: {
            "result": result,
            "vector_score": result.relevance_score,
            "hybrid_score": 0.0,
        } for result in vector_results}

        for hybrid in hybrid_results:
            chunk_id = hybrid.get("chunk_id")
            if chunk_id in combined:
                combined[chunk_id]["hybrid_score"] = hybrid.get("score", 0.0)
            else:
                metadata = self._metadata_store.get_chunk_by_chunk_id(chunk_id)
                if not metadata:
                    continue
                vector_result = SearchResult(
                    chunk_id=metadata.chunk_id,
                    source_path=metadata.source_path,
                    source_type=metadata.source_type.value
                    if isinstance(metadata.source_type, SourceType)
                    else metadata.source_type,
                    text=metadata.chunk_text,
                    relevance_score=hybrid.get("score", 0.0),
                    scores=create_structured_scores(
                        vector_score=0.0,
                        original_score=hybrid.get("score", 0.0),
                    ),
                    metadata=(
                        create_structured_metadata(**metadata.metadata)
                        if getattr(metadata, "metadata", None)
                        else None
                    ),
                    provenance=create_search_provenance(
                        search_features=["hybrid_search"],
                        search_stage="hybrid_only",
                    ),
                    source_info=self._create_source_info_from_chunk(metadata),
                )
                combined[chunk_id] = {
                    "result": vector_result,
                    "vector_score": 0.0,
                    "hybrid_score": hybrid.get("score", 0.0),
                }

        final_results = []
        for values in combined.values():
            result = values["result"]
            vector_score = values["vector_score"]
            hybrid_score = values["hybrid_score"]
            combined_score = (vector_score * 0.7) + (hybrid_score * 0.3)
            result.relevance_score = combined_score
            result.scores = create_structured_scores(
                vector_score=vector_score,
                hybrid_score=hybrid_score,
                original_score=combined_score,
            )
            result.provenance = create_search_provenance(
                search_features=["vector_search", "hybrid_search"],
                search_stage="combined",
            )
            final_results.append(result)
        final_results.sort(key=lambda item: item.relevance_score, reverse=True)
        return final_results

    def _create_source_info_from_chunk(self, chunk) -> Dict[str, Any]:
        """Extract structured source information from a chunk."""
        source_path = getattr(chunk, "source_path", "/unknown")
        source_type_val = getattr(chunk, "source_type", None)
        source_type_str = (
            source_type_val.value
            if hasattr(source_type_val, "value")
            else str(source_type_val) if source_type_val else "unknown"
        )
        source_info: Dict[str, Any] = {
            "file_path": str(source_path) if source_path else "/unknown",
            "source_type": source_type_str,
        }

        if isinstance(source_path, str) and source_path != "/unknown":
            try:
                path_obj = Path(source_path)
                if path_obj.exists() and path_obj.is_file():
                    stat = path_obj.stat()
                    source_info.update(
                        {
                            "filename": path_obj.name,
                            "file_extension": path_obj.suffix.lower(),
                            "file_size_bytes": stat.st_size,
                            "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "created_at": stat.st_ctime,
                            "modified_at": stat.st_mtime,
                        }
                    )
                else:
                    source_info["filename"] = path_obj.name
                    source_info["file_extension"] = path_obj.suffix.lower()
            except (OSError, ValueError, TypeError):
                self._logger.debug("Could not extract file metadata for %s", source_path)
                source_info["filename"] = "unknown"
                if isinstance(source_path, str) and source_path.startswith(("http://", "https://")):
                    source_info["source_type"] = "webpage"

            if source_type_str == "document" and source_path.lower().endswith(".pdf"):
                try:
                    pdf_loader = PDFLoader()
                    pdf_metadata = pdf_loader.get_pdf_info(source_path)
                    source_info.update(pdf_metadata)
                except Exception as exc:
                    self._logger.debug("Could not extract PDF metadata: %s", exc)

        chunk_text = getattr(chunk, "chunk_text", "")
        if chunk_text and isinstance(chunk_text, str):
            try:
                pdf_loader = PDFLoader()
                page_info = pdf_loader.extract_page_context(chunk_text)
                if page_info:
                    source_info.update(page_info)
            except Exception as exc:
                self._logger.debug("Could not extract page context: %s", exc)

        if hasattr(chunk, "parent_section"):
            parent_section = getattr(chunk, "parent_section", None)
            if parent_section:
                source_info["section_title"] = str(parent_section)

        if hasattr(chunk, "metadata"):
            metadata = getattr(chunk, "metadata", None)
            if metadata and hasattr(metadata, "items"):
                for key, value in metadata.items():
                    if key not in source_info and value is not None:
                        source_info[key] = value
        return source_info
