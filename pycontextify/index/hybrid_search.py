"""Hybrid search module combining vector similarity and keyword matching.

This module implements hybrid search that combines FAISS vector similarity search
with traditional keyword search using TF-IDF and BM25 algorithms.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with scores and metadata."""

    chunk_id: str
    text: str
    vector_score: float
    keyword_score: float
    combined_score: float
    source_path: str
    source_type: str
    metadata: Dict[str, Any]


class HybridSearchEngine:
    """Hybrid search engine combining vector and keyword search."""

    def __init__(self, keyword_weight: float = 0.3):
        """Initialize hybrid search engine.

        Args:
            keyword_weight: Weight for keyword search (0.0-1.0). Vector weight = 1 - keyword_weight
        """
        self.keyword_weight = keyword_weight
        self.vector_weight = 1.0 - keyword_weight

        # Keyword search components
        self.tfidf_vectorizer = None
        self.bm25_index = None
        self.document_texts = []
        self.chunk_ids = []

        # Initialize components
        self._initialize_keyword_search()

    def _initialize_keyword_search(self):
        """Initialize keyword search components."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),  # Include bigrams
                max_features=10000,
                min_df=1,
                max_df=0.95,
            )
            logger.info("Keyword search components initialized")
        except ImportError as e:
            logger.warning(f"Could not initialize keyword search: {e}")
            self.tfidf_vectorizer = None

    def add_documents(self, chunk_ids: List[str], texts: List[str]):
        """Add documents to the keyword search index.

        Args:
            chunk_ids: List of chunk identifiers
            texts: List of document texts
        """
        if not self.tfidf_vectorizer:
            logger.warning("Keyword search not available - skipping document indexing")
            return

        self.chunk_ids = chunk_ids
        self.document_texts = texts

        try:
            # Build TF-IDF index
            if texts:
                self.tfidf_vectorizer.fit(texts)
                logger.info(f"TF-IDF index built with {len(texts)} documents")

                # Build BM25 index
                from rank_bm25 import BM25Okapi

                tokenized_texts = [text.lower().split() for text in texts]
                self.bm25_index = BM25Okapi(tokenized_texts)
                logger.info(f"BM25 index built with {len(texts)} documents")

        except Exception as e:
            logger.error(f"Error building keyword search index: {e}")
            self.tfidf_vectorizer = None
            self.bm25_index = None

    def search(
        self,
        query: str,
        vector_scores: List[Tuple[int, float]],
        metadata_store,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query
            vector_scores: List of (faiss_id, vector_score) from vector search
            metadata_store: MetadataStore instance for retrieving chunk metadata
            top_k: Number of results to return

        Returns:
            List of SearchResult objects ranked by combined score
        """
        if not vector_scores:
            return []

        # Get keyword scores if available
        keyword_scores = {}
        if self.tfidf_vectorizer and self.bm25_index and self.document_texts:
            keyword_scores = self._get_keyword_scores(query)

        # Combine scores and create results
        results = []
        for faiss_id, vector_score in vector_scores[
            : top_k * 2
        ]:  # Get extra candidates for keyword rescoring
            try:
                chunk_metadata = metadata_store.get_chunk(faiss_id)
                if not chunk_metadata:
                    continue

                chunk_id = chunk_metadata.chunk_id
                keyword_score = keyword_scores.get(chunk_id, 0.0)

                # Combine scores
                combined_score = (
                    self.vector_weight * vector_score
                    + self.keyword_weight * keyword_score
                )

                result = SearchResult(
                    chunk_id=chunk_id,
                    text=chunk_metadata.chunk_text,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    combined_score=combined_score,
                    source_path=chunk_metadata.source_path,
                    source_type=(
                        chunk_metadata.source_type.value
                        if hasattr(chunk_metadata.source_type, "value")
                        else str(chunk_metadata.source_type)
                    ),
                    metadata=self._extract_metadata(chunk_metadata),
                )
                results.append(result)

            except Exception as e:
                logger.warning(
                    f"Error processing search result for faiss_id {faiss_id}: {e}"
                )
                continue

        # Sort by combined score and return top results
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]

    def _get_keyword_scores(self, query: str) -> Dict[str, float]:
        """Get keyword search scores for all documents.

        Args:
            query: Search query

        Returns:
            Dictionary mapping chunk_ids to keyword scores
        """
        keyword_scores = {}

        try:
            # TF-IDF search
            tfidf_scores = self._tfidf_search(query)

            # BM25 search
            bm25_scores = self._bm25_search(query)

            # Combine TF-IDF and BM25 scores
            for i, chunk_id in enumerate(self.chunk_ids):
                tfidf_score = tfidf_scores.get(i, 0.0)
                bm25_score = bm25_scores.get(i, 0.0)

                # Average the two keyword scores
                keyword_scores[chunk_id] = (tfidf_score + bm25_score) / 2.0

        except Exception as e:
            logger.error(f"Error computing keyword scores: {e}")

        return keyword_scores

    def _tfidf_search(self, query: str) -> Dict[int, float]:
        """Perform TF-IDF search.

        Args:
            query: Search query

        Returns:
            Dictionary mapping document indices to TF-IDF scores
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])

            # Transform documents
            doc_vectors = self.tfidf_vectorizer.transform(self.document_texts)

            # Calculate similarities
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()

            # Return scores as dictionary
            return {
                i: float(score) for i, score in enumerate(similarities) if score > 0
            }

        except Exception as e:
            logger.error(f"TF-IDF search error: {e}")
            return {}

    def _bm25_search(self, query: str) -> Dict[int, float]:
        """Perform BM25 search.

        Args:
            query: Search query

        Returns:
            Dictionary mapping document indices to BM25 scores
        """
        try:
            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)

            # Normalize scores to 0-1 range
            max_score = max(scores) if scores.max() > 0 else 1.0
            normalized_scores = scores / max_score

            # Return scores as dictionary
            return {
                i: float(score)
                for i, score in enumerate(normalized_scores)
                if score > 0
            }

        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return {}

    def _extract_metadata(self, chunk_metadata) -> Dict[str, Any]:
        """Extract relevant metadata from chunk metadata.

        Args:
            chunk_metadata: ChunkMetadata object

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "source_type": (
                chunk_metadata.source_type.value
                if hasattr(chunk_metadata.source_type, "value")
                else str(chunk_metadata.source_type)
            ),
            "source_path": chunk_metadata.source_path,
            "chunk_id": chunk_metadata.chunk_id,
            "created_at": (
                chunk_metadata.created_at.isoformat()
                if hasattr(chunk_metadata.created_at, "isoformat")
                else str(chunk_metadata.created_at)
            ),
        }

        # Add optional metadata if available
        if hasattr(chunk_metadata, "file_extension") and chunk_metadata.file_extension:
            metadata["file_extension"] = chunk_metadata.file_extension

        if (
            hasattr(chunk_metadata, "embedding_provider")
            and chunk_metadata.embedding_provider
        ):
            metadata["embedding_provider"] = chunk_metadata.embedding_provider

        if (
            hasattr(chunk_metadata, "embedding_model")
            and chunk_metadata.embedding_model
        ):
            metadata["embedding_model"] = chunk_metadata.embedding_model

        return metadata

    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid search statistics.

        Returns:
            Dictionary with search engine statistics
        """
        stats = {
            "keyword_weight": self.keyword_weight,
            "vector_weight": self.vector_weight,
            "tfidf_available": self.tfidf_vectorizer is not None,
            "bm25_available": self.bm25_index is not None,
            "indexed_documents": len(self.document_texts),
            "indexed_chunk_ids": len(self.chunk_ids),
        }

        if self.tfidf_vectorizer and hasattr(self.tfidf_vectorizer, "vocabulary_"):
            stats["tfidf_vocabulary_size"] = len(self.tfidf_vectorizer.vocabulary_)

        return stats
