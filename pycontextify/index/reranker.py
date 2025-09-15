"""Reranking module using cross-encoder models for improved search result relevance.

This module implements result reranking using pre-trained cross-encoder models
that can better understand the relationship between queries and documents.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass 
class RerankResult:
    """Reranked search result."""
    chunk_id: str
    text: str
    original_score: float
    rerank_score: float
    final_score: float
    source_path: str
    source_type: str
    metadata: Dict[str, Any]


class CrossEncoderReranker:
    """Cross-encoder based reranker for improving search result relevance."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None
        self.is_available = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.is_available = True
            logger.info("Cross-encoder reranker initialized successfully")
            
        except ImportError:
            logger.warning("sentence-transformers not available - reranking disabled")
            self.is_available = False
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
            self.is_available = False
    
    def rerank(
        self, 
        query: str, 
        search_results: List[Any], 
        top_k: Optional[int] = None,
        combine_scores: bool = True,
        score_weight: float = 0.7
    ) -> List[RerankResult]:
        """Rerank search results using cross-encoder model.
        
        Args:
            query: Original search query
            search_results: List of search results to rerank
            top_k: Number of results to return (None = all)
            combine_scores: Whether to combine original and rerank scores
            score_weight: Weight for rerank score in combination (0.0-1.0)
            
        Returns:
            List of reranked results
        """
        if not self.is_available or not search_results:
            logger.warning("Reranker not available or no results to rerank")
            return self._convert_to_rerank_results(search_results, query)
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = []
            for result in search_results:
                # Handle different result types
                text = self._extract_text_from_result(result)
                query_doc_pairs.append([query, text])
            
            # Get rerank scores
            logger.debug(f"Reranking {len(query_doc_pairs)} results for query: '{query}'")
            rerank_scores = self.model.predict(query_doc_pairs)
            
            # Convert to rerank results
            rerank_results = []
            for i, (result, rerank_score) in enumerate(zip(search_results, rerank_scores)):
                original_score = self._extract_original_score(result)
                
                # Calculate final score
                if combine_scores:
                    final_score = (
                        score_weight * float(rerank_score) + 
                        (1 - score_weight) * original_score
                    )
                else:
                    final_score = float(rerank_score)
                
                rerank_result = RerankResult(
                    chunk_id=self._extract_chunk_id(result),
                    text=self._extract_text_from_result(result),
                    original_score=original_score,
                    rerank_score=float(rerank_score),
                    final_score=final_score,
                    source_path=self._extract_source_path(result),
                    source_type=self._extract_source_type(result),
                    metadata=self._extract_metadata(result)
                )
                rerank_results.append(rerank_result)
            
            # Sort by final score
            rerank_results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Apply top_k limit
            if top_k is not None:
                rerank_results = rerank_results[:top_k]
            
            logger.info(f"Reranked {len(search_results)} results to {len(rerank_results)}")
            return rerank_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback to original results
            return self._convert_to_rerank_results(search_results, query)
    
    def _extract_text_from_result(self, result: Any) -> str:
        """Extract text content from search result."""
        if hasattr(result, 'text'):
            return result.text
        elif hasattr(result, 'chunk_text'):
            return result.chunk_text
        elif hasattr(result, 'content'):
            return result.content
        elif isinstance(result, dict):
            return result.get('text', result.get('content', result.get('chunk_text', '')))
        else:
            return str(result)
    
    def _extract_original_score(self, result: Any) -> float:
        """Extract original score from search result."""
        if hasattr(result, 'combined_score'):
            return result.combined_score
        elif hasattr(result, 'score'):
            return result.score
        elif hasattr(result, 'vector_score'):
            return result.vector_score
        elif isinstance(result, dict):
            return float(result.get('score', result.get('combined_score', result.get('vector_score', 0.0))))
        else:
            return 0.0
    
    def _extract_chunk_id(self, result: Any) -> str:
        """Extract chunk ID from search result."""
        if hasattr(result, 'chunk_id'):
            return result.chunk_id
        elif isinstance(result, dict):
            return result.get('chunk_id', 'unknown')
        else:
            return 'unknown'
    
    def _extract_source_path(self, result: Any) -> str:
        """Extract source path from search result."""
        if hasattr(result, 'source_path'):
            return result.source_path
        elif isinstance(result, dict):
            return result.get('source_path', 'unknown')
        else:
            return 'unknown'
    
    def _extract_source_type(self, result: Any) -> str:
        """Extract source type from search result."""
        if hasattr(result, 'source_type'):
            return result.source_type
        elif isinstance(result, dict):
            return result.get('source_type', 'unknown')
        else:
            return 'unknown'
    
    def _extract_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract metadata from search result."""
        if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
            return result.metadata.copy()
        elif isinstance(result, dict):
            # Extract metadata fields if available
            metadata = {}
            metadata_keys = ['embedding_provider', 'embedding_model', 'file_extension', 'created_at']
            for key in metadata_keys:
                if key in result:
                    metadata[key] = result[key]
            return metadata
        else:
            return {}
    
    def _convert_to_rerank_results(self, search_results: List[Any], query: str) -> List[RerankResult]:
        """Convert search results to rerank results without reranking."""
        rerank_results = []
        
        for result in search_results:
            original_score = self._extract_original_score(result)
            
            rerank_result = RerankResult(
                chunk_id=self._extract_chunk_id(result),
                text=self._extract_text_from_result(result),
                original_score=original_score,
                rerank_score=original_score,  # Use original score as rerank score
                final_score=original_score,
                source_path=self._extract_source_path(result),
                source_type=self._extract_source_type(result),
                metadata=self._extract_metadata(result)
            )
            rerank_results.append(rerank_result)
        
        return rerank_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "is_available": self.is_available,
            "model_loaded": self.model is not None
        }
    
    def warmup(self, sample_query: str = "test query", sample_text: str = "test document") -> bool:
        """Warm up the model with a sample prediction.
        
        Args:
            sample_query: Sample query for warmup
            sample_text: Sample document text for warmup
            
        Returns:
            True if warmup successful, False otherwise
        """
        if not self.is_available:
            return False
            
        try:
            logger.info("Warming up cross-encoder model...")
            _ = self.model.predict([[sample_query, sample_text]])
            logger.info("Model warmup completed successfully")
            return True
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics.
        
        Returns:
            Dictionary with reranker statistics
        """
        stats = {
            "model_name": self.model_name,
            "is_available": self.is_available,
            "model_loaded": self.model is not None
        }
        
        if self.is_available and self.model:
            try:
                # Try to get model info if available
                if hasattr(self.model, 'get_sentence_embedding_dimension'):
                    stats["embedding_dimension"] = self.model.get_sentence_embedding_dimension()
            except Exception:
                pass  # Some models may not have this method
        
        return stats