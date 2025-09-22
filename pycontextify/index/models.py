"""Search result models for PyContextify.

This module defines standardized data structures for search responses,
providing consistent formatting across all search modes and features.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from pathlib import Path
import time
import logging
from enum import Enum


class SearchErrorCode(Enum):
    """Standardized error codes for search operations."""
    
    # Content/Index Errors
    NO_CONTENT = "NO_CONTENT"
    INDEX_EMPTY = "INDEX_EMPTY"
    INDEX_CORRUPTED = "INDEX_CORRUPTED"
    
    # Query Errors
    INVALID_QUERY = "INVALID_QUERY"
    QUERY_TOO_SHORT = "QUERY_TOO_SHORT"
    QUERY_TOO_LONG = "QUERY_TOO_LONG"
    
    # Component Errors
    EMBEDDER_UNAVAILABLE = "EMBEDDER_UNAVAILABLE"
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"
    HYBRID_SEARCH_FAILED = "HYBRID_SEARCH_FAILED"
    RERANKER_FAILED = "RERANKER_FAILED"
    RELATIONSHIP_ERROR = "RELATIONSHIP_ERROR"
    
    # Performance/Resource Errors
    TIMEOUT = "TIMEOUT"
    MEMORY_ERROR = "MEMORY_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    
    # Generic Errors
    SEARCH_ERROR = "SEARCH_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class SearchResult:
    """Standardized search result object.
    
    This structure provides consistent formatting regardless of which
    search features are enabled (hybrid, reranking, relationships).
    """
    # Core fields (always present)
    chunk_id: str
    source_path: str 
    source_type: str  # "codebase", "document", "webpage"
    text: str  # Unified field name (was chunk_text)
    relevance_score: float  # Primary unified score
    
    # Position information (structured for better access)
    position: Optional[Dict[str, Any]] = None  # start_char, end_char, line_start, line_end
    
    # Score breakdown (detailed when available)  
    scores: Optional[Dict[str, float]] = None  # vector, keyword, rerank, combined
    
    # Enhanced metadata (structured with categories)
    metadata: Optional[Dict[str, Any]] = None
    
    # Context/relationships (when enabled)
    context: Optional[Dict[str, Any]] = None  # relationships, related_chunks
    
    # Search provenance (tracks how this result was found)
    provenance: Optional[Dict[str, Any]] = None  # search_features_used, ranking_factors
    
    # Phase 3.1: Result Ranking & Explanation
    rank: Optional[int] = None  # Position in result set (1-based)
    relevance_explanation: Optional[Dict[str, Any]] = None  # Why this result is relevant
    confidence_scores: Optional[Dict[str, float]] = None  # Confidence by match type
    
    # Phase 4.2: Enhanced Source Information
    source_info: Optional[Dict[str, Any]] = None  # Rich source metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result


@dataclass 
class SearchResponse:
    """Standardized search response wrapper.
    
    Provides consistent top-level structure with query metadata,
    performance information, and standardized results.
    """
    # Response metadata
    success: bool
    query: str
    
    # Search configuration used
    search_config: Dict[str, Any]
    
    # Results
    results: List[SearchResult]
    total_results: int
    
    # Optional fields (must come after required fields)
    query_analysis: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    
    # Phase 3.4: API Versioning  
    api_version: str = "3.2"  # Updated for new formatting features
    
    # Phase 4.1: Display formatting
    display_format: str = "readable"  # Default to human-readable format
    formatted_output: Optional[str] = None  # Pre-formatted output when requested

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "api_version": self.api_version,
            "success": self.success,
            "query": self.query,
            "search_config": self.search_config,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results
        }
        
        if self.performance is not None:
            result["performance"] = self.performance
            
        if self.error is not None:
            result["error"] = self.error
            
        if self.error_code is not None:
            result["error_code"] = self.error_code
            
        if self.query_analysis is not None:
            result["query_analysis"] = self.query_analysis
            
        # Add display format information
        result["display_format"] = self.display_format
        if self.formatted_output is not None:
            result["formatted_output"] = self.formatted_output
            
        return result

    @classmethod
    def create_success(
        cls,
        query: str,
        results: List[SearchResult],
        search_config: Dict[str, Any],
        performance: Optional[Dict[str, Any]] = None,
        query_analysis: Optional[Dict[str, Any]] = None
    ) -> "SearchResponse":
        """Create a successful search response."""
        return cls(
            success=True,
            query=query,
            search_config=search_config,
            results=results,
            total_results=len(results),
            performance=performance,
            query_analysis=query_analysis
        )

    @classmethod
    def create_error(
        cls,
        query: str,
        error: str,
        error_code: Optional[str] = None,
        search_config: Optional[Dict[str, Any]] = None,
        partial_results: Optional[List[SearchResult]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        query_analysis: Optional[Dict[str, Any]] = None
    ) -> "SearchResponse":
        """Create an error search response with optional partial results.
        
        Args:
            query: The original search query
            error: Human-readable error description
            error_code: Machine-readable error code
            search_config: Configuration that was used
            partial_results: Any partial results that were obtained before error
            recovery_suggestions: Suggestions for how to resolve the error
        """
        response = cls(
            success=False,
            query=query,
            search_config=search_config or {},
            results=partial_results or [],
            total_results=len(partial_results) if partial_results else 0,
            error=error,
            error_code=error_code,
            query_analysis=query_analysis
        )
        
        # Add recovery suggestions to performance metadata if provided
        if recovery_suggestions:
            response.performance = {
                "error_recovery": {
                    "suggestions": recovery_suggestions,
                    "partial_results_available": len(partial_results) > 0 if partial_results else False
                }
            }
            
        return response
    
    def format_for_display(self, format_type: str = "readable") -> Union[str, Dict[str, Any]]:
        """Format search results for different display types.
        
        Args:
            format_type: Type of formatting ('readable', 'structured', 'summary')
            
        Returns:
            Formatted output as string (for readable/summary) or dict (for structured)
        """
        if format_type == "readable":
            return self._format_readable()
        elif format_type == "summary":
            return self._format_summary()
        elif format_type == "structured":
            return self.to_dict()
        else:
            # Default to readable for unknown formats
            return self._format_readable()
    
    def _format_readable(self) -> str:
        """Create human-readable formatted output similar to rust-local-rag."""
        if not self.success:
            error_msg = f"❌ Search failed: {self.error or 'Unknown error'}"
            if self.query_analysis and 'search_strategies' in self.query_analysis:
                suggestions = self.query_analysis['search_strategies']
                if suggestions:
                    error_msg += "\n\n💡 Suggestions:"
                    for strategy in suggestions[:2]:  # Limit to 2 suggestions
                        error_msg += f"\n• {strategy.get('description', '')}"
            return error_msg
        
        if not self.results:
            no_results = f"🔍 No results found for '{self.query}'"
            
            # Add query suggestions if available
            if self.query_analysis and 'enhancements' in self.query_analysis:
                suggestions = self.query_analysis['enhancements'].get('suggested_alternatives', [])
                if suggestions:
                    no_results += "\n\n💡 Try these alternatives:"
                    for suggestion in suggestions[:3]:  # Limit to 3 suggestions
                        no_results += f"\n• '{suggestion}'"
            
            return no_results
        
        # Header with search info
        formatted = [f"🔍 Found {len(self.results)} result{'s' if len(self.results) != 1 else ''} for '{self.query}'"]
        
        # Add performance info if available
        if self.performance and 'search_time_ms' in self.performance:
            search_time = self.performance['search_time_ms']
            search_mode = self.performance.get('search_mode', 'unknown')
            formatted[0] += f" ({search_time}ms, {search_mode})"
        
        formatted.append("")  # Empty line
        
        # Format each result
        for i, result in enumerate(self.results, 1):
            formatted.append(self._format_result_readable(result, i))
            
            # Add separator between results (except for last one)
            if i < len(self.results):
                formatted.append("\n" + "─" * 60 + "\n")
        
        # Add query insights if available
        if self.query_analysis and len(self.results) > 0:
            insights = self._format_query_insights()
            if insights:
                formatted.append("\n" + "═" * 60)
                formatted.append(insights)
        
        return "\n".join(formatted)
    
    def _format_result_readable(self, result: 'SearchResult', rank: int) -> str:
        """Format a single search result for readable display."""
        # Get enhanced source info
        source_name = self._get_display_source_name(result.source_path, result.source_info)
        confidence = result.confidence_scores.get('overall', 0.0) if result.confidence_scores else 0.0
        
        # Header with result info
        header = f"**Result {rank}** (Score: {result.relevance_score:.3f}"
        if confidence > 0:
            header += f", Confidence: {confidence:.1%}"
        header += ")"
        
        # Enhanced source and type info
        source_icon = self._get_source_icon(result.source_type)
        source_line = self._format_enhanced_source_line(source_icon, source_name, result)
        
        # Main content preview
        content_preview = self._create_content_preview(result.text)
        
        # Key insights
        insights = []
        
        # Add match information
        if result.relevance_explanation and 'match_details' in result.relevance_explanation:
            matched_terms = result.relevance_explanation['match_details'].get('matched_terms', [])
            if matched_terms:
                insights.append(f"🎯 Key matches: {', '.join(matched_terms)}")        
        
        # Add context information
        if result.context and 'relationships' in result.context:
            refs = result.context['relationships'].get('references', [])
            if refs:
                key_refs = refs[:3]  # Limit to 3 key references
                insights.append(f"🔗 Related: {', '.join(key_refs)}")
        
        # Add related chunks count
        if result.context and 'related_chunks' in result.context:
            related_count = len(result.context['related_chunks'])
            if related_count > 0:
                insights.append(f"📚 {related_count} related chunk{'s' if related_count != 1 else ''} available")
        
        # Combine all parts
        parts = [header, source_line, "", content_preview]
        
        if insights:
            parts.append("")
            parts.extend(insights)
        
        return "\n".join(parts)
    
    def _format_summary(self) -> str:
        """Create a condensed summary format."""
        if not self.success:
            return f"❌ Search failed: {self.error or 'Unknown error'}"
        
        if not self.results:
            return f"🔍 No results found for '{self.query}'"
        
        # Brief header
        summary = f"🔍 {len(self.results)} result{'s' if len(self.results) != 1 else ''} for '{self.query}'\n"
        
        # Top result summary
        top_result = self.results[0]
        source_name = self._get_display_source_name(top_result.source_path)
        preview = self._create_content_preview(top_result.text, max_length=150)
        
        summary += f"\n📄 Top result from {source_name}:\n{preview}"
        
        # Additional results count
        if len(self.results) > 1:
            summary += f"\n\n➕ {len(self.results) - 1} more result{'s' if len(self.results) > 2 else ''} available"
        
        return summary
    
    def _format_query_insights(self) -> str:
        """Format query analysis insights."""
        if not self.query_analysis:
            return ""
        
        insights = ["💡 Search Insights:"]
        
        # Query characteristics
        if 'characteristics' in self.query_analysis:
            chars = self.query_analysis['characteristics']
            intent = chars.get('intent', 'unknown')
            complexity = chars.get('complexity_score', 0)
            
            insights.append(f"• Query type: {intent.title()} (complexity: {complexity:.2f})")
        
        # Search strategies
        if 'search_strategies' in self.query_analysis:
            strategies = self.query_analysis['search_strategies'][:2]  # Limit to 2
            for strategy in strategies:
                desc = strategy.get('description', '')
                if desc:
                    insights.append(f"• {desc}")
        
        return "\n".join(insights) if len(insights) > 1 else ""
    
    def _get_display_source_name(self, source_path: str, source_info: Optional[Dict[str, Any]] = None) -> str:
        """Extract a display-friendly name from source path and metadata."""
        # Try to get document title from metadata first
        if source_info:
            doc_title = source_info.get('document_title')
            if doc_title:
                return doc_title
        
        # Fallback to filename
        return Path(source_path).name
    
    def _get_source_icon(self, source_type: str) -> str:
        """Get an appropriate icon for the source type."""
        icons = {
            "document": "📄",
            "codebase": "💻", 
            "webpage": "🌐",
            "pdf": "📄",
            "markdown": "📝",
            "text": "📄"
        }
        return icons.get(source_type.lower(), "📄")
    
    def _create_content_preview(self, text: str, max_length: int = 400) -> str:
        """Create a preview of the content with smart truncation."""
        if not text:
            return "[No content available]"
        
        # Clean up the text
        cleaned = text.strip().replace('\n\n\n', '\n\n').replace('\t', ' ')
        
        # If text is short enough, return as-is
        if len(cleaned) <= max_length:
            return cleaned
        
        # Find a good break point near the limit
        truncated = cleaned[:max_length]
        
        # Try to break at sentence end
        last_sentence = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_sentence > max_length * 0.7:  # If we found a sentence end in the last 30%
            return truncated[:last_sentence + 1] + "..."
        
        # Otherwise break at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # If we found a space in the last 20%
            return truncated[:last_space] + "..."
        
        # Last resort: hard truncate
        return truncated + "..."
    
    def _format_enhanced_source_line(self, source_icon: str, source_name: str, result: 'SearchResult') -> str:
        """Format an enhanced source line with metadata."""
        parts = [f"{source_icon} [{source_name}]"]
        
        # Add page number if available
        if result.source_info:
            page_num = result.source_info.get('page_number')
            if page_num:
                parts.append(f"(Page {page_num})")
            
            # Add section if available
            section = result.source_info.get('section_title')
            if section:
                parts.append(f"- {section}")
            
            # Add file size for context
            file_size = result.source_info.get('file_size_mb')
            if file_size and file_size > 0:
                parts.append(f"({file_size}MB)")
        
        # Add chunk ID
        if result.chunk_id:
            parts.append(f"[{result.chunk_id[:8]}...]")
            
        return " ".join(parts)
    
    @classmethod
    def create_degraded(
        cls,
        query: str,
        results: List[SearchResult],
        search_config: Dict[str, Any],
        performance: Dict[str, Any],
        degradation_reason: str,
        failed_components: List[str]
    ) -> "SearchResponse":
        """Create a response for degraded search (partial success).
        
        Args:
            query: The search query
            results: Results that were successfully obtained
            search_config: Configuration used
            performance: Performance metrics
            degradation_reason: Why search was degraded
            failed_components: List of components that failed
        """
        # Add degradation info to performance
        enhanced_performance = dict(performance)
        enhanced_performance["degradation"] = {
            "reason": degradation_reason,
            "failed_components": failed_components,
            "success_partial": True
        }
        
        return cls(
            success=True,  # Still considered successful with partial results
            query=query,
            search_config=search_config,
            results=results,
            total_results=len(results),
            performance=enhanced_performance,
            error=f"Degraded search: {degradation_reason}",
            error_code="DEGRADED_SEARCH"
        )




def create_search_performance_info(
    start_time: float,
    search_mode: str,
    total_candidates: int = 0,
    rerank_time: Optional[float] = None,
    vector_time: Optional[float] = None,
    keyword_time: Optional[float] = None,
    relationship_time: Optional[float] = None,
    embedding_time: Optional[float] = None
) -> Dict[str, Any]:
    """Create enhanced performance information dictionary.
    
    Args:
        start_time: Start time from time.time()
        search_mode: Search mode used ("vector", "hybrid", "hybrid_reranked", etc.)
        total_candidates: Number of candidates before filtering
        rerank_time: Time spent on reranking (seconds)
        vector_time: Time spent on vector search (seconds)
        keyword_time: Time spent on keyword search (seconds)
        relationship_time: Time spent processing relationships (seconds)
        embedding_time: Time spent generating query embedding (seconds)
        
    Returns:
        Enhanced performance info dictionary
    """
    end_time = time.time()
    total_time_ms = int((end_time - start_time) * 1000)
    
    perf_info = {
        "search_time_ms": total_time_ms,
        "search_mode": search_mode,
        "total_candidates": total_candidates,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components_used": []
    }
    
    # Add detailed timing breakdown
    timing_breakdown = {}
    components_used = []
    
    if embedding_time is not None:
        timing_breakdown["embedding_ms"] = int(embedding_time * 1000)
        components_used.append("embedding")
    
    if vector_time is not None:
        timing_breakdown["vector_search_ms"] = int(vector_time * 1000)
        components_used.append("vector_search")
    
    if keyword_time is not None:
        timing_breakdown["keyword_search_ms"] = int(keyword_time * 1000)
        components_used.append("keyword_search")
    
    if rerank_time is not None:
        timing_breakdown["rerank_ms"] = int(rerank_time * 1000)
        components_used.append("reranking")
    
    if relationship_time is not None:
        timing_breakdown["relationship_processing_ms"] = int(relationship_time * 1000)
        components_used.append("relationships")
    
    if timing_breakdown:
        perf_info["timing_breakdown"] = timing_breakdown
    
    perf_info["components_used"] = components_used
    
    # Add performance quality metrics
    if total_candidates > 0:
        perf_info["search_efficiency"] = {
            "candidates_per_ms": round(total_candidates / max(total_time_ms, 1), 2),
            "filtering_ratio": round(total_candidates / max(len(components_used), 1), 2)
        }
    
    return perf_info


def create_structured_position(
    start_char: Optional[int] = None,
    end_char: Optional[int] = None,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Create structured position information.
    
    Args:
        start_char: Starting character position
        end_char: Ending character position  
        line_start: Starting line number
        line_end: Ending line number
        
    Returns:
        Structured position dictionary or None if no position data
    """
    position = {}
    
    if start_char is not None:
        position["start_char"] = start_char
    if end_char is not None:
        position["end_char"] = end_char
    if line_start is not None:
        position["line_start"] = line_start
    if line_end is not None:
        position["line_end"] = line_end
    
    # Add computed fields
    if start_char is not None and end_char is not None:
        position["char_length"] = end_char - start_char
    if line_start is not None and line_end is not None:
        position["line_count"] = max(1, line_end - line_start + 1)
    
    return position if position else None


def create_structured_scores(
    vector_score: Optional[float] = None,
    keyword_score: Optional[float] = None,
    rerank_score: Optional[float] = None,
    original_score: Optional[float] = None,
    combined_score: Optional[float] = None
) -> Optional[Dict[str, float]]:
    """Create structured score breakdown.
    
    Args:
        vector_score: Semantic similarity score
        keyword_score: Keyword/lexical match score
        rerank_score: Cross-encoder reranking score
        original_score: Original score before reranking
        combined_score: Final combined score
        
    Returns:
        Structured scores dictionary or None if no scores
    """
    scores = {}
    
    if vector_score is not None:
        scores["vector"] = float(vector_score)
    if keyword_score is not None:
        scores["keyword"] = float(keyword_score)
    if rerank_score is not None:
        scores["rerank"] = float(rerank_score)
    if original_score is not None:
        scores["original"] = float(original_score)
    if combined_score is not None:
        scores["combined"] = float(combined_score)
    
    return scores if scores else None


def create_structured_metadata(
    created_at: Optional[str] = None,
    file_extension: Optional[str] = None,
    word_count: Optional[int] = None,
    char_count: Optional[int] = None,
    chunk_index: Optional[int] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    **additional_metadata: Any
) -> Dict[str, Any]:
    """Create structured metadata with categorization.
    
    Args:
        created_at: Creation timestamp
        file_extension: File extension if applicable
        word_count: Number of words in chunk
        char_count: Number of characters in chunk
        chunk_index: Index of this chunk in the source
        embedding_provider: Provider used for embedding
        embedding_model: Model used for embedding
        **additional_metadata: Any additional metadata fields
        
    Returns:
        Structured metadata dictionary
    """
    metadata = {
        "content": {},
        "source": {},
        "processing": {}
    }
    
    # Content-related metadata
    if word_count is not None:
        metadata["content"]["word_count"] = word_count
    if char_count is not None:
        metadata["content"]["char_count"] = char_count
    if chunk_index is not None:
        metadata["content"]["chunk_index"] = chunk_index
        
    # Source-related metadata
    if created_at is not None:
        metadata["source"]["created_at"] = created_at
    if file_extension is not None:
        metadata["source"]["file_extension"] = file_extension
        
    # Processing-related metadata
    if embedding_provider is not None:
        metadata["processing"]["embedding_provider"] = embedding_provider
    if embedding_model is not None:
        metadata["processing"]["embedding_model"] = embedding_model
        
    # Add any additional metadata to appropriate category or root
    for key, value in additional_metadata.items():
        if key.startswith(("word_", "char_", "line_")):
            metadata["content"][key] = value
        elif key.startswith(("file_", "source_", "created_")):
            metadata["source"][key] = value
        elif key.startswith(("embedding_", "model_", "processing_")):
            metadata["processing"][key] = value
        else:
            metadata[key] = value
    
    # Remove empty categories
    return {k: v for k, v in metadata.items() if v}


def create_search_provenance(
    search_features: List[str],
    ranking_factors: Optional[Dict[str, Any]] = None,
    search_stage: str = "final",
    confidence: Optional[float] = None
) -> Dict[str, Any]:
    """Create search provenance information.
    
    Args:
        search_features: List of features used (vector, keyword, rerank, etc.)
        ranking_factors: Factors that influenced ranking
        search_stage: Stage of search (initial, reranked, filtered, etc.)
        confidence: Confidence in the result matching
        
    Returns:
        Structured provenance dictionary
    """
    provenance = {
        "features_used": search_features,
        "search_stage": search_stage,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    if ranking_factors:
        provenance["ranking_factors"] = ranking_factors
        
    if confidence is not None:
        provenance["confidence"] = round(confidence, 3)
        
    return provenance


def create_query_analysis(
    original_query: str,
    normalized_query: Optional[str] = None,
    query_length: Optional[int] = None,
    word_count: Optional[int] = None,
    detected_intent: Optional[str] = None,
    language: Optional[str] = None,
    complexity_score: Optional[float] = None,
    suggested_alternatives: Optional[List[str]] = None,
    expansion_terms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create query analysis information.
    
    Args:
        original_query: The original search query
        normalized_query: Cleaned/normalized version of the query
        query_length: Length of the query in characters
        word_count: Number of words in the query
        detected_intent: Detected search intent (informational, navigational, etc.)
        language: Detected language of the query
        complexity_score: Query complexity score (0.0 to 1.0)
        suggested_alternatives: Alternative query suggestions
        expansion_terms: Terms that could expand the query
        
    Returns:
        Query analysis dictionary
    """
    analysis = {
        "original_query": original_query,
        "processed_at": datetime.now(timezone.utc).isoformat()
    }
    
    if normalized_query is not None and normalized_query != original_query:
        analysis["normalized_query"] = normalized_query
        
    # Basic query statistics
    stats = {}
    if query_length is not None:
        stats["char_count"] = query_length
    else:
        stats["char_count"] = len(original_query)
        
    if word_count is not None:
        stats["word_count"] = word_count
    else:
        stats["word_count"] = len(original_query.split())
        
    analysis["query_stats"] = stats
    
    # Query characteristics
    characteristics = {}
    if detected_intent:
        characteristics["intent"] = detected_intent
    if language:
        characteristics["language"] = language
    if complexity_score is not None:
        characteristics["complexity_score"] = round(complexity_score, 3)
        
    if characteristics:
        analysis["characteristics"] = characteristics
    
    # Query enhancements
    enhancements = {}
    if suggested_alternatives:
        enhancements["suggested_alternatives"] = suggested_alternatives[:5]  # Limit suggestions
    if expansion_terms:
        enhancements["expansion_terms"] = expansion_terms[:10]  # Limit terms
        
    if enhancements:
        analysis["enhancements"] = enhancements
        
    return analysis


def create_versioned_response(
    response: "SearchResponse", 
    requested_version: Optional[str] = None,
    include_deprecation_warnings: bool = True
) -> Dict[str, Any]:
    """Create a versioned response with optional format adaptation.
    
    Args:
        response: The SearchResponse object
        requested_version: Specific API version requested (None for latest)
        include_deprecation_warnings: Whether to include deprecation warnings
        
    Returns:
        Dictionary with version-appropriate format
    """
    # Get base response dictionary
    result = response.to_dict()
    
    # Handle version-specific formatting
    if requested_version:
        # Parse requested version
        try:
            req_major, req_minor = map(int, requested_version.split('.'))
            current_major, current_minor = 3, 1  # Current version 3.1
            
            # Add deprecation warning if using old version
            if include_deprecation_warnings and (req_major < current_major or 
                (req_major == current_major and req_minor < current_minor)):
                
                if "warnings" not in result:
                    result["warnings"] = []
                
                result["warnings"].append({
                    "type": "deprecation",
                    "message": f"API version {requested_version} is deprecated. "
                              f"Please upgrade to version {current_major}.{current_minor}",
                    "current_version": f"{current_major}.{current_minor}",
                    "requested_version": requested_version,
                    "upgrade_guide": "https://pycontextify.readthedocs.io/api-migration/"
                })
            
            # Format for older versions (remove new features)
            if req_major < 3 or (req_major == 3 and req_minor < 1):
                # Remove Phase 3.1+ features for backward compatibility
                for search_result in result.get("results", []):
                    search_result.pop("rank", None)
                    search_result.pop("relevance_explanation", None)
                    search_result.pop("confidence_scores", None)
            
            if req_major < 2:
                # Remove Phase 2+ features for very old versions
                result.pop("query_analysis", None)
                for search_result in result.get("results", []):
                    search_result.pop("scores", None)
                    search_result.pop("provenance", None)
                    
        except ValueError:
            # Invalid version format - add warning
            if "warnings" not in result:
                result["warnings"] = []
            result["warnings"].append({
                "type": "invalid_version",
                "message": f"Invalid API version format: {requested_version}. Using latest version."
            })
    
    return result


def get_supported_api_versions() -> Dict[str, Any]:
    """Get information about supported API versions.
    
    Returns:
        Dictionary with version information
    """
    return {
        "current_version": "3.1",
        "supported_versions": [
            {
                "version": "3.1",
                "status": "current",
                "features": ["ranking", "explanations", "confidence_scores", "query_expansion"],
                "release_date": "2025-09-21"
            },
            {
                "version": "3.0",
                "status": "supported", 
                "features": ["relationships", "query_analysis", "performance_metrics"],
                "release_date": "2025-09-20"
            },
            {
                "version": "2.4",
                "status": "deprecated",
                "features": ["query_analysis"],
                "deprecation_date": "2025-09-21",
                "end_of_life": "2025-12-21"
            },
            {
                "version": "2.0",
                "status": "deprecated",
                "features": ["structured_metadata", "error_handling"],
                "deprecation_date": "2025-09-20",
                "end_of_life": "2025-12-20"
            }
        ],
        "migration_guide": "https://pycontextify.readthedocs.io/api-migration/",
        "breaking_changes": {
            "3.0_to_3.1": ["Added ranking fields to SearchResult"],
            "2.x_to_3.0": ["Restructured metadata format", "Enhanced error responses"]
        }
    }


def create_relevance_explanation(
    query: str,
    match_type: str = "semantic",
    matched_terms: Optional[List[str]] = None,
    match_score: Optional[float] = None,
    reason: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    semantic_concepts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create relevance explanation for a search result.
    
    Args:
        query: The original search query
        match_type: Type of match ("semantic", "keyword", "hybrid", "exact")
        matched_terms: Specific terms that matched
        match_score: Score of the match
        reason: Human-readable explanation of why result is relevant
        keywords: Keywords that contributed to relevance
        semantic_concepts: Semantic concepts that matched
        
    Returns:
        Structured relevance explanation
    """
    explanation = {
        "query": query,
        "match_type": match_type,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
    
    if match_score is not None:
        explanation["match_score"] = round(match_score, 4)
    
    # Add specific match details
    match_details = {}
    if matched_terms:
        match_details["matched_terms"] = matched_terms[:10]  # Limit terms
    if keywords:
        match_details["keywords"] = keywords[:8]  # Limit keywords
    if semantic_concepts:
        match_details["semantic_concepts"] = semantic_concepts[:5]  # Limit concepts
    
    if match_details:
        explanation["match_details"] = match_details
    
    # Generate automatic reason if not provided
    if reason:
        explanation["reason"] = reason
    else:
        explanation["reason"] = _generate_relevance_reason(
            match_type, matched_terms, keywords, semantic_concepts
        )
    
    return explanation


def _generate_relevance_reason(
    match_type: str,
    matched_terms: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    semantic_concepts: Optional[List[str]] = None
) -> str:
    """Generate automatic relevance explanation based on match details."""
    reasons = []
    
    if match_type == "exact":
        reasons.append("Contains exact match of query terms")
    elif match_type == "keyword":
        if matched_terms:
            reasons.append(f"Matches {len(matched_terms)} query terms")
        if keywords:
            reasons.append(f"Contains relevant keywords: {', '.join(keywords[:3])}")
    elif match_type == "semantic":
        reasons.append("Semantically similar to query")
        if semantic_concepts:
            reasons.append(f"Related concepts: {', '.join(semantic_concepts[:2])}")
    elif match_type == "hybrid":
        reasons.append("Combined keyword and semantic matching")
        if matched_terms and len(matched_terms) > 0:
            reasons.append(f"Matches {len(matched_terms)} terms")
    
    return ". ".join(reasons) if reasons else "Relevance determined by search algorithm"


def create_confidence_scores(
    semantic_confidence: Optional[float] = None,
    keyword_confidence: Optional[float] = None,
    exact_match_confidence: Optional[float] = None,
    context_confidence: Optional[float] = None,
    overall_confidence: Optional[float] = None
) -> Optional[Dict[str, float]]:
    """Create confidence scores for different match types.
    
    Args:
        semantic_confidence: Confidence in semantic similarity (0.0-1.0)
        keyword_confidence: Confidence in keyword matching (0.0-1.0)
        exact_match_confidence: Confidence in exact matches (0.0-1.0)
        context_confidence: Confidence based on context/relationships (0.0-1.0)
        overall_confidence: Overall confidence in result relevance (0.0-1.0)
        
    Returns:
        Structured confidence scores or None if no scores provided
    """
    confidence = {}
    
    if semantic_confidence is not None:
        confidence["semantic"] = round(max(0.0, min(1.0, semantic_confidence)), 3)
    if keyword_confidence is not None:
        confidence["keyword"] = round(max(0.0, min(1.0, keyword_confidence)), 3)
    if exact_match_confidence is not None:
        confidence["exact_match"] = round(max(0.0, min(1.0, exact_match_confidence)), 3)
    if context_confidence is not None:
        confidence["context"] = round(max(0.0, min(1.0, context_confidence)), 3)
    if overall_confidence is not None:
        confidence["overall"] = round(max(0.0, min(1.0, overall_confidence)), 3)
    
    return confidence if confidence else None


def create_result_ranking_info(
    total_results: int,
    rank_position: int,
    score_percentile: Optional[float] = None,
    relative_score: Optional[float] = None
) -> Dict[str, Any]:
    """Create ranking information for a search result.
    
    Args:
        total_results: Total number of results in the set
        rank_position: 1-based position of this result
        score_percentile: Percentile ranking by score (0.0-1.0)
        relative_score: Score relative to top result (0.0-1.0)
        
    Returns:
        Ranking information dictionary
    """
    ranking_info = {
        "position": rank_position,
        "total_results": total_results,
        "top_percentile": round((total_results - rank_position + 1) / total_results * 100, 1)
    }
    
    if score_percentile is not None:
        ranking_info["score_percentile"] = round(score_percentile * 100, 1)
    if relative_score is not None:
        ranking_info["relative_score"] = round(relative_score, 3)
    
    return ranking_info


def enhance_search_results_with_ranking(
    results: List["SearchResult"],
    query: str,
    include_explanations: bool = True,
    include_confidence: bool = True
) -> List["SearchResult"]:
    """Enhance search results with ranking information and explanations.
    
    Args:
        results: List of search results to enhance
        query: Original search query
        include_explanations: Whether to add relevance explanations
        include_confidence: Whether to add confidence scores
        
    Returns:
        Enhanced results with ranking information
    """
    if not results:
        return results
    
    # Sort results by relevance score (descending) to ensure proper ranking
    sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)
    
    # Calculate score statistics for relative ranking
    scores = [r.relevance_score for r in sorted_results]
    max_score = max(scores) if scores else 1.0
    
    enhanced_results = []
    for i, result in enumerate(sorted_results, 1):
        # Create a copy to avoid modifying original
        enhanced_result = SearchResult(
            chunk_id=result.chunk_id,
            source_path=result.source_path,
            source_type=result.source_type,
            text=result.text,
            relevance_score=result.relevance_score,
            position=result.position,
            scores=result.scores,
            metadata=result.metadata,
            context=result.context,
            provenance=result.provenance,
            source_info=result.source_info  # Preserve source_info
        )
        
        # Add ranking information
        enhanced_result.rank = i
        
        # Add relevance explanation if requested
        if include_explanations:
            match_type = _determine_match_type(result)
            matched_terms = _extract_matched_terms(result, query)
            
            enhanced_result.relevance_explanation = create_relevance_explanation(
                query=query,
                match_type=match_type,
                matched_terms=matched_terms,
                match_score=result.relevance_score
            )
        
        # Add confidence scores if requested
        if include_confidence:
            enhanced_result.confidence_scores = _calculate_confidence_scores(result)
        
        enhanced_results.append(enhanced_result)
    
    return enhanced_results


def _determine_match_type(result: "SearchResult") -> str:
    """Determine the primary match type based on result scores and provenance."""
    if result.scores:
        # Check if reranking was used
        if "rerank" in result.scores:
            return "hybrid"  # Hybrid + reranked
        elif "vector" in result.scores and "keyword" in result.scores:
            return "hybrid"  # Hybrid search
        elif "vector" in result.scores:
            return "semantic"  # Vector/semantic only
        elif "keyword" in result.scores:
            return "keyword"  # Keyword only
    
    # Check provenance for features used
    if result.provenance and "features_used" in result.provenance:
        features = result.provenance["features_used"]
        if "rerank" in features:
            return "hybrid"
        elif "vector" in features and "keyword" in features:
            return "hybrid"
        elif "vector" in features:
            return "semantic"
        elif "keyword" in features:
            return "keyword"
    
    # Default based on score
    if result.relevance_score >= 0.95:
        return "exact"
    elif result.relevance_score >= 0.7:
        return "semantic"
    else:
        return "keyword"


def _extract_matched_terms(result: "SearchResult", query: str) -> List[str]:
    """Extract matched terms from result text and query."""
    query_terms = query.lower().split()
    result_text = result.text.lower()
    
    matched = []
    for term in query_terms:
        if term in result_text:
            matched.append(term)
    
    return matched


def _calculate_confidence_scores(result: "SearchResult") -> Optional[Dict[str, float]]:
    """Calculate confidence scores based on result characteristics."""
    # Base confidence on relevance score
    base_confidence = min(1.0, result.relevance_score)
    
    # Semantic confidence based on vector score
    semantic_confidence = None
    if result.scores and "vector" in result.scores:
        semantic_confidence = min(1.0, result.scores["vector"])
    else:
        semantic_confidence = base_confidence * 0.7  # Lower if no vector score
    
    # Keyword confidence based on keyword score  
    keyword_confidence = None
    if result.scores and "keyword" in result.scores:
        keyword_confidence = min(1.0, result.scores["keyword"])
    else:
        keyword_confidence = base_confidence * 0.5  # Lower if no keyword score
    
    # Exact match confidence based on high relevance
    exact_match_confidence = 1.0 if result.relevance_score >= 0.95 else 0.0
    
    # Overall confidence
    overall_confidence = base_confidence
    
    return create_confidence_scores(
        semantic_confidence=semantic_confidence,
        keyword_confidence=keyword_confidence,
        exact_match_confidence=exact_match_confidence,
        overall_confidence=overall_confidence
    )


class SearchPerformanceLogger:
    """Logger for tracking search performance patterns and metrics."""
    
    def __init__(self, logger_name: str = "pycontextify.search.performance"):
        self.logger = logging.getLogger(logger_name)
        self.search_stats = {
            "total_searches": 0,
            "average_response_time_ms": 0.0,
            "search_modes_used": {},
            "error_rate": 0.0,
            "total_errors": 0
        }
    
    def log_search_performance(self, response: "SearchResponse") -> None:
        """Log performance metrics for a search response."""
        self.search_stats["total_searches"] += 1
        
        if response.success and response.performance:
            # Update average response time
            search_time = response.performance.get("search_time_ms", 0)
            current_avg = self.search_stats["average_response_time_ms"]
            total = self.search_stats["total_searches"]
            self.search_stats["average_response_time_ms"] = (
                (current_avg * (total - 1) + search_time) / total
            )
            
            # Track search mode usage
            search_mode = response.performance.get("search_mode", "unknown")
            if search_mode not in self.search_stats["search_modes_used"]:
                self.search_stats["search_modes_used"][search_mode] = 0
            self.search_stats["search_modes_used"][search_mode] += 1
            
            # Log detailed performance info if enabled
            query_display = f"{response.query[:50]}..." if len(response.query) > 50 else response.query
            self.logger.debug(
                f"Search performance: {search_time}ms, mode: {search_mode}, "
                f"results: {len(response.results)}, query: '{query_display}'"
            )
        else:
            # Track errors
            self.search_stats["total_errors"] += 1
            self.search_stats["error_rate"] = (
                self.search_stats["total_errors"] / self.search_stats["total_searches"]
            )
            
            # Log error details
            query_display = f"{response.query[:50]}..." if len(response.query) > 50 else response.query
            self.logger.warning(
                f"Search failed: {response.error} (code: {response.error_code}) "
                f"for query: '{query_display}'"
            )
    
    def log_slow_search(self, response: "SearchResponse", threshold_ms: int = 1000) -> None:
        """Log searches that exceed the performance threshold."""
        if response.success and response.performance:
            search_time = response.performance.get("search_time_ms", 0)
            if search_time > threshold_ms:
                components = response.performance.get("components_used", [])
                breakdown = response.performance.get("timing_breakdown", {})
                
                query_display = f"{response.query[:50]}..." if len(response.query) > 50 else response.query
                self.logger.warning(
                    f"Slow search detected: {search_time}ms > {threshold_ms}ms threshold. "
                    f"Query: '{query_display}', "
                    f"Components: {components}, Breakdown: {breakdown}"
                )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of search performance statistics."""
        return {
            "performance_summary": {
                **self.search_stats,
                "success_rate": round(1.0 - self.search_stats["error_rate"], 3),
                "average_response_time_ms": round(self.search_stats["average_response_time_ms"], 2)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
