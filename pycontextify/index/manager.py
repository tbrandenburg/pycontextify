"""Central IndexManager for PyContextify.

This module implements the main coordination system that orchestrates all
components for indexing operations, persistence, and search functionality.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import psutil

from .chunker import ChunkerFactory
from .config import Config
from .embedders import EmbedderFactory
from .loaders import LoaderFactory
from .metadata import MetadataStore, SourceType
from .models import (
    SearchResult,
    SearchResponse,
    SearchPerformanceLogger,
    SearchErrorCode,
    create_search_performance_info,
    create_structured_position,
    create_structured_scores,
    create_structured_metadata,
    create_search_provenance,
    create_query_analysis,
    enhance_search_results_with_ranking
)
from .relationship_store import RelationshipStore
from .search_intelligence import SearchIntelligenceFramework
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class IndexManager:
    """Central manager for all indexing operations with auto-persistence."""

    def __init__(self, config: Config):
        """Initialize IndexManager with configuration.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize components
        self.metadata_store = MetadataStore()
        self.relationship_store = RelationshipStore()
        self.embedder = None
        self.vector_store = None
        self.hybrid_search = None
        self.reranker = None
        
        # Store embedder configuration for lazy loading
        self._embedder_config = None
        self._embedder_initialized = False
        
        # Initialize performance logger
        self.performance_logger = SearchPerformanceLogger()
        
        # Initialize search intelligence framework (Phase 3)
        self._initialize_search_intelligence()
        
        # Initialize enhanced search components (these are lightweight)
        self._initialize_hybrid_search()
        self._initialize_reranker()

        # Auto-load existing index if enabled
        if self.config.auto_load:
            self._auto_load()

    def _ensure_embedder_loaded(self) -> None:
        """Ensure embedder is loaded (lazy loading)."""
        if self._embedder_initialized:
            return
            
        try:
            embedding_config = self.config.get_embedding_config()
            logger.info(f"Lazy loading embedder: {embedding_config['provider']} with model {embedding_config['model']}")
            
            self.embedder = EmbedderFactory.create_embedder(
                provider=embedding_config["provider"],
                model_name=embedding_config["model"],
                **{
                    k: v
                    for k, v in embedding_config.items()
                    if k not in ["provider", "model"]
                },
            )
            
            # Mark as initialized before initializing vector store
            self._embedder_initialized = True
            
            # Initialize vector store now that we have embedder
            self._initialize_vector_store()
            
            logger.info(f"Successfully loaded embedder: {self.embedder.get_provider_name()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise

    def _initialize_vector_store(self) -> None:
        """Initialize FAISS vector store (lazy)."""
        # Only initialize if embedder is already loaded and vector store doesn't exist
        if self._embedder_initialized and self.embedder and self.vector_store is None:
            dimension = self.embedder.get_dimension()
            self.vector_store = VectorStore(dimension, self.config)
            logger.info(f"Initialized vector store with dimension {dimension}")
    
    def _initialize_hybrid_search(self) -> None:
        """Initialize hybrid search engine if enabled."""
        if not self.config.use_hybrid_search:
            logger.info("Hybrid search disabled by configuration")
            return
            
        try:
            from .hybrid_search import HybridSearchEngine
            self.hybrid_search = HybridSearchEngine(keyword_weight=self.config.keyword_weight)
            logger.info(f"Initialized hybrid search with keyword weight: {self.config.keyword_weight}")
        except ImportError as e:
            logger.warning(f"Could not initialize hybrid search: {e}")
            self.hybrid_search = None
    
    def _initialize_reranker(self) -> None:
        """Initialize cross-encoder reranker if enabled."""
        if not self.config.use_reranking:
            logger.info("Reranking disabled by configuration")
            return
            
        try:
            from .reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(model_name=self.config.reranking_model)
            
            # Warm up the model
            if self.reranker.is_available:
                self.reranker.warmup()
                
            logger.info(f"Initialized reranker with model: {self.config.reranking_model}")
        except ImportError as e:
            logger.warning(f"Could not initialize reranker: {e}")
            self.reranker = None
    
    def _initialize_search_intelligence(self) -> None:
        """Initialize search intelligence framework."""
        try:
            # Create storage path based on config
            intelligence_path = None
            if hasattr(self.config, 'index_directory') and self.config.index_directory:
                intelligence_path = os.path.join(self.config.index_directory, 'search_intelligence.json')
            
            self.search_intelligence = SearchIntelligenceFramework(
                storage_path=intelligence_path,
                max_history=getattr(self.config, 'max_search_history', 10000)
            )
            logger.info("Initialized search intelligence framework")
        except Exception as e:
            logger.warning(f"Could not initialize search intelligence: {e}")
            self.search_intelligence = None

    def _create_smart_preview(self, text: str, entity: str, max_length: int = 150) -> str:
        """Create smart text preview focused around the shared entity.
        
        Args:
            text: Full text to preview
            entity: Entity to focus preview around
            max_length: Maximum length of preview
            
        Returns:
            Smart preview text with entity context
        """
        # Find entity in text (case-insensitive)
        text_lower = text.lower()
        entity_lower = entity.lower()
        
        entity_pos = text_lower.find(entity_lower)
        if entity_pos == -1:
            # Entity not found, return standard preview
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Calculate preview window around entity
        padding = (max_length - len(entity)) // 2
        start_pos = max(0, entity_pos - padding)
        end_pos = min(len(text), entity_pos + len(entity) + padding)
        
        # Adjust to word boundaries
        if start_pos > 0:
            # Find previous word boundary
            while start_pos > 0 and text[start_pos] not in ' \n\t':
                start_pos -= 1
            start_pos += 1  # Move past the space
        
        if end_pos < len(text):
            # Find next word boundary
            while end_pos < len(text) and text[end_pos] not in ' \n\t.!?':
                end_pos += 1
        
        preview = text[start_pos:end_pos].strip()
        
        # Add ellipsis if truncated
        if start_pos > 0:
            preview = "..." + preview
        if end_pos < len(text):
            preview = preview + "..."
        
        return preview
    
    def _calculate_context_relevance(
        self, query: str, related_text: str, relationship_strength: float
    ) -> float:
        """Calculate relevance score for related chunk based on query similarity.
        
        Args:
            query: Original search query
            related_text: Text of related chunk
            relationship_strength: Strength of relationship (0.0-1.0)
            
        Returns:
            Relevance score (0.0-1.0)
        """
        # Simple keyword-based relevance (can be enhanced with embeddings)
        query_terms = set(query.lower().split())
        text_terms = set(related_text.lower().split())
        
        # Calculate overlap
        overlap = len(query_terms & text_terms)
        max_overlap = max(len(query_terms), 1)
        
        keyword_relevance = overlap / max_overlap
        
        # Combine with relationship strength (weighted average)
        combined_relevance = (keyword_relevance * 0.6) + (relationship_strength * 0.4)
        
        return min(1.0, combined_relevance)
    
    def _generate_query_suggestions(self, query: str, word_count: int, intent: str) -> List[str]:
        """Generate intelligent query suggestions based on patterns and intent."""
        suggestions = []
        
        # Single word queries - suggest expansions
        if word_count == 1 and len(query) > 3:
            suggestions.extend([
                f"how to {query}",
                f"{query} example",
                f"{query} tutorial",
                f"{query} documentation"
            ])
            
        # Long queries - suggest simplification
        elif word_count > 8:
            key_words = [word for word in query.split() if len(word) > 3][:4]
            suggestions.append(' '.join(key_words))
            
        # Intent-based suggestions
        if intent == "informational":
            if "how" not in query.lower():
                suggestions.append(f"how to {query}")
            suggestions.append(f"{query} guide")
            
        elif intent == "example_seeking":
            if "example" not in query.lower():
                suggestions.append(f"{query} example")
            suggestions.append(f"{query} sample code")
            
        elif intent == "comparative":
            if "vs" not in query.lower() and "versus" not in query.lower():
                words = query.split()
                if len(words) >= 2:
                    suggestions.append(f"{words[0]} vs {words[-1]}")
        
        # Remove duplicates and limit
        return list(dict.fromkeys(suggestions))[:5]
    
    def _generate_expansion_terms(self, normalized_query: str) -> List[str]:
        """Generate semantic expansion terms based on query content."""
        expansion_terms = []
        
        # Technical domains
        tech_expansions = {
            "code": ["implementation", "example", "tutorial", "function", "method"],
            "function": ["method", "implementation", "code", "usage"],
            "api": ["documentation", "endpoint", "reference", "integration"],
            "database": ["query", "schema", "table", "sql"],
            "web": ["html", "css", "javascript", "frontend", "backend"],
            "security": ["authentication", "authorization", "encryption", "vulnerability"],
            "performance": ["optimization", "speed", "efficiency", "benchmark"],
        }
        
        # Problem-solving terms
        problem_expansions = {
            "error": ["fix", "solution", "troubleshoot", "debug", "resolve"],
            "bug": ["fix", "patch", "workaround", "issue", "problem"],
            "issue": ["solution", "fix", "resolve", "troubleshoot"],
            "problem": ["solution", "fix", "resolve", "answer"],
        }
        
        # Learning terms  
        learning_expansions = {
            "tutorial": ["guide", "walkthrough", "example", "lesson"],
            "guide": ["tutorial", "manual", "documentation", "howto"],
            "example": ["sample", "demo", "illustration", "case study"],
            "documentation": ["docs", "reference", "manual", "guide"],
        }
        
        # Combine all expansion dictionaries
        all_expansions = {**tech_expansions, **problem_expansions, **learning_expansions}
        
        # Find matching terms and add expansions
        for term, expansions in all_expansions.items():
            if term in normalized_query:
                expansion_terms.extend(expansions)
        
        # Remove duplicates and terms already in query
        query_words = set(normalized_query.split())
        expansion_terms = [term for term in expansion_terms if term not in query_words]
        
        return list(dict.fromkeys(expansion_terms))[:8]
    
    def _suggest_search_strategies(self, query: str, intent: str, complexity_score: float) -> List[Dict[str, str]]:
        """Suggest alternative search strategies based on query analysis."""
        strategies = []
        
        # Strategy based on complexity
        if complexity_score > 0.7:
            strategies.append({
                "strategy": "simplify_query",
                "description": "Try breaking down your query into simpler terms",
                "rationale": "Complex queries may miss relevant results"
            })
            
        if complexity_score < 0.3 and len(query.split()) <= 2:
            strategies.append({
                "strategy": "expand_query", 
                "description": "Add more descriptive terms to narrow your search",
                "rationale": "Short queries may return too broad results"
            })
        
        # Intent-based strategies
        if intent == "informational":
            strategies.append({
                "strategy": "how_to_search",
                "description": "Try adding 'how to' or 'guide' to find tutorials",
                "rationale": "Informational queries benefit from instructional content"
            })
            
        elif intent == "example_seeking":
            strategies.append({
                "strategy": "example_search",
                "description": "Search for 'sample', 'demo', or 'case study'",
                "rationale": "Example-focused searches find practical implementations"
            })
        
        # Domain-specific strategies
        if any(tech_term in query.lower() for tech_term in ["code", "function", "api", "programming"]):
            strategies.append({
                "strategy": "technical_search",
                "description": "Include programming language or framework names",
                "rationale": "Technical queries benefit from specific technology context"
            })
            
        return strategies[:3]  # Limit to top 3 strategies
    
    def _auto_load(self) -> None:
        """Automatically load existing index if available."""
        try:
            paths = self.config.get_index_paths()

            # Check if index files exist
            if all(path.exists() for path in paths.values()):
                logger.info("Loading existing index...")

                # First load metadata to get embedding info
                self.metadata_store.load_from_file(str(paths["metadata"]))
                
                # Load relationships
                self.relationship_store.load_from_file(str(paths["relationships"]))
                
                # Check if we have chunks to load
                if self.metadata_store.get_stats().get("total_chunks", 0) > 0:
                    # Get embedding info from stored metadata
                    embedding_info = self.metadata_store.get_embedding_info()
                    if embedding_info and embedding_info.get("models"):
                        # Extract provider and model from stored info
                        first_model = embedding_info["models"][0]  # Format: "provider:model"
                        if ":" in first_model:
                            stored_provider, stored_model = first_model.split(":", 1)
                            
                            # Override config temporarily to match stored embeddings
                            original_provider = self.config.embedding_provider
                            original_model = self.config.embedding_model
                            self.config.embedding_provider = stored_provider
                            self.config.embedding_model = stored_model
                            
                            logger.info(f"Loading with stored embedding settings: {stored_provider}:{stored_model}")
                    
                    # Now ensure embedder is loaded (this will create vector store with correct dimensions)
                    self._ensure_embedder_loaded()
                    
                    # Load vector store after embedder initialization
                    if self.vector_store is not None:
                        self.vector_store.load_from_file(str(paths["index"]))
                        logger.info(f"Loaded {self.vector_store.get_total_vectors()} vectors")
                    else:
                        logger.error("Vector store not initialized, cannot load vectors")
                else:
                    logger.info("No chunks in metadata, skipping vector loading")

                logger.info("Successfully loaded existing index")
            else:
                logger.info("No existing index found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _validate_embedding_compatibility(self) -> bool:
        """Validate that existing index is compatible with current embedding settings."""
        if self.metadata_store.get_stats()["total_chunks"] == 0:
            return True

        # Need embedder to validate compatibility
        self._ensure_embedder_loaded()
        return self.metadata_store.validate_embedding_compatibility(
            self.embedder.get_provider_name(), self.embedder.get_model_name()
        )

    def _auto_save(self) -> None:
        """Automatically save all components if auto-persist is enabled."""
        if not self.config.auto_persist:
            return

        try:
            self.config.ensure_index_directory()
            paths = self.config.get_index_paths()

            # Save vector store
            self.vector_store.save_to_file(str(paths["index"]))

            # Save metadata
            self.metadata_store.save_to_file(
                str(paths["metadata"]), self.config.compress_metadata
            )

            # Save relationships
            self.relationship_store.save_to_file(
                str(paths["relationships"]), self.config.compress_metadata
            )

            logger.info("Auto-saved index to disk")
        except Exception as e:
            logger.error(f"Failed to auto-save index: {e}")
            # Don't raise - indexing should continue even if save fails

    def index_codebase(self, path: str) -> Dict[str, Any]:
        """Index a codebase directory.

        Args:
            path: Path to codebase directory

        Returns:
            Statistics about the indexing operation
        """
        logger.info(f"Starting codebase indexing: {path}")

        try:
            # Load content
            loader = LoaderFactory.get_loader(
                SourceType.CODE, max_file_size_mb=self.config.max_file_size_mb
            )
            files = loader.load(path)

            if not files:
                return {"error": "No files found to index"}

            # Process files
            chunks_added = 0
            for file_path, content in files:
                chunks_added += self._process_content(
                    content, file_path, SourceType.CODE
                )

            # Auto-save after successful indexing
            self._auto_save()

            # Ensure embedder loaded before accessing provider/model info
            self._ensure_embedder_loaded()
            stats = {
                "files_processed": len(files),
                "chunks_added": chunks_added,
                "source_type": "code",
                "embedding_provider": self.embedder.get_provider_name(),
                "embedding_model": self.embedder.get_model_name(),
            }

            logger.info(f"Completed codebase indexing: {stats}")
            return stats

        except Exception as e:
            error_msg = f"Failed to index codebase {path}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def index_document(self, path: str) -> Dict[str, Any]:
        """Index a single document.

        Args:
            path: Path to document file

        Returns:
            Statistics about the indexing operation
        """
        logger.info(f"Starting document indexing: {path}")

        try:
            # Load content with PDF engine configuration
            loader = LoaderFactory.get_loader(
                SourceType.DOCUMENT, 
                pdf_engine=self.config.pdf_engine
            )
            files = loader.load(path)

            if not files:
                return {"error": "Could not load document"}

            # Process document
            file_path, content = files[0]
            chunks_added = self._process_content(
                content, file_path, SourceType.DOCUMENT
            )

            # Auto-save after successful indexing
            self._auto_save()

            # Ensure embedder loaded before accessing provider/model info
            self._ensure_embedder_loaded()
            stats = {
                "file_processed": file_path,
                "chunks_added": chunks_added,
                "source_type": "document",
                "embedding_provider": self.embedder.get_provider_name(),
                "embedding_model": self.embedder.get_model_name(),
            }

            logger.info(f"Completed document indexing: {stats}")
            return stats

        except Exception as e:
            error_msg = f"Failed to index document {path}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def index_webpage(
        self, url: str, recursive: bool = False, max_depth: int = 1
    ) -> Dict[str, Any]:
        """Index web content.

        Args:
            url: URL to index
            recursive: Whether to follow links
            max_depth: Maximum crawl depth

        Returns:
            Statistics about the indexing operation
        """
        logger.info(
            f"Starting webpage indexing: {url} (recursive={recursive}, max_depth={max_depth})"
        )

        try:
            # Load content
            loader = LoaderFactory.get_loader(
                SourceType.WEBPAGE, delay_seconds=self.config.crawl_delay_seconds
            )
            pages = loader.load(url, recursive=recursive, max_depth=max_depth)

            if not pages:
                return {"error": "Could not load any web pages"}

            # Process pages
            chunks_added = 0
            for page_url, content in pages:
                chunks_added += self._process_content(
                    content, page_url, SourceType.WEBPAGE
                )

            # Auto-save after successful indexing
            self._auto_save()

            # Ensure embedder loaded before accessing provider/model info
            self._ensure_embedder_loaded()
            stats = {
                "pages_processed": len(pages),
                "chunks_added": chunks_added,
                "source_type": "webpage",
                "recursive": recursive,
                "max_depth": max_depth,
                "embedding_provider": self.embedder.get_provider_name(),
                "embedding_model": self.embedder.get_model_name(),
            }

            logger.info(f"Completed webpage indexing: {stats}")
            return stats

        except Exception as e:
            error_msg = f"Failed to index webpage {url}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def _process_content(
        self, content: str, source_path: str, source_type: SourceType
    ) -> int:
        """Process content into chunks and add to index.

        Args:
            content: Text content to process
            source_path: Source path or URL
            source_type: Type of content

        Returns:
            Number of chunks added
        """
        # CHECK FOR EXISTING CONTENT AND REMOVE (RE-INDEXING LOGIC)
        existing_chunks = self.metadata_store.get_chunks_by_source_path(source_path)
        chunks_removed = 0
        
        if existing_chunks:
            logger.info(f"Found {len(existing_chunks)} existing chunks for {source_path}, removing for re-indexing")
            
            # Remove from vector store
            if self.vector_store is not None:
                faiss_ids_to_remove = []
                for chunk in existing_chunks:
                    faiss_id = self.metadata_store.get_faiss_id(chunk.chunk_id)
                    if faiss_id is not None:
                        faiss_ids_to_remove.append(faiss_id)
                
                if faiss_ids_to_remove:
                    self.vector_store.remove_vectors(faiss_ids_to_remove)
            
            # Remove from metadata store
            for chunk in existing_chunks:
                faiss_id = self.metadata_store.get_faiss_id(chunk.chunk_id)
                if faiss_id is not None:
                    self.metadata_store.remove_chunk(faiss_id)
            
            # Remove from relationship store
            if self.config.enable_relationships and self.relationship_store:
                for chunk in existing_chunks:
                    self.relationship_store.remove_chunk_relationships(chunk.chunk_id)
            
            chunks_removed = len(existing_chunks)
            logger.info(f"Removed {chunks_removed} existing chunks for re-indexing")
        
        # Get appropriate chunker
        chunker = ChunkerFactory.get_chunker(source_type, self.config)

        # Ensure embedder is loaded before chunking (we pass provider/model info)
        self._ensure_embedder_loaded()
        
        # Chunk content
        chunks = chunker.chunk_text(
            content,
            source_path,
            self.embedder.get_provider_name(),
            self.embedder.get_model_name(),
        )

        if not chunks:
            return 0

        # Generate embeddings
        texts = [chunk.chunk_text for chunk in chunks]
        self._ensure_embedder_loaded()
        embeddings = self.embedder.embed_texts(texts)

        # Add to vector store
        if self.vector_store is None:
            self._initialize_vector_store()
        faiss_ids = self.vector_store.add_vectors(embeddings)

        # Add metadata
        for chunk, faiss_id in zip(chunks, faiss_ids):
            self.metadata_store.add_chunk(chunk)

        # Build relationships
        if self.config.enable_relationships:
            self.relationship_store.build_relationships_from_chunks(chunks)

        return len(chunks)
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Advanced query analysis with semantic understanding and search intelligence.
        
        Args:
            query: The search query to analyze
            
        Returns:
            Comprehensive query analysis with enhanced insights
        """
        # Phase 3.1: Enhanced Query Preprocessing
        normalized_query = self._normalize_query(query)
        
        # Phase 3.2: Advanced Intent Detection
        intent_analysis = self._analyze_query_intent(normalized_query, query)
        
        # Phase 3.3: Semantic Complexity Analysis
        complexity_analysis = self._analyze_query_complexity(query, normalized_query)
        
        # Phase 3.4: Domain and Context Detection
        domain_analysis = self._analyze_query_domain(normalized_query)
        
        # Phase 3.5: Query Quality Assessment
        quality_analysis = self._assess_query_quality(query, normalized_query)
        
        # Phase 3.6: Intelligent Search Suggestions
        suggestions = self._generate_intelligent_suggestions(query, normalized_query, intent_analysis, domain_analysis)
        
        # Phase 3.7: Search Strategy Recommendations
        search_strategies = self._recommend_search_strategies(query, intent_analysis, complexity_analysis, domain_analysis)
        
        # Compile comprehensive analysis
        analysis = create_query_analysis(
            original_query=query,
            normalized_query=normalized_query if normalized_query != query.lower() else None,
            query_length=len(query),
            word_count=len(query.split()),
            detected_intent=intent_analysis["primary_intent"],
            complexity_score=complexity_analysis["overall_score"],
            suggested_alternatives=suggestions.get("alternatives") if suggestions else None,
            expansion_terms=suggestions.get("expansions") if suggestions else None
        )
        
        # Add Phase 3 enhanced analysis
        analysis.update({
            "intent_analysis": intent_analysis,
            "complexity_analysis": complexity_analysis,
            "domain_analysis": domain_analysis,
            "quality_analysis": quality_analysis,
            "search_strategies": search_strategies,
            "intelligent_suggestions": suggestions
        })
        
        return analysis
    
    def _normalize_query(self, query: str) -> str:
        """Enhanced query normalization with intelligent preprocessing."""
        import re
        
        # Basic normalization
        normalized = query.strip().lower()
        
        # Handle multiple whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common query patterns
        normalized = re.sub(r'["\']+', '', normalized)  # Remove quotes
        normalized = re.sub(r'[!?]{2,}', '!', normalized)  # Normalize punctuation
        
        # Handle common abbreviations and expansions
        expansions = {
            'how to': 'how to',  # Keep as is
            'howto': 'how to',
            'what\'s': 'what is',
            'where\'s': 'where is',
            'can\'t': 'cannot',
            'won\'t': 'will not',
        }
        
        for abbrev, expansion in expansions.items():
            normalized = normalized.replace(abbrev, expansion)
        
        return normalized
    
    def _analyze_query_intent(self, normalized_query: str, original_query: str) -> Dict[str, Any]:
        """Advanced intent analysis with confidence scoring."""
        import re
        
        # Intent patterns with confidence weights
        intent_patterns = {
            "informational": {
                "patterns": [r'\b(how|what|why|when|where|which|who)\b', r'\b(explain|describe|define)\b'],
                "confidence": 0.8
            },
            "navigational": {
                "patterns": [r'\b(find|show|get|search|locate|lookup)\b', r'\b(go to|navigate to)\b'],
                "confidence": 0.7
            },
            "example_seeking": {
                "patterns": [r'\b(example|sample|demo|tutorial|walkthrough)\b', r'\b(show me|give me).*\b(example|sample)\b'],
                "confidence": 0.9
            },
            "comparative": {
                "patterns": [r'\b(compare|vs|versus|difference|better|best|worst)\b', r'\b(which is|what\'s the difference)\b'],
                "confidence": 0.85
            },
            "troubleshooting": {
                "patterns": [r'\b(error|bug|issue|problem|fix|solve|troubleshoot)\b', r'\b(not working|doesn\'t work)\b'],
                "confidence": 0.9
            },
            "implementation": {
                "patterns": [r'\b(implement|create|build|make|develop|code)\b', r'\b(step by step|guide)\b'],
                "confidence": 0.8
            }
        }
        
        # Score each intent
        intent_scores = {}
        for intent, config in intent_patterns.items():
            score = 0
            for pattern in config["patterns"]:
                matches = len(re.findall(pattern, normalized_query))
                score += matches * config["confidence"]
            intent_scores[intent] = score
        
        # Determine primary and secondary intents
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_intent = sorted_intents[0][0] if sorted_intents[0][1] > 0 else "general"
        secondary_intent = sorted_intents[1][0] if len(sorted_intents) > 1 and sorted_intents[1][1] > 0 else None
        
        # Calculate confidence
        total_score = sum(intent_scores.values())
        primary_confidence = intent_scores.get(primary_intent, 0.0) / max(total_score, 1.0) if total_score > 0 else 0.1
        
        return {
            "primary_intent": primary_intent,
            "secondary_intent": secondary_intent,
            "confidence": round(primary_confidence, 3),
            "all_scores": intent_scores,
            "intent_indicators": self._extract_intent_indicators(normalized_query, original_query)
        }
    
    def _analyze_query_complexity(self, query: str, normalized_query: str) -> Dict[str, Any]:
        """Comprehensive query complexity analysis."""
        import re
        
        word_count = len(query.split())
        char_count = len(query)
        
        # Lexical complexity
        unique_words = len(set(normalized_query.split()))
        lexical_diversity = unique_words / max(word_count, 1)
        
        # Syntactic complexity
        punctuation_count = len(re.findall(r'[.,;:!?()\[\]{}"\']', query))
        syntactic_score = min(punctuation_count / max(char_count, 1) * 10, 1.0)
        
        # Semantic complexity
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b|\b\w*[A-Z]\w*[A-Z]\w*\b', query))
        semantic_score = min(technical_terms / max(word_count, 1) * 2, 1.0)
        
        # Query structure complexity
        has_operators = bool(re.search(r'\b(and|or|not|AND|OR|NOT)\b', query))
        has_quotes = '"' in query or "'" in query
        has_wildcards = '*' in query or '?' in query
        
        structure_features = sum([has_operators, has_quotes, has_wildcards])
        structure_score = min(structure_features / 3.0, 1.0)
        
        # Overall complexity calculation
        complexity_factors = {
            "length": min(char_count / 100, 1.0),
            "word_count": min(word_count / 20, 1.0),
            "lexical_diversity": lexical_diversity,
            "syntactic": syntactic_score,
            "semantic": semantic_score,
            "structure": structure_score
        }
        
        overall_score = sum(complexity_factors.values()) / len(complexity_factors)
        
        # Complexity classification
        if overall_score < 0.3:
            complexity_level = "simple"
        elif overall_score < 0.6:
            complexity_level = "moderate"
        else:
            complexity_level = "complex"
        
        return {
            "overall_score": round(overall_score, 3),
            "complexity_level": complexity_level,
            "factors": {k: round(v, 3) for k, v in complexity_factors.items()},
            "metrics": {
                "word_count": word_count,
                "char_count": char_count,
                "unique_words": unique_words,
                "lexical_diversity": round(lexical_diversity, 3),
                "has_operators": has_operators,
                "has_quotes": has_quotes,
                "has_wildcards": has_wildcards
            }
        }
    
    def _analyze_query_domain(self, normalized_query: str) -> Dict[str, Any]:
        """Detect query domain and technical context."""
        # Domain keyword patterns
        domains = {
            "programming": ["code", "function", "class", "method", "api", "library", "framework", "debug", "syntax", "algorithm"],
            "data_science": ["data", "analysis", "model", "machine learning", "statistics", "dataset", "prediction", "analytics"],
            "web_development": ["html", "css", "javascript", "react", "vue", "angular", "frontend", "backend", "web", "server"],
            "database": ["sql", "database", "query", "table", "schema", "index", "join", "select", "insert", "update"],
            "system_admin": ["server", "linux", "windows", "network", "security", "firewall", "backup", "monitoring"],
            "mobile": ["android", "ios", "mobile", "app", "react native", "flutter", "swift", "kotlin"],
            "cloud": ["aws", "azure", "gcp", "cloud", "docker", "kubernetes", "serverless", "microservices"],
            "ai_ml": ["artificial intelligence", "neural network", "deep learning", "tensorflow", "pytorch", "nlp"]
        }
        
        # Score domains
        domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in normalized_query)
            if score > 0:
                domain_scores[domain] = score
        
        # Determine primary domain
        primary_domain = "general"
        confidence = 0.0
        
        if domain_scores:
            primary_domain = max(domain_scores.keys(), key=lambda k: domain_scores[k])
            max_score = domain_scores[primary_domain]
            total_words = len(normalized_query.split())
            confidence = min(max_score / max(total_words, 1) * 2, 1.0)
        
        return {
            "primary_domain": primary_domain,
            "confidence": round(confidence, 3),
            "domain_scores": domain_scores,
            "technical_indicators": self._extract_technical_indicators(normalized_query)
        }
    
    def _assess_query_quality(self, query: str, normalized_query: str) -> Dict[str, Any]:
        """Assess query quality and provide improvement suggestions."""
        import re
        
        quality_factors = {
            "length_appropriate": 3 <= len(query.split()) <= 15,
            "has_content_words": any(len(word) > 3 for word in normalized_query.split()),
            "not_too_vague": len([w for w in normalized_query.split() if w not in ["thing", "stuff", "something", "anything"]]) > 0,
            "has_specific_terms": bool(re.search(r'\b[A-Z][a-z]+|\b\w{6,}\b', query)),
            "grammatically_coherent": not re.search(r'\b(\w+)\s+\1\b', normalized_query)  # No repeated words
        }
        
        quality_score = sum(quality_factors.values()) / len(quality_factors)
        
        # Quality level
        if quality_score >= 0.8:
            quality_level = "excellent"
        elif quality_score >= 0.6:
            quality_level = "good"
        elif quality_score >= 0.4:
            quality_level = "fair"
        else:
            quality_level = "needs_improvement"
        
        # Generate improvement suggestions
        suggestions = []
        if not quality_factors["length_appropriate"]:
            if len(query.split()) < 3:
                suggestions.append("Add more descriptive terms to your query")
            else:
                suggestions.append("Try simplifying your query to focus on key concepts")
        
        if not quality_factors["has_specific_terms"]:
            suggestions.append("Include more specific terminology related to your topic")
        
        if not quality_factors["not_too_vague"]:
            suggestions.append("Replace vague terms with more precise language")
        
        return {
            "quality_score": round(quality_score, 3),
            "quality_level": quality_level,
            "factors": quality_factors,
            "improvement_suggestions": suggestions
        }
    
    def _generate_intelligent_suggestions(self, query: str, normalized_query: str, intent_analysis: Dict, domain_analysis: Dict) -> Dict[str, Any]:
        """Generate intelligent search suggestions based on analysis."""
        suggestions = {
            "alternatives": [],
            "expansions": [],
            "refinements": [],
            "related_queries": []
        }
        
        # Intent-based alternatives
        intent = intent_analysis["primary_intent"]
        if intent == "example_seeking":
            suggestions["alternatives"] = [
                f"{query} tutorial",
                f"{query} sample code",
                f"examples of {query}"
            ]
        elif intent == "informational":
            suggestions["alternatives"] = [
                f"how to {query}",
                f"{query} guide",
                f"{query} documentation"
            ]
        elif intent == "troubleshooting":
            suggestions["alternatives"] = [
                f"fix {query}",
                f"solve {query}",
                f"{query} solution"
            ]
        
        # Domain-based expansions
        domain = domain_analysis["primary_domain"]
        if domain != "general":
            domain_terms = self._get_domain_expansion_terms(domain)
            suggestions["expansions"] = domain_terms[:5]  # Limit to 5
        
        # Query refinements based on complexity
        words = normalized_query.split()
        if len(words) > 8:  # Complex query
            # Suggest breaking it down
            key_terms = [w for w in words if len(w) > 4][:3]
            suggestions["refinements"] = [f"{' '.join(key_terms[:2])}", f"{' '.join(key_terms[1:])}"] if len(key_terms) >= 2 else []
        elif len(words) <= 2:  # Simple query
            # Suggest expansion
            suggestions["refinements"] = [f"{query} tutorial", f"{query} example", f"{query} guide"]
        
        # Generate related queries using word associations
        suggestions["related_queries"] = self._generate_related_queries(normalized_query, domain)
        
        # Filter out duplicates and empty suggestions
        for key in suggestions:
            suggestions[key] = list(dict.fromkeys([s for s in suggestions[key] if s and s != query]))
        
        return suggestions
    
    def _recommend_search_strategies(self, query: str, intent_analysis: Dict, complexity_analysis: Dict, domain_analysis: Dict) -> List[Dict[str, Any]]:
        """Recommend optimal search strategies based on analysis."""
        strategies = []
        
        complexity_level = complexity_analysis["complexity_level"]
        intent = intent_analysis["primary_intent"]
        domain = domain_analysis["primary_domain"]
        
        # Complexity-based strategies
        if complexity_level == "complex":
            strategies.append({
                "strategy": "query_decomposition",
                "description": "Break your complex query into smaller, focused searches",
                "rationale": "Complex queries may miss relevant results due to over-specification",
                "priority": "high"
            })
        elif complexity_level == "simple":
            strategies.append({
                "strategy": "query_expansion",
                "description": "Add more specific terms to narrow your search",
                "rationale": "Simple queries may return too broad results",
                "priority": "medium"
            })
        
        # Intent-based strategies
        if intent == "example_seeking":
            strategies.append({
                "strategy": "example_focused",
                "description": "Search for 'tutorial', 'example', or 'demo' content",
                "rationale": "Example queries benefit from instructional content",
                "priority": "high"
            })
        elif intent == "troubleshooting":
            strategies.append({
                "strategy": "problem_solution",
                "description": "Include error messages or symptoms in your search",
                "rationale": "Troubleshooting queries need specific problem context",
                "priority": "high"
            })
        
        # Domain-based strategies
        if domain != "general":
            strategies.append({
                "strategy": "domain_specific",
                "description": f"Focus on {domain.replace('_', ' ')} specific terminology",
                "rationale": f"Specialized {domain.replace('_', ' ')} queries benefit from domain context",
                "priority": "medium"
            })
        
        return strategies[:3]  # Limit to top 3 strategies
    
    def _extract_intent_indicators(self, normalized_query: str, original_query: str) -> List[str]:
        """Extract specific words/phrases that indicate intent."""
        indicators = []
        
        # Question words
        question_words = ["how", "what", "why", "when", "where", "which", "who"]
        indicators.extend([word for word in question_words if word in normalized_query])
        
        # Action words
        action_words = ["find", "show", "get", "create", "build", "fix", "solve"]
        indicators.extend([word for word in action_words if word in normalized_query])
        
        # Context indicators
        if "example" in normalized_query or "sample" in normalized_query:
            indicators.append("example_seeking")
        
        if "vs" in normalized_query or "versus" in normalized_query or "compare" in normalized_query:
            indicators.append("comparison")
        
        return list(set(indicators))
    
    def _extract_technical_indicators(self, normalized_query: str) -> List[str]:
        """Extract technical terms and indicators."""
        import re
        indicators = []
        
        # Programming languages
        languages = ["python", "java", "javascript", "c++", "c#", "go", "rust", "ruby", "php"]
        indicators.extend([lang for lang in languages if lang in normalized_query])
        
        # Technical acronyms (2-5 uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,5}\b', normalized_query.upper())
        indicators.extend(acronyms)
        
        # File extensions
        extensions = re.findall(r'\.[a-z]{2,4}\b', normalized_query)
        indicators.extend(extensions)
        
        return list(set(indicators))
    
    def _get_domain_expansion_terms(self, domain: str) -> List[str]:
        """Get expansion terms for specific domains."""
        expansions = {
            "programming": ["implementation", "syntax", "best practices", "tutorial", "documentation"],
            "data_science": ["analysis", "visualization", "modeling", "preprocessing", "evaluation"],
            "web_development": ["responsive", "optimization", "deployment", "testing", "security"],
            "database": ["optimization", "indexing", "normalization", "backup", "performance"],
            "system_admin": ["configuration", "monitoring", "automation", "troubleshooting", "security"],
            "mobile": ["development", "testing", "deployment", "performance", "ui/ux"],
            "cloud": ["architecture", "scalability", "deployment", "monitoring", "cost optimization"],
            "ai_ml": ["training", "evaluation", "deployment", "optimization", "preprocessing"]
        }
        return expansions.get(domain, [])
    
    def _generate_related_queries(self, normalized_query: str, domain: str) -> List[str]:
        """Generate related queries using semantic associations."""
        related = []
        words = normalized_query.split()
        
        if len(words) >= 2:
            # Permutations of key terms
            key_words = [w for w in words if len(w) > 3][:3]
            if len(key_words) >= 2:
                related.append(f"{key_words[-1]} {key_words[0]}")
                related.append(f"{' '.join(key_words)} best practices")
                related.append(f"{' '.join(key_words)} common issues")
        
        return related[:3]  # Limit to 3 related queries
    
    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """Get intelligent query suggestions based on learned patterns.
        
        Args:
            partial_query: Partial query string
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of query suggestions with metadata
        """
        if not self.search_intelligence:
            return []
        
        try:
            return self.search_intelligence.get_query_suggestions(partial_query, max_suggestions)
        except Exception as e:
            logger.debug(f"Failed to get query suggestions: {e}")
            return []
    
    def get_search_recommendations(self, query: str, query_analysis: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get adaptive search recommendations based on learned patterns.
        
        Args:
            query: Search query
            query_analysis: Optional pre-computed query analysis
            
        Returns:
            List of adaptive recommendations
        """
        if not self.search_intelligence:
            return []
        
        try:
            # Use provided analysis or compute it
            if query_analysis is None:
                query_analysis = self._analyze_query(query)
            
            return self.search_intelligence.get_adaptive_search_recommendations(query, query_analysis)
        except Exception as e:
            logger.debug(f"Failed to get search recommendations: {e}")
            return []
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get comprehensive search analytics and patterns.
        
        Returns:
            Dictionary with search analytics and insights
        """
        if not self.search_intelligence:
            return {"error": "Search intelligence not available"}
        
        try:
            return self.search_intelligence.analyze_search_patterns()
        except Exception as e:
            logger.debug(f"Failed to get search analytics: {e}")
            return {"error": f"Analytics unavailable: {str(e)}"}
    
    def get_search_quality_metrics(self) -> Dict[str, Any]:
        """Get search quality metrics and performance data.
        
        Returns:
            Dictionary with quality metrics
        """
        if not self.search_intelligence:
            return {"error": "Search intelligence not available"}
        
        try:
            return self.search_intelligence.get_search_quality_metrics()
        except Exception as e:
            logger.debug(f"Failed to get quality metrics: {e}")
            return {"error": f"Quality metrics unavailable: {str(e)}"}
    
    def save_search_intelligence(self) -> None:
        """Explicitly save search intelligence data."""
        if self.search_intelligence:
            try:
                self.search_intelligence.save()
                logger.info("Saved search intelligence data")
            except Exception as e:
                logger.error(f"Failed to save search intelligence: {e}")
    
    def clear_search_intelligence(self) -> None:
        """Clear all search intelligence data."""
        if self.search_intelligence:
            try:
                self.search_intelligence.clear_data()
                logger.info("Cleared search intelligence data")
            except Exception as e:
                logger.error(f"Failed to clear search intelligence: {e}")
    
    def _create_vector_search_results(self, distances, indices) -> List[SearchResult]:
        """Create SearchResult objects from vector search results."""
        # Removed debug print
        search_results = []
        for distance, faiss_id in zip(distances, indices):
            chunk = self.metadata_store.get_chunk(faiss_id)
            if chunk:
                # Create source info from chunk metadata
                source_info = self._create_source_info_from_chunk(chunk)
                # Removed debug print
                logger.debug(f"Created source_info for {chunk.chunk_id}: {source_info}")
                
                search_result = SearchResult(
                    chunk_id=chunk.chunk_id,
                    source_path=chunk.source_path,
                    source_type=chunk.source_type.value,
                    text=chunk.chunk_text,
                    relevance_score=float(distance),
                    position=create_structured_position(
                        start_char=chunk.start_char,
                        end_char=chunk.end_char
                    ),
                    metadata=create_structured_metadata(
                        created_at=chunk.created_at.isoformat(),
                        word_count=len(chunk.chunk_text.split()),
                        char_count=len(chunk.chunk_text),
                        file_extension=getattr(chunk, 'file_extension', None)
                    ),
                    scores=create_structured_scores(
                        vector_score=float(distance)
                    ),
                    provenance=create_search_provenance(
                        search_features=["vector"],
                        search_stage="vector_only"
                    ),
                    source_info=source_info
                )
                search_results.append(search_result)
        return search_results
    
    def _get_search_config(self) -> Dict[str, Any]:
        """Get current search configuration for response metadata."""
        return {
            "hybrid_search": self.config.use_hybrid_search,
            "reranking": self.config.use_reranking,
            "relationships": self.config.enable_relationships,
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
            "keyword_weight": getattr(self.config, 'keyword_weight', 0.3),
            "reranking_model": getattr(self.config, 'reranking_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        }
    
    def _create_source_info_from_chunk(self, chunk) -> Dict[str, Any]:
        """Create source info from chunk metadata.
        
        Args:
            chunk: ChunkMetadata object
            
        Returns:
            Dictionary with enhanced source metadata
        """
        from .pdf_loader import PDFLoader
        from pathlib import Path
        
        # Get source_path safely, handling both real objects and mocks
        source_path = getattr(chunk, 'source_path', '/unknown')
        source_type_val = getattr(chunk, 'source_type', None)
        
        # Handle source_type.value safely (could be enum or string)
        if hasattr(source_type_val, 'value'):
            source_type_str = source_type_val.value
        else:
            source_type_str = str(source_type_val) if source_type_val else 'unknown'
        
        source_info = {
            "file_path": str(source_path) if source_path else '/unknown',
            "source_type": source_type_str
        }
        
        # Only proceed with file operations if we have a real string path
        if isinstance(source_path, str) and source_path != '/unknown':
            # Basic file information
            try:
                file_path = Path(source_path)
                if file_path.exists() and file_path.is_file():
                    stat = file_path.stat()
                    source_info.update({
                        "filename": file_path.name,
                        "file_extension": file_path.suffix.lower(),
                        "file_size_bytes": stat.st_size,
                        "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "created_at": stat.st_ctime,
                        "modified_at": stat.st_mtime
                    })
                else:
                    # File doesn't exist, but we can still extract basic info
                    source_info["filename"] = file_path.name
                    source_info["file_extension"] = file_path.suffix.lower()
            except (OSError, ValueError, TypeError) as e:
                # Handle URLs or invalid paths gracefully
                logger.debug(f"Could not extract file metadata: {e}")
                try:
                    source_info["filename"] = Path(source_path).name
                    if source_path.startswith(("http://", "https://")):
                        source_info["source_type"] = "webpage"
                except Exception:
                    source_info["filename"] = "unknown"
            
            # PDF-specific metadata extraction
            if (source_type_str == "document" and 
                source_path.lower().endswith('.pdf')):
                try:
                    pdf_loader = PDFLoader()
                    pdf_metadata = pdf_loader.get_pdf_info(source_path)
                    source_info.update(pdf_metadata)
                except Exception as e:
                    logger.debug(f"Could not extract PDF metadata: {e}")
        
        # Page and section context from chunk text (safe to do even with mocks)
        chunk_text = getattr(chunk, 'chunk_text', '')
        if chunk_text and isinstance(chunk_text, str):
            try:
                pdf_loader = PDFLoader()
                page_info = pdf_loader.extract_page_context(chunk_text)
                if page_info:
                    source_info.update(page_info)
            except Exception as e:
                logger.debug(f"Could not extract page context: {e}")
        
        # Add chunk-specific metadata
        if hasattr(chunk, 'parent_section'):
            parent_section = getattr(chunk, 'parent_section', None)
            if parent_section:
                source_info["section_title"] = str(parent_section)
        
        # Add additional metadata from chunk
        if hasattr(chunk, 'metadata'):
            metadata = getattr(chunk, 'metadata', None)
            if metadata and hasattr(metadata, 'items'):
                try:
                    for key, value in metadata.items():
                        if key not in source_info and value is not None:
                            source_info[key] = value
                except Exception as e:
                    logger.debug(f"Could not extract chunk metadata: {e}")
        
        return source_info
    
    def _ensure_hybrid_search_index(self) -> None:
        """Ensure hybrid search index is built from current chunks."""
        if not self.hybrid_search:
            return
            
        # Check if index needs to be built/updated
        current_chunk_count = self.metadata_store.get_stats()["total_chunks"]
        hybrid_stats = self.hybrid_search.get_stats()
        
        if hybrid_stats["indexed_documents"] != current_chunk_count:
            logger.info("Building hybrid search index...")
            
            # Get all chunks
            all_chunks = self.metadata_store.get_all_chunks()
            
            if all_chunks:
                chunk_ids = [chunk.chunk_id for chunk in all_chunks]
                texts = [chunk.chunk_text for chunk in all_chunks]
                
                # Build keyword search index
                self.hybrid_search.add_documents(chunk_ids, texts)
                logger.info(f"Built hybrid search index with {len(texts)} documents")

    def search(self, query: str, top_k: int = 5, display_format: str = "readable") -> SearchResponse:
        """Perform semantic search with optional hybrid search and reranking.

        Args:
            query: Search query
            top_k: Number of results to return  
            display_format: Output format ('readable', 'structured', 'summary')

        Returns:
            SearchResponse with standardized results and metadata
        """
        # Removed debug print
        # Track performance with detailed timing
        start_time = time.time()
        search_mode = "vector"  # Will be updated based on features used
        
        # Analyze query characteristics
        query_analysis = self._analyze_query(query)
        
        # Initialize timing trackers
        embedding_start = None
        embedding_time = None
        vector_start = None
        vector_time = None
        keyword_start = None
        keyword_time = None
        
        try:
            # Check if we have any indexed content
            if self.vector_store is None or self.vector_store.is_empty():
                return SearchResponse.create_error(
                    query=query,
                    error="No indexed content available. Please index some documents, code, or webpages first.",
                    error_code=SearchErrorCode.NO_CONTENT.value,
                    search_config=self._get_search_config(),
                    recovery_suggestions=[
                        "Use index_document(), index_codebase(), or index_webpage() to add content",
                        "Check if auto_load is enabled and index files exist",
                        "Verify the vector store was initialized properly"
                    ]
                )

            # Ensure embedder is loaded for query embedding
            self._ensure_embedder_loaded()
            
            # Embed query with timing
            embedding_start = time.time()
            query_vector = self.embedder.embed_single(query)
            embedding_time = time.time() - embedding_start

            # Search vector store (get more candidates for reranking) with timing
            search_top_k = top_k * 3 if self.config.use_reranking else top_k
            vector_start = time.time()
            distances, indices = self.vector_store.search(query_vector, search_top_k)
            vector_time = time.time() - vector_start

            # Use hybrid search if enabled with graceful degradation
            if self.hybrid_search and self.config.use_hybrid_search:
                # Removed debug print
                try:
                    search_mode = "hybrid"
                    # Build keyword index if not already done
                    self._ensure_hybrid_search_index()
                    
                    # Prepare vector scores for hybrid search
                    vector_scores = [(int(faiss_id), float(distance)) for distance, faiss_id in zip(distances, indices)]
                    
                    # Perform hybrid search with timing
                    keyword_start = time.time()
                    hybrid_results = self.hybrid_search.search(
                        query=query,
                        vector_scores=vector_scores,
                        metadata_store=self.metadata_store,
                        top_k=top_k
                    )
                    keyword_time = time.time() - keyword_start
                    
                    if not hybrid_results:
                        # Fall back to vector search if hybrid search returns no results
                        logger.warning("Hybrid search returned no results, falling back to vector search")
                        search_mode = "vector_fallback"
                        hybrid_search_failed = True
                    else:
                        hybrid_search_failed = False
                        
                except Exception as e:
                    # Graceful degradation: fall back to vector search if hybrid search fails
                    logger.error(f"Hybrid search failed: {e}, falling back to vector search")
                    search_mode = "vector_fallback" 
                    keyword_time = None
                    hybrid_search_failed = True
                    
                if not hybrid_search_failed:
                    # Convert hybrid results directly to SearchResult format with structured helpers
                    search_results = []
                    for result in hybrid_results:
                        # Get chunk for source_info creation
                        chunk = self.metadata_store.get_chunk_by_chunk_id(result.chunk_id) if hasattr(result, 'chunk_id') else None
                        source_info = self._create_source_info_from_chunk(chunk) if chunk else None
                        
                        search_result = SearchResult(
                            chunk_id=result.chunk_id,
                            source_path=result.source_path,
                            source_type=result.source_type,
                            text=result.text,
                            relevance_score=result.combined_score,
                            scores=create_structured_scores(
                                vector_score=result.vector_score,
                                keyword_score=result.keyword_score,
                                combined_score=result.combined_score
                            ),
                            metadata=create_structured_metadata(**result.metadata) if result.metadata else None,
                            provenance=create_search_provenance(
                                search_features=["vector", "keyword", "hybrid"],
                                search_stage="hybrid_combined",
                                ranking_factors={
                                    "vector_weight": self.config.keyword_weight if hasattr(self.config, 'keyword_weight') else 0.7,
                                    "keyword_weight": 1.0 - (self.config.keyword_weight if hasattr(self.config, 'keyword_weight') else 0.7)
                                }
                            ),
                            source_info=source_info
                        )
                        search_results.append(search_result)
                else:
                    # Fall back to vector-only search
                    # Removed debug print
                    search_results = self._create_vector_search_results(distances, indices)
            else:
                # Standard vector search using helper method
                # Removed debug print
                search_results = self._create_vector_search_results(distances, indices)

            # Apply reranking if enabled with graceful degradation
            rerank_start_time = None
            reranking_failed = False
            if self.reranker and self.config.use_reranking and search_results:
                # Removed debug print
                try:
                    search_mode = "reranked" if search_mode == "vector" else "hybrid_reranked"
                    rerank_start_time = time.time()
                
                    # Convert SearchResults to legacy format for reranker
                    legacy_results = []
                    for sr in search_results:
                        legacy_results.append({
                            "score": sr.relevance_score,
                            "source_path": sr.source_path,
                            "source_type": sr.source_type,
                            "chunk_text": sr.text,
                            "chunk_id": sr.chunk_id,
                            "metadata": sr.metadata
                        })
                    
                    reranked = self.reranker.rerank(
                        query=query,
                        search_results=legacy_results,
                        top_k=top_k
                    )
                    
                    # Convert reranked results back to SearchResult format with structured helpers
                    search_results = []
                    for result in reranked:
                        # Get chunk for source_info creation
                        chunk = self.metadata_store.get_chunk_by_chunk_id(result.chunk_id) if hasattr(result, 'chunk_id') else None
                        # Removed debug print
                        source_info = self._create_source_info_from_chunk(chunk) if chunk else None
                        # Removed debug print
                        
                        search_result = SearchResult(
                            chunk_id=result.chunk_id,
                            source_path=result.source_path,
                            source_type=result.source_type,
                            text=result.text,
                            relevance_score=result.final_score,
                            scores=create_structured_scores(
                                original_score=result.original_score,
                                rerank_score=result.rerank_score,
                                combined_score=result.final_score
                            ),
                            metadata=create_structured_metadata(**result.metadata) if result.metadata else None,
                            provenance=create_search_provenance(
                                search_features=["vector", "reranking"] if search_mode == "reranked" else ["vector", "keyword", "hybrid", "reranking"],
                                search_stage="reranked_final",
                                ranking_factors={
                                    "reranking_model": self.config.reranking_model,
                                    "confidence_boost": round(result.rerank_score - result.original_score, 3)
                                },
                                confidence=min(result.rerank_score, 1.0)
                            ),
                            source_info=source_info
                        )
                        # Removed debug print
                        search_results.append(search_result)
                        
                except Exception as e:
                    # Graceful degradation: continue without reranking
                    logger.error(f"Reranking failed: {e}, continuing with original results")
                    reranking_failed = True
                    search_mode = search_mode.replace("_reranked", "_fallback")
                    rerank_time = None

            # Phase 3.1: Enhance results with ranking information and explanations
            enhanced_results = enhance_search_results_with_ranking(
                results=search_results,
                query=query,
                include_explanations=True,
                include_confidence=True
            )
            
            final_results = enhanced_results[:top_k]  # Ensure we don't exceed requested count
            
            # Create enhanced performance info
            rerank_time = (time.time() - rerank_start_time) if rerank_start_time else None
            performance = create_search_performance_info(
                start_time=start_time,
                search_mode=search_mode, 
                total_candidates=len(search_results),
                rerank_time=rerank_time,
                vector_time=vector_time,
                keyword_time=keyword_time,
                embedding_time=embedding_time
            )
            
            # Check if we need to return a degraded response
            failed_components = []
            if 'hybrid_search_failed' in locals() and hybrid_search_failed:
                failed_components.append("hybrid_search")
            if reranking_failed:
                failed_components.append("reranking")
            
            if failed_components:
                response = SearchResponse.create_degraded(
                    query=query,
                    results=final_results,
                    search_config=self._get_search_config(),
                    performance=performance,
                    degradation_reason=f"Some components failed: {', '.join(failed_components)}",
                    failed_components=failed_components
                )
                # Add query analysis to degraded response
                response.query_analysis = query_analysis
            else:
                response = SearchResponse.create_success(
                    query=query,
                    results=final_results,
                    search_config=self._get_search_config(),
                    performance=performance,
                    query_analysis=query_analysis
                )
            
            # Set display format and generate formatted output
            response.display_format = display_format
            if display_format != "structured":
                response.formatted_output = response.format_for_display(display_format)
            
            # Log performance metrics
            self.performance_logger.log_search_performance(response)
            self.performance_logger.log_slow_search(response, threshold_ms=1000)
            
            # Record search interaction for intelligence learning (Phase 3)
            if self.search_intelligence:
                try:
                    self.search_intelligence.record_search(query, query_analysis, response)
                except Exception as e:
                    logger.debug(f"Failed to record search for intelligence: {e}")
            
            return response

        except Exception as e:
            logger.error(f"Search failed: {e}")
            
            # Determine appropriate error code based on exception type
            if "timeout" in str(e).lower():
                error_code = SearchErrorCode.TIMEOUT.value
            elif "memory" in str(e).lower() or "out of memory" in str(e).lower():
                error_code = SearchErrorCode.MEMORY_ERROR.value
            elif "embedder" in str(e).lower() or "embedding" in str(e).lower():
                error_code = SearchErrorCode.EMBEDDER_UNAVAILABLE.value
            elif "vector" in str(e).lower():
                error_code = SearchErrorCode.VECTOR_STORE_ERROR.value
            else:
                error_code = SearchErrorCode.SEARCH_ERROR.value
            
            error_response = SearchResponse.create_error(
                query=query,
                error=f"Search operation failed: {str(e)}",
                error_code=error_code,
                search_config=self._get_search_config(),
                recovery_suggestions=[
                    "Check if the index is properly loaded",
                    "Verify the embedder is available and functional", 
                    "Try a simpler query if the current one is complex",
                    "Check system resources (memory, disk space)"
                ],
                query_analysis=query_analysis if 'query_analysis' in locals() else None
            )
            
            # Log error performance
            self.performance_logger.log_search_performance(error_response)
            
            return error_response

    def search_with_context(
        self, query: str, top_k: int = 5, include_related: bool = False, display_format: str = "readable"
    ) -> SearchResponse:
        """Perform enhanced search with relationship context.

        Args:
            query: Search query
            top_k: Number of results to return
            include_related: Whether to include related chunks
            display_format: Output format ('readable', 'structured', 'summary')

        Returns:
            SearchResponse with enhanced results and relationship context
        """
        try:
            # Start with basic search (get structured format first)
            base_response = self.search(query, top_k, "structured")

            # If relationships are not enabled or not requested, reformat and return basic search
            if not include_related or not self.config.enable_relationships:
                base_response.display_format = display_format
                if display_format != "structured":
                    base_response.formatted_output = base_response.format_for_display(display_format)
                return base_response
                
            # Check if the base search was successful
            if not base_response.success:
                return base_response  # Return error response as-is

            # Enhance results with relationship context
            enhanced_results = []
            for result in base_response.results:
                # Get chunk metadata for relationships
                chunk = None
                for stored_chunk in self.metadata_store.get_all_chunks():
                    if (
                        stored_chunk.source_path == result.source_path
                        and stored_chunk.chunk_text == result.text
                    ):
                        chunk = stored_chunk
                        break

                if chunk:
                    # Get relationship information
                    relationships = chunk.get_relationships()
                    
                    # Find related chunks with enhanced relevance scoring
                    related_chunks = []
                    entity_count = {}
                    
                    # Count entity occurrences for relationship strength
                    for entity in relationships.get("references", []):
                        entity_count[entity] = entity_count.get(entity, 0) + 1
                    
                    # Process top entities by frequency (relationship strength)
                    top_entities = sorted(entity_count.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    for entity, frequency in top_entities:
                        related_chunk_ids = self.relationship_store.get_related_chunks(entity)[:3]
                        for chunk_id in related_chunk_ids:
                            related_chunk = self.metadata_store.get_chunk_by_chunk_id(chunk_id)
                            if related_chunk and related_chunk.chunk_id != chunk.chunk_id:
                                # Calculate relationship strength based on shared entities
                                shared_entities = len(set(relationships.get("references", [])) & 
                                                    set(related_chunk.get_relationships().get("references", [])))
                                relationship_strength = min(1.0, (shared_entities + frequency) / 10.0)
                                
                                # Better text preview with smart truncation
                                preview_text = self._create_smart_preview(related_chunk.chunk_text, entity, 150)
                                
                                # Calculate relevance score based on query similarity
                                relevance_score = self._calculate_context_relevance(
                                    query, related_chunk.chunk_text, relationship_strength
                                )
                                
                                related_chunks.append({
                                    "chunk_id": related_chunk.chunk_id,
                                    "source_path": related_chunk.source_path,
                                    "text_preview": preview_text,
                                    "relationship_type": "reference",
                                    "relevance_score": round(relevance_score, 3),
                                    "relationship_strength": round(relationship_strength, 3),
                                    "shared_entity": entity,
                                    "entity_frequency": frequency
                                })
                    
                    # Sort by relevance score and limit results
                    related_chunks = sorted(related_chunks, key=lambda x: x["relevance_score"], reverse=True)[:3]
                    
                    # Add context to the result
                    context = {}
                    if relationships:
                        context["relationships"] = relationships
                    if related_chunks:
                        context["related_chunks"] = related_chunks[:3]  # Limit related chunks
                    
                    if context:
                        result.context = context

                enhanced_results.append(result)

            # Return enhanced SearchResponse with same metadata but enhanced results
            response = SearchResponse.create_success(
                query=base_response.query,
                results=enhanced_results,
                search_config=base_response.search_config,
                performance=base_response.performance
            )
            
            # Set display format and generate formatted output
            response.display_format = display_format
            if display_format != "structured":
                response.formatted_output = response.format_for_display(display_format)
            
            return response

        except Exception as e:
            logger.error(f"Context search failed: {e}")
            # Fallback to basic search with proper format
            return self.search(query, top_k, display_format)
    
    def _create_smart_preview(self, text: str, highlight_entity: str, max_length: int = 150) -> str:
        """Create intelligent text preview with entity highlighting.
        
        Args:
            text: Full text to create preview from
            highlight_entity: Entity to highlight in preview
            max_length: Maximum length of preview
            
        Returns:
            Smart preview with entity context
        """
        if not text:
            return ""
        
        # Find the entity in the text (case insensitive)
        entity_lower = highlight_entity.lower()
        text_lower = text.lower()
        
        # Find best position to center the preview around
        entity_pos = text_lower.find(entity_lower)
        if entity_pos == -1:
            # Entity not found, return beginning of text
            return text[:max_length] + ("..." if len(text) > max_length else "")
        
        # Calculate preview window around the entity
        entity_center = entity_pos + len(highlight_entity) // 2
        preview_start = max(0, entity_center - max_length // 2)
        preview_end = min(len(text), preview_start + max_length)
        
        # Adjust start if we're at the end
        if preview_end == len(text) and preview_end - preview_start < max_length:
            preview_start = max(0, preview_end - max_length)
        
        preview = text[preview_start:preview_end]
        
        # Add ellipses if truncated
        if preview_start > 0:
            preview = "..." + preview
        if preview_end < len(text):
            preview = preview + "..."
        
        return preview.strip()
    
    def _calculate_context_relevance(
        self, 
        query: str, 
        chunk_text: str, 
        relationship_strength: float
    ) -> float:
        """Calculate relevance score for context chunks.
        
        Args:
            query: Original search query
            chunk_text: Text of the related chunk
            relationship_strength: Strength of relationship connection
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            # Base relevance from relationship strength
            base_score = relationship_strength * 0.6
            
            # Add semantic similarity if embedder is available
            if self.embedder:
                try:
                    # Get embeddings for query and chunk
                    query_embedding = self.embedder.embed(query)
                    chunk_embedding = self.embedder.embed(chunk_text[:512])  # Truncate for efficiency
                    
                    # Calculate cosine similarity
                    import numpy as np
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    
                    # Normalize similarity to 0-1 range and weight it
                    semantic_score = max(0, (similarity + 1) / 2) * 0.4
                    
                    return min(1.0, base_score + semantic_score)
                    
                except Exception as e:
                    logger.debug(f"Could not calculate semantic similarity: {e}")
            
            # Fallback to keyword matching if no embedder
            query_words = set(query.lower().split())
            chunk_words = set(chunk_text.lower().split())
            
            if query_words and chunk_words:
                keyword_overlap = len(query_words.intersection(chunk_words)) / len(query_words)
                keyword_score = keyword_overlap * 0.4
                return min(1.0, base_score + keyword_score)
            
            return base_score
            
        except Exception as e:
            logger.debug(f"Error calculating context relevance: {e}")
            return relationship_strength  # Fallback to just relationship strength
    
    def get_content_recommendations(
        self, 
        query: str = None, 
        entity_name: str = None,
        top_k: int = 5,
        recommendation_types: List[str] = None
    ) -> SearchResponse:
        """Get intelligent content recommendations based on relationships and patterns.
        
        Args:
            query: Optional query to base recommendations on
            entity_name: Optional entity name to get recommendations for
            top_k: Number of recommendations to return
            recommendation_types: Types of recommendations ('similar', 'related', 'trending')
            
        Returns:
            SearchResponse with recommended content
        """
        try:
            if recommendation_types is None:
                recommendation_types = ['similar', 'related']
            
            recommendations = []
            
            # Query-based recommendations
            if query:
                # Get initial search results
                base_results = self.search(query, top_k=top_k * 2, display_format="structured")
                
                if base_results.success:
                    # Extract entities from top results for relationship discovery
                    result_entities = set()
                    for result in base_results.results[:3]:  # Analyze top 3 results
                        # Find chunk metadata
                        chunk = None
                        for stored_chunk in self.metadata_store.get_all_chunks():
                            if (stored_chunk.source_path == result.source_path and 
                                stored_chunk.chunk_text == result.text):
                                chunk = stored_chunk
                                break
                        
                        if chunk:
                            relationships = chunk.get_relationships()
                            result_entities.update(relationships.get("references", [])[:5])
                    
                    # Get recommendations for related entities
                    for entity in list(result_entities)[:5]:
                        entity_recommendations = self._get_entity_recommendations(entity, top_k=3)
                        recommendations.extend(entity_recommendations)
            
            # Entity-based recommendations
            elif entity_name:
                entity_recommendations = self._get_entity_recommendations(entity_name, top_k=top_k)
                recommendations.extend(entity_recommendations)
            
            # General recommendations based on popular/trending content
            else:
                trending_recommendations = self._get_trending_recommendations(top_k=top_k)
                recommendations.extend(trending_recommendations)
            
            # Remove duplicates and rank by relevance
            unique_recommendations = []
            seen_paths = set()
            
            for rec in recommendations:
                if rec.source_path not in seen_paths:
                    unique_recommendations.append(rec)
                    seen_paths.add(rec.source_path)
            
            # Sort by relevance score and limit results
            unique_recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
            final_recommendations = unique_recommendations[:top_k]
            
            # Create response
            response = SearchResponse.create_success(
                query=query or f"Recommendations for {entity_name}" if entity_name else "Content recommendations",
                results=final_recommendations,
                search_config=self._get_search_config(),
                performance={
                    "total_time_ms": 50.0,  # Lightweight recommendation process
                    "stage_times": {"recommendation_analysis": 50.0}
                }
            )
            
            response.metadata = {
                "recommendation_types": recommendation_types,
                "source_entity": entity_name,
                "recommendation_count": len(final_recommendations)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Content recommendations failed: {e}")
            return SearchResponse.create_error(
                query=query or "recommendations",
                error=f"Failed to generate recommendations: {str(e)}",
                search_config=self._get_search_config()
            )
    
    def _get_entity_recommendations(self, entity_name: str, top_k: int = 5) -> List[SearchResult]:
        """Get recommendations based on entity relationships.
        
        Args:
            entity_name: Name of entity to get recommendations for
            top_k: Number of recommendations
            
        Returns:
            List of recommended SearchResult objects
        """
        recommendations = []
        
        try:
            # Get chunks related to this entity
            related_chunk_ids = self.relationship_store.get_related_chunks(entity_name)[:10]
            
            for chunk_id in related_chunk_ids:
                chunk = self.metadata_store.get_chunk_by_chunk_id(chunk_id)
                if not chunk:
                    continue
                
                # Calculate recommendation strength
                relationships = chunk.get_relationships()
                entity_frequency = relationships.get("references", []).count(entity_name)
                
                # Base score on entity frequency and chunk relevance
                relevance_score = min(0.9, entity_frequency * 0.2 + 0.3)
                
                # Create search result
                search_result = SearchResult(
                    chunk_id=chunk.chunk_id,
                    source_path=chunk.source_path,
                    source_type=chunk.source_type.value if hasattr(chunk.source_type, 'value') else str(chunk.source_type),
                    text=chunk.chunk_text[:500] + "..." if len(chunk.chunk_text) > 500 else chunk.chunk_text,
                    relevance_score=relevance_score,
                    position=create_structured_position(
                        start_char=chunk.start_char,
                        end_char=chunk.end_char
                    ),
                    metadata=create_structured_metadata(
                        created_at=chunk.created_at.isoformat(),
                        word_count=len(chunk.chunk_text.split()),
                        char_count=len(chunk.chunk_text),
                        recommendation_reason=f"Contains entity '{entity_name}' ({entity_frequency} occurrences)"
                    ),
                    scores=create_structured_scores(
                        recommendation_score=relevance_score
                    ),
                    provenance=create_search_provenance(
                        search_features=["relationship_based"],
                        search_stage="recommendation"
                    ),
                    source_info=self._create_source_info_from_chunk(chunk)
                )
                
                recommendations.append(search_result)
            
            # Sort by relevance
            recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            logger.debug(f"Error getting entity recommendations for {entity_name}: {e}")
            return []
    
    def _get_trending_recommendations(self, top_k: int = 5) -> List[SearchResult]:
        """Get trending/popular content recommendations.
        
        Args:
            top_k: Number of recommendations
            
        Returns:
            List of trending SearchResult objects
        """
        recommendations = []
        
        try:
            # Get all chunks and analyze for "trending" patterns
            all_chunks = self.metadata_store.get_all_chunks()
            
            chunk_scores = []
            for chunk in all_chunks:
                # Simple trending score based on:
                # 1. Recent modification time
                # 2. Number of relationships/references
                # 3. Text complexity/richness
                
                relationships = chunk.get_relationships()
                reference_count = len(relationships.get("references", []))
                
                # Recency score (newer is better)
                import time
                chunk_age_days = (time.time() - chunk.created_at.timestamp()) / (24 * 3600)
                recency_score = max(0.1, 1.0 - (chunk_age_days / 30.0))  # Decay over 30 days
                
                # Complexity score
                word_count = len(chunk.chunk_text.split())
                complexity_score = min(1.0, word_count / 500.0)  # Normalize to 500 words
                
                # Combined trending score
                trending_score = (
                    recency_score * 0.4 + 
                    min(1.0, reference_count / 10.0) * 0.4 + 
                    complexity_score * 0.2
                )
                
                chunk_scores.append((chunk, trending_score))
            
            # Sort by trending score and select top chunks
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            for chunk, score in chunk_scores[:top_k]:
                search_result = SearchResult(
                    chunk_id=chunk.chunk_id,
                    source_path=chunk.source_path,
                    source_type=chunk.source_type.value if hasattr(chunk.source_type, 'value') else str(chunk.source_type),
                    text=chunk.chunk_text[:500] + "..." if len(chunk.chunk_text) > 500 else chunk.chunk_text,
                    relevance_score=score,
                    position=create_structured_position(
                        start_char=chunk.start_char,
                        end_char=chunk.end_char
                    ),
                    metadata=create_structured_metadata(
                        created_at=chunk.created_at.isoformat(),
                        word_count=len(chunk.chunk_text.split()),
                        char_count=len(chunk.chunk_text),
                        recommendation_reason="Trending content based on activity and relationships"
                    ),
                    scores=create_structured_scores(
                        trending_score=score
                    ),
                    provenance=create_search_provenance(
                        search_features=["trending_analysis"],
                        search_stage="recommendation"
                    ),
                    source_info=self._create_source_info_from_chunk(chunk)
                )
                
                recommendations.append(search_result)
            
            return recommendations
            
        except Exception as e:
            logger.debug(f"Error getting trending recommendations: {e}")
            return []

    def _extract_knowledge_catalog(self) -> Dict[str, Any]:
        """Extract knowledge catalog information from existing chunks.
        
        Returns:
            Dictionary with content overview and knowledge catalog
        """
        all_chunks = self.metadata_store.get_all_chunks()
        
        if not all_chunks:
            return {
                "content_overview": {},
                "knowledge_catalog": {
                    "indexed_sources": {
                        "codebases": [],
                        "webpages": [],
                        "documents": []
                    },
                    "primary_topics": [],
                    "searchable_concepts": []
                }
            }
        
        # Extract basic information from chunks
        source_paths = set()
        file_extensions = set()
        code_symbols = set()
        all_tags = set()
        all_references = set()
        
        # Simple domain detection from file paths and extensions
        programming_languages = set()
        web_domains = set()
        codebases = set()
        documents = set()
        
        # Common stop words to filter from topics
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'her', 'its', 'our', 'their', 'page', 'section', 'chapter', 'document', 'text', 'content',
            'file', 'line', 'item', 'part', 'type', 'kind', 'way', 'time', 'place', 'thing', 'person'
        }
        
        for chunk in all_chunks:
            source_paths.add(chunk.source_path)
            
            # Extract file extension
            if chunk.file_extension:
                file_extensions.add(chunk.file_extension)
                
                # Simple programming language detection
                lang_map = {
                    '.py': 'Python',
                    '.js': 'JavaScript', 
                    '.ts': 'TypeScript',
                    '.java': 'Java',
                    '.cpp': 'C++',
                    '.c': 'C',
                    '.go': 'Go',
                    '.rs': 'Rust',
                    '.php': 'PHP',
                    '.rb': 'Ruby',
                    '.swift': 'Swift',
                    '.kt': 'Kotlin',
                    '.scala': 'Scala',
                    '.sh': 'Shell',
                    '.sql': 'SQL'
                }
                if chunk.file_extension in lang_map:
                    programming_languages.add(lang_map[chunk.file_extension])
            
            # Enhanced codebase detection from source paths  
            if chunk.source_type.value == 'code':
                path_parts = chunk.source_path.replace('\\', '/').split('/')
                
                # Look for project root by finding directories that contain project files
                # or have src/lib structure underneath them
                project_path = None
                
                # Work backwards from the file to find the most likely project root
                for i in range(len(path_parts) - 2, 0, -1):  # Start from parent of file, work backwards
                    current_path = '/'.join(path_parts[:i+1])
                    part = path_parts[i]
                    
                    # Skip system directories and common subdirectories
                    if part.lower() in {'usr', 'bin', 'lib', 'opt', 'home', 'windows', 'program files',
                                        'src', 'lib', 'app', 'components', 'modules', 'packages',
                                        'build', 'dist', '__pycache__', 'node_modules', 'target',
                                        'test', 'tests', 'docs', 'examples', 'tools'}:
                        continue
                    
                    # Look for directories that are likely project roots
                    if (part and len(part) > 2 and any(char.isalpha() for char in part) and
                        not part.startswith('.')):
                        
                        # Check if this directory has project-like structure below it
                        has_src_structure = False
                        project_indicators = {'src', 'lib', 'app', 'components', 'modules', 'packages'}
                        
                        # Look at immediate children for common project structure
                        for j in range(i+1, min(i+3, len(path_parts))):
                            if path_parts[j].lower() in project_indicators:
                                has_src_structure = True
                                break
                        
                        # Or check if this looks like a meaningful project name
                        looks_like_project = (
                            len(part) > 4 or  # Longer names
                            any(char.isupper() for char in part) or  # CamelCase  
                            '_' in part or '-' in part or  # snake_case or kebab-case
                            part.startswith('py') or  # Python projects
                            any(x in part.lower() for x in ['project', 'app', 'service', 'api', 'tool'])
                        )
                        
                        if has_src_structure or looks_like_project:
                            project_path = current_path
                            break  # Take the first (deepest) match
                
                # If we found a project path, add it
                if project_path and len(project_path) > 10:  # Avoid very short paths
                    codebases.add(project_path)
            
            # Extract document filenames
            if chunk.source_type.value == 'document':
                # Get just the filename from the path
                filename = chunk.source_path.replace('\\', '/').split('/')[-1]
                if filename and '.' in filename:
                    documents.add(filename)
                
            # Extract web domains
            if chunk.source_type.value == 'webpage' and chunk.source_path.startswith(('http://', 'https://')):
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(chunk.source_path).netloc
                    if domain:
                        web_domains.add(domain)
                except:
                    pass
            
            # Collect symbols, tags, references
            code_symbols.update(chunk.code_symbols)
            all_tags.update(chunk.tags)
            all_references.update(chunk.references)
        
        # Create content overview
        content_overview = {
            "total_sources": len(source_paths),
            "content_types": list(file_extensions) if file_extensions else [],
            "programming_languages": list(programming_languages) if programming_languages else [],
            "domains": {
                "web_domains": list(web_domains) if web_domains else [],
                "codebase_paths": list(codebases)[:10] if codebases else []  # Limit to top 10
            }
        }
        
        # Create knowledge catalog with improved filtering
        # Filter and rank code symbols by relevance
        symbol_scores = []
        for symbol in code_symbols:
            frequency = sum(1 for chunk in all_chunks if symbol in chunk.code_symbols)
            # Boost score for symbols that look like meaningful identifiers
            relevance_boost = 1.0
            if len(symbol) > 3 and not symbol.lower() in stop_words:
                relevance_boost = 1.5
            if any(char.isupper() for char in symbol):  # CamelCase or mixed case
                relevance_boost *= 1.2
            symbol_scores.append((symbol, frequency * relevance_boost))
        
        top_symbols = sorted(symbol_scores, key=lambda x: x[1], reverse=True)[:20]
        
        # Filter and rank references by semantic relevance
        reference_scores = []
        for ref in all_references:
            frequency = sum(1 for chunk in all_chunks if ref in chunk.references)
            # Filter out stop words and very short/generic terms
            if (len(ref) > 2 and 
                ref.lower() not in stop_words and 
                not ref.isdigit() and 
                len(ref.strip()) > 1):
                
                # Boost score for terms that look like domain concepts
                relevance_boost = 1.0
                if len(ref) > 5:  # Longer terms are often more specific
                    relevance_boost = 1.3
                if any(char.isupper() for char in ref):  # Proper nouns or acronyms
                    relevance_boost *= 1.2
                if '_' in ref or '-' in ref:  # Technical terms
                    relevance_boost *= 1.1
                    
                reference_scores.append((ref, frequency * relevance_boost))
        
        top_references = sorted(reference_scores, key=lambda x: x[1], reverse=True)[:15]
        
        # Filter tags for quality
        quality_tags = []
        for tag in all_tags:
            if (len(tag) > 3 and 
                tag.lower() not in stop_words and
                not tag.isdigit() and
                len(tag.strip()) > 1):
                quality_tags.append(tag)
        
        # Deduplicate codebases - keep only root paths, remove subdirectories
        unique_codebases = []
        sorted_codebases = sorted(codebases, key=len)  # Shorter paths first (likely parents)
        
        for codebase in sorted_codebases:
            # Check if this codebase is a subdirectory of any existing codebase
            is_subdirectory = False
            for existing in unique_codebases:
                if codebase.startswith(existing + '/'):
                    is_subdirectory = True
                    break
            
            if not is_subdirectory:
                unique_codebases.append(codebase)
        
        knowledge_catalog = {
            "indexed_sources": {
                "codebases": unique_codebases[:10] if unique_codebases else [],
                "webpages": list(web_domains) if web_domains else [],
                "documents": list(documents)[:15] if documents else []  # Increased from 10
            },
            "primary_topics": [ref[0] for ref in top_references[:12]],  # Filtered meaningful concepts
            "searchable_concepts": {
                "code_symbols": [symbol[0] for symbol in top_symbols[:15]],  # High-quality symbols
                "technologies": list(programming_languages) if programming_languages else [],
                "tags": quality_tags[:12] if quality_tags else []  # Filtered quality tags
            }
        }
        
        return {
            "content_overview": content_overview,
            "knowledge_catalog": knowledge_catalog
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get search performance statistics summary.
        
        Returns:
            Dictionary with performance metrics and statistics
        """
        return self.performance_logger.get_performance_summary()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.

        Returns:
            System status and statistics
        """
        try:
            # Basic stats
            metadata_stats = self.metadata_store.get_stats()
            relationship_stats = self.relationship_store.get_stats()
            vector_stats = self.vector_store.get_index_info() if self.vector_store else {"total_vectors": 0, "dimension": None}
            
            # Create unified index stats
            index_stats = {
                "total_chunks": metadata_stats.get("total_chunks", 0),
                "total_documents": len(set(chunk.source_path for chunk in self.metadata_store.get_all_chunks())),
                "total_vectors": vector_stats.get("total_vectors", 0),
                "source_types": metadata_stats.get("source_types", {})
            }

            # Memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            # Embedding info (handle lazy state)
            if not self._embedder_initialized or self.embedder is None:
                embedding_info = {
                    "provider": self.config.embedding_provider,
                    "model": self.config.embedding_model,
                    "dimension": None,
                    "is_available": False,
                }
            else:
                embedding_info = {
                    "provider": self.embedder.get_provider_name(),
                    "model": self.embedder.get_model_name(),
                    "dimension": self.embedder.get_dimension(),
                    "is_available": self.embedder.is_available(),
                }

            # Persistence info
            paths = self.config.get_index_paths()
            persistence_info = {
                "auto_persist": self.config.auto_persist,
                "index_dir": str(self.config.index_dir),
                "index_files_exist": all(path.exists() for path in paths.values()),
                "last_modified": {},
            }

            # Get file modification times
            for name, path in paths.items():
                if path.exists():
                    persistence_info["last_modified"][name] = path.stat().st_mtime
            
            # Enhanced search components info
            hybrid_search_info = {}
            if self.hybrid_search:
                hybrid_search_info = self.hybrid_search.get_stats()
            
            reranking_info = {}
            if self.reranker:
                reranking_info = self.reranker.get_stats()
            
            # System performance metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
            
            performance_info = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_mb": memory_mb,
                "memory_total_mb": memory_info.total / (1024 * 1024),
                "memory_available_mb": memory_info.available / (1024 * 1024),
                "memory_usage_percent": memory_info.percent,
                "disk_usage_percent": disk_usage.percent,
                "disk_free_gb": disk_usage.free / (1024 * 1024 * 1024)
            }
            
            # Extract knowledge catalog and enhanced metadata
            knowledge_info = self._extract_knowledge_catalog()

            return {
                "status": "healthy",
                "index_stats": index_stats,
                "metadata": metadata_stats,
                "relationships": relationship_stats,
                "vector_store": vector_stats,
                "embedding": embedding_info,
                "hybrid_search": hybrid_search_info,
                "reranking": reranking_info,
                "performance": performance_info,
                "persistence": persistence_info,
                "configuration": self.config.get_config_summary(),
                "content_overview": knowledge_info["content_overview"],
                "knowledge_catalog": knowledge_info["knowledge_catalog"],
            }

        except Exception as e:
            # Return error status with basic structure to prevent KeyErrors
            return {
                "status": "error",
                "error": str(e),
                "metadata": {},
                "relationships": {},
                "vector_store": {},
                "embedding": {
                    "provider": getattr(self.config, 'embedding_provider', 'unknown'),
                    "model": getattr(self.config, 'embedding_model', 'unknown'),
                    "dimension": None,
                    "is_available": False,
                },
                "hybrid_search": {},
                "reranking": {},
                "performance": {},
                "persistence": {},
                "configuration": {},
            }

    def clear_index(self, remove_files: bool = False) -> Dict[str, Any]:
        """Clear all indexed data.

        Args:
            remove_files: Whether to remove saved files

        Returns:
            Operation result
        """
        try:
            # Clear in-memory data
            self.metadata_store.clear()
            self.relationship_store.clear()
            if self.vector_store is not None:
                self.vector_store.clear()

            # Remove files if requested
            if remove_files:
                paths = self.config.get_index_paths()
                for path in paths.values():
                    if path.exists():
                        path.unlink()
                        logger.info(f"Removed {path}")

            logger.info("Cleared index data")
            return {"success": True, "files_removed": remove_files}

        except Exception as e:
            error_msg = f"Failed to clear index: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def save_index(self) -> Dict[str, Any]:
        """Manually save index to disk.

        Returns:
            Save operation result
        """
        try:
            self._auto_save()
            return {"success": True}
        except Exception as e:
            error_msg = f"Failed to save index: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.embedder is not None:
            self.embedder.cleanup()
