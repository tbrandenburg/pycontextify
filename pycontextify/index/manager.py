"""Central IndexManager for PyContextify.

This module implements the main coordination system that orchestrates all
components for indexing operations, persistence, and search functionality.
"""

import logging
from typing import Any, Dict, List

import psutil

from .chunker import ChunkerFactory
from .config import Config
from .embedders import EmbedderFactory
from .loaders import LoaderFactory
from .metadata import MetadataStore, SourceType
from .relationship_store import RelationshipStore
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

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search with optional hybrid search and reranking.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results
        """
        try:
            # Check if we have any indexed content
            if self.vector_store is None or self.vector_store.is_empty():
                return []

            # Ensure embedder is loaded for query embedding
            self._ensure_embedder_loaded()
            
            # Embed query
            query_vector = self.embedder.embed_single(query)

            # Search vector store (get more candidates for reranking)
            search_top_k = top_k * 3 if self.config.use_reranking else top_k
            distances, indices = self.vector_store.search(query_vector, search_top_k)

            # Use hybrid search if enabled
            if self.hybrid_search and self.config.use_hybrid_search:
                # Build keyword index if not already done
                self._ensure_hybrid_search_index()
                
                # Prepare vector scores for hybrid search
                vector_scores = [(int(faiss_id), float(distance)) for distance, faiss_id in zip(distances, indices)]
                
                # Perform hybrid search
                hybrid_results = self.hybrid_search.search(
                    query=query,
                    vector_scores=vector_scores,
                    metadata_store=self.metadata_store,
                    top_k=top_k
                )
                
                # Convert hybrid results to standard format
                results = []
                for result in hybrid_results:
                    results.append({
                        "score": result.combined_score,
                        "vector_score": result.vector_score,
                        "keyword_score": result.keyword_score,
                        "source_path": result.source_path,
                        "source_type": result.source_type,
                        "chunk_text": result.text,
                        "chunk_id": result.chunk_id,
                        "metadata": result.metadata
                    })
            else:
                # Standard vector search
                results = []
                for distance, faiss_id in zip(distances, indices):
                    chunk = self.metadata_store.get_chunk(faiss_id)
                    if chunk:
                        results.append({
                            "score": float(distance),
                            "source_path": chunk.source_path,
                            "source_type": chunk.source_type.value,
                            "chunk_text": chunk.chunk_text,
                            "chunk_id": chunk.chunk_id,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "created_at": chunk.created_at.isoformat(),
                        })

            # Apply reranking if enabled
            if self.reranker and self.config.use_reranking and results:
                reranked = self.reranker.rerank(
                    query=query,
                    search_results=results,
                    top_k=top_k
                )
                
                # Convert reranked results back to standard format
                results = []
                for result in reranked:
                    results.append({
                        "score": result.final_score,
                        "original_score": result.original_score,
                        "rerank_score": result.rerank_score,
                        "source_path": result.source_path,
                        "source_type": result.source_type,
                        "chunk_text": result.text,
                        "chunk_id": result.chunk_id,
                        "metadata": result.metadata
                    })

            return results[:top_k]  # Ensure we don't exceed requested count

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_with_context(
        self, query: str, top_k: int = 5, include_related: bool = False
    ) -> List[Dict[str, Any]]:
        """Perform enhanced search with relationship context.

        Args:
            query: Search query
            top_k: Number of results to return
            include_related: Whether to include related chunks

        Returns:
            List of enhanced search results with relationship context
        """
        try:
            # Start with basic search
            results = self.search(query, top_k)

            if not include_related or not self.config.enable_relationships:
                return results

            # Enhance with relationship context
            enhanced_results = []
            for result in results:
                # Get chunk metadata
                chunk = None
                for stored_chunk in self.metadata_store.get_all_chunks():
                    if (
                        stored_chunk.source_path == result["source_path"]
                        and stored_chunk.chunk_text == result["chunk_text"]
                    ):
                        chunk = stored_chunk
                        break

                if chunk:
                    # Add relationship information
                    relationships = chunk.get_relationships()
                    result["relationships"] = relationships

                    # Find related chunks
                    related_chunks = []
                    for entity in relationships.get("references", [])[
                        :3
                    ]:  # Limit to avoid noise
                        related_chunk_ids = self.relationship_store.get_related_chunks(
                            entity
                        )[:2]
                        for chunk_id in related_chunk_ids:
                            related_chunk = self.metadata_store.get_chunk_by_chunk_id(
                                chunk_id
                            )
                            if (
                                related_chunk
                                and related_chunk.chunk_id != chunk.chunk_id
                            ):
                                related_chunks.append(
                                    {
                                        "source_path": related_chunk.source_path,
                                        "chunk_text": (
                                            related_chunk.chunk_text[:200] + "..."
                                            if len(related_chunk.chunk_text) > 200
                                            else related_chunk.chunk_text
                                        ),
                                        "relationship_type": "reference",
                                    }
                                )

                    result["related_chunks"] = related_chunks[
                        :3
                    ]  # Limit related chunks

                enhanced_results.append(result)

            return enhanced_results

        except Exception as e:
            logger.error(f"Context search failed: {e}")
            return self.search(query, top_k)  # Fallback to basic search

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
                    "indexed_sources": [],
                    "primary_topics": [],
                    "code_repositories": [],
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
        project_names = set()
        
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
                    '.php': 'PHP'
                }
                if chunk.file_extension in lang_map:
                    programming_languages.add(lang_map[chunk.file_extension])
            
            # Extract project names from paths
            if chunk.source_type.value == 'code':
                path_parts = chunk.source_path.replace('\\', '/').split('/')
                for part in path_parts:
                    if part and not part.startswith('.') and len(part) > 2:
                        # Look for likely project names (directories with meaningful names)
                        if any(char.isalpha() for char in part) and part.lower() not in ['src', 'lib', 'bin', 'test', 'tests']:
                            project_names.add(part)
            
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
                "project_names": list(project_names)[:10] if project_names else []  # Limit to top 10
            }
        }
        
        # Create knowledge catalog
        # Get top items by frequency for better discovery
        top_symbols = sorted([(symbol, sum(1 for chunk in all_chunks if symbol in chunk.code_symbols)) 
                            for symbol in code_symbols], key=lambda x: x[1], reverse=True)[:20]
        top_references = sorted([(ref, sum(1 for chunk in all_chunks if ref in chunk.references)) 
                               for ref in all_references], key=lambda x: x[1], reverse=True)[:15]
        
        knowledge_catalog = {
            "indexed_sources": {
                "code_repositories": list(project_names)[:5] if project_names else [],
                "web_resources": list(web_domains) if web_domains else [],
                "document_types": [ext for ext in file_extensions if ext in ['.md', '.txt', '.pdf', '.doc', '.docx']]
            },
            "primary_topics": [ref[0] for ref in top_references[:10]],  # Most referenced concepts
            "searchable_concepts": {
                "code_symbols": [symbol[0] for symbol in top_symbols[:10]],  # Most common symbols
                "technologies": list(programming_languages) if programming_languages else [],
                "tags": list(all_tags)[:10] if all_tags else []
            }
        }
        
        return {
            "content_overview": content_overview,
            "knowledge_catalog": knowledge_catalog
        }

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
