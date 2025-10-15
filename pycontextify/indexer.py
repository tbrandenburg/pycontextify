"""Refactored IndexManager with IndexingPipeline - coordination and pipeline.

This module provides IndexingPipeline for processing filebases and IndexManager
for coordinating indexing and search operations. IndexManager delegates to
specialized services for embedder management and search.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from .bootstrap import BootstrapService
from .chunker import ChunkerFactory
from .config import Config
from .crawler import FileCrawler
from .embedder import EmbedderService
from .loader import FileLoaderFactory
from .postprocess import postprocess_file, postprocess_filebase
from .search import SearchService
from .search_hybrid import HybridSearchEngine
from .search_models import SearchPerformanceLogger, SearchResponse
from .storage_metadata import ChunkMetadata, MetadataStore
from .storage_vector import VectorStore
from .types import SourceType

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Executes the 8-step indexing pipeline for filebases.

    Pipeline steps:
    1. Crawl file tree
    2-5. Load, chunk, and postprocess files
    6. Postprocess entire filebase
    7. Generate embeddings
    8. Store vectors and metadata
    """

    def __init__(self, config, embedder_service, vector_store, metadata_store):
        """Initialize indexing pipeline.

        Args:
            config: Configuration object
            embedder_service: EmbedderService instance
            vector_store: VectorStore instance
            metadata_store: MetadataStore instance
        """
        self.config = config
        self.embedder_service = embedder_service
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    def index_filebase(
        self,
        base_path: str,
        tags: str,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute complete indexing pipeline.

        Args:
            base_path: Root directory or single file to index
            tags: Comma-separated tags (required, used for organization)
            include: List of fnmatch patterns to include
            exclude: List of fnmatch patterns to exclude
            exclude_dirs: List of directory patterns to exclude

        Returns:
            Dictionary with pipeline statistics

        Raises:
            ValueError: If tags are empty or None
            FileNotFoundError: If base_path does not exist
        """
        # Validate inputs
        if not tags or not isinstance(tags, str) or not tags.strip():
            raise ValueError("tags are required and must be a comma-separated string")

        parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        if not parsed_tags:
            raise ValueError("tags must include at least one non-empty tag")

        base_path_obj = Path(base_path).resolve()
        if not base_path_obj.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

        if base_path_obj.is_dir():
            crawl_target = base_path_obj
            loader_base_path = base_path_obj
        elif base_path_obj.is_file():
            crawl_target = base_path_obj
            loader_base_path = base_path_obj.parent
        else:
            raise ValueError(f"Base path must be a file or directory: {base_path}")

        tags = tags.strip()
        tag_summary = ", ".join(parsed_tags)
        start_time = time.time()
        started_at = datetime.now(timezone.utc)

        logger.info(f"Starting filebase indexing: {base_path} (tags: {tag_summary})")

        # Initialize stats
        stats = {
            "tags_input": tags,
            "tags": parsed_tags,
            "base_path": str(base_path_obj),
            "started_at": started_at.isoformat(),
            "files_crawled": 0,
            "files_loaded": 0,
            "chunks_created": 0,
            "vectors_embedded": 0,
            "errors": 0,
            "error_samples": [],
        }

        try:
            # Step 1: Crawl
            file_paths = self._crawl_files(
                crawl_target, include, exclude, exclude_dirs, stats
            )

            if not file_paths:
                return self._finalize_stats(stats, start_time)

            # Step 2-5: Load, Chunk, Postprocess
            all_chunks = self._process_files(
                file_paths, loader_base_path, tags, parsed_tags, stats
            )

            if not all_chunks:
                return self._finalize_stats(stats, start_time)

            # Step 6: Filebase-level postprocessing
            logger.info("Step 6/8: Postprocessing entire filebase")
            all_chunks = postprocess_filebase(all_chunks)

            # Step 7: Embed
            embeddings = self._embed_chunks(all_chunks, stats)
            if embeddings is None:
                return self._finalize_stats(stats, start_time)

            # Step 8: Store
            self._store_vectors(all_chunks, embeddings, stats)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            stats["errors"] += 1
            stats["error_samples"].append({"stage": "pipeline", "error": str(e)})

        return self._finalize_stats(stats, start_time)

    def _crawl_files(
        self,
        base_path: Path,
        include: Optional[List[str]],
        exclude: Optional[List[str]],
        exclude_dirs: Optional[List[str]],
        stats: Dict[str, Any],
    ) -> List[Path]:
        """Step 1: Crawl file tree."""
        logger.info("Step 1/8: Crawling file tree")
        try:
            crawler = FileCrawler(
                include=include,
                exclude=exclude,
                exclude_dirs=exclude_dirs,
            )
            file_paths = crawler.crawl(str(base_path))
            stats["files_crawled"] = len(file_paths)
            logger.info(f"Crawled {len(file_paths)} files")
            return file_paths
        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            stats["errors"] += 1
            stats["error_samples"].append({"stage": "crawl", "error": str(e)})
            return []

    def _process_files(
        self,
        file_paths: List[Path],
        base_dir: Path,
        tags_input: str,
        tags: List[str],
        stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Steps 2-5: Load, chunk, and postprocess files.

        Args:
            file_paths: Iterable of file paths to process
            base_dir: Directory used for relative path calculations
            tags_input: Raw tag string used during indexing
            tags: Parsed tags for indexing
            stats: Mutable statistics dictionary
        """
        logger.info("Step 2-5: Loading, chunking, and postprocessing files")

        loader = FileLoaderFactory()
        embedder = self.embedder_service.get_embedder()
        embedding_provider = embedder.get_provider_name()
        embedding_model = embedder.get_model_name()

        all_chunks = []

        for idx, file_path in enumerate(file_paths):
            file_path_obj = Path(file_path)
            try:
                # Step 2: Load
                logger.debug(f"Loading {idx + 1}/{len(file_paths)}: {file_path_obj}")
                normalized_docs = loader.load(
                    path=str(file_path_obj),
                    tags=tags_input,
                    base_path=str(base_dir),
                )

                for doc in normalized_docs:
                    metadata = doc.get("metadata", {})
                    metadata["tags"] = list(tags)

                if not normalized_docs:
                    continue

                # Add embedding info to metadata
                for doc in normalized_docs:
                    doc["metadata"]["embedding_provider"] = embedding_provider
                    doc["metadata"]["embedding_model"] = embedding_model

                # Step 3: Chunk
                file_chunks = ChunkerFactory.chunk_normalized_docs(
                    normalized_docs=normalized_docs,
                    config=self.config,
                )

                if not file_chunks:
                    continue

                for chunk in file_chunks:
                    metadata = chunk.get("metadata", {})
                    metadata.setdefault("tags", list(tags))

                # Step 4: Postprocess (file-level)
                file_chunks = postprocess_file(file_chunks)

                all_chunks.extend(file_chunks)
                stats["files_loaded"] += 1

            except Exception as e:
                logger.error(f"Error processing {file_path_obj}: {e}")
                stats["errors"] += 1
                if len(stats["error_samples"]) < 10:
                    stats["error_samples"].append(
                        {
                            "stage": "load/chunk",
                            "file": str(file_path_obj),
                            "error": str(e),
                        }
                    )

        stats["chunks_created"] = len(all_chunks)
        logger.info(
            f"Created {len(all_chunks)} chunks from {stats['files_loaded']} files"
        )
        return all_chunks

    def _embed_chunks(
        self, chunks: List[Dict[str, Any]], stats: Dict[str, Any]
    ) -> Optional[List]:
        """Step 7: Generate embeddings."""
        logger.info("Step 7/8: Generating embeddings")
        try:
            embedder = self.embedder_service.get_embedder()
            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_texts(texts)
            stats["vectors_embedded"] = len(embeddings)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            stats["errors"] += 1
            stats["error_samples"].append({"stage": "embed", "error": str(e)})
            return None

    def _store_vectors(
        self,
        chunks: List[Dict[str, Any]],
        embeddings,
        stats: Dict[str, Any],
    ) -> None:
        """Step 8: Store vectors and metadata."""
        logger.info("Step 8/8: Storing vectors and metadata")
        try:
            faiss_ids = self.vector_store.add_vectors(embeddings)

            for chunk_dict, faiss_id in zip(chunks, faiss_ids):
                chunk_metadata = self._create_chunk_metadata(chunk_dict)
                self.metadata_store.add_chunk(chunk_metadata)

            logger.info(f"Stored {len(faiss_ids)} vectors and metadata")

        except Exception as e:
            logger.error(f"Storage failed: {e}")
            stats["errors"] += 1
            stats["error_samples"].append({"stage": "store", "error": str(e)})

    def _create_chunk_metadata(self, chunk_dict: Dict[str, Any]) -> ChunkMetadata:
        """Convert normalized chunk to ChunkMetadata."""
        metadata = chunk_dict["metadata"]
        file_ext = metadata.get("file_extension", "")

        # Infer source type from file extension
        code_exts = {
            "py",
            "js",
            "ts",
            "jsx",
            "tsx",
            "java",
            "cpp",
            "c",
            "h",
            "hpp",
            "cs",
            "go",
            "rs",
            "swift",
            "kt",
            "scala",
            "rb",
            "php",
        }
        source_type = SourceType.CODE if file_ext in code_exts else SourceType.DOCUMENT

        return ChunkMetadata(
            source_path=metadata.get("full_path", ""),
            source_type=source_type,
            chunk_text=chunk_dict["text"],
            start_char=metadata.get("start_char", 0),
            end_char=metadata.get("end_char", 0),
            file_extension=file_ext,
            embedding_provider=metadata.get(
                "embedding_provider", "sentence_transformers"
            ),
            embedding_model=metadata.get("embedding_model", "all-mpnet-base-v2"),
            tags=metadata.get("tags", []),
            references=metadata.get("references", []),
            parent_section=metadata.get("chunk_name"),
            code_symbols=metadata.get("code_symbols", []),
            metadata=metadata.copy(),
        )

    def _finalize_stats(
        self, stats: Dict[str, Any], start_time: float
    ) -> Dict[str, Any]:
        """Finalize pipeline statistics with timing info."""
        duration_seconds = time.time() - start_time
        stats["finished_at"] = datetime.now(timezone.utc).isoformat()
        stats["duration_seconds"] = round(duration_seconds, 2)

        # Limit error samples
        if len(stats.get("error_samples", [])) > 10:
            stats["error_samples"] = stats["error_samples"][:10]

        logger.info(
            f"Filebase indexing complete: "
            f"{stats['chunks_created']} chunks from {stats['files_loaded']} files "
            f"in {duration_seconds:.2f}s (tags: {', '.join(stats.get('tags', []))})"
        )

        return stats


class IndexManager:
    """Central coordinator for indexing and search operations.

    This refactored version delegates to specialized services rather than
    implementing everything directly. This improves testability, maintainability,
    and separation of concerns.
    """

    def __init__(self, config: Config):
        """Initialize IndexManager with configuration.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize storage (always needed)
        self.metadata_store = MetadataStore()
        self.vector_store = None  # Lazy init after embedder loads

        # Initialize services
        self.embedder_service = EmbedderService(config)
        self.bootstrap_service = BootstrapService(config)

        # Hybrid search (lightweight initialization)
        self.hybrid_search = None
        if config.use_hybrid_search:
            try:
                self.hybrid_search = HybridSearchEngine(
                    keyword_weight=config.keyword_weight
                )
                logger.info(
                    f"Initialized hybrid search with keyword weight: "
                    f"{config.keyword_weight}"
                )
            except Exception as e:
                logger.warning(f"Could not initialize hybrid search: {e}")

        # Performance tracking
        self.performance_logger = SearchPerformanceLogger()

        # Pipeline and search services (initialized when needed)
        self.indexing_pipeline = None
        self.search_service = None

        # Auto-load if configured
        if config.auto_load:
            self._auto_load()

    def _auto_load(self) -> None:
        """Auto-load existing index or bootstrap if needed."""
        paths = self.config.get_index_paths()
        essential_paths = {k: v for k, v in paths.items() if k in ["metadata", "index"]}
        missing = {k: v for k, v in essential_paths.items() if not v.exists()}

        if not missing:
            # All artifacts exist - load them
            self._load_existing_index(paths)
            return

        logger.info(
            f"Detected missing index artifacts: "
            f"{', '.join(p.name for p in missing.values())}"
        )

        # Try bootstrap
        if self.bootstrap_service.bootstrap_if_needed(paths):
            # Bootstrap succeeded - load the index
            self._load_existing_index(paths)
        else:
            logger.info("No existing index, starting fresh")

    def _load_existing_index(self, paths: Dict) -> None:
        """Load index artifacts into memory.

        Args:
            paths: Dictionary of artifact paths
        """
        logger.info("Loading existing index...")

        # Load metadata
        self.metadata_store.load_from_file(str(paths["metadata"]))

        # Load embedder and vector store if we have chunks
        if self.metadata_store.get_stats().get("total_chunks", 0) > 0:
            # Get embedding info from stored metadata
            embedding_info = self.metadata_store.get_embedding_info()
            if embedding_info and embedding_info.get("models"):
                first_model = embedding_info["models"][0]
                if ":" in first_model:
                    provider, model = first_model.split(":", 1)
                    self.config.embedding_provider = provider
                    self.config.embedding_model = model
                    logger.info(
                        f"Loading with stored embedding settings: {provider}:{model}"
                    )

            # Load embedder (this will trigger lazy loading)
            embedder = self.embedder_service.get_embedder()

            # Initialize vector store
            self._ensure_vector_store()
            self.vector_store.load_from_file(str(paths["index"]))

            logger.info(f"Loaded {self.vector_store.get_total_vectors()} vectors")

    def _ensure_vector_store(self) -> None:
        """Ensure vector store is initialized."""
        if self.vector_store is None:
            embedder = self.embedder_service.get_embedder()
            dimension = embedder.get_dimension()
            self.vector_store = VectorStore(dimension, self.config)
            logger.info(f"Initialized vector store with dimension {dimension}")

    def _ensure_services(self) -> None:
        """Ensure pipeline and search services are initialized."""
        if self.indexing_pipeline is None:
            self._ensure_vector_store()
            self.indexing_pipeline = IndexingPipeline(
                config=self.config,
                embedder_service=self.embedder_service,
                vector_store=self.vector_store,
                metadata_store=self.metadata_store,
            )

        if self.search_service is None:
            self._ensure_vector_store()
            self.search_service = SearchService(
                config=self.config,
                embedder_service=self.embedder_service,
                vector_store=self.vector_store,
                metadata_store=self.metadata_store,
                hybrid_search=self.hybrid_search,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Public API - Delegate to Services
    # ═══════════════════════════════════════════════════════════════════════

    def index_filebase(
        self,
        base_path: str,
        tags: str,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Index a filebase from a directory or a single file.

        Args:
            base_path: Root directory or file to index
            tags: Comma-separated tags (required)
            include: Patterns to include
            exclude: Patterns to exclude
            exclude_dirs: Directory patterns to exclude

        Returns:
            Pipeline statistics dictionary
        """
        self._ensure_services()
        result = self.indexing_pipeline.index_filebase(
            base_path, tags, include, exclude, exclude_dirs
        )
        self._auto_save()
        return result

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
        self._ensure_services()
        response = self.search_service.search(query, top_k, display_format)

        # Log performance
        self.performance_logger.log_search_performance(response)

        return response

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.

        Returns:
            System status and statistics
        """
        try:
            metadata_stats = self.metadata_store.get_stats()
            vector_stats = (
                self.vector_store.get_index_info()
                if self.vector_store
                else {"total_vectors": 0, "dimension": None}
            )

            # Embedding info
            if self.embedder_service.is_loaded():
                embedder = self.embedder_service.get_embedder()
                embedding_info = {
                    "provider": embedder.get_provider_name(),
                    "model": embedder.get_model_name(),
                    "dimension": embedder.get_dimension(),
                    "is_available": embedder.is_available(),
                }
            else:
                embedding_info = {
                    "provider": self.config.embedding_provider,
                    "model": self.config.embedding_model,
                    "dimension": None,
                    "is_available": False,
                }

            # Hybrid search info
            hybrid_search_info = {}
            if self.hybrid_search:
                hybrid_search_info = self.hybrid_search.get_stats()

            # Performance info
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            memory_info = psutil.virtual_memory()

            performance_info = {
                "memory_usage_mb": memory_mb,
                "memory_total_mb": memory_info.total / (1024 * 1024),
                "memory_available_mb": memory_info.available / (1024 * 1024),
                "memory_usage_percent": memory_info.percent,
            }

            # Persistence info
            paths = self.config.get_index_paths()
            essential_paths = {
                k: v for k, v in paths.items() if k in ["metadata", "index"]
            }
            persistence_info = {
                "auto_persist": self.config.auto_persist,
                "index_dir": str(self.config.index_dir),
                "index_files_exist": all(
                    path.exists() for path in essential_paths.values()
                ),
            }

            # Create index_stats for compatibility
            index_stats = {
                "total_chunks": metadata_stats.get("total_chunks", 0),
                "total_documents": len(
                    set(c.source_path for c in self.metadata_store.get_all_chunks())
                ),
                "total_vectors": vector_stats.get("total_vectors", 0),
                "source_types": metadata_stats.get("source_types", {}),
            }

            return {
                "status": "healthy",
                "metadata": metadata_stats,  # Keep original key for compatibility
                "vector_store": vector_stats,  # Keep original key for compatibility
                "index_stats": index_stats,  # Keep for compatibility
                "embedding": embedding_info,
                "hybrid_search": hybrid_search_info,
                "performance": performance_info,
                "persistence": persistence_info,
                "configuration": self.config.get_config_summary(),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "embedding": {
                    "provider": getattr(self.config, "embedding_provider", "unknown"),
                    "model": getattr(self.config, "embedding_model", "unknown"),
                    "dimension": None,
                    "is_available": False,
                },
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

    def _auto_save(self) -> None:
        """Auto-save if configured."""
        if not self.config.auto_persist:
            return

        try:
            self.config.ensure_index_directory()
            paths = self.config.get_index_paths()

            # Save vector store
            if self.vector_store:
                self.vector_store.save_to_file(str(paths["index"]))

            # Save metadata
            self.metadata_store.save_to_file(
                str(paths["metadata"]), self.config.compress_metadata
            )

            logger.info("Auto-saved index")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

    # Legacy compatibility methods and properties
    @property
    def embedder(self):
        """Get embedder instance (legacy compatibility)."""
        if self.embedder_service.is_loaded():
            return self.embedder_service.get_embedder()
        return None

    def ensure_embedder_loaded(self) -> None:
        """Ensure the embedder is ready (legacy compatibility)."""
        self.embedder_service.get_embedder()

    def auto_save(self) -> None:
        """Public facade for conditional persistence (legacy compatibility)."""
        self._auto_save()

    # Context manager support
    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Auto-save if configured and no errors
        if self.config.auto_persist and exc_type is None:
            try:
                logger.info("Auto-saving index during cleanup")
                self.save_index()
            except Exception as e:
                logger.warning(f"Auto-save failed during cleanup: {e}")

        # Clean up embedder resources
        try:
            self.embedder_service.cleanup()
        except Exception as e:
            logger.warning(f"Embedder cleanup failed: {e}")

        logger.info("IndexManager cleanup completed")
        return False
