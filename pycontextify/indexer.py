"""Central IndexManager for PyContextify.

This module implements the main coordination system that orchestrates all
components for indexing operations, persistence, and search functionality.
"""

import hashlib
import logging
import os
import shutil
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import psutil
import requests

from .chunker import ChunkerFactory
from .config import Config
from .embedder_factory import EmbedderFactory
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
from .storage_metadata import ChunkMetadata, MetadataStore
from .storage_vector import VectorStore
from .types import SourceType

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
        self.embedder = None
        self.vector_store = None
        self.hybrid_search = None
        # Store embedder configuration for lazy loading
        self._embedder_config = None
        self._embedder_initialized = False
        self._bootstrap_thread: Optional[threading.Thread] = None
        self._bootstrap_lock = threading.Lock()
        self._load_lock = threading.Lock()

        # Initialize performance logger
        self.performance_logger = SearchPerformanceLogger()

        # Initialize hybrid search (lightweight)
        self._initialize_hybrid_search()

        # Auto-load existing index if enabled
        if self.config.auto_load:
            self._auto_load()

    def _ensure_embedder_loaded(self) -> None:
        """Ensure embedder is loaded (lazy loading)."""
        if self._embedder_initialized:
            return

        try:
            embedding_config = self.config.get_embedding_config()
            logger.info(
                f"Lazy loading embedder: {embedding_config['provider']} with model {embedding_config['model']}"
            )

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

            logger.info(
                f"Successfully loaded embedder: {self.embedder.get_provider_name()}"
            )

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
            from .search_hybrid import HybridSearchEngine

            self.hybrid_search = HybridSearchEngine(
                keyword_weight=self.config.keyword_weight
            )
            logger.info(
                f"Initialized hybrid search with keyword weight: {self.config.keyword_weight}"
            )
        except ImportError as e:
            logger.warning(f"Could not initialize hybrid search: {e}")
            self.hybrid_search = None

    def _generate_query_suggestions(
        self, query: str, word_count: int, intent: str
    ) -> List[str]:
        """Generate intelligent query suggestions based on patterns and intent."""
        suggestions = []

        # Single word queries - suggest expansions
        if word_count == 1 and len(query) > 3:
            suggestions.extend(
                [
                    f"how to {query}",
                    f"{query} example",
                    f"{query} tutorial",
                    f"{query} documentation",
                ]
            )

        # Long queries - suggest simplification
        elif word_count > 8:
            key_words = [word for word in query.split() if len(word) > 3][:4]
            suggestions.append(" ".join(key_words))

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
            "security": [
                "authentication",
                "authorization",
                "encryption",
                "vulnerability",
            ],
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
        all_expansions = {
            **tech_expansions,
            **problem_expansions,
            **learning_expansions,
        }

        # Find matching terms and add expansions
        for term, expansions in all_expansions.items():
            if term in normalized_query:
                expansion_terms.extend(expansions)

        # Remove duplicates and terms already in query
        query_words = set(normalized_query.split())
        expansion_terms = [term for term in expansion_terms if term not in query_words]

        return list(dict.fromkeys(expansion_terms))[:8]

    def _suggest_search_strategies(
        self, query: str, intent: str, complexity_score: float
    ) -> List[Dict[str, str]]:
        """Suggest alternative search strategies based on query analysis."""
        strategies = []

        # Strategy based on complexity
        if complexity_score > 0.7:
            strategies.append(
                {
                    "strategy": "simplify_query",
                    "description": "Try breaking down your query into simpler terms",
                    "rationale": "Complex queries may miss relevant results",
                }
            )

        if complexity_score < 0.3 and len(query.split()) <= 2:
            strategies.append(
                {
                    "strategy": "expand_query",
                    "description": "Add more descriptive terms to narrow your search",
                    "rationale": "Short queries may return too broad results",
                }
            )

        # Intent-based strategies
        if intent == "informational":
            strategies.append(
                {
                    "strategy": "how_to_search",
                    "description": "Try adding 'how to' or 'guide' to find tutorials",
                    "rationale": "Informational queries benefit from instructional content",
                }
            )

        elif intent == "example_seeking":
            strategies.append(
                {
                    "strategy": "example_search",
                    "description": "Search for 'sample', 'demo', or 'case study'",
                    "rationale": "Example-focused searches find practical implementations",
                }
            )

        # Domain-specific strategies
        if any(
            tech_term in query.lower()
            for tech_term in ["code", "function", "api", "programming"]
        ):
            strategies.append(
                {
                    "strategy": "technical_search",
                    "description": "Include programming language or framework names",
                    "rationale": "Technical queries benefit from specific technology context",
                }
            )

        return strategies[:3]  # Limit to top 3 strategies

    def _auto_load(self) -> None:
        """Automatically load existing index if available."""
        try:
            paths = self.config.get_index_paths()

            essential_paths = {
                key: value
                for key, value in paths.items()
                if key in ["metadata", "index"]
            }
            missing_paths = {
                key: value
                for key, value in essential_paths.items()
                if not value.exists()
            }

            if missing_paths:
                logger.info(
                    "Detected missing index artifacts: %s",
                    ", ".join(sorted(path.name for path in missing_paths.values())),
                )
                if self._restore_from_backups(missing_paths):
                    missing_paths = {
                        key: value
                        for key, value in essential_paths.items()
                        if not value.exists()
                    }

            if not missing_paths:
                self._load_existing_index(paths)
                return

            logger.info("No existing index found, starting fresh")
            self._schedule_bootstrap(paths)
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            import traceback

            logger.debug(traceback.format_exc())

    def _load_existing_index(self, paths: Dict[str, Path]) -> None:
        """Load existing index artifacts into memory."""
        with self._load_lock:
            logger.info("Loading existing index...")

            self.metadata_store.load_from_file(str(paths["metadata"]))

            if self.metadata_store.get_stats().get("total_chunks", 0) > 0:
                embedding_info = self.metadata_store.get_embedding_info()
                if embedding_info and embedding_info.get("models"):
                    first_model = embedding_info["models"][0]
                    if ":" in first_model:
                        stored_provider, stored_model = first_model.split(":", 1)
                        self.config.embedding_provider = stored_provider
                        self.config.embedding_model = stored_model
                        logger.info(
                            "Loading with stored embedding settings: %s:%s",
                            stored_provider,
                            stored_model,
                        )

                self._ensure_embedder_loaded()

                if self.vector_store is not None:
                    self.vector_store.load_from_file(str(paths["index"]))
                    logger.info(
                        f"Loaded {self.vector_store.get_total_vectors()} vectors"
                    )
                else:
                    logger.error("Vector store not initialized, cannot load vectors")
            else:
                logger.info("No chunks in metadata, skipping vector loading")

            logger.info("Successfully loaded existing index")

    def _restore_from_backups(self, missing_paths: Dict[str, Path]) -> bool:
        """Attempt to restore missing artifacts from VectorStore backups."""

        restored_any = False
        for path in missing_paths.values():
            if VectorStore.restore_latest_backup(path):
                restored_any = True

        if restored_any:
            logger.info("Restored one or more index artifacts from backups")

        return restored_any

    def _schedule_bootstrap(self, paths: Dict[str, Path]) -> None:
        """Schedule asynchronous bootstrap of index artifacts when configured."""

        sources = self.config.get_bootstrap_sources()
        if not sources:
            logger.info("Bootstrap archive URL not configured; skipping bootstrap")
            return

        with self._bootstrap_lock:
            if self._bootstrap_thread and self._bootstrap_thread.is_alive():
                logger.debug("Bootstrap worker already running; skipping reschedule")
                return

            logger.info("Scheduling bootstrap download for missing index artifacts")
            self._bootstrap_thread = threading.Thread(
                target=self._bootstrap_index_from_archive,
                args=(paths, sources),
                name="pycontextify-index-bootstrap",
                daemon=True,
            )
            self._bootstrap_thread.start()

    def _bootstrap_index_from_archive(
        self, paths: Dict[str, Path], sources: Dict[str, str]
    ) -> None:
        """Download, verify, and extract bootstrap archive into place."""

        archive_url = sources.get("archive")
        checksum_url = sources.get("checksum")

        if not archive_url or not checksum_url:
            logger.warning("Bootstrap sources incomplete, skipping bootstrap")
            return

        try:
            essential_paths = {
                key: value
                for key, value in paths.items()
                if key in ["metadata", "index"]
            }
            missing_paths = {
                key: value
                for key, value in essential_paths.items()
                if not value.exists()
            }
            if not missing_paths:
                logger.info("Bootstrap skipped because index artifacts already exist")
                return

            if self._restore_from_backups(missing_paths):
                missing_paths = {
                    key: value
                    for key, value in essential_paths.items()
                    if not value.exists()
                }
                if not missing_paths:
                    logger.info(
                        "Bootstrap cancelled after restoring artifacts from backups"
                    )
                    self._load_existing_index(paths)
                    return

            with TemporaryDirectory(prefix="pycontextify-bootstrap-") as tmpdir:
                temp_dir = Path(tmpdir)
                archive_path = self._download_to_path(archive_url, temp_dir)
                checksum_value = self._fetch_checksum(checksum_url)
                self._verify_checksum(archive_path, checksum_value)

                extract_dir = temp_dir / "extracted"
                extract_dir.mkdir(parents=True, exist_ok=True)
                self._extract_archive(archive_path, extract_dir)
                self._move_bootstrap_artifacts(extract_dir, paths)

            remaining_missing = {
                key: value
                for key, value in essential_paths.items()
                if not value.exists()
            }
            if remaining_missing:
                logger.warning(
                    "Bootstrap archive did not provide all required artifacts: %s",
                    ", ".join(sorted(path.name for path in remaining_missing.values())),
                )
                return

            logger.info("Bootstrap archive applied successfully; loading index")
            self._load_existing_index(paths)
        except Exception as exc:
            logger.warning(f"Failed to bootstrap index from {archive_url}: {exc}")

    def _download_to_path(
        self, url: str, destination_dir: Path, max_retries: int = 3
    ) -> Path:
        """Download or copy the given URL into destination_dir with retry logic.

        Args:
            url: URL to download from (http://, https://, or file://)
            destination_dir: Directory to save the downloaded file
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Path to the downloaded file

        Raises:
            FileNotFoundError: If file:// URL points to non-existent file
            ValueError: If URL scheme is not supported
            Exception: After max retries exhausted for transient errors
        """
        parsed = urlparse(url)
        filename = Path(unquote(parsed.path or "")).name or "bootstrap_archive"
        destination_dir.mkdir(parents=True, exist_ok=True)
        target_path = destination_dir / filename

        # Handle file:// URLs without retry (local filesystem)
        if parsed.scheme == "file":
            # Use url2pathname to handle Windows paths properly
            # (e.g., file:///C:/path becomes C:\path on Windows)
            local_path = url2pathname(parsed.path)
            source_path = Path(local_path)
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Bootstrap archive file not found: {source_path}"
                )
            logger.info("Copying bootstrap archive from %s", source_path)
            shutil.copy2(source_path, target_path)
            return target_path

        # Handle http:// and https:// URLs with retry logic
        if parsed.scheme in ("http", "https"):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    if attempt > 1:
                        # Exponential backoff: 1s, 2s, 4s, ...
                        delay = 2 ** (attempt - 2)
                        logger.info(
                            f"Retrying download after {delay}s delay (attempt "
                            f"{attempt}/{max_retries})"
                        )
                        time.sleep(delay)

                    logger.info(
                        f"Downloading bootstrap archive from {url} "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()

                    # Write to temporary file first, then rename atomically
                    temp_path = target_path.with_suffix(".tmp")
                    with temp_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)

                    # Atomic rename
                    os.replace(temp_path, target_path)
                    logger.info("Download completed successfully")
                    return target_path

                except requests.exceptions.Timeout as e:
                    last_exception = e
                    logger.warning(
                        f"Download timeout on attempt {attempt}/{max_retries}: {e}"
                    )
                    # Retry on timeout
                    continue

                except requests.exceptions.ConnectionError as e:
                    last_exception = e
                    logger.warning(
                        f"Connection error on attempt {attempt}/{max_retries}: {e}"
                    )
                    # Retry on connection errors
                    continue

                except requests.exceptions.HTTPError as e:
                    # Check if error is retriable (408, 429, 5xx)
                    if e.response is not None:
                        status_code = e.response.status_code
                        if status_code in (408, 429) or status_code >= 500:
                            last_exception = e
                            logger.warning(
                                f"Retriable HTTP error {status_code} on attempt "
                                f"{attempt}/{max_retries}: {e}"
                            )
                            continue  # Retry
                        else:
                            # Non-retriable 4xx error
                            logger.error(f"Non-retriable HTTP error {status_code}: {e}")
                            raise
                    else:
                        # No response, treat as retriable
                        last_exception = e
                        logger.warning(
                            f"HTTP error on attempt {attempt}/{max_retries}: {e}"
                        )
                        continue

                except Exception as e:
                    # Unexpected errors - log and re-raise immediately
                    logger.error(
                        f"Unexpected error during download (attempt "
                        f"{attempt}/{max_retries}): {e}"
                    )
                    raise

            # All retries exhausted
            error_msg = (
                f"Failed to download {url} after {max_retries} attempts. "
                f"Last error: {last_exception}"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

        raise ValueError(f"Unsupported bootstrap scheme: {parsed.scheme}")

    def _fetch_checksum(self, url: str, max_retries: int = 3) -> str:
        """Retrieve checksum text from the provided URL with retry logic.

        Args:
            url: URL to fetch checksum from (http://, https://, or file://)
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            The SHA256 checksum as a hex string

        Raises:
            FileNotFoundError: If file:// URL points to non-existent file
            ValueError: If checksum file is empty or malformed
            Exception: After max retries exhausted for transient errors
        """
        parsed = urlparse(url)

        # Handle file:// URLs
        if parsed.scheme == "file":
            # Use url2pathname to handle Windows paths properly
            local_path = url2pathname(parsed.path)
            checksum_path = Path(local_path)
            if not checksum_path.exists():
                raise FileNotFoundError(
                    f"Bootstrap checksum file not found: {checksum_path}"
                )
            content = checksum_path.read_text(encoding="utf-8")
        # Handle http:// and https:// URLs with retry
        elif parsed.scheme in ("http", "https"):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    if attempt > 1:
                        delay = 2 ** (attempt - 2)
                        logger.info(
                            f"Retrying checksum fetch after {delay}s "
                            f"(attempt {attempt}/{max_retries})"
                        )
                        time.sleep(delay)

                    logger.debug(f"Fetching checksum from {url}")
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    content = response.text
                    break

                except (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                ) as e:
                    last_exception = e
                    logger.warning(
                        f"Error fetching checksum on attempt "
                        f"{attempt}/{max_retries}: {e}"
                    )
                    if attempt == max_retries:
                        raise Exception(
                            f"Failed to fetch checksum after {max_retries} attempts. "
                            f"Last error: {e}"
                        )
                    continue

                except requests.exceptions.HTTPError as e:
                    if e.response is not None:
                        status_code = e.response.status_code
                        if status_code in (408, 429) or status_code >= 500:
                            last_exception = e
                            logger.warning(
                                f"Retriable HTTP error {status_code} "
                                f"(attempt {attempt}/{max_retries})"
                            )
                            if attempt == max_retries:
                                raise Exception(
                                    f"Failed to fetch checksum after "
                                    f"{max_retries} attempts"
                                )
                            continue
                        else:
                            logger.error(f"Non-retriable HTTP error {status_code}")
                            raise
                    raise
        else:
            raise ValueError(f"Unsupported checksum URL scheme: {parsed.scheme}")

        # Parse checksum from content
        if not content.strip():
            raise ValueError("Bootstrap checksum file is empty")

        # Support two formats:
        # 1. "<hex_digest>" (just the hash)
        # 2. "<hex_digest>  <filename>" (hash with filename)
        parts = content.strip().split()
        if not parts:
            raise ValueError("Bootstrap checksum file is empty")

        checksum = parts[0].lower()

        # Validate it's a valid hex string
        if len(checksum) != 64 or not all(c in "0123456789abcdef" for c in checksum):
            raise ValueError(
                f"Invalid SHA256 checksum format: {checksum[:20]}... "
                "(expected 64 hex characters)"
            )

        logger.info(f"Fetched checksum: {checksum[:16]}...")
        return checksum

    def _verify_checksum(self, archive_path: Path, expected_checksum: str) -> None:
        """Verify archive SHA256 checksum."""

        digest = hashlib.sha256()
        with archive_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

        actual_checksum = digest.hexdigest()
        if actual_checksum.lower() != expected_checksum.lower():
            raise ValueError(
                "Bootstrap archive checksum mismatch: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )

    def _extract_archive(self, archive_path: Path, destination: Path) -> None:
        """Extract supported archive formats into destination directory."""

        lower_name = archive_path.name.lower()
        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zip_ref:
                zip_ref.extractall(destination)
            return

        if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz"):
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(destination)
            return

        raise ValueError(f"Unsupported bootstrap archive format: {archive_path}")

    def _move_bootstrap_artifacts(
        self, extract_dir: Path, paths: Dict[str, Path]
    ) -> None:
        """Move extracted files into their final locations if missing."""

        for key in ["metadata", "index"]:
            destination = paths[key]
            if destination.exists():
                continue

            matches = list(extract_dir.rglob(destination.name))
            if not matches:
                logger.warning(
                    "Bootstrap archive missing expected %s file %s",
                    key,
                    destination.name,
                )
                continue

            source_path = matches[0]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source_path, destination)
            logger.info("Bootstrapped %s", destination.name)

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

            logger.info("Auto-saved index to disk")
        except Exception as e:
            logger.error(f"Failed to auto-save index: {e}")
            # Don't raise - indexing should continue even if save fails

    def auto_save(self) -> None:
        """Public facade for conditional persistence used by indexers."""

        self._auto_save()

    def ensure_embedder_loaded(self) -> None:
        """Ensure the embedder is ready before reporting statistics."""

        self._ensure_embedder_loaded()

    def index_filebase(
        self,
        base_path: str,
        topic: str,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Index a filebase (directory) using unified pipeline.

        This is the new unified indexing pipeline that replaces index_code() and
        index_document() with a single, consistent flow.

        Args:
            base_path: Root directory to index
            topic: Topic name (required, used for organization)
            include: List of fnmatch patterns to include
            exclude: List of fnmatch patterns to exclude
            exclude_dirs: List of directory patterns to exclude

        Returns:
            Dictionary with pipeline statistics

        Raises:
            ValueError: If topic is empty or None
            FileNotFoundError: If base_path does not exist
        """
        from datetime import datetime, timezone

        from .crawler import FileCrawler
        from .loader import FileLoaderFactory
        from .postprocess import postprocess_file, postprocess_filebase

        # Validate topic (required)
        if not topic or not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic is required and must be a non-empty string")

        topic = topic.strip()

        # Validate base_path
        base_path_obj = Path(base_path).resolve()
        if not base_path_obj.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")
        if not base_path_obj.is_dir():
            raise ValueError(f"Base path is not a directory: {base_path}")

        # Track timing
        started_at = datetime.now(timezone.utc)
        start_time = time.time()

        logger.info(f"Starting filebase indexing: {base_path} (topic: {topic})")

        # Initialize stats
        stats = {
            "topic": topic,
            "base_path": str(base_path_obj),
            "started_at": started_at.isoformat(),
            "files_crawled": 0,
            "files_loaded": 0,
            "chunks_created": 0,
            "vectors_embedded": 0,
            "errors": 0,
            "error_samples": [],
        }

        # Step 1: Crawl file tree
        logger.info("Step 1/8: Crawling file tree")
        try:
            crawler = FileCrawler(
                include=include,
                exclude=exclude,
                exclude_dirs=exclude_dirs,
            )
            file_paths = crawler.crawl(str(base_path_obj))
            stats["files_crawled"] = len(file_paths)
            logger.info(f"Crawled {len(file_paths)} files")
        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            stats["errors"] += 1
            stats["error_samples"].append({"stage": "crawl", "error": str(e)})
            return self._finalize_stats(stats, start_time)

        if not file_paths:
            logger.warning("No files found to index")
            return self._finalize_stats(stats, start_time)

        # Prepare loader and ensure embedder is ready
        loader = FileLoaderFactory()
        self._ensure_embedder_loaded()

        # Get embedding info for chunks
        embedding_provider = self.embedder.get_provider_name()
        embedding_model = self.embedder.get_model_name()

        all_chunks_to_store = []

        # Step 2-5: Load, Chunk, Postprocess (per-file)
        logger.info("Step 2-5: Loading, chunking, and postprocessing files")
        for file_idx, file_path in enumerate(file_paths):
            try:
                # Step 2: Load/Normalize
                logger.debug(f"Loading {file_idx + 1}/{len(file_paths)}: {file_path}")
                normalized_docs = loader.load(
                    path=file_path,
                    topic=topic,
                    base_path=str(base_path_obj),
                )

                if not normalized_docs:
                    # File skipped (binary or error), already logged by loader
                    continue

                # Add embedding info to each doc's metadata
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

                # Step 4: Postprocess (file-level)
                file_chunks = postprocess_file(file_chunks)

                # Collect chunks
                all_chunks_to_store.extend(file_chunks)
                stats["files_loaded"] += 1

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                stats["errors"] += 1
                if len(stats["error_samples"]) < 10:
                    stats["error_samples"].append(
                        {
                            "stage": "load/chunk",
                            "file": str(file_path),
                            "error": str(e),
                        }
                    )

        if not all_chunks_to_store:
            logger.warning("No chunks created from files")
            return self._finalize_stats(stats, start_time)

        logger.info(
            f"Created {len(all_chunks_to_store)} chunks from {stats['files_loaded']} files"
        )
        stats["chunks_created"] = len(all_chunks_to_store)

        # Step 6: Postprocess (filebase-level)
        logger.info("Step 6/8: Postprocessing entire filebase")
        all_chunks_to_store = postprocess_filebase(all_chunks_to_store)

        # Step 7: Embed
        logger.info("Step 7/8: Generating embeddings")
        try:
            texts_to_embed = [chunk["text"] for chunk in all_chunks_to_store]
            embeddings = self.embedder.embed_texts(texts_to_embed)
            stats["vectors_embedded"] = len(embeddings)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            stats["errors"] += 1
            stats["error_samples"].append({"stage": "embed", "error": str(e)})
            return self._finalize_stats(stats, start_time)

        # Step 8: Store
        logger.info("Step 8/8: Storing vectors and metadata")
        try:
            # Ensure vector store is initialized
            if self.vector_store is None:
                self._initialize_vector_store()

            # Add vectors to FAISS
            faiss_ids = self.vector_store.add_vectors(embeddings)

            # Add metadata (convert normalized chunks to ChunkMetadata)
            for chunk_dict, faiss_id in zip(all_chunks_to_store, faiss_ids):
                # Create ChunkMetadata from normalized chunk
                chunk_metadata = self._create_chunk_metadata_from_dict(chunk_dict)
                self.metadata_store.add_chunk(chunk_metadata)

            # Auto-save if enabled
            self._auto_save()

            logger.info(f"Stored {len(faiss_ids)} vectors and metadata")

        except Exception as e:
            logger.error(f"Storage failed: {e}")
            stats["errors"] += 1
            stats["error_samples"].append({"stage": "store", "error": str(e)})
            return self._finalize_stats(stats, start_time)

        # Finalize stats
        return self._finalize_stats(stats, start_time)

    def _create_chunk_metadata_from_dict(
        self, chunk_dict: Dict[str, Any]
    ) -> ChunkMetadata:
        """Convert normalized chunk dict to ChunkMetadata.

        Args:
            chunk_dict: Normalized chunk with {"text": str, "metadata": dict}

        Returns:
            ChunkMetadata object ready for storage
        """
        metadata = chunk_dict["metadata"]

        # Determine source type from file extension
        file_ext = metadata.get("file_extension", "")
        source_type = self._infer_source_type_from_extension(file_ext)

        # Create ChunkMetadata
        chunk_metadata = ChunkMetadata(
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
            metadata=metadata.copy(),  # Store all metadata
        )

        return chunk_metadata

    def _infer_source_type_from_extension(self, file_extension: str) -> SourceType:
        """Infer SourceType from file extension.

        Args:
            file_extension: File extension without dot

        Returns:
            SourceType enum value
        """
        code_extensions = {
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

        if file_extension in code_extensions:
            return SourceType.CODE
        else:
            return SourceType.DOCUMENT

    def _finalize_stats(
        self, stats: Dict[str, Any], start_time: float
    ) -> Dict[str, Any]:
        """Finalize pipeline statistics with timing info.

        Args:
            stats: Statistics dictionary to finalize
            start_time: Start time from time.time()

        Returns:
            Finalized statistics dictionary
        """
        from datetime import datetime, timezone

        duration_seconds = time.time() - start_time
        stats["finished_at"] = datetime.now(timezone.utc).isoformat()
        stats["duration_seconds"] = round(duration_seconds, 2)

        # Limit error samples
        if len(stats.get("error_samples", [])) > 10:
            stats["error_samples"] = stats["error_samples"][:10]

        logger.info(
            f"Filebase indexing complete: "
            f"{stats['chunks_created']} chunks from {stats['files_loaded']} files "
            f"in {duration_seconds:.2f}s (topic: {stats['topic']})"
        )

        return stats

    def process_content(
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
            logger.info(
                f"Found {len(existing_chunks)} existing chunks for {source_path}, removing for re-indexing"
            )

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

        # Add metadata (convert Chunk DTO to ChunkMetadata for storage)
        from .storage_metadata import ChunkMetadata

        for chunk, faiss_id in zip(chunks, faiss_ids):
            chunk_metadata = ChunkMetadata.from_chunk(chunk)
            self.metadata_store.add_chunk(chunk_metadata)

        return len(chunks)

    def _get_search_config(self) -> Dict[str, Any]:
        """Get current search configuration for response metadata."""
        return {
            "hybrid_search": self.config.use_hybrid_search,
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
            "keyword_weight": getattr(self.config, "keyword_weight", 0.3),
        }

    def _create_source_info_from_chunk(self, chunk) -> Dict[str, Any]:
        """Create source info from chunk metadata.

        Args:
            chunk: ChunkMetadata object

        Returns:
            Dictionary with enhanced source metadata
        """
        from pathlib import Path

        # Get source_path safely, handling both real objects and mocks
        source_path = getattr(chunk, "source_path", "/unknown")
        source_type_val = getattr(chunk, "source_type", None)

        # Handle source_type.value safely (could be enum or string)
        if hasattr(source_type_val, "value"):
            source_type_str = source_type_val.value
        else:
            source_type_str = str(source_type_val) if source_type_val else "unknown"

        source_info = {
            "file_path": str(source_path) if source_path else "/unknown",
            "source_type": source_type_str,
        }

        # Only proceed with file operations if we have a real string path
        if isinstance(source_path, str) and source_path != "/unknown":
            # Basic file information
            try:
                file_path = Path(source_path)
                if file_path.exists() and file_path.is_file():
                    stat = file_path.stat()
                    source_info.update(
                        {
                            "filename": file_path.name,
                            "file_extension": file_path.suffix.lower(),
                            "file_size_bytes": stat.st_size,
                            "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "created_at": stat.st_ctime,
                            "modified_at": stat.st_mtime,
                        }
                    )
                else:
                    # File doesn't exist, but we can still extract basic info
                    source_info["filename"] = file_path.name
                    source_info["file_extension"] = file_path.suffix.lower()
            except (OSError, ValueError, TypeError) as e:
                # Handle invalid paths gracefully
                logger.debug(f"Could not extract file metadata: {e}")
                try:
                    source_info["filename"] = Path(source_path).name
                except Exception:
                    source_info["filename"] = "unknown"

            # PDF-specific metadata extraction
            # Note: PDF metadata extraction has been moved to the loader module
            if source_type_str == "document" and source_path.lower().endswith(".pdf"):
                # PDF info is already in chunk metadata if available
                pass

        # Page and section context from chunk text (safe to do even with mocks)
        # Note: Page context is now extracted during loading phase
        chunk_text = getattr(chunk, "chunk_text", "")
        if chunk_text and isinstance(chunk_text, str):
            # Page info should already be in chunk metadata if available
            pass

        # Add chunk-specific metadata
        if hasattr(chunk, "parent_section"):
            parent_section = getattr(chunk, "parent_section", None)
            if parent_section:
                source_info["section_title"] = str(parent_section)

        # Add additional metadata from chunk
        if hasattr(chunk, "metadata"):
            metadata = getattr(chunk, "metadata", None)
            if metadata and hasattr(metadata, "items"):
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

    def search(
        self, query: str, top_k: int = 5, display_format: str = "readable"
    ) -> SearchResponse:
        """Perform semantic search with optional hybrid enhancement.

        Args:
            query: Search query text
            top_k: Number of results to return
            display_format: Output format ('readable', 'structured', 'summary')

        Returns:
            SearchResponse with results and metadata
        """
        return self._simple_search(query, top_k, display_format)

    def _simple_search(
        self, query: str, top_k: int = 5, display_format: str = "readable"
    ) -> SearchResponse:
        """Simplified search implementation with optional hybrid enhancement."""
        import time

        start_time = time.time()

        try:
            # Check if we have any indexed content
            if self.vector_store is None or self.vector_store.is_empty():
                return SearchResponse.create_error(
                    query=query,
                    error="No indexed content available. Please index some documents or code first.",
                    error_code=SearchErrorCode.NO_CONTENT.value,
                    search_config=self._get_search_config(),
                    recovery_suggestions=[
                        "Use index_filebase() to add content",
                        "Check if auto_load is enabled and index files exist",
                        "Verify the vector store was initialized properly",
                    ],
                )

            # Ensure embedder is loaded
            self._ensure_embedder_loaded()

            # Generate query embedding
            query_embedding = self.embedder.embed_single(query)

            # Perform vector search
            distances, indices = self.vector_store.search(
                query_embedding, top_k * 2
            )  # Get more for hybrid filtering

            # Create initial results
            search_results = self._create_vector_search_results(distances, indices)

            # Apply hybrid search enhancement if enabled
            if self.config.use_hybrid_search and self.hybrid_search:
                try:
                    self._ensure_hybrid_search_index()
                    hybrid_results = self.hybrid_search.search(query, top_k * 2)

                    # Combine semantic and keyword results
                    combined_results = self._combine_hybrid_results(
                        search_results, hybrid_results
                    )
                    search_results = combined_results[:top_k]
                except Exception as e:
                    logger.warning(f"Hybrid search failed, using vector-only: {e}")
                    search_results = search_results[:top_k]
            else:
                search_results = search_results[:top_k]

            # Enhance results with ranking information
            enhanced_results = enhance_search_results_with_ranking(
                results=search_results,
                query=query,
                include_explanations=True,
                include_confidence=True,
            )

            # Create performance information
            performance = create_search_performance_info(
                start_time=start_time,
                search_mode="hybrid" if self.config.use_hybrid_search else "vector",
                total_candidates=len(search_results),
            )

            # Create successful response
            response = SearchResponse.create_success(
                query=query,
                results=enhanced_results,
                search_config=self._get_search_config(),
                performance=performance,
            )

            # Set display format and generate formatted output
            response.display_format = display_format
            if display_format != "structured":
                response.formatted_output = response.format_for_display(display_format)

            # Log performance
            self.performance_logger.log_search_performance(response)

            return response

        except Exception as e:
            logger.error(f"Search failed: {e}")

            error_response = SearchResponse.create_error(
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

            return error_response

    def _create_vector_search_results(self, distances, indices):
        """Convert vector search results to SearchResult objects."""
        results = []
        for i, (distance, idx) in enumerate(zip(distances, indices)):
            if idx >= 0:  # Valid index
                chunk = self.metadata_store.get_chunk(idx)
                if chunk:
                    source_info = self._create_source_info_from_chunk(chunk)

                    # Convert distance to similarity score (cosine distance -> similarity)
                    relevance_score = max(0.0, 1.0 - distance)

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

    def _combine_hybrid_results(self, vector_results, hybrid_results):
        """Combine vector and hybrid search results with simple scoring."""
        # Create a dictionary to track chunks by ID
        combined_chunks = {}

        # Add vector results
        for result in vector_results:
            combined_chunks[result.chunk_id] = {
                "result": result,
                "vector_score": result.relevance_score,
                "keyword_score": 0.0,
                "combined_score": result.relevance_score
                * 0.7,  # Give vector search 70% weight
            }

        # Add/enhance with hybrid results
        for hybrid_result in hybrid_results:
            chunk_id = hybrid_result.get("chunk_id")
            keyword_score = hybrid_result.get("score", 0.0)

            if chunk_id in combined_chunks:
                # Update existing result with hybrid score
                combined_chunks[chunk_id]["keyword_score"] = keyword_score
                combined_chunks[chunk_id]["combined_score"] = (
                    combined_chunks[chunk_id]["vector_score"] * 0.7
                    + keyword_score * 0.3
                )
            else:
                # Create new result from hybrid search
                chunk = self.metadata_store.get_chunk_by_chunk_id(chunk_id)
                if chunk:
                    source_info = self._create_source_info_from_chunk(chunk)

                    result = SearchResult(
                        chunk_id=chunk.chunk_id,
                        source_path=chunk.source_path,
                        source_type=chunk.source_type.value,
                        text=chunk.chunk_text,
                        relevance_score=keyword_score,
                        scores=create_structured_scores(
                            keyword_score=keyword_score,
                            combined_score=keyword_score
                            * 0.3,  # Pure keyword gets 30% weight
                        ),
                        metadata=(
                            create_structured_metadata(**chunk.metadata)
                            if chunk.metadata
                            else None
                        ),
                        provenance=create_search_provenance(
                            search_features=["hybrid_search"],
                            search_stage="hybrid_only",
                        ),
                        source_info=source_info,
                    )

                    combined_chunks[chunk_id] = {
                        "result": result,
                        "vector_score": 0.0,
                        "keyword_score": keyword_score,
                        "combined_score": keyword_score * 0.3,
                    }

        # Sort by combined score and return results
        sorted_items = sorted(
            combined_chunks.values(), key=lambda x: x["combined_score"], reverse=True
        )

        # Update scores in results and return
        final_results = []
        for item in sorted_items:
            result = item["result"]
            result.relevance_score = item["combined_score"]

            # Update scores with combined information
            result.scores = create_structured_scores(
                vector_score=item["vector_score"],
                keyword_score=item["keyword_score"],
                combined_score=item["combined_score"],
                original_score=(
                    result.scores.original_score
                    if result.scores
                    else item["combined_score"]
                ),
            )

            final_results.append(result)

        return final_results

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
            relationship_stats = {}  # Removed relationship store
            vector_stats = (
                self.vector_store.get_index_info()
                if self.vector_store
                else {"total_vectors": 0, "dimension": None}
            )

            # Create unified index stats
            index_stats = {
                "total_chunks": metadata_stats.get("total_chunks", 0),
                "total_documents": len(
                    set(
                        chunk.source_path
                        for chunk in self.metadata_store.get_all_chunks()
                    )
                ),
                "total_vectors": vector_stats.get("total_vectors", 0),
                "source_types": metadata_stats.get("source_types", {}),
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
            essential_paths = {
                k: v for k, v in paths.items() if k in ["metadata", "index"]
            }
            persistence_info = {
                "auto_persist": self.config.auto_persist,
                "index_dir": str(self.config.index_dir),
                "index_files_exist": all(
                    path.exists() for path in essential_paths.values()
                ),
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

            # System performance metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage("/")

            performance_info = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_mb": memory_mb,
                "memory_total_mb": memory_info.total / (1024 * 1024),
                "memory_available_mb": memory_info.available / (1024 * 1024),
                "memory_usage_percent": memory_info.percent,
                "disk_usage_percent": disk_usage.percent,
                "disk_free_gb": disk_usage.free / (1024 * 1024 * 1024),
            }

            return {
                "status": "healthy",
                "index_stats": index_stats,
                "metadata": metadata_stats,
                "relationships": relationship_stats,
                "vector_store": vector_stats,
                "embedding": embedding_info,
                "hybrid_search": hybrid_search_info,
                "performance": performance_info,
                "persistence": persistence_info,
                "configuration": self.config.get_config_summary(),
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
                    "provider": getattr(self.config, "embedding_provider", "unknown"),
                    "model": getattr(self.config, "embedding_model", "unknown"),
                    "dimension": None,
                    "is_available": False,
                },
                "hybrid_search": {},
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
            self.auto_save()
            return {"success": True}
        except Exception as e:
            error_msg = f"Failed to save index: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with comprehensive cleanup sequence."""
        import logging

        logger = logging.getLogger(__name__)

        # Auto-save if enabled and no critical errors
        if hasattr(self, "config") and self.config.auto_persist and exc_type is None:
            try:
                logger.info("Auto-saving index during cleanup")
                self.save_index()
            except Exception as e:
                logger.warning(f"Auto-save failed during cleanup: {e}")

        # Clean up embedder resources
        if hasattr(self, "embedder") and self.embedder is not None:
            try:
                logger.debug("Cleaning up embedder resources")
                self.embedder.cleanup()
            except Exception as e:
                logger.warning(f"Embedder cleanup failed: {e}")

        # Clean up vector store if needed
        if hasattr(self, "vector_store") and self.vector_store is not None:
            try:
                # Vector store might have cleanup methods in the future
                logger.debug("Vector store cleanup completed")
            except Exception as e:
                logger.warning(f"Vector store cleanup failed: {e}")

        # Clear any cached data
        if (
            hasattr(self, "search_intelligence")
            and self.search_intelligence is not None
        ):
            try:
                # Save search intelligence data before cleanup
                logger.debug("Finalizing search intelligence data")
                # Don't raise exceptions during cleanup
            except Exception as e:
                logger.warning(f"Search intelligence cleanup failed: {e}")

        logger.info("IndexManager cleanup completed")

        # Return False to not suppress exceptions
        return False
