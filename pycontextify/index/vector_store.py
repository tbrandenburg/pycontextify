"""FAISS vector store wrapper for PyContextify.

This module provides a clean abstraction over FAISS for vector similarity
search with persistence, backup support, and embedding dimension validation.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS vector store wrapper with persistence and backup support.

    This class encapsulates FAISS operations and provides automatic
    persistence, backup management, and embedding dimension validation.
    """

    def __init__(self, dimension: int, config: Config) -> None:
        """Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension for the vector index
            config: Configuration object with persistence settings
        """
        self.dimension = dimension
        self.config = config
        self._index = None
        self._total_vectors = 0

        # Initialize FAISS index
        self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize FAISS index with IndexFlatIP for cosine similarity."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

        # Use IndexFlatIP for exact cosine similarity search
        self._index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Initialized FAISS IndexFlatIP with dimension {self.dimension}")

    def add_vectors(self, vectors: np.ndarray) -> List[int]:
        """Add vectors to the index.

        Args:
            vectors: Numpy array of vectors with shape (n_vectors, dimension)

        Returns:
            List of FAISS IDs for the added vectors

        Raises:
            ValueError: If vectors have wrong dimensions
        """
        if vectors.ndim != 2:
            raise ValueError("Vectors must be 2D array")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}"
            )

        # Ensure vectors are float32
        vectors = vectors.astype(np.float32)

        # Get starting ID for new vectors
        start_id = self._total_vectors

        # Add to FAISS index
        self._index.add(vectors)
        self._total_vectors += len(vectors)

        # Return list of IDs
        return list(range(start_id, start_id + len(vectors)))

    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors.

        Args:
            query_vector: Query vector with shape (dimension,)
            top_k: Number of top results to return

        Returns:
            Tuple of (distances, indices) arrays
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query vector dimension {query_vector.shape[1]} doesn't match index dimension {self.dimension}"
            )

        # Ensure float32
        query_vector = query_vector.astype(np.float32)

        # Search
        distances, indices = self._index.search(
            query_vector, min(top_k, self._total_vectors)
        )

        # Return 1D arrays for single query
        return distances[0], indices[0]

    def get_total_vectors(self) -> int:
        """Return number of indexed vectors."""
        return self._total_vectors

    def get_embedding_dimension(self) -> int:
        """Return the dimension of stored embeddings."""
        return self.dimension

    def get_index_info(self) -> Dict[str, Any]:
        """Return FAISS index statistics."""
        return {
            "total_vectors": self._total_vectors,
            "dimension": self.dimension,
            "index_type": "IndexFlatIP",
            "memory_usage_mb": self.get_memory_usage() / (1024 * 1024),
            "is_trained": self._index.is_trained if self._index else False,
        }

    def get_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        if self._total_vectors == 0:
            return 0

        # Rough estimation: vectors + index overhead
        vector_memory = self._total_vectors * self.dimension * 4  # float32 = 4 bytes
        index_overhead = 1024 * 1024  # 1MB overhead estimate
        return vector_memory + index_overhead

    def is_empty(self) -> bool:
        """Check if index contains any vectors."""
        return self._total_vectors == 0

    def remove_vectors(self, faiss_ids: List[int]) -> None:
        """Remove vectors by FAISS IDs.

        Note: FAISS IndexFlatIP doesn't support removal of individual vectors.
        This method logs the request but cannot actually remove vectors.
        The vector store would need to be rebuilt to truly remove vectors.

        Args:
            faiss_ids: List of FAISS IDs to remove
        """
        if not faiss_ids:
            return

        # FAISS IndexFlatIP doesn't support vector removal
        # Log this limitation
        logger.warning(
            f"Cannot remove {len(faiss_ids)} vectors from IndexFlatIP. "
            "FAISS IndexFlatIP does not support individual vector removal. "
            "Consider using clear() and rebuilding the entire index for true removal."
        )

        # In a production system, you might want to:
        # 1. Keep track of removed IDs and filter them during search
        # 2. Use a different FAISS index type that supports removal
        # 3. Periodically rebuild the index to physically remove vectors

    def clear(self) -> None:
        """Clear all vectors from index."""
        self._initialize_index()
        self._total_vectors = 0
        logger.info("Vector store cleared")

    def save_to_file(self, filepath: str) -> None:
        """Save FAISS index to disk with optional backup.

        Args:
            filepath: Path to save the index
        """
        if self._index is None:
            logger.warning("No index to save")
            return

        filepath = Path(filepath)

        # Create backup if requested and file exists
        if self.config.backup_indices and filepath.exists():
            self._create_backup(filepath)

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save index
        try:
            import faiss

            faiss.write_index(self._index, str(filepath))
            logger.info(f"Saved FAISS index to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise

    def load_from_file(self, filepath: str) -> None:
        """Load FAISS index from disk.

        Args:
            filepath: Path to load the index from
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.info(
                f"Index file {filepath} does not exist, starting with empty index"
            )
            return

        try:
            import faiss

            self._index = faiss.read_index(str(filepath))
            self._total_vectors = self._index.ntotal

            # Verify dimension matches
            if self._index.d != self.dimension:
                raise ValueError(
                    f"Loaded index dimension {self._index.d} doesn't match expected {self.dimension}"
                )

            logger.info(
                f"Loaded FAISS index from {filepath} with {self._total_vectors} vectors"
            )
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {filepath}: {e}")
            # Initialize empty index on failure
            self._initialize_index()
            raise

    def _create_backup(self, filepath: Path) -> None:
        """Create backup of existing index file.

        Args:
            filepath: Path to the index file to backup
        """
        if not filepath.exists():
            return

        # Create backup filename with timestamp
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_name(
            f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
        )

        try:
            shutil.copy2(filepath, backup_path)
            logger.info(f"Created backup: {backup_path}")

            # Clean up old backups
            self._cleanup_old_backups(filepath.parent, filepath.stem)
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    def _cleanup_old_backups(self, directory: Path, base_name: str) -> None:
        """Remove old backup files beyond max limit.

        Args:
            directory: Directory containing backup files
            base_name: Base name of the original file
        """
        if self.config.max_backups <= 0:
            return

        # Find all backup files for this base name
        pattern = f"{base_name}_backup_*"
        backup_files = list(directory.glob(pattern))

        # Sort by modification time (newest first)
        backup_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Remove old backups beyond the limit
        for old_backup in backup_files[self.config.max_backups :]:
            try:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {old_backup}: {e}")

    def validate_dimension(self, vectors: np.ndarray) -> bool:
        """Validate that vectors match the index dimension.

        Args:
            vectors: Vectors to validate

        Returns:
            True if dimensions match
        """
        if vectors.ndim == 1:
            return len(vectors) == self.dimension
        elif vectors.ndim == 2:
            return vectors.shape[1] == self.dimension
        else:
            return False

    def get_vector(self, faiss_id: int) -> Optional[np.ndarray]:
        """Get a vector by its FAISS ID.

        Args:
            faiss_id: FAISS ID of the vector

        Returns:
            Vector array or None if not found
        """
        if faiss_id >= self._total_vectors or faiss_id < 0:
            return None

        try:
            # For IndexFlatIP, we can reconstruct the vector
            vector = self._index.reconstruct(faiss_id)
            return vector
        except Exception as e:
            logger.warning(f"Failed to reconstruct vector {faiss_id}: {e}")
            return None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # FAISS handles cleanup automatically
        pass
