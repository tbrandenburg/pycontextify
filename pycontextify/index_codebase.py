"""Codebase indexing orchestration for PyContextify."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .storage_metadata import SourceType

logger = logging.getLogger(__name__)


class CodeLoader:
    """Loader for codebase directories with dependency extraction."""

    SUPPORTED_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".h",
        ".c",
        ".cs",
        ".rb",
        ".php",
        ".go",
        ".rs",
        ".swift",
        ".kt",
        ".scala",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
    }

    EXCLUDED_DIRS = {
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".env",
        "build",
        "dist",
        "target",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".tox",
        "htmlcov",
    }

    def __init__(self, max_file_size_mb: int = 10):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def load(self, directory_path: str) -> List[Tuple[str, str]]:
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        files: List[Tuple[str, str]] = []
        processed_count = 0

        for file_path in self._walk_directory(directory):
            try:
                content = self._read_file(file_path)
                if content:
                    files.append((str(file_path), content))
                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.info("Processed %d code files", processed_count)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to read %s: %s", file_path, exc)

        logger.info("Loaded %d code files from %s", len(files), directory_path)
        return files

    def _walk_directory(self, directory: Path) -> List[Path]:
        files: List[Path] = []
        for path in directory.rglob("*"):
            if path.is_dir():
                continue
            if any(excluded in path.parts for excluded in self.EXCLUDED_DIRS):
                continue
            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    if path.stat().st_size <= self.max_file_size_bytes:
                        files.append(path)
                    else:
                        logger.warning("Skipping large file: %s", path)
                except OSError:  # pragma: no cover - defensive
                    continue
        return files

    def _read_file(self, file_path: Path) -> Optional[str]:
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as handle:
                    content = handle.read()
                    if "\x00" in content:
                        return None
                    return content
            except UnicodeDecodeError:
                continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Error reading %s: %s", file_path, exc)
                return None
        logger.warning("Could not decode file: %s", file_path)
        return None


class CodebaseIndexer:
    """Coordinate codebase ingestion using an :class:`IndexManager`."""

    def __init__(self, manager: "IndexManager") -> None:
        self._manager = manager
        self._loader = CodeLoader(max_file_size_mb=manager.config.max_file_size_mb)

    def index(self, path: str) -> Dict[str, int | str]:
        logger.info("Starting codebase indexing: %s", path)
        try:
            files = self._loader.load(path)
            if not files:
                return {"error": "No files found to index"}

            chunks_added = 0
            for file_path, content in files:
                chunks_added += self._manager.process_content(
                    content, file_path, SourceType.CODE
                )

            self._manager.auto_save()
            self._manager.ensure_embedder_loaded()

            stats: Dict[str, int | str] = {
                "files_processed": len(files),
                "chunks_added": chunks_added,
                "source_type": "code",
                "embedding_provider": self._manager.embedder.get_provider_name(),
                "embedding_model": self._manager.embedder.get_model_name(),
            }
            logger.info("Completed codebase indexing: %s", stats)
            return stats
        except Exception as exc:  # pragma: no cover - defensive
            error_msg = f"Failed to index codebase {path}: {exc}"
            logger.error(error_msg)
            return {"error": error_msg}


__all__ = ["CodebaseIndexer", "CodeLoader"]
