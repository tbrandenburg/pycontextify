"""Bootstrap and loading logic for persisted indices."""

from __future__ import annotations

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
from typing import Dict, Optional
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import requests

from ...storage.metadata import MetadataStore
from ...storage.vector import VectorStore
from ...orchestrator.config import Config
from .embedding import EmbeddingService


class BootstrapService:
    """Coordinate loading and bootstrap of persisted index artefacts."""

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

        self._bootstrap_thread: Optional[threading.Thread] = None
        self._bootstrap_lock = threading.Lock()
        self._load_lock = threading.Lock()

    def auto_load(self) -> None:
        """Attempt to load existing index artefacts, scheduling bootstrap if needed."""
        try:
            paths = self._config.get_index_paths()
            essential = {
                key: value
                for key, value in paths.items()
                if key in {"metadata", "index"}
            }
            missing = {key: value for key, value in essential.items() if not value.exists()}

            if missing and self._restore_from_backups(missing):
                missing = {
                    key: value
                    for key, value in essential.items()
                    if not value.exists()
                }

            if not missing:
                self._load_existing_index(paths)
                return

            self._logger.info("No existing index found, starting fresh")
            self._schedule_bootstrap(paths)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning("Failed to load existing index: %s", exc)

    def _load_existing_index(self, paths: Dict[str, Path]) -> None:
        """Load stored metadata and vector files into memory."""
        with self._load_lock:
            self._logger.info("Loading existing index...")
            metadata_path = str(paths["metadata"])
            self._metadata_store.load_from_file(metadata_path)

            if self._metadata_store.get_stats().get("total_chunks", 0) == 0:
                self._logger.info("No chunks in metadata, skipping vector loading")
                return

            embedding_info = self._metadata_store.get_embedding_info()
            if embedding_info and embedding_info.get("models"):
                first_model = embedding_info["models"][0]
                if ":" in first_model:
                    stored_provider, stored_model = first_model.split(":", 1)
                    self._config.embedding_provider = stored_provider
                    self._config.embedding_model = stored_model
                    self._logger.info(
                        "Loading with stored embedding settings: %s:%s",
                        stored_provider,
                        stored_model,
                    )

            self._embedding_service.ensure_loaded()
            vector_path = str(paths["index"])
            if self._embedding_service.vector_store is not None:
                self._embedding_service.load_vector_store(vector_path)
                self._logger.info(
                    "Loaded %s vectors",
                    self._embedding_service.vector_store.get_total_vectors(),
                )
            else:
                self._logger.error("Vector store not initialized, cannot load vectors")

            self._logger.info("Successfully loaded existing index")

    def _restore_from_backups(self, missing_paths: Dict[str, Path]) -> bool:
        """Restore missing artefacts from configured backups when available."""
        restored = False
        for path in missing_paths.values():
            if VectorStore.restore_latest_backup(path):
                restored = True
        if restored:
            self._logger.info("Restored one or more index artifacts from backups")
        return restored

    def _schedule_bootstrap(self, paths: Dict[str, Path]) -> None:
        """Start asynchronous bootstrap download when configured."""
        sources = self._config.get_bootstrap_sources()
        if not sources:
            self._logger.info("Bootstrap archive URL not configured; skipping bootstrap")
            return

        with self._bootstrap_lock:
            if self._bootstrap_thread and self._bootstrap_thread.is_alive():
                self._logger.debug("Bootstrap worker already running; skipping reschedule")
                return

            self._logger.info("Scheduling bootstrap download for missing index artifacts")
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
        """Download, verify and extract bootstrap archives."""
        archive_url = sources.get("archive")
        checksum_url = sources.get("checksum")

        if not archive_url or not checksum_url:
            self._logger.warning("Bootstrap sources incomplete, skipping bootstrap")
            return

        try:
            essential = {
                key: value
                for key, value in paths.items()
                if key in {"metadata", "index"}
            }
            missing = {key: value for key, value in essential.items() if not value.exists()}

            if not missing:
                self._logger.info("Bootstrap skipped because index artifacts already exist")
                return

            if self._restore_from_backups(missing):
                missing = {
                    key: value
                    for key, value in essential.items()
                    if not value.exists()
                }
                if not missing:
                    self._logger.info(
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

            remaining = {key: value for key, value in essential.items() if not value.exists()}
            if remaining:
                self._logger.warning(
                    "Bootstrap archive did not provide all required artifacts: %s",
                    ", ".join(sorted(path.name for path in remaining.values())),
                )
                return

            self._logger.info("Bootstrap archive applied successfully; loading index")
            self._load_existing_index(paths)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning(
                "Failed to bootstrap index from %s: %s", archive_url, exc
            )

    def _download_to_path(
        self, url: str, destination_dir: Path, max_retries: int = 3
    ) -> Path:
        """Download or copy the given URL into destination_dir with retry logic."""
        parsed = urlparse(url)
        filename = Path(unquote(parsed.path or "")).name or "bootstrap_archive"
        destination_dir.mkdir(parents=True, exist_ok=True)
        target_path = destination_dir / filename

        if parsed.scheme == "file":
            local_path = url2pathname(parsed.path)
            source_path = Path(local_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Bootstrap archive file not found: {source_path}")
            self._logger.info("Copying bootstrap archive from %s", source_path)
            shutil.copy2(source_path, target_path)
            return target_path

        if parsed.scheme in ("http", "https"):
            last_exception: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    if attempt > 1:
                        delay = 2 ** (attempt - 2)
                        self._logger.info(
                            "Retrying download after %ss delay (attempt %s/%s)",
                            delay,
                            attempt,
                            max_retries,
                        )
                        time.sleep(delay)

                    with requests.get(url, stream=True, timeout=30) as response:
                        response.raise_for_status()
                        with open(target_path, "wb") as handle:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    handle.write(chunk)
                    return target_path
                except Exception as exc:  # pragma: no cover - network errors
                    last_exception = exc
                    self._logger.warning("Download failed: %s", exc)
            if last_exception:
                raise last_exception
            raise RuntimeError("Failed to download bootstrap archive")

        raise ValueError(f"Unsupported URL scheme for bootstrap archive: {parsed.scheme}")

    def _fetch_checksum(self, url: str) -> str:
        """Fetch the checksum value from remote location."""
        parsed = urlparse(url)
        if parsed.scheme == "file":
            local_path = url2pathname(parsed.path)
            with open(local_path, "r", encoding="utf-8") as handle:
                content = handle.read()
        elif parsed.scheme in ("http", "https"):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = response.text
        else:
            raise ValueError(f"Unsupported checksum URL scheme: {parsed.scheme}")

        if not content.strip():
            raise ValueError("Bootstrap checksum file is empty")

        parts = content.strip().split()
        if not parts:
            raise ValueError("Bootstrap checksum file is empty")

        checksum = parts[0].lower()
        if len(checksum) != 64 or not all(c in "0123456789abcdef" for c in checksum):
            raise ValueError(
                "Invalid SHA256 checksum format: %s" % checksum[:20]
            )

        self._logger.info("Fetched checksum: %s...", checksum[:16])
        return checksum

    def _verify_checksum(self, archive_path: Path, expected_checksum: str) -> None:
        """Verify archive SHA256 checksum."""
        digest = hashlib.sha256()
        with archive_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual.lower() != expected_checksum.lower():
            raise ValueError(
                f"Bootstrap archive checksum mismatch: expected {expected_checksum}, got {actual}"
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

    def _move_bootstrap_artifacts(self, extract_dir: Path, paths: Dict[str, Path]) -> None:
        """Move extracted files into their final locations if missing."""
        for key in ["metadata", "index"]:
            destination = paths[key]
            if destination.exists():
                continue

            matches = list(extract_dir.rglob(destination.name))
            if not matches:
                self._logger.warning(
                    "Bootstrap archive missing expected %s file %s",
                    key,
                    destination.name,
                )
                continue

            source_path = matches[0]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source_path, destination)
            self._logger.info("Bootstrapped %s", destination.name)
