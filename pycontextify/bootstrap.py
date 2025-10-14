"""Bootstrap service for downloading and extracting index archives.

This service handles the bootstrap process for initializing index artifacts
from remote or local archive sources, including download retry logic,
checksum verification, and archive extraction.
"""

import hashlib
import logging
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import requests

logger = logging.getLogger(__name__)


class BootstrapService:
    """Manages bootstrap index downloading and extraction."""

    def __init__(self, config):
        """Initialize bootstrap service with configuration.

        Args:
            config: Configuration object with bootstrap settings
        """
        self.config = config

    def bootstrap_if_needed(self, index_paths: Dict[str, Path]) -> bool:
        """Bootstrap index if artifacts are missing and sources configured.

        Args:
            index_paths: Dictionary mapping artifact names to paths

        Returns:
            True if bootstrap succeeded or not needed, False if failed
        """
        # Check if essential artifacts are missing
        essential_paths = {
            k: v for k, v in index_paths.items() if k in ["metadata", "index"]
        }
        missing_paths = {k: v for k, v in essential_paths.items() if not v.exists()}

        if not missing_paths:
            logger.info("All index artifacts exist, no bootstrap needed")
            return True

        # Get bootstrap sources from config
        sources = self.config.get_bootstrap_sources()
        if not sources or not sources.get("archive") or not sources.get("checksum"):
            logger.info("Bootstrap not configured, skipping")
            return False

        logger.info(
            f"Starting bootstrap for missing artifacts: "
            f"{', '.join(p.name for p in missing_paths.values())}"
        )

        try:
            with TemporaryDirectory(prefix="pycontextify-bootstrap-") as tmpdir:
                temp_dir = Path(tmpdir)

                # Step 1: Download archive
                archive_path = self._download_archive(sources["archive"], temp_dir)

                # Step 2: Fetch and verify checksum
                checksum_value = self._fetch_checksum(sources["checksum"])
                self._verify_checksum(archive_path, checksum_value)

                # Step 3: Extract archive
                extract_dir = temp_dir / "extracted"
                extract_dir.mkdir(parents=True, exist_ok=True)
                self._extract_archive(archive_path, extract_dir)

                # Step 4: Move artifacts to final locations
                self._move_artifacts(extract_dir, index_paths)

            logger.info("Bootstrap completed successfully")
            return True

        except Exception as exc:
            logger.error(f"Bootstrap failed: {exc}")
            return False

    def _download_archive(self, url: str, dest_dir: Path, max_retries: int = 3) -> Path:
        """Download archive with retry logic.

        Args:
            url: URL to download from (http://, https://, or file://)
            dest_dir: Directory to save the downloaded file
            max_retries: Maximum number of retry attempts

        Returns:
            Path to the downloaded file

        Raises:
            FileNotFoundError: If file:// URL points to non-existent file
            ValueError: If URL scheme is not supported
            Exception: After max retries exhausted for transient errors
        """
        parsed = urlparse(url)
        filename = Path(unquote(parsed.path or "")).name or "bootstrap_archive"
        dest_dir.mkdir(parents=True, exist_ok=True)
        target_path = dest_dir / filename

        # Handle file:// URLs (no retry needed)
        if parsed.scheme == "file":
            # Handle Windows paths properly - handle both correct and malformed file URLs
            if parsed.netloc and not parsed.path:
                # Malformed URL like file://C:\path - netloc has the path
                local_path = parsed.netloc
            elif parsed.path:
                # Correct URL like file:///C:\path - path has the path
                local_path = url2pathname(parsed.path)
            else:
                # Fallback - try to extract from URL directly
                local_path = url.replace("file://", "", 1)
                if (
                    local_path.startswith("/")
                    and len(local_path) > 1
                    and local_path[2] == ":"
                ):
                    # Remove leading slash from /C: style paths
                    local_path = local_path[1:]

            source_path = Path(local_path)
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Bootstrap archive file not found: {source_path}"
                )
            logger.info(f"Copying bootstrap archive from {source_path}")
            shutil.copy2(source_path, target_path)
            return target_path

        # Handle http(s):// URLs with retry
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    delay = 2 ** (attempt - 2)
                    logger.info(
                        f"Retrying download after {delay}s delay "
                        f"(attempt {attempt}/{max_retries})"
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
                with temp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                # Atomic rename
                os.replace(temp_path, target_path)
                logger.info("Download completed successfully")
                return target_path

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(
                    f"Download timeout on attempt {attempt}/{max_retries}: {e}"
                )
                continue

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(
                    f"Connection error on attempt {attempt}/{max_retries}: {e}"
                )
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
                        continue
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
                    f"Unexpected error during download "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                raise

        # All retries exhausted
        error_msg = (
            f"Failed to download {url} after {max_retries} attempts. "
            f"Last error: {last_exception}"
        )
        logger.error(error_msg)
        raise Exception(error_msg)

    def _fetch_checksum(self, url: str, max_retries: int = 3) -> str:
        """Retrieve checksum text from the provided URL with retry logic.

        Args:
            url: URL to fetch checksum from (http://, https://, or file://)
            max_retries: Maximum number of retry attempts

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
            # Handle Windows paths properly - handle both correct and malformed file URLs
            if parsed.netloc and not parsed.path:
                # Malformed URL like file://C:\path - netloc has the path
                local_path = parsed.netloc
            elif parsed.path:
                # Correct URL like file:///C:\path - path has the path
                local_path = url2pathname(parsed.path)
            else:
                # Fallback - try to extract from URL directly
                local_path = url.replace("file://", "", 1)
                if (
                    local_path.startswith("/")
                    and len(local_path) > 1
                    and local_path[2] == ":"
                ):
                    # Remove leading slash from /C: style paths
                    local_path = local_path[1:]

            checksum_path = Path(local_path)
            if not checksum_path.exists():
                raise FileNotFoundError(
                    f"Bootstrap checksum file not found: {checksum_path}"
                )
            content = checksum_path.read_text(encoding="utf-8")

        # Handle http(s):// URLs with retry
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
        """Verify archive SHA256 checksum.

        Args:
            archive_path: Path to archive file
            expected_checksum: Expected SHA256 hex digest

        Raises:
            ValueError: If checksum doesn't match
        """
        digest = hashlib.sha256()
        with archive_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)

        actual_checksum = digest.hexdigest()
        if actual_checksum.lower() != expected_checksum.lower():
            raise ValueError(
                "Bootstrap archive checksum mismatch: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )
        logger.info("Checksum verification passed")

    def _extract_archive(self, archive_path: Path, destination: Path) -> None:
        """Extract supported archive formats into destination directory.

        Args:
            archive_path: Path to archive file
            destination: Directory to extract to

        Raises:
            ValueError: If archive format is not supported
        """
        lower_name = archive_path.name.lower()

        if lower_name.endswith(".zip"):
            logger.info(f"Extracting ZIP archive: {archive_path.name}")
            with zipfile.ZipFile(archive_path) as zip_ref:
                zip_ref.extractall(destination)
            return

        if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz"):
            logger.info(f"Extracting TAR.GZ archive: {archive_path.name}")
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(destination)
            return

        raise ValueError(f"Unsupported bootstrap archive format: {archive_path}")

    def _move_artifacts(self, extract_dir: Path, index_paths: Dict[str, Path]) -> None:
        """Move extracted files into their final locations if missing.

        Args:
            extract_dir: Directory containing extracted files
            index_paths: Dictionary mapping artifact names to final paths
        """
        for key in ["metadata", "index"]:
            destination = index_paths[key]
            if destination.exists():
                logger.debug(f"Artifact {destination.name} already exists, skipping")
                continue

            matches = list(extract_dir.rglob(destination.name))
            if not matches:
                logger.warning(
                    f"Bootstrap archive missing expected {key} file {destination.name}"
                )
                continue

            source_path = matches[0]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source_path, destination)
            logger.info(f"Bootstrapped {destination.name}")
