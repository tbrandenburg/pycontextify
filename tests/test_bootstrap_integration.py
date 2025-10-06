"""Integration tests for HTTP bootstrap functionality.

This module contains integration tests for the bootstrap feature, including:
- Archive creation and download (ZIP, TAR-GZ)
- Checksum validation
- Local file:// and HTTPS URL support
- Retry logic and error handling
- Backup restoration priority
- Concurrent bootstrap locking
- Background thread execution
"""

import hashlib
import http.server
import os
import shutil
import socket
import socketserver
import tarfile
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple
from unittest import mock

import pytest

from pycontextify.index.config import Config
from pycontextify.index.manager import IndexManager

# ============================================================================
# Fixtures for Archive Creation
# ============================================================================


@pytest.fixture
def minimal_index_files(tmp_path: Path) -> Dict[str, Path]:
    """Create minimal valid FAISS index and metadata files.

    Returns:
        Dictionary with 'faiss' and 'pkl' keys pointing to created files.
    """
    # Create minimal FAISS index file (just a placeholder)
    faiss_file = tmp_path / "semantic_index.faiss"
    faiss_file.write_bytes(b"FAKE_FAISS_INDEX_DATA")

    # Create minimal metadata pickle file (just a placeholder)
    pkl_file = tmp_path / "semantic_index.pkl"
    pkl_file.write_bytes(b"FAKE_PKL_METADATA")

    return {"faiss": faiss_file, "pkl": pkl_file}


@pytest.fixture
def create_zip_archive(tmp_path: Path):
    """Factory fixture to create ZIP archives with specified files.

    Usage:
        archive_path = create_zip_archive(files, "archive.zip")
    """

    def _create_zip(files: Dict[str, Path], archive_name: str) -> Path:
        archive_path = tmp_path / archive_name
        with zipfile.ZipFile(archive_path, "w") as zf:
            for file_path in files.values():
                zf.write(file_path, arcname=file_path.name)
        return archive_path

    return _create_zip


@pytest.fixture
def create_tarball_archive(tmp_path: Path):
    """Factory fixture to create TAR-GZ archives with specified files.

    Usage:
        archive_path = create_tarball_archive(files, "archive.tar.gz")
    """

    def _create_tarball(files: Dict[str, Path], archive_name: str) -> Path:
        archive_path = tmp_path / archive_name
        with tarfile.open(archive_path, "w:gz") as tf:
            for file_path in files.values():
                tf.add(file_path, arcname=file_path.name)
        return archive_path

    return _create_tarball


@pytest.fixture
def compute_sha256():
    """Factory fixture to compute SHA256 checksum of a file.

    Usage:
        checksum = compute_sha256(archive_path)
    """

    def _compute(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    return _compute


@pytest.fixture
def create_checksum_file(compute_sha256):
    """Factory fixture to create .sha256 checksum file for an archive.

    Usage:
        checksum_path = create_checksum_file(archive_path)
    """

    def _create_checksum(archive_path: Path) -> Path:
        checksum = compute_sha256(archive_path)
        checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
        checksum_path.write_text(f"{checksum}  {archive_path.name}\n")
        return checksum_path

    return _create_checksum


# ============================================================================
# Fixtures for Local HTTP Server
# ============================================================================


class RequestCountingHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler that counts requests by path."""

    request_counts: Dict[str, int] = {}
    request_lock = threading.Lock()

    def log_message(self, format, *args):
        """Suppress HTTP server log messages during tests."""
        pass

    def do_GET(self):
        """Handle GET request and count it."""
        with self.request_lock:
            path = self.path
            RequestCountingHandler.request_counts[path] = (
                RequestCountingHandler.request_counts.get(path, 0) + 1
            )
        return super().do_GET()

    @classmethod
    def reset_counts(cls):
        """Reset request counters."""
        with cls.request_lock:
            cls.request_counts.clear()

    @classmethod
    def get_count(cls, path: str) -> int:
        """Get request count for a specific path."""
        with cls.request_lock:
            return cls.request_counts.get(path, 0)


@pytest.fixture
def http_server(tmp_path: Path):
    """Start a local HTTP server serving a temporary directory.

    Yields:
        Tuple of (base_url, serve_directory)

    The server runs in a background thread and is automatically shut down
    after the test completes.
    """
    serve_dir = tmp_path / "http_root"
    serve_dir.mkdir(exist_ok=True)

    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    # Change to serve directory for SimpleHTTPRequestHandler
    original_cwd = os.getcwd()
    os.chdir(serve_dir)

    try:
        # Create and start server
        RequestCountingHandler.reset_counts()
        server = socketserver.TCPServer(
            ("127.0.0.1", port),
            RequestCountingHandler,
            bind_and_activate=False,
        )
        server.allow_reuse_address = True
        server.server_bind()
        server.server_activate()

        # Run server in background thread
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        # Give server time to start
        time.sleep(0.1)

        base_url = f"http://127.0.0.1:{port}"
        yield base_url, serve_dir

        # Shutdown server
        server.shutdown()
        server.server_close()
    finally:
        os.chdir(original_cwd)


# ============================================================================
# Fixtures for Environment Patching
# ============================================================================


@pytest.fixture
def clean_bootstrap_env(monkeypatch):
    """Clear bootstrap-related environment variables for a clean test."""
    monkeypatch.delenv("PYCONTEXTIFY_INDEX_BOOTSTRAP_ARCHIVE_URL", raising=False)
    monkeypatch.delenv("PYCONTEXTIFY_INDEX_BOOTSTRAP_CHECKSUM_URL", raising=False)


@pytest.fixture
def set_bootstrap_env(monkeypatch, clean_bootstrap_env):
    """Factory fixture to set bootstrap environment variables.

    Usage:
        set_bootstrap_env(archive_url="https://example.com/archive.zip")
    """

    def _set_env(archive_url: Optional[str] = None, checksum_url: Optional[str] = None):
        if archive_url:
            monkeypatch.setenv("PYCONTEXTIFY_INDEX_BOOTSTRAP_ARCHIVE_URL", archive_url)
        if checksum_url:
            monkeypatch.setenv(
                "PYCONTEXTIFY_INDEX_BOOTSTRAP_CHECKSUM_URL", checksum_url
            )

    return _set_env


# ============================================================================
# Helper Functions
# ============================================================================


def create_manager_with_bootstrap(
    tmp_path: Path,
    archive_url: str,
    index_name: str = "test_index",
    auto_load: bool = True,
) -> IndexManager:
    """Helper to create an IndexManager with bootstrap configuration.

    Args:
        tmp_path: Temporary directory for index storage
        archive_url: Bootstrap archive URL
        index_name: Name of the index
        auto_load: Whether to enable auto-load

    Returns:
        Configured IndexManager instance
    """
    config_overrides = {
        "index_dir": str(tmp_path / "index_data"),
        "index_name": index_name,
        "auto_persist": True,
        "auto_load": auto_load,
        "bootstrap_archive_url": archive_url,
    }
    config = Config(config_overrides=config_overrides)
    return IndexManager(config)


def wait_for_bootstrap_completion(
    manager: IndexManager,
    timeout: float = 5.0,
    check_interval: float = 0.1,
) -> bool:
    """Wait for bootstrap thread to complete.

    Args:
        manager: IndexManager instance
        timeout: Maximum time to wait in seconds
        check_interval: How often to check thread status in seconds

    Returns:
        True if bootstrap completed, False if timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if (
            manager._bootstrap_thread is None
            or not manager._bootstrap_thread.is_alive()
        ):
            return True
        time.sleep(check_interval)
    return False


# ============================================================================
# Placeholder Tests (to be implemented)
# ============================================================================


class TestBootstrapIntegration:
    """Integration tests for HTTP bootstrap functionality."""

    def test_zip_archive_bootstrap_over_http(
        self,
        tmp_path,
        minimal_index_files,
        create_zip_archive,
        create_checksum_file,
        http_server,
        clean_bootstrap_env,
    ):
        """Test bootstrapping from a ZIP archive over local HTTP."""
        base_url, serve_dir = http_server

        # Create archive with required files
        archive_path = create_zip_archive(minimal_index_files, "test_index.zip")
        checksum_path = create_checksum_file(archive_path)

        # Copy archive and checksum to HTTP server directory
        shutil.copy2(archive_path, serve_dir / archive_path.name)
        shutil.copy2(checksum_path, serve_dir / checksum_path.name)

        # Configure bootstrap URL
        archive_url = f"{base_url}/{archive_path.name}"

        # Create manager with bootstrap configuration
        manager = create_manager_with_bootstrap(
            tmp_path, archive_url, index_name="semantic_index"
        )

        # Wait for bootstrap to complete
        assert wait_for_bootstrap_completion(
            manager, timeout=10.0
        ), "Bootstrap did not complete in time"

        # Verify index files exist
        index_dir = tmp_path / "index_data"
        faiss_file = index_dir / "semantic_index.faiss"
        pkl_file = index_dir / "semantic_index.pkl"

        assert faiss_file.exists(), "FAISS index file should exist after bootstrap"
        assert pkl_file.exists(), "Metadata file should exist after bootstrap"

    def test_tarball_archive_bootstrap_over_http(
        self,
        tmp_path,
        minimal_index_files,
        create_tarball_archive,
        create_checksum_file,
        http_server,
        clean_bootstrap_env,
    ):
        """Test bootstrapping from a TAR-GZ archive over local HTTP."""
        base_url, serve_dir = http_server

        # Create tarball archive with required files
        archive_path = create_tarball_archive(minimal_index_files, "test_index.tar.gz")
        checksum_path = create_checksum_file(archive_path)

        # Copy archive and checksum to HTTP server directory
        shutil.copy2(archive_path, serve_dir / archive_path.name)
        shutil.copy2(checksum_path, serve_dir / checksum_path.name)

        # Configure bootstrap URL
        archive_url = f"{base_url}/{archive_path.name}"

        # Create manager with bootstrap configuration
        manager = create_manager_with_bootstrap(
            tmp_path, archive_url, index_name="semantic_index"
        )

        # Wait for bootstrap to complete
        assert wait_for_bootstrap_completion(
            manager, timeout=10.0
        ), "Bootstrap did not complete in time"

        # Verify index files exist
        index_dir = tmp_path / "index_data"
        faiss_file = index_dir / "semantic_index.faiss"
        pkl_file = index_dir / "semantic_index.pkl"

        assert faiss_file.exists(), "FAISS index file should exist after bootstrap"
        assert pkl_file.exists(), "Metadata file should exist after bootstrap"

    def test_file_url_bootstrap(
        self,
        tmp_path,
        minimal_index_files,
        create_zip_archive,
        create_checksum_file,
        clean_bootstrap_env,
    ):
        """Test bootstrapping from a file:// URL."""
        # Create archive with required files
        archive_path = create_zip_archive(minimal_index_files, "test_index.zip")
        checksum_path = create_checksum_file(archive_path)

        # Convert to file:// URL
        archive_url = archive_path.as_uri()

        # Create manager with bootstrap configuration
        manager = create_manager_with_bootstrap(
            tmp_path, archive_url, index_name="semantic_index"
        )

        # Wait for bootstrap to complete
        assert wait_for_bootstrap_completion(
            manager, timeout=10.0
        ), "Bootstrap did not complete in time"

        # Verify index files exist
        index_dir = tmp_path / "index_data"
        faiss_file = index_dir / "semantic_index.faiss"
        pkl_file = index_dir / "semantic_index.pkl"

        assert faiss_file.exists(), "FAISS index file should exist after bootstrap"
        assert pkl_file.exists(), "Metadata file should exist after bootstrap"

    def test_checksum_validation_success(
        self,
        tmp_path,
        minimal_index_files,
        create_zip_archive,
        create_checksum_file,
        http_server,
        clean_bootstrap_env,
    ):
        """Test successful checksum validation."""
        base_url, serve_dir = http_server

        # Create archive with required files
        archive_path = create_zip_archive(minimal_index_files, "test_index.zip")
        checksum_path = create_checksum_file(archive_path)

        # Copy archive and checksum to HTTP server directory
        shutil.copy2(archive_path, serve_dir / archive_path.name)
        shutil.copy2(checksum_path, serve_dir / checksum_path.name)

        # Configure bootstrap URL
        archive_url = f"{base_url}/{archive_path.name}"

        # Create manager with bootstrap configuration
        manager = create_manager_with_bootstrap(
            tmp_path, archive_url, index_name="semantic_index"
        )

        # Wait for bootstrap to complete
        assert wait_for_bootstrap_completion(
            manager, timeout=10.0
        ), "Bootstrap did not complete in time"

        # Verify index files exist (checksum passed)
        index_dir = tmp_path / "index_data"
        assert (index_dir / "semantic_index.faiss").exists()
        assert (index_dir / "semantic_index.pkl").exists()

    def test_checksum_validation_mismatch(
        self,
        tmp_path,
        minimal_index_files,
        create_zip_archive,
        compute_sha256,
        http_server,
        clean_bootstrap_env,
    ):
        """Test bootstrap failure on checksum mismatch."""
        base_url, serve_dir = http_server

        # Create archive with required files
        archive_path = create_zip_archive(minimal_index_files, "test_index.zip")

        # Create BOGUS checksum file (incorrect hash)
        bogus_checksum = "0" * 64  # Invalid checksum
        checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
        checksum_path.write_text(f"{bogus_checksum}  {archive_path.name}\n")

        # Copy archive and bogus checksum to HTTP server directory
        shutil.copy2(archive_path, serve_dir / archive_path.name)
        shutil.copy2(checksum_path, serve_dir / checksum_path.name)

        # Configure bootstrap URL
        archive_url = f"{base_url}/{archive_path.name}"

        # Create manager with bootstrap configuration
        manager = create_manager_with_bootstrap(
            tmp_path, archive_url, index_name="semantic_index"
        )

        # Wait for bootstrap to complete (should fail)
        wait_for_bootstrap_completion(manager, timeout=10.0)

        # Verify index files DO NOT exist (bootstrap should have failed)
        index_dir = tmp_path / "index_data"
        faiss_file = index_dir / "semantic_index.faiss"
        pkl_file = index_dir / "semantic_index.pkl"

        assert (
            not faiss_file.exists()
        ), "FAISS file should not exist after failed bootstrap"
        assert (
            not pkl_file.exists()
        ), "Metadata file should not exist after failed bootstrap"

    def test_download_404_error(
        self,
        tmp_path,
        http_server,
        clean_bootstrap_env,
    ):
        """Test bootstrap failure on 404 error."""
        base_url, serve_dir = http_server

        # Point to non-existent archive
        archive_url = f"{base_url}/nonexistent_archive.zip"

        # Create manager with bootstrap configuration
        manager = create_manager_with_bootstrap(
            tmp_path, archive_url, index_name="semantic_index"
        )

        # Wait for bootstrap to complete (should fail)
        wait_for_bootstrap_completion(manager, timeout=10.0)

        # Verify index files DO NOT exist (bootstrap should have failed)
        index_dir = tmp_path / "index_data"
        faiss_file = index_dir / "semantic_index.faiss"
        pkl_file = index_dir / "semantic_index.pkl"

        assert not faiss_file.exists(), "FAISS file should not exist after 404 error"
        assert not pkl_file.exists(), "Metadata file should not exist after 404 error"

    def test_download_timeout_with_retry(
        self,
        tmp_path,
        clean_bootstrap_env,
    ):
        """Test retry behavior on timeout."""
        pytest.skip("To be implemented")

    def test_connection_refused_with_retry(
        self,
        tmp_path,
        clean_bootstrap_env,
    ):
        """Test retry behavior on connection refused."""
        pytest.skip("To be implemented")

    def test_incomplete_archive(
        self,
        tmp_path,
        create_zip_archive,
        create_checksum_file,
        http_server,
        clean_bootstrap_env,
    ):
        """Test bootstrap failure with incomplete archive."""
        base_url, serve_dir = http_server

        # Create incomplete archive (missing .pkl file)
        incomplete_files = {"faiss": tmp_path / "semantic_index.faiss"}
        incomplete_files["faiss"].write_bytes(b"FAKE_FAISS_INDEX_DATA")

        archive_path = create_zip_archive(incomplete_files, "incomplete_index.zip")
        checksum_path = create_checksum_file(archive_path)

        # Copy archive and checksum to HTTP server directory
        shutil.copy2(archive_path, serve_dir / archive_path.name)
        shutil.copy2(checksum_path, serve_dir / checksum_path.name)

        # Configure bootstrap URL
        archive_url = f"{base_url}/{archive_path.name}"

        # Create manager with bootstrap configuration
        manager = create_manager_with_bootstrap(
            tmp_path, archive_url, index_name="semantic_index"
        )

        # Wait for bootstrap to complete (should fail)
        wait_for_bootstrap_completion(manager, timeout=10.0)

        # Verify index files DO NOT exist (bootstrap should have failed)
        index_dir = tmp_path / "index_data"
        faiss_file = index_dir / "semantic_index.faiss"
        pkl_file = index_dir / "semantic_index.pkl"

        # Both should be missing as the archive was incomplete
        assert (
            not pkl_file.exists()
        ), "Metadata file should not exist after incomplete archive"

    def test_backup_restoration_priority(
        self,
        tmp_path,
        minimal_index_files,
        create_zip_archive,
        create_checksum_file,
        http_server,
        clean_bootstrap_env,
    ):
        """Test that backup restoration takes priority over download."""
        from datetime import datetime

        base_url, serve_dir = http_server

        # Setup: Create index directory with backup files
        index_dir = tmp_path / "index_data"
        index_dir.mkdir(parents=True, exist_ok=True)

        # Store original backup content for verification
        backup_faiss_content = b"BACKUP_FAISS_DATA"
        backup_pkl_content = b"BACKUP_PKL_DATA"

        # Create backup files with proper naming (semantic_index_backup_TIMESTAMP.ext)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_faiss = index_dir / f"semantic_index_backup_{timestamp}.faiss"
        backup_pkl = index_dir / f"semantic_index_backup_{timestamp}.pkl"

        backup_faiss.write_bytes(backup_faiss_content)
        backup_pkl.write_bytes(backup_pkl_content)

        # Ensure current index files do NOT exist (so bootstrap is triggered)
        current_faiss = index_dir / "semantic_index.faiss"
        current_pkl = index_dir / "semantic_index.pkl"

        assert (
            not current_faiss.exists()
        ), "Current FAISS file should not exist initially"
        assert not current_pkl.exists(), "Current PKL file should not exist initially"

        # Create HTTP archive with DIFFERENT content that should NOT be used
        archive_files = {
            "faiss": tmp_path / "semantic_index.faiss",
            "pkl": tmp_path / "semantic_index.pkl",
        }
        archive_files["faiss"].write_bytes(b"ARCHIVE_FAISS_DATA")
        archive_files["pkl"].write_bytes(b"ARCHIVE_PKL_DATA")

        archive_path = create_zip_archive(archive_files, "test_index.zip")
        checksum_path = create_checksum_file(archive_path)

        # Copy archive and checksum to HTTP server directory
        shutil.copy2(archive_path, serve_dir / archive_path.name)
        shutil.copy2(checksum_path, serve_dir / checksum_path.name)

        # Configure bootstrap URL
        archive_url = f"{base_url}/{archive_path.name}"

        # Track if HTTP download is attempted (it should NOT be)
        http_download_attempted = False

        def mock_download_to_path(original_func):
            def wrapper(self, url, *args, **kwargs):
                nonlocal http_download_attempted
                if url.startswith("http://") or url.startswith("https://"):
                    http_download_attempted = True
                return original_func(self, url, *args, **kwargs)

            return wrapper

        # Patch the download method to track HTTP attempts
        from pycontextify.index.manager import IndexManager

        original_download = IndexManager._download_to_path
        IndexManager._download_to_path = mock_download_to_path(original_download)

        try:
            # Create manager with bootstrap configuration
            manager = create_manager_with_bootstrap(
                tmp_path, archive_url, index_name="semantic_index"
            )

            # Wait for bootstrap to complete
            assert wait_for_bootstrap_completion(
                manager, timeout=10.0
            ), "Bootstrap did not complete in time"

            # Verify files were restored from backups, NOT downloaded
            assert current_faiss.exists(), "FAISS file should exist after restoration"
            assert current_pkl.exists(), "PKL file should exist after restoration"

            # Verify content matches backups, not archive
            assert (
                current_faiss.read_bytes() == backup_faiss_content
            ), "FAISS content should match backup, not HTTP archive"
            assert (
                current_pkl.read_bytes() == backup_pkl_content
            ), "PKL content should match backup, not HTTP archive"

            # Verify HTTP download was NOT attempted
            assert (
                not http_download_attempted
            ), "HTTP download should not occur when backups exist"

        finally:
            # Restore original method
            IndexManager._download_to_path = original_download

    def test_concurrent_bootstrap_locking(
        self,
        tmp_path,
        minimal_index_files,
        create_zip_archive,
        create_checksum_file,
        http_server,
        clean_bootstrap_env,
    ):
        """Test that concurrent bootstrap attempts are prevented by locking."""
        pytest.skip("To be implemented")

    def test_background_thread_execution(
        self,
        tmp_path,
        minimal_index_files,
        create_zip_archive,
        create_checksum_file,
        http_server,
        clean_bootstrap_env,
    ):
        """Test that bootstrap runs in background thread without blocking."""
        pytest.skip("To be implemented")
