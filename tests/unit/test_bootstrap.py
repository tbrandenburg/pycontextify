"""Unit tests for BootstrapService."""

import hashlib
import tarfile
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from pycontextify.bootstrap import BootstrapService


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.get_bootstrap_sources.return_value = {
        "archive": "http://example.com/archive.zip",
        "checksum": "http://example.com/checksum.txt",
    }
    return config


@pytest.fixture
def bootstrap_service(mock_config):
    """Create BootstrapService instance."""
    return BootstrapService(mock_config)


@pytest.fixture
def temp_index_paths():
    """Create temporary index paths."""
    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        yield {
            "metadata": base / "metadata.json",
            "index": base / "index.faiss",
        }


class TestBootstrapIfNeeded:
    """Tests for bootstrap_if_needed method."""

    def test_skips_when_artifacts_exist(self, bootstrap_service, temp_index_paths):
        """Should skip bootstrap when all artifacts exist."""
        # Create files
        temp_index_paths["metadata"].touch()
        temp_index_paths["index"].touch()

        result = bootstrap_service.bootstrap_if_needed(temp_index_paths)

        assert result is True

    def test_skips_when_not_configured(self, mock_config, temp_index_paths):
        """Should skip bootstrap when sources not configured."""
        mock_config.get_bootstrap_sources.return_value = {}
        service = BootstrapService(mock_config)

        result = service.bootstrap_if_needed(temp_index_paths)

        assert result is False

    def test_skips_when_sources_incomplete(self, mock_config, temp_index_paths):
        """Should skip when sources missing archive or checksum."""
        mock_config.get_bootstrap_sources.return_value = {"archive": "url"}
        service = BootstrapService(mock_config)

        result = service.bootstrap_if_needed(temp_index_paths)

        assert result is False

    @patch.object(BootstrapService, "_download_archive")
    @patch.object(BootstrapService, "_fetch_checksum")
    @patch.object(BootstrapService, "_verify_checksum")
    @patch.object(BootstrapService, "_extract_archive")
    @patch.object(BootstrapService, "_move_artifacts")
    def test_successful_bootstrap(
        self,
        mock_move,
        mock_extract,
        mock_verify,
        mock_fetch,
        mock_download,
        bootstrap_service,
        temp_index_paths,
    ):
        """Should successfully bootstrap when artifacts missing."""
        mock_download.return_value = Path("/tmp/archive.zip")
        mock_fetch.return_value = "abc123"

        result = bootstrap_service.bootstrap_if_needed(temp_index_paths)

        assert result is True
        mock_download.assert_called_once()
        mock_fetch.assert_called_once()
        mock_verify.assert_called_once()
        mock_extract.assert_called_once()
        mock_move.assert_called_once()

    @patch.object(BootstrapService, "_download_archive")
    def test_returns_false_on_download_failure(
        self, mock_download, bootstrap_service, temp_index_paths
    ):
        """Should return False if download fails."""
        mock_download.side_effect = Exception("Download failed")

        result = bootstrap_service.bootstrap_if_needed(temp_index_paths)

        assert result is False


class TestDownloadArchive:
    """Tests for _download_archive method."""

    def test_file_url_success(self, bootstrap_service):
        """Should copy file from file:// URL."""
        with TemporaryDirectory() as tmpdir:
            # Create source file
            source = Path(tmpdir) / "source.zip"
            source.write_bytes(b"test data")

            # Download to destination
            dest_dir = Path(tmpdir) / "dest"
            url = f"file://{source}"

            result = bootstrap_service._download_archive(url, dest_dir)

            assert result.exists()
            assert result.read_bytes() == b"test data"

    def test_file_url_not_found(self, bootstrap_service):
        """Should raise FileNotFoundError for non-existent file."""
        with TemporaryDirectory() as tmpdir:
            dest_dir = Path(tmpdir)
            url = "file:///nonexistent/file.zip"

            with pytest.raises(FileNotFoundError):
                bootstrap_service._download_archive(url, dest_dir)

    @patch("requests.get")
    def test_http_url_success(self, mock_get, bootstrap_service):
        """Should download from http:// URL."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_get.return_value = mock_response

        with TemporaryDirectory() as tmpdir:
            dest_dir = Path(tmpdir)
            url = "http://example.com/archive.zip"

            result = bootstrap_service._download_archive(url, dest_dir)

            assert result.exists()
            assert result.read_bytes() == b"chunk1chunk2"
            mock_get.assert_called_once_with(url, stream=True, timeout=30)

    @patch("requests.get")
    def test_http_retry_on_timeout(self, mock_get, bootstrap_service):
        """Should retry on timeout."""
        # First call times out, second succeeds
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"data"]

        mock_get.side_effect = [
            requests.exceptions.Timeout("Timeout"),
            mock_response,
        ]

        with TemporaryDirectory() as tmpdir:
            dest_dir = Path(tmpdir)
            url = "http://example.com/archive.zip"

            result = bootstrap_service._download_archive(url, dest_dir, max_retries=2)

            assert result.exists()
            assert mock_get.call_count == 2

    @patch("requests.get")
    def test_http_retry_on_connection_error(self, mock_get, bootstrap_service):
        """Should retry on connection error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"data"]

        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            mock_response,
        ]

        with TemporaryDirectory() as tmpdir:
            dest_dir = Path(tmpdir)
            url = "http://example.com/archive.zip"

            result = bootstrap_service._download_archive(url, dest_dir, max_retries=2)

            assert result.exists()
            assert mock_get.call_count == 2

    @patch("requests.get")
    def test_http_retry_on_5xx_error(self, mock_get, bootstrap_service):
        """Should retry on 5xx errors."""
        # First call: raise HTTPError with 500 status
        mock_error_response = Mock()
        mock_error_response.status_code = 500
        mock_error_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_error_response
        )

        # Second call: succeed
        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.iter_content.return_value = [b"data"]
        mock_success_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_error_response, mock_success_response]

        with TemporaryDirectory() as tmpdir:
            dest_dir = Path(tmpdir)
            url = "http://example.com/archive.zip"

            result = bootstrap_service._download_archive(url, dest_dir, max_retries=2)

            assert result.exists()
            assert mock_get.call_count == 2

    @patch("requests.get")
    def test_http_fails_after_max_retries(self, mock_get, bootstrap_service):
        """Should fail after max retries exhausted."""
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")

        with TemporaryDirectory() as tmpdir:
            dest_dir = Path(tmpdir)
            url = "http://example.com/archive.zip"

            with pytest.raises(Exception, match="Failed to download"):
                bootstrap_service._download_archive(url, dest_dir, max_retries=2)

            assert mock_get.call_count == 2

    def test_unsupported_scheme(self, bootstrap_service):
        """Should raise ValueError for unsupported URL scheme."""
        with TemporaryDirectory() as tmpdir:
            dest_dir = Path(tmpdir)
            url = "ftp://example.com/archive.zip"

            with pytest.raises(ValueError, match="Unsupported URL scheme"):
                bootstrap_service._download_archive(url, dest_dir)


class TestFetchChecksum:
    """Tests for _fetch_checksum method."""

    def test_file_url_success(self, bootstrap_service):
        """Should read checksum from file:// URL."""
        with TemporaryDirectory() as tmpdir:
            checksum_file = Path(tmpdir) / "checksum.txt"
            expected = "a" * 64
            checksum_file.write_text(expected)

            url = f"file://{checksum_file}"
            result = bootstrap_service._fetch_checksum(url)

            assert result == expected

    def test_file_url_not_found(self, bootstrap_service):
        """Should raise FileNotFoundError for missing file."""
        url = "file:///nonexistent/checksum.txt"

        with pytest.raises(FileNotFoundError):
            bootstrap_service._fetch_checksum(url)

    @patch("requests.get")
    def test_http_url_success(self, mock_get, bootstrap_service):
        """Should fetch checksum from http:// URL."""
        expected = "b" * 64
        mock_response = Mock()
        mock_response.text = expected
        mock_get.return_value = mock_response

        url = "http://example.com/checksum.txt"
        result = bootstrap_service._fetch_checksum(url)

        assert result == expected

    def test_handles_checksum_with_filename(self, bootstrap_service):
        """Should parse checksum with filename format."""
        with TemporaryDirectory() as tmpdir:
            checksum_file = Path(tmpdir) / "checksum.txt"
            expected = "c" * 64
            checksum_file.write_text(f"{expected}  archive.zip")

            url = f"file://{checksum_file}"
            result = bootstrap_service._fetch_checksum(url)

            assert result == expected

    def test_empty_checksum_file(self, bootstrap_service):
        """Should raise ValueError for empty checksum file."""
        with TemporaryDirectory() as tmpdir:
            checksum_file = Path(tmpdir) / "checksum.txt"
            checksum_file.write_text("")

            url = f"file://{checksum_file}"

            with pytest.raises(ValueError, match="empty"):
                bootstrap_service._fetch_checksum(url)

    def test_invalid_checksum_format(self, bootstrap_service):
        """Should raise ValueError for invalid checksum."""
        with TemporaryDirectory() as tmpdir:
            checksum_file = Path(tmpdir) / "checksum.txt"
            checksum_file.write_text("not_a_valid_checksum")

            url = f"file://{checksum_file}"

            with pytest.raises(ValueError, match="Invalid SHA256"):
                bootstrap_service._fetch_checksum(url)

    def test_unsupported_scheme(self, bootstrap_service):
        """Should raise ValueError for unsupported scheme."""
        url = "ftp://example.com/checksum.txt"

        with pytest.raises(ValueError, match="Unsupported checksum URL scheme"):
            bootstrap_service._fetch_checksum(url)


class TestVerifyChecksum:
    """Tests for _verify_checksum method."""

    def test_valid_checksum(self, bootstrap_service):
        """Should pass for valid checksum."""
        with TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "archive.zip"
            data = b"test data"
            archive.write_bytes(data)

            expected = hashlib.sha256(data).hexdigest()

            # Should not raise
            bootstrap_service._verify_checksum(archive, expected)

    def test_invalid_checksum(self, bootstrap_service):
        """Should raise ValueError for invalid checksum."""
        with TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "archive.zip"
            archive.write_bytes(b"test data")

            wrong_checksum = "a" * 64

            with pytest.raises(ValueError, match="checksum mismatch"):
                bootstrap_service._verify_checksum(archive, wrong_checksum)


class TestExtractArchive:
    """Tests for _extract_archive method."""

    def test_extract_zip(self, bootstrap_service):
        """Should extract ZIP archive."""
        with TemporaryDirectory() as tmpdir:
            # Create ZIP archive
            archive_path = Path(tmpdir) / "archive.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("test.txt", "content")

            # Extract
            extract_dir = Path(tmpdir) / "extracted"
            extract_dir.mkdir()
            bootstrap_service._extract_archive(archive_path, extract_dir)

            # Verify
            extracted_file = extract_dir / "test.txt"
            assert extracted_file.exists()
            assert extracted_file.read_text() == "content"

    def test_extract_tar_gz(self, bootstrap_service):
        """Should extract TAR.GZ archive."""
        with TemporaryDirectory() as tmpdir:
            # Create TAR.GZ archive
            archive_path = Path(tmpdir) / "archive.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                # Create a temporary file to add
                test_file = Path(tmpdir) / "test.txt"
                test_file.write_text("content")
                tar.add(test_file, arcname="test.txt")

            # Extract
            extract_dir = Path(tmpdir) / "extracted"
            extract_dir.mkdir()
            bootstrap_service._extract_archive(archive_path, extract_dir)

            # Verify
            extracted_file = extract_dir / "test.txt"
            assert extracted_file.exists()
            assert extracted_file.read_text() == "content"

    def test_unsupported_format(self, bootstrap_service):
        """Should raise ValueError for unsupported format."""
        with TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "archive.rar"
            archive_path.touch()
            extract_dir = Path(tmpdir) / "extracted"
            extract_dir.mkdir()

            with pytest.raises(ValueError, match="Unsupported"):
                bootstrap_service._extract_archive(archive_path, extract_dir)


class TestMoveArtifacts:
    """Tests for _move_artifacts method."""

    def test_moves_artifacts(self, bootstrap_service):
        """Should move artifacts to final locations."""
        with TemporaryDirectory() as tmpdir:
            # Setup
            extract_dir = Path(tmpdir) / "extracted"
            extract_dir.mkdir()
            (extract_dir / "metadata.json").write_text("{}")
            (extract_dir / "index.faiss").write_bytes(b"index")

            dest_dir = Path(tmpdir) / "dest"
            index_paths = {
                "metadata": dest_dir / "metadata.json",
                "index": dest_dir / "index.faiss",
            }

            # Execute
            bootstrap_service._move_artifacts(extract_dir, index_paths)

            # Verify
            assert index_paths["metadata"].exists()
            assert index_paths["index"].exists()
            assert index_paths["metadata"].read_text() == "{}"
            assert index_paths["index"].read_bytes() == b"index"

    def test_skips_existing_artifacts(self, bootstrap_service):
        """Should skip artifacts that already exist."""
        with TemporaryDirectory() as tmpdir:
            # Setup
            extract_dir = Path(tmpdir) / "extracted"
            extract_dir.mkdir()
            (extract_dir / "metadata.json").write_text("new")

            dest_dir = Path(tmpdir) / "dest"
            dest_dir.mkdir()
            index_paths = {"metadata": dest_dir / "metadata.json", "index": Path()}

            # Pre-create destination file
            index_paths["metadata"].write_text("old")

            # Execute
            bootstrap_service._move_artifacts(extract_dir, index_paths)

            # Verify - should keep old content
            assert index_paths["metadata"].read_text() == "old"

    def test_warns_on_missing_artifacts(self, bootstrap_service, caplog):
        """Should warn if expected artifacts not found in archive."""
        with TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir) / "extracted"
            extract_dir.mkdir()
            # Don't create any files

            dest_dir = Path(tmpdir) / "dest"
            index_paths = {
                "metadata": dest_dir / "metadata.json",
                "index": dest_dir / "index.faiss",
            }

            # Execute
            bootstrap_service._move_artifacts(extract_dir, index_paths)

            # Verify warnings logged
            assert "missing expected" in caplog.text.lower()
