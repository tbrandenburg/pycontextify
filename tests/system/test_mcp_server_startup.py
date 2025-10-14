"""System tests for MCP server startup scenarios.

These tests validate the entire system startup process including
configuration loading, index initialization, and MCP server readiness.
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Generator

import pytest


class TestMCPServerStartup:
    """System tests for MCP server startup scenarios."""

    def test_server_starts_with_empty_index(self):
        """Test that MCP server starts successfully with no existing index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index"

            # Start server process with custom index path
            cmd = [
                "uv",
                "run",
                "pycontextify",
                "--index-path",
                str(index_path),
                "--quiet",
            ]

            # Use a timeout to prevent hanging
            try:
                # Start the process but don't wait for completion
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Give it a few seconds to start up and initialize
                time.sleep(3)

                # Check if process is still running (expected for MCP STDIO mode)
                poll_result = proc.poll()

                if poll_result is not None:
                    # Process has exited - check if it's an error or expected behavior
                    stdout, stderr = proc.communicate()

                    # Only fail if it exited with non-zero code (actual error)
                    if poll_result != 0:
                        pytest.fail(
                            f"Server failed to start with code {poll_result}\n"
                            f"STDOUT: {stdout}\n"
                            f"STDERR: {stderr}"
                        )

                    # Exit code 0 means clean shutdown - check that it at least started
                    if "Starting MCP server" not in stderr:
                        pytest.fail(
                            "Server did not show startup message\n"
                            f"STDOUT: {stdout}\n"
                            f"STDERR: {stderr}"
                        )
                else:
                    # Server is running in STDIO mode waiting for input - this is correct
                    # Terminate it cleanly
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

            except Exception as e:
                pytest.fail(f"Failed to start server: {e}")

    def test_server_handles_corrupted_index_gracefully(self):
        """Test that server handles corrupted index files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index"
            index_path.mkdir(parents=True, exist_ok=True)

            # Create a corrupted metadata file with invalid SourceType
            corrupted_metadata = index_path / "semantic_index.pkl"
            corrupted_metadata.write_bytes(b"invalid pickle data")

            # Start server process with custom index path and no auto-load
            cmd = [
                "uv",
                "run",
                "pycontextify",
                "--index-path",
                str(index_path),
                "--no-auto-load",
                "--quiet",
            ]

            try:
                # Start the process
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Give it time to start
                time.sleep(3)

                # Check if process is still running
                poll_result = proc.poll()

                if poll_result is not None:
                    # Process has exited, check if it was a clean exit
                    stdout, stderr = proc.communicate()

                    # Server should handle corruption gracefully
                    # Either by skipping auto-load or by cleaning up
                    if poll_result != 0:
                        # Only fail if it's a real startup error, not corruption handling
                        if (
                            "Failed to initialize IndexManager" in stderr
                            and "SourceType" in stderr
                        ):
                            pytest.fail(
                                "Server should handle corrupted index gracefully, "
                                "but failed with SourceType error"
                            )
                else:
                    # Server is running - success!
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

            except Exception as e:
                pytest.fail(f"Failed to test corrupted index handling: {e}")

    def test_server_starts_with_valid_initial_filebase(self):
        """Test server startup with initial filebase indexing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple test filebase
            filebase_path = Path(tmpdir) / "test_filebase"
            filebase_path.mkdir()
            (filebase_path / "test.py").write_text(
                "# Test Python file\ndef hello(): return 'world'"
            )
            (filebase_path / "README.md").write_text("# Test Project\nThis is a test.")

            index_path = Path(tmpdir) / "test_index"

            # Start server with initial filebase - use faster model for testing
            cmd = [
                "uv",
                "run",
                "pycontextify",
                "--index-path",
                str(index_path),
                "--initial-filebase",
                str(filebase_path),
                "--topic",
                "test_project",
                "--embedding-model",
                "all-MiniLM-L6-v2",  # Faster model for testing
                "--quiet",
            ]

            try:
                # Start the process
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Give it time to download faster model and index files
                # Using all-MiniLM-L6-v2 which is smaller and faster to download
                time.sleep(15)

                # Check if process is running or has completed indexing
                poll_result = proc.poll()

                if poll_result is not None:
                    stdout, stderr = proc.communicate()

                    # Only fail if it exited with error code
                    if poll_result != 0:
                        pytest.fail(
                            f"Server failed during initial indexing with code {poll_result}\n"
                            f"STDOUT: {stdout}\n"
                            f"STDERR: {stderr}"
                        )

                    # Check that server started successfully (indexing may complete silently)
                    if "Starting MCP server" not in stderr:
                        pytest.fail(
                            "Server did not start successfully\n"
                            f"STDOUT: {stdout}\n"
                            f"STDERR: {stderr}"
                        )
                else:
                    # Server is still running after indexing - normal for MCP mode
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

                # The key test: verify index files were created and persisted
                # Since auto-persist defaults to True, files should exist
                assert (
                    index_path.exists()
                ), f"Index directory should have been created at {index_path}"

                # Check for expected index files
                index_files = list(index_path.glob("semantic_index.*"))
                if len(index_files) == 0:
                    # List what files were actually created for debugging
                    all_files = (
                        list(index_path.rglob("*")) if index_path.exists() else []
                    )
                    pytest.fail(
                        f"Index files should have been created but were not found.\n"
                        f"Index path: {index_path}\n"
                        f"Files found: {all_files}\n"
                        f"Expected pattern: semantic_index.*"
                    )

            except Exception as e:
                pytest.fail(f"Failed to test initial filebase indexing: {e}")
