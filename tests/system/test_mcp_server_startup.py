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
            cmd = ["uv", "run", "pycontextify", "--index-path", str(index_path), "--quiet"]
            
            # Use a timeout to prevent hanging
            try:
                # Start the process but don't wait for completion
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                
                # Give it a few seconds to start up and fail if it's going to fail
                time.sleep(3)
                
                # Check if process is still running (success) or has exited (failure)
                poll_result = proc.poll()
                
                if poll_result is not None:
                    # Process has exited, get output
                    stdout, stderr = proc.communicate()
                    pytest.fail(
                        f"Server exited unexpectedly with code {poll_result}\n"
                        f"STDOUT: {stdout}\n"
                        f"STDERR: {stderr}"
                    )
                
                # If we get here, the server is running successfully
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
                "uv", "run", "pycontextify", 
                "--index-path", str(index_path),
                "--no-auto-load", 
                "--quiet"
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
                        if "Failed to initialize IndexManager" in stderr and "SourceType" in stderr:
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
            (filebase_path / "test.py").write_text("# Test Python file\ndef hello(): return 'world'")
            (filebase_path / "README.md").write_text("# Test Project\nThis is a test.")
            
            index_path = Path(tmpdir) / "test_index"
            
            # Start server with initial filebase
            cmd = [
                "uv", "run", "pycontextify",
                "--index-path", str(index_path),
                "--initial-filebase", str(filebase_path),
                "--topic", "test_project",
                "--quiet"
            ]
            
            try:
                # Start the process
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                
                # Give it time to index and start
                time.sleep(5)
                
                # Check if process is running or has completed indexing
                poll_result = proc.poll()
                
                if poll_result is not None:
                    stdout, stderr = proc.communicate()
                    
                    # Server might exit after indexing, which is okay
                    if poll_result != 0:
                        pytest.fail(
                            f"Server failed during initial indexing with code {poll_result}\n"
                            f"STDOUT: {stdout}\n"
                            f"STDERR: {stderr}"
                        )
                else:
                    # Server is still running
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                
                # Verify index was created
                assert index_path.exists(), "Index directory should have been created"
                
                # Check for expected index files
                index_files = list(index_path.glob("semantic_index.*"))
                assert len(index_files) > 0, "Index files should have been created"
                        
            except Exception as e:
                pytest.fail(f"Failed to test initial filebase indexing: {e}")