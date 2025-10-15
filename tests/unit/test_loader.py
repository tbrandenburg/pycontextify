"""Unit tests for FileLoaderFactory module."""

from datetime import datetime
from pathlib import Path

import pytest

from pycontextify.loader import FileLoaderFactory


class TestFileLoaderFactory:
    """Test FileLoaderFactory functionality."""

    def test_load_text_file(self, tmp_path):
        """Test loading a simple text file."""
        test_file = tmp_path / "test.txt"
        test_content = "This is a test file.\nWith multiple lines."
        test_file.write_text(test_content, encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test_tag")

        assert len(docs) == 1
        assert docs[0]["text"] == test_content
        assert "metadata" in docs[0]

        # Check required metadata fields
        meta = docs[0]["metadata"]
        assert meta["tags"] == ["test_tag"]
        assert meta["full_path"] == str(test_file.resolve())
        assert meta["filename_stem"] == "test"
        assert meta["file_extension"] == "txt"
        assert "date_loaded" in meta
        assert "loading_time_ms" in meta
        assert isinstance(meta["loading_time_ms"], float)

    def test_load_python_file(self, tmp_path):
        """Test loading a Python source file."""
        test_file = tmp_path / "script.py"
        test_content = "def hello():\n    return 'world'"
        test_file.write_text(test_content, encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="code")

        assert len(docs) == 1
        assert docs[0]["text"] == test_content
        assert docs[0]["metadata"]["file_extension"] == "py"
        assert docs[0]["metadata"]["tags"] == ["code"]

    def test_load_markdown_file(self, tmp_path):
        """Test loading a Markdown file."""
        test_file = tmp_path / "readme.md"
        test_content = "# Header\n\nSome content here."
        test_file.write_text(test_content, encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="docs")

        assert len(docs) == 1
        assert docs[0]["text"] == test_content
        assert docs[0]["metadata"]["file_extension"] == "md"

    def test_metadata_fields_complete(self, tmp_path):
        """Test that all required metadata fields are present."""
        test_file = tmp_path / "data.json"
        test_file.write_text('{"key": "value"}', encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="data")

        assert len(docs) == 1
        meta = docs[0]["metadata"]

        # Required metadata fields
        required_fields = [
            "mime_type",
            "full_path",
            "filename_stem",
            "file_extension",
            "loading_time_ms",
            "date_loaded",
            "tags",
        ]

        for field in required_fields:
            assert field in meta, f"Missing required field: {field}"

    def test_mime_type_detection(self, tmp_path):
        """Test MIME type detection for different file types."""
        # Text file
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("text", encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(txt_file), tags="test")
        assert (
            "text" in docs[0]["metadata"]["mime_type"]
            or docs[0]["metadata"]["mime_type"] is None
        )

        # Python file
        py_file = tmp_path / "file.py"
        py_file.write_text("print('hi')", encoding="utf-8")

        docs = loader.load(str(py_file), tags="test")
        # MIME type might be text/x-python or similar
        mime = docs[0]["metadata"]["mime_type"]
        assert mime is None or "text" in mime or "python" in mime

    def test_date_loaded_format(self, tmp_path):
        """Test that date_loaded is in ISO 8601 format."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content", encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test")

        date_str = docs[0]["metadata"]["date_loaded"]
        # Should be parseable as ISO 8601
        try:
            parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            assert parsed is not None
        except ValueError:
            pytest.fail(f"date_loaded is not valid ISO 8601: {date_str}")

    def test_loading_time_measurement(self, tmp_path):
        """Test that loading_time_ms is reasonable."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content", encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test")

        loading_time = docs[0]["metadata"]["loading_time_ms"]
        assert loading_time >= 0
        assert loading_time < 10000  # Should be less than 10 seconds

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        loader = FileLoaderFactory()

        with pytest.raises((FileNotFoundError, OSError)):
            loader.load("/nonexistent/file.txt", tags="test")

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test")

        # Should still return a doc with empty text
        assert len(docs) == 1
        assert docs[0]["text"] == ""
        assert docs[0]["metadata"]["tags"] == ["test"]

    def test_skip_binary_file(self, tmp_path):
        """Test that binary files are skipped."""
        test_file = tmp_path / "binary.bin"
        # Write binary content
        test_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test")

        # Binary files should be skipped (return empty list)
        assert len(docs) == 0

    def test_unicode_content(self, tmp_path):
        """Test loading file with Unicode content."""
        test_file = tmp_path / "unicode.txt"
        test_content = "Hello 世界 🌍 émojis and ñáéíóú"
        test_file.write_text(test_content, encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test")

        assert len(docs) == 1
        assert docs[0]["text"] == test_content

    def test_large_text_file(self, tmp_path):
        """Test loading a larger text file."""
        test_file = tmp_path / "large.txt"
        # Create a file with 10,000 lines
        content_lines = [f"Line {i}: Some content here" for i in range(10000)]
        test_content = "\n".join(content_lines)
        test_file.write_text(test_content, encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test")

        assert len(docs) == 1
        assert docs[0]["text"] == test_content
        assert len(docs[0]["text"]) > 100000

    def test_different_encoding(self, tmp_path):
        """Test loading file with default encoding."""
        test_file = tmp_path / "encoded.txt"
        test_content = "ASCII text content"
        test_file.write_text(test_content, encoding="utf-8")

        loader = FileLoaderFactory(default_encoding="utf-8")
        docs = loader.load(str(test_file), tags="test")

        assert len(docs) == 1
        assert docs[0]["text"] == test_content

    def test_relative_path_metadata(self, tmp_path):
        """Test that full_path is absolute."""
        test_file = tmp_path / "subdir" / "file.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("content", encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test")

        full_path = docs[0]["metadata"]["full_path"]
        assert Path(full_path).is_absolute()

    def test_filename_stem_extraction(self, tmp_path):
        """Test filename stem extraction."""
        test_file = tmp_path / "my_file.test.txt"
        test_file.write_text("content", encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test")

        # Path.stem returns everything before the last dot
        assert docs[0]["metadata"]["filename_stem"] == "my_file.test"

    def test_no_extension_file(self, tmp_path):
        """Test file without extension.

        Note: Files without extensions may be skipped if they don't have a
        recognized text/ MIME type. This is expected behavior.
        """
        test_file = tmp_path / "README"
        test_file.write_text("readme content", encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test")

        # File without extension might be skipped or loaded depending on MIME detection
        # If loaded, verify metadata
        if docs:
            assert docs[0]["metadata"]["file_extension"] == ""
            assert docs[0]["metadata"]["filename_stem"] == "README"
        # If not loaded (empty list), that's also acceptable

    def test_tag_propagation(self, tmp_path):
        """Test that tags are correctly propagated to metadata."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("content", encoding="utf-8")

        loader = FileLoaderFactory()

        # Test different tags
        docs1 = loader.load(str(test_file), tags="tag1")
        assert docs1[0]["metadata"]["tags"] == ["tag1"]

        docs2 = loader.load(str(test_file), tags="tag2")
        assert docs2[0]["metadata"]["tags"] == ["tag2"]

    def test_multiple_files_same_loader(self, tmp_path):
        """Test loading multiple files with the same loader instance."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("content1", encoding="utf-8")

        file2 = tmp_path / "file2.txt"
        file2.write_text("content2", encoding="utf-8")

        loader = FileLoaderFactory()

        docs1 = loader.load(str(file1), tags="test")
        docs2 = loader.load(str(file2), tags="test")

        assert len(docs1) == 1
        assert len(docs2) == 1
        assert docs1[0]["text"] == "content1"
        assert docs2[0]["text"] == "content2"

    def test_whitespace_only_file(self, tmp_path):
        """Test loading file with only whitespace."""
        test_file = tmp_path / "whitespace.txt"
        test_file.write_text("   \n\n\t\t\n   ", encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test")

        # Should still load the whitespace content
        assert len(docs) == 1
        assert docs[0]["text"] == "   \n\n\t\t\n   "

    def test_special_characters_in_path(self, tmp_path):
        """Test loading file with special characters in path."""
        special_dir = tmp_path / "dir with spaces"
        special_dir.mkdir()
        test_file = special_dir / "file-name_test.txt"
        test_file.write_text("content", encoding="utf-8")

        loader = FileLoaderFactory()
        docs = loader.load(str(test_file), tags="test")

        assert len(docs) == 1
        assert docs[0]["text"] == "content"
        assert "file-name_test" in docs[0]["metadata"]["full_path"]
