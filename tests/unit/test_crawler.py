"""Unit tests for FileCrawler module."""

from pathlib import Path

import pytest

from pycontextify.crawler import FileCrawler


class TestFileCrawler:
    """Test FileCrawler functionality."""

    def test_crawler_basic_functionality(self, tmp_path):
        """Test basic file crawling."""
        # Create test structure
        (tmp_path / "file1.py").write_text("print('hello')")
        (tmp_path / "file2.txt").write_text("text content")
        (tmp_path / "file3.md").write_text("# Markdown")

        crawler = FileCrawler()
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 3
        assert all(isinstance(f, str) for f in files)
        # Should be sorted
        assert files == sorted(files)

    def test_crawler_single_file_path(self, tmp_path):
        """Test crawling when given a single file path."""
        file_path = tmp_path / "single.py"
        file_path.write_text("print('hi')")

        crawler = FileCrawler()
        files = crawler.crawl(str(file_path))

        assert files == [str(file_path.resolve())]

        # Respect include/exclude filters for single file
        include_crawler = FileCrawler(include=["*.txt"])
        assert include_crawler.crawl(str(file_path)) == []

        exclude_crawler = FileCrawler(exclude=["*.py"])
        assert exclude_crawler.crawl(str(file_path)) == []

    def test_crawler_include_pattern(self, tmp_path):
        """Test include pattern filtering."""
        (tmp_path / "file1.py").write_text("python code")
        (tmp_path / "file2.py").write_text("more python")
        (tmp_path / "file3.txt").write_text("text")
        (tmp_path / "file4.js").write_text("javascript")

        # Include only Python files
        crawler = FileCrawler(include=["*.py"])
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 2
        assert all(f.endswith(".py") for f in files)

    def test_crawler_exclude_pattern(self, tmp_path):
        """Test exclude pattern filtering."""
        (tmp_path / "file1.py").write_text("python code")
        (tmp_path / "file2.pyc").write_text("compiled")
        (tmp_path / "file3.txt").write_text("text")

        # Exclude compiled Python files
        crawler = FileCrawler(exclude=["*.pyc"])
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 2
        assert not any(f.endswith(".pyc") for f in files)

    def test_crawler_include_and_exclude(self, tmp_path):
        """Test include and exclude patterns together."""
        (tmp_path / "src.py").write_text("source")
        (tmp_path / "test.py").write_text("test")
        (tmp_path / "data.txt").write_text("data")

        # Include .py files but exclude test files
        crawler = FileCrawler(include=["*.py"], exclude=["test*"])
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 1
        assert files[0].endswith("src.py")

    def test_crawler_exclude_dirs(self, tmp_path):
        """Test directory exclusion."""
        # Create directory structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("main")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lib.js").write_text("lib")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test.py").write_text("test")

        # Exclude node_modules and tests
        crawler = FileCrawler(exclude_dirs=["node_modules", "tests"])
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 1
        assert "main.py" in files[0]
        assert "node_modules" not in files[0]
        assert "tests" not in files[0]

    def test_crawler_nested_directories(self, tmp_path):
        """Test crawling nested directory structure."""
        # Create nested structure
        (tmp_path / "level1").mkdir()
        (tmp_path / "level1" / "file1.py").write_text("l1")
        (tmp_path / "level1" / "level2").mkdir()
        (tmp_path / "level1" / "level2" / "file2.py").write_text("l2")
        (tmp_path / "level1" / "level2" / "level3").mkdir()
        (tmp_path / "level1" / "level2" / "level3" / "file3.py").write_text("l3")

        crawler = FileCrawler()
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 3
        # All nested files should be found
        assert any("file1.py" in f for f in files)
        assert any("file2.py" in f for f in files)
        assert any("file3.py" in f for f in files)

    def test_crawler_empty_directory(self, tmp_path):
        """Test crawling empty directory."""
        crawler = FileCrawler()
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 0
        assert files == []

    def test_crawler_no_matches(self, tmp_path):
        """Test when no files match the patterns."""
        (tmp_path / "file1.txt").write_text("text")
        (tmp_path / "file2.md").write_text("markdown")

        # Look for Python files only
        crawler = FileCrawler(include=["*.py"])
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 0

    def test_crawler_multiple_include_patterns(self, tmp_path):
        """Test multiple include patterns."""
        (tmp_path / "file1.py").write_text("python")
        (tmp_path / "file2.js").write_text("javascript")
        (tmp_path / "file3.txt").write_text("text")
        (tmp_path / "file4.md").write_text("markdown")

        # Include Python and JavaScript files
        crawler = FileCrawler(include=["*.py", "*.js"])
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 2
        assert any(f.endswith(".py") for f in files)
        assert any(f.endswith(".js") for f in files)

    def test_crawler_multiple_exclude_patterns(self, tmp_path):
        """Test multiple exclude patterns."""
        (tmp_path / "file1.py").write_text("python")
        (tmp_path / "file2.pyc").write_text("compiled")
        (tmp_path / "file3.log").write_text("log")
        (tmp_path / "file4.txt").write_text("text")

        # Exclude .pyc and .log files
        crawler = FileCrawler(exclude=["*.pyc", "*.log"])
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 2
        assert not any(f.endswith(".pyc") for f in files)
        assert not any(f.endswith(".log") for f in files)

    def test_crawler_sorting_behavior(self, tmp_path):
        """Test that results are sorted."""
        (tmp_path / "zebra.py").write_text("z")
        (tmp_path / "apple.py").write_text("a")
        (tmp_path / "banana.py").write_text("b")

        crawler = FileCrawler()
        files = crawler.crawl(str(tmp_path))

        # Should be sorted alphabetically
        assert files == sorted(files)
        assert "apple" in files[0]
        assert "banana" in files[1]
        assert "zebra" in files[2]

    def test_crawler_hidden_files(self, tmp_path):
        """Test behavior with hidden files (dot files)."""
        (tmp_path / "visible.py").write_text("visible")
        (tmp_path / ".hidden").write_text("hidden")

        crawler = FileCrawler()
        files = crawler.crawl(str(tmp_path))

        # Should include all files by default
        assert len(files) == 2

        # Can exclude hidden files
        crawler_no_hidden = FileCrawler(exclude=[".*"])
        files_no_hidden = crawler_no_hidden.crawl(str(tmp_path))
        assert len(files_no_hidden) == 1
        assert "visible.py" in files_no_hidden[0]

    def test_crawler_complex_patterns(self, tmp_path):
        """Test complex fnmatch patterns."""
        (tmp_path / "test_unit.py").write_text("test")
        (tmp_path / "test_integration.py").write_text("test")
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "utils.py").write_text("utils")

        # Include all .py but exclude test files
        crawler = FileCrawler(include=["*.py"], exclude=["test_*"])
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 2
        assert any("main.py" in f for f in files)
        assert any("utils.py" in f for f in files)
        # Check filename only, not full path (to avoid path separator issues)
        assert not any(Path(f).name.startswith("test_") for f in files)

    def test_crawler_absolute_paths(self, tmp_path):
        """Test that returned paths are absolute."""
        (tmp_path / "file.py").write_text("code")

        crawler = FileCrawler()
        files = crawler.crawl(str(tmp_path))

        assert len(files) == 1
        # Path should be absolute
        assert Path(files[0]).is_absolute()

    def test_crawler_nonexistent_path(self):
        """Test crawling nonexistent path."""
        crawler = FileCrawler()

        with pytest.raises((FileNotFoundError, OSError)):
            crawler.crawl("/nonexistent/path/to/directory")
