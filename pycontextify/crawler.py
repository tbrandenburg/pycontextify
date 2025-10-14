"""File tree crawler for PyContextify filebase indexing.

This module implements file discovery with wildcard filtering support.
"""

import fnmatch
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class FileCrawler:
    """Crawls a file tree with include/exclude filtering using fnmatch wildcards.

    This crawler walks a directory tree and applies fnmatch-based filters to
    discover files for indexing. It supports:
    - Include patterns (whitelist specific files)
    - Exclude patterns (blacklist specific files)
    - Exclude directories (skip entire directory subtrees)

    Examples:
        >>> crawler = FileCrawler(include=["*.py", "*.md"], exclude=["*_test.py"])
        >>> files = crawler.crawl("/path/to/project")
        >>> # Returns sorted list of matching .py and .md files, excluding tests

        >>> crawler = FileCrawler(exclude_dirs=["node_modules", ".git"])
        >>> files = crawler.crawl("/path/to/project")
        >>> # Returns all files except those in node_modules or .git directories
    """

    def __init__(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ):
        """Initialize crawler with filter patterns.

        Args:
            include: List of fnmatch patterns to include (whitelist). If None or empty,
                    all files are initially included.
            exclude: List of fnmatch patterns to exclude (blacklist). Applied after includes.
            exclude_dirs: List of directory name patterns to skip entirely.
        """
        self.include = include or []
        self.exclude = exclude or []
        self.exclude_dirs = exclude_dirs or []

        logger.debug(
            f"FileCrawler initialized: include={self.include}, "
            f"exclude={self.exclude}, exclude_dirs={self.exclude_dirs}"
        )

    def crawl(self, base_path: str) -> List[str]:
        """Crawl directory tree and return matching file paths.

        Args:
            base_path: Root directory to start crawling from

        Returns:
            Sorted list of absolute file paths matching the filters

        Raises:
            FileNotFoundError: If base_path does not exist
            NotADirectoryError: If base_path is not a directory
        """
        base = Path(base_path).resolve()

        if not base.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

        if not base.is_dir():
            raise NotADirectoryError(f"Base path is not a directory: {base_path}")

        logger.info(f"Starting crawl of: {base}")

        matched_files = []
        total_files = 0
        filtered_files = 0
        skipped_dirs = 0

        for root, dirs, files in os.walk(base, followlinks=False):
            # Prune excluded directories in-place
            original_dir_count = len(dirs)
            dirs[:] = [d for d in dirs if not self._should_exclude_dir(d)]
            skipped_dirs += original_dir_count - len(dirs)

            # Process files in current directory
            for filename in files:
                total_files += 1
                file_path = Path(root) / filename

                # Compute relative path for pattern matching (use POSIX style)
                try:
                    rel_path = file_path.relative_to(base).as_posix()
                except ValueError:
                    # Should not happen but handle gracefully
                    logger.warning(f"Could not compute relative path for {file_path}")
                    continue

                # Apply filters
                if self._should_include_file(rel_path):
                    matched_files.append(str(file_path.resolve()))
                else:
                    filtered_files += 1

        # Sort for stability
        matched_files.sort()

        logger.info(
            f"Crawl complete: {len(matched_files)} files matched, "
            f"{filtered_files} filtered, {skipped_dirs} dirs skipped, "
            f"{total_files} total files examined"
        )

        if logger.isEnabledFor(logging.DEBUG) and matched_files:
            sample_size = min(5, len(matched_files))
            logger.debug(f"Sample matched files: {matched_files[:sample_size]}")

        return matched_files

    def _should_exclude_dir(self, dirname: str) -> bool:
        """Check if directory should be excluded.

        Args:
            dirname: Directory name (not full path)

        Returns:
            True if directory matches any exclude_dirs pattern
        """
        if not self.exclude_dirs:
            return False

        for pattern in self.exclude_dirs:
            if fnmatch.fnmatch(dirname, pattern):
                logger.debug(f"Excluding directory: {dirname} (pattern: {pattern})")
                return True

        return False

    def _should_include_file(self, rel_path: str) -> bool:
        """Check if file should be included based on filters.

        Filtering logic:
        1. If include patterns exist, file must match at least one
        2. If exclude patterns exist, file must not match any
        3. Exclusions always override inclusions

        Args:
            rel_path: File path relative to base (POSIX-style)

        Returns:
            True if file passes all filters
        """
        # Step 1: Apply include filter (if specified)
        if self.include:
            included = False
            for pattern in self.include:
                if fnmatch.fnmatch(rel_path, pattern):
                    included = True
                    break

            if not included:
                logger.debug(f"File does not match include patterns: {rel_path}")
                return False

        # Step 2: Apply exclude filter (always checked, overrides includes)
        if self.exclude:
            for pattern in self.exclude:
                if fnmatch.fnmatch(rel_path, pattern):
                    logger.debug(
                        f"File matches exclude pattern: {rel_path} ({pattern})"
                    )
                    return False

        return True


__all__ = ["FileCrawler"]
