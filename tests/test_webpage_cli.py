"""Tests for webpage CLI functionality and configuration overrides."""

import argparse
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args_with_custom_argv(argv):
    """Parse command-line arguments with custom argv for testing."""
    parser = argparse.ArgumentParser(
        prog="pycontextify",
        description="PyContextify MCP Server - Semantic search over codebases, documents, and webpages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Index configuration
    parser.add_argument("--index-path", type=str, help="Directory path for vector storage")
    parser.add_argument("--index-name", type=str, help="Custom index name")

    # Initial indexing
    parser.add_argument("--initial-documents", nargs="*", type=str, help="File paths to documents")
    parser.add_argument("--initial-codebase", nargs="*", type=str, help="Directory paths to codebases")
    parser.add_argument("--initial-webpages", nargs="*", type=str, help="URLs to webpages")

    # Webpage crawling options
    parser.add_argument("--recursive-crawling", action="store_true", help="Enable recursive crawling")
    parser.add_argument("--max-crawl-depth", type=int, default=1, help="Maximum crawl depth (1-3)")
    parser.add_argument("--crawl-delay", type=int, help="Delay between web requests")

    # Server configuration
    parser.add_argument("--no-auto-persist", action="store_true", help="Disable automatic persistence")
    parser.add_argument("--embedding-provider", choices=["sentence_transformers", "ollama", "openai"])

    return parser.parse_args(argv)


def args_to_config_overrides(args):
    """Convert CLI arguments to configuration overrides dictionary."""
    overrides = {}
    
    if args.index_path:
        overrides["index_dir"] = args.index_path
    if args.index_name:
        overrides["index_name"] = args.index_name
    if args.no_auto_persist:
        overrides["auto_persist"] = False
    if args.embedding_provider:
        overrides["embedding_provider"] = args.embedding_provider
    if hasattr(args, 'crawl_delay') and args.crawl_delay is not None:
        overrides["crawl_delay_seconds"] = args.crawl_delay
        
    return overrides


def simulate_webpage_indexing(args, mgr_mock):
    """Simulate the webpage indexing logic from perform_initial_indexing."""
    total_indexed = 0
    
    if args.initial_webpages:
        for webpage_url in args.initial_webpages:
            # Validate URL
            if not webpage_url.startswith(("http://", "https://")):
                continue
            
            # Apply crawling settings
            recursive = args.recursive_crawling if hasattr(args, 'recursive_crawling') else False
            max_depth = getattr(args, 'max_crawl_depth', 1)
            
            # Validate and limit max_depth
            if max_depth < 1 or max_depth > 3:
                max_depth = min(max(max_depth, 1), 3)
            
            # Mock the webpage indexing result
            mock_result = {
                "pages_processed": 2 if recursive else 1,
                "chunks_added": 10 if recursive else 5
            }
            mgr_mock.index_webpage.return_value = mock_result
            
            result = mgr_mock.index_webpage(webpage_url, recursive=recursive, max_depth=max_depth)
            total_indexed += result.get("chunks_added", 0)
    
    return total_indexed


class TestWebpageCLIArguments(unittest.TestCase):
    """Test webpage CLI argument parsing functionality."""

    def test_no_webpage_arguments(self):
        """Test parsing with no webpage arguments."""
        args = parse_args_with_custom_argv([])
        self.assertIsNone(args.initial_webpages)
        self.assertFalse(args.recursive_crawling)
        self.assertEqual(args.max_crawl_depth, 1)
        self.assertIsNone(args.crawl_delay)

    def test_single_webpage_argument(self):
        """Test single webpage URL."""
        args = parse_args_with_custom_argv([
            "--initial-webpages", "https://docs.python.org"
        ])
        self.assertEqual(args.initial_webpages, ["https://docs.python.org"])
        self.assertFalse(args.recursive_crawling)
        self.assertEqual(args.max_crawl_depth, 1)

    def test_multiple_webpage_arguments(self):
        """Test multiple webpage URLs."""
        args = parse_args_with_custom_argv([
            "--initial-webpages", 
            "https://docs.python.org", 
            "https://github.com/python/cpython",
            "https://pypi.org"
        ])
        expected_urls = [
            "https://docs.python.org", 
            "https://github.com/python/cpython",
            "https://pypi.org"
        ]
        self.assertEqual(args.initial_webpages, expected_urls)

    def test_recursive_crawling_flag(self):
        """Test recursive crawling flag."""
        args = parse_args_with_custom_argv([
            "--initial-webpages", "https://docs.example.com",
            "--recursive-crawling"
        ])
        self.assertTrue(args.recursive_crawling)
        self.assertEqual(args.max_crawl_depth, 1)  # Default value

    def test_max_crawl_depth_argument(self):
        """Test max crawl depth argument."""
        args = parse_args_with_custom_argv([
            "--initial-webpages", "https://docs.example.com",
            "--recursive-crawling",
            "--max-crawl-depth", "3"
        ])
        self.assertTrue(args.recursive_crawling)
        self.assertEqual(args.max_crawl_depth, 3)

    def test_crawl_delay_argument(self):
        """Test crawl delay argument."""
        args = parse_args_with_custom_argv([
            "--initial-webpages", "https://docs.example.com",
            "--crawl-delay", "5"
        ])
        self.assertEqual(args.crawl_delay, 5)

    def test_combined_webpage_arguments(self):
        """Test all webpage arguments together."""
        args = parse_args_with_custom_argv([
            "--initial-webpages", "https://docs.example.com", "https://api.example.com",
            "--recursive-crawling",
            "--max-crawl-depth", "2",
            "--crawl-delay", "3"
        ])
        self.assertEqual(args.initial_webpages, ["https://docs.example.com", "https://api.example.com"])
        self.assertTrue(args.recursive_crawling)
        self.assertEqual(args.max_crawl_depth, 2)
        self.assertEqual(args.crawl_delay, 3)

    def test_mixed_content_types(self):
        """Test webpage arguments mixed with other content types."""
        args = parse_args_with_custom_argv([
            "--initial-documents", "README.md", "docs/api.md",
            "--initial-codebase", "src", "tests",
            "--initial-webpages", "https://docs.example.com",
            "--recursive-crawling"
        ])
        self.assertEqual(args.initial_documents, ["README.md", "docs/api.md"])
        self.assertEqual(args.initial_codebase, ["src", "tests"])
        self.assertEqual(args.initial_webpages, ["https://docs.example.com"])
        self.assertTrue(args.recursive_crawling)


class TestWebpageConfigOverrides(unittest.TestCase):
    """Test webpage configuration override conversion."""

    def test_no_webpage_overrides(self):
        """Test no webpage overrides when no CLI args are provided."""
        args = parse_args_with_custom_argv([])
        overrides = args_to_config_overrides(args)
        self.assertNotIn("crawl_delay_seconds", overrides)

    def test_crawl_delay_override(self):
        """Test crawl delay configuration override."""
        args = parse_args_with_custom_argv([
            "--crawl-delay", "10"
        ])
        overrides = args_to_config_overrides(args)
        self.assertEqual(overrides["crawl_delay_seconds"], 10)

    def test_mixed_overrides_with_crawl_delay(self):
        """Test mixed configuration overrides including crawl delay."""
        args = parse_args_with_custom_argv([
            "--index-name", "web_search",
            "--crawl-delay", "2",
            "--no-auto-persist"
        ])
        overrides = args_to_config_overrides(args)
        
        expected = {
            "index_name": "web_search",
            "crawl_delay_seconds": 2,
            "auto_persist": False
        }
        self.assertEqual(overrides, expected)


class TestWebpageIndexingLogic(unittest.TestCase):
    """Test webpage indexing logic simulation."""

    def test_no_webpages_indexing(self):
        """Test no indexing when no webpages specified."""
        args = parse_args_with_custom_argv([])
        mgr_mock = MagicMock()
        
        total_indexed = simulate_webpage_indexing(args, mgr_mock)
        
        self.assertEqual(total_indexed, 0)
        mgr_mock.index_webpage.assert_not_called()

    def test_single_webpage_indexing(self):
        """Test single webpage indexing."""
        args = parse_args_with_custom_argv([
            "--initial-webpages", "https://docs.python.org"
        ])
        mgr_mock = MagicMock()
        
        total_indexed = simulate_webpage_indexing(args, mgr_mock)
        
        self.assertEqual(total_indexed, 5)  # Mock returns 5 chunks for non-recursive
        mgr_mock.index_webpage.assert_called_once_with(
            "https://docs.python.org", recursive=False, max_depth=1
        )

    def test_recursive_webpage_indexing(self):
        """Test recursive webpage indexing."""
        args = parse_args_with_custom_argv([
            "--initial-webpages", "https://docs.example.com",
            "--recursive-crawling",
            "--max-crawl-depth", "2"
        ])
        mgr_mock = MagicMock()
        
        total_indexed = simulate_webpage_indexing(args, mgr_mock)
        
        self.assertEqual(total_indexed, 10)  # Mock returns 10 chunks for recursive
        mgr_mock.index_webpage.assert_called_once_with(
            "https://docs.example.com", recursive=True, max_depth=2
        )

    def test_multiple_webpages_indexing(self):
        """Test indexing multiple webpages."""
        args = parse_args_with_custom_argv([
            "--initial-webpages", 
            "https://docs.python.org", 
            "https://github.com/python/cpython"
        ])
        mgr_mock = MagicMock()
        
        total_indexed = simulate_webpage_indexing(args, mgr_mock)
        
        self.assertEqual(total_indexed, 10)  # 2 webpages × 5 chunks each
        self.assertEqual(mgr_mock.index_webpage.call_count, 2)

    def test_invalid_url_filtering(self):
        """Test that invalid URLs are filtered out."""
        args = parse_args_with_custom_argv([
            "--initial-webpages", 
            "https://valid.com",
            "ftp://invalid.com",  # Invalid protocol
            "not-a-url",          # Not a URL
            "http://also-valid.com"
        ])
        mgr_mock = MagicMock()
        
        total_indexed = simulate_webpage_indexing(args, mgr_mock)
        
        self.assertEqual(total_indexed, 10)  # Only 2 valid URLs × 5 chunks each
        self.assertEqual(mgr_mock.index_webpage.call_count, 2)

    def test_max_depth_validation(self):
        """Test max depth validation and adjustment."""
        # Test depth too high
        args = parse_args_with_custom_argv([
            "--initial-webpages", "https://docs.example.com",
            "--recursive-crawling",
            "--max-crawl-depth", "10"  # Too high, should be limited to 3
        ])
        mgr_mock = MagicMock()
        
        simulate_webpage_indexing(args, mgr_mock)
        
        # Should be called with max_depth=3 (limited)
        mgr_mock.index_webpage.assert_called_once_with(
            "https://docs.example.com", recursive=True, max_depth=3
        )

    def test_max_depth_too_low(self):
        """Test max depth validation for values too low."""
        # Test depth too low
        args = parse_args_with_custom_argv([
            "--initial-webpages", "https://docs.example.com",
            "--recursive-crawling",
            "--max-crawl-depth", "0"  # Too low, should be adjusted to 1
        ])
        mgr_mock = MagicMock()
        
        simulate_webpage_indexing(args, mgr_mock)
        
        # Should be called with max_depth=1 (adjusted)
        mgr_mock.index_webpage.assert_called_once_with(
            "https://docs.example.com", recursive=True, max_depth=1
        )


if __name__ == "__main__":
    unittest.main()